# ==============================================================================
# 파일: scripts/run_custom_dir_eval.py
# 역할: 커스텀 폴더( *.wav 와 같은 이름의 *.txt ) 전체 평가
#       - 긴 오디오는 슬라이딩 윈도우 전사 + 간단 스티칭
#       - parallel 디코더/γ/정규화/CER/결과 저장(summary.csv, 전사 텍스트)
# 사용 예:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.run_custom_dir_eval \
#   --data_dir data/insurerance_data \
#   --arch parallel --gamma 0.0 \
#   --base_model openai/whisper-large-v3-turbo \
#   --model_path saved/merged-parallel-ko-a0_1 \
#   --language ko --window_sec 30 --hop_sec 25
# ==============================================================================
import os, glob, time, csv
import argparse
from datetime import datetime

import numpy as np
import soundfile as sf
from tqdm import tqdm

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer

from safetensors.torch import safe_open, load_file

# 병렬 디코더 (있을 시)
try:
    from models.parallel_decoder import ParallelWhisperDecoderLayer
    HAS_PARALLEL = True
except Exception:
    HAS_PARALLEL = False


# -------------------- 파일/가중치 유틸 --------------------
def load_checkpoint_state_dict(model_dir: str):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        from collections import OrderedDict
        sd = OrderedDict()
        for shard in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        return sd
    st = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st):
        return load_file(st, device="cpu")
    pt = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu")
    raise FileNotFoundError(f"No weights found in: {model_dir}")


def sanitize_model_tag(s: str) -> str:
    if not s:
        return "base"
    if os.path.isdir(s):
        s = os.path.basename(os.path.normpath(s))
    return s.replace("/", "-")


# -------------------- 오디오/스티칭 --------------------
def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        n = int(round(len(wav) * target_sr / sr))
        x_old = np.linspace(0.0, 1.0, len(wav), dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, n, dtype=np.float64)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
        sr = target_sr
    else:
        wav = wav.astype(np.float32)
    return wav, sr


def longest_suffix_prefix(a: str, b: str, max_len=80) -> int:
    a_tail = a[-max_len:]
    for k in range(len(a_tail), 0, -1):
        if b.startswith(a_tail[-k:]):
            return k
    return 0


# -------------------- 모델 구성 --------------------
def build_model(base_model_id: str, arch: str, torch_dtype):
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
    )
    if arch == "parallel":
        if not HAS_PARALLEL:
            raise RuntimeError("Parallel decoder not found. Add models.parallel_decoder or use --arch original.")
        for i in range(model.config.decoder_layers):
            model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(model.config, layer_idx=i)
    return model


def retie_output_proj(model: WhisperForConditionalGeneration):
    # proj_out 또는 lm_head를 embed_tokens와 weight tie
    if hasattr(model, "proj_out"):
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
    elif hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.decoder.embed_tokens.weight


def set_decoder_gamma(model, gamma: float):
    for layer in model.model.decoder.layers:
        if hasattr(layer, "gamma"):
            layer.gamma.fill_(float(gamma))


# -------------------- 슬라이딩-윈도우 전사 --------------------
def transcribe_long(
    model, processor, wav: np.ndarray, sr: int,
    language: str, window_sec: float, hop_sec: float, prefix_keep_chars: int
):
    # 언어 프롬프트는 config에 세팅 (generate에 전달 X)
    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = forced

    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    assert 0 < hop <= win, "hop_sec must be >0 and <= window_sec"

    stops = max(1, len(wav) - win + 1)
    starts = list(range(0, stops, hop))
    # 꼬리 처리
    if starts and (len(wav) - starts[-1]) > int(0.5 * win) and (len(wav) - starts[-1]) < win:
        starts.append(len(wav) - win)

    pieces = []
    for start in starts:
        chunk = wav[start: start + win]
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
        inputs = inputs.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            ids = model.generate(inputs)
        txt = processor.batch_decode(ids, skip_special_tokens=True)[0]
        pieces.append(txt)

    # 간단 스티칭
    stitched = ""
    for seg in pieces:
        seg = seg.strip()
        if not seg:
            continue
        if not stitched:
            stitched = seg
            continue
        k = longest_suffix_prefix(stitched, seg, max_len=prefix_keep_chars)
        stitched = stitched + seg[k:]
    return stitched, pieces


# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser("Evaluate custom directory of wav/txt pairs (long-audio ready)")
    ap.add_argument("--data_dir", type=str, required=True, help="폴더 내 *.wav 와 동일 stem의 *.txt (정답)")
    ap.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    ap.add_argument("--model_path", type=str, default="", help="병합/파인튜닝 체크포인트 디렉토리")
    ap.add_argument("--arch", choices=["original", "parallel"], default="parallel")
    ap.add_argument("--gamma", type=float, default=0., help="디코더 병렬 전환 계수. 1.0=원본(순차), 0.0=완전 병렬. 미지정 시 체크포인트값 사용")

    ap.add_argument("--language", type=str, default="ko")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=225)

    ap.add_argument("--window_sec", type=float, default=30.0)
    ap.add_argument("--hop_sec", type=float, default=25.0)
    ap.add_argument("--prefix_keep_chars", type=int, default=80)

    args = ap.parse_args()

    # 성능 스위치
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 프로세서/모델
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = build_model(args.base_model, args.arch, torch_dtype)

    if args.model_path:
        sd = load_checkpoint_state_dict(args.model_path)
        incomp = model.load_state_dict(sd, strict=False)
        print(f"[load_state] missing={len(incomp.missing_keys)} unexpected={len(incomp.unexpected_keys)}")

    retie_output_proj(model)
    if args.gamma is not None and args.arch == "parallel":
        set_decoder_gamma(model, args.gamma)

    model.to(device=device, dtype=torch_dtype)
    model.eval()
    model.config.use_cache = True
    if model.generation_config is not None:
        model.generation_config.use_cache = True
        model.generation_config.num_beams = args.num_beams
        model.generation_config.max_new_tokens = args.max_new_tokens
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    normalizer = EnglishTextNormalizer() if args.language.lower().startswith("en") else BasicTextNormalizer()

    # 출력 디렉토리
    model_tag = sanitize_model_tag(args.model_path or args.base_model)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("results", f"custom_{model_tag}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 파일 나열
    wavs = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    assert wavs, f"No wav files in {args.data_dir}"

    rows = []
    for wav_path in tqdm(wavs, desc="files"):
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        txt_path = os.path.join(args.data_dir, stem + ".txt")
        has_ref = os.path.exists(txt_path)

        # 로드/전사
        t0 = time.time()
        wav, sr = load_audio(wav_path, target_sr=16000)
        stitched, pieces = transcribe_long(
            model, processor, wav, sr,
            language=args.language,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            prefix_keep_chars=args.prefix_keep_chars,
        )
        t1 = time.time()
        dur_sec = len(wav) / sr
        hyp_raw = stitched
        hyp_norm = normalizer(hyp_raw)

        # 저장
        with open(os.path.join(out_dir, f"{stem}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(hyp_raw + "\n")
        with open(os.path.join(out_dir, f"{stem}_norm.txt"), "w", encoding="utf-8") as f:
            f.write(hyp_norm + "\n")

        # CER
        cer = None
        if has_ref:
            ref = open(txt_path, "r", encoding="utf-8").read().strip()
            ref_norm = normalizer(ref)
            import jiwer
            cer = jiwer.cer([ref_norm], [hyp_norm])

        rows.append({
            "file": os.path.basename(wav_path),
            "duration_sec": f"{dur_sec:.2f}",
            "num_chunks": len(pieces),
            "time_sec": f"{(t1 - t0):.2f}",
            "CER": (f"{cer*100:.2f}%" if cer is not None else "")
        })

    # summary 저장 + 평균 CER
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "duration_sec", "num_chunks", "time_sec", "CER"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    cer_vals = []
    for r in rows:
        if r["CER"]:
            cer_vals.append(float(r["CER"].rstrip("%")))
    if cer_vals:
        print(f"\n[RESULT] Avg CER over {len(cer_vals)} files: {np.mean(cer_vals):.2f}%")
    print(f"[SAVED] outputs → {out_dir}  (summary.csv & *_raw.txt / *_norm.txt)")


if __name__ == "__main__":
    main()
