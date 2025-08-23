# ==============================================================================
# 파일: scripts/run_long_audio_eval.py
# 역할: 긴 오디오(10~20분)를 슬라이딩 윈도우로 나눠 순차 디코딩 후 스티칭
#       - faster-whisper 없이 HF Whisper 모델로 장문 오디오 평가
#       - 옵션: 병합모델/병렬디코더/γ 강제/빔서치/정규화/CER
#
# 사용법:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.run_long_audio_eval \
#   --audio_path /path/to/20min.wav \
#   --base_model openai/whisper-large-v3-turbo \
#   --model_path saved/merged-parallel-ko-a0_3 \
#   --arch parallel --gamma 0.0 \
#   --language ko --window_sec 30 --hop_sec 25
# ==============================================================================

import argparse
import os
import glob
import math
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datetime import datetime  # ← 추가

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer

from safetensors.torch import safe_open, load_file

# (선택) 병렬 디코더 필요 시 import
try:
    from models.parallel_decoder import ParallelWhisperDecoderLayer
    HAS_PARALLEL = True
except Exception:
    HAS_PARALLEL = False


# ------------------------------
# 파일 유틸
# ------------------------------
def load_checkpoint_state_dict(model_dir: str):
    """saved/xxx 디렉토리에서 safetensors/pt 가중치를 합쳐 로드"""
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


# ------------------------------
# 오디오/텍스트 유틸
# ------------------------------
def load_audio(path, target_sr=16000):
    """모노 변환 + 16kHz 리샘플(선형 보간)"""
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
    """a의 접미사와 b의 접두사가 가장 길게 겹치는 길이(아주 단순 스티칭용)"""
    a_tail = a[-max_len:]
    for k in range(len(a_tail), 0, -1):
        if b.startswith(a_tail[-k:]):
            return k
    return 0


def sanitize_model_tag(s: str) -> str:
    """
    파일명에 쓸 수 있도록 모델 식별자를 정리:
    - 디렉토리면 마지막 폴더명
    - 슬래시는 대시로 치환
    """
    if not s:
        return "base"
    if os.path.isdir(s):
        s = os.path.basename(os.path.normpath(s))
    return s.replace("/", "-")


# ------------------------------
# 모델 구성/보조
# ------------------------------
def build_model(base_model_id: str, arch: str, torch_dtype):
    """
    항상 base를 먼저 로드 → 필요 시 평행 레이어로 교체.
    (체크포인트는 나중에 strict=False로 로드)
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
    )
    if arch == "parallel":
        if not HAS_PARALLEL:
            raise RuntimeError(
                "Parallel decoder not found. Add models.parallel_decoder or use --arch original."
            )
        for i in range(model.config.decoder_layers):
            model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(
                model.config, layer_idx=i
            )
    return model


def retie_output_proj(model: WhisperForConditionalGeneration):
    """
    Whisper는 출력 프로젝션을 임베딩과 tie하는 게 일반적.
    (proj_out 또는 lm_head 어느 쪽이든 embed_tokens와 weight 공유 보장)
    """
    if hasattr(model, "proj_out"):
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
    elif hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.decoder.embed_tokens.weight


def set_decoder_gamma(model, gamma: float):
    """Parallel 디코더일 때만 의미 있는 γ를 강제 주입"""
    for layer in model.model.decoder.layers:
        if hasattr(layer, "gamma"):
            layer.gamma.fill_(float(gamma))


# ------------------------------
# 메인
# ------------------------------
def run():
    ap = argparse.ArgumentParser("Long-audio evaluation via sliding window")
    # 모델
    ap.add_argument("--audio_path", type=str, required=True, help="입력 WAV/FLAC 경로")
    ap.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    ap.add_argument("--model_path", type=str, default="", help="병합/파인튜닝 체크포인트 디렉토리")
    ap.add_argument("--arch", choices=["original", "parallel"], default="original")
    ap.add_argument("--gamma", type=float, default=None, help="parallel일 때 디코더 γ 고정값")

    # 디코딩
    ap.add_argument("--language", type=str, default="ko")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=225)

    # 윈도우
    ap.add_argument("--window_sec", type=float, default=30.0, help="분할 창 길이(초)")
    ap.add_argument("--hop_sec", type=float, default=25.0, help="슬라이딩 간격(초), window_sec보다 짧게")

    # 스티칭
    ap.add_argument("--prefix_keep_chars", type=int, default=80, help="문자 스티칭 시 접미/접두 비교 최대 길이")

    # 평가(CER)
    ap.add_argument("--ref_txt", type=str, default="", help="정답 텍스트 파일 경로(있으면 CER 출력)")

    args = ap.parse_args()

    # 디바이스/dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 1) 프로세서/모델 로드
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = build_model(args.base_model, args.arch, torch_dtype)

    # 2) (중요) 평행 레이어로 교체 후에 체크포인트 로드
    if args.model_path:
        sd = load_checkpoint_state_dict(args.model_path)
        incomp = model.load_state_dict(sd, strict=False)
        print(f"Missing keys: {len(incomp.missing_keys)} | Unexpected keys: {len(incomp.unexpected_keys)}")

    # 3) 출력 proj weight tying 보장 + γ 세팅
    retie_output_proj(model)
    if args.gamma is not None and args.arch == "parallel":
        set_decoder_gamma(model, args.gamma)

    # 4) 디코딩 설정
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    model.config.use_cache = True
    if model.generation_config is not None:
        model.generation_config.use_cache = True
        model.generation_config.num_beams = args.num_beams
        model.generation_config.max_new_tokens = args.max_new_tokens
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # 언어 프롬프트는 config에 설정 (generate 인자 X)
    forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = forced

    # 정규화기
    normalizer = EnglishTextNormalizer() if args.language.lower().startswith("en") else BasicTextNormalizer()

    # 5) 오디오 로드/분할
    wav, sr = load_audio(args.audio_path, target_sr=16000)
    win = int(args.window_sec * sr)
    hop = int(args.hop_sec * sr)
    assert 0 < hop <= win, "hop_sec must be > 0 and <= window_sec"

    # 슬라이싱 인덱스 생성 (마지막 꼬리 너무 짧으면 스킵)
    stops = max(1, len(wav) - win + 1)
    starts = list(range(0, stops, hop))
    if starts and (len(wav) - starts[-1]) > int(0.5 * win) and (len(wav) - starts[-1]) < win:
        # 꼬리가 절반 이상이면 마지막 창 하나 더 추가
        starts.append(len(wav) - win)

    # 6) 분할 디코딩
    pieces = []
    for start in tqdm(starts, desc="chunks"):
        chunk = wav[start: start + win]
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
        inputs = inputs.to(device=device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            ids = model.generate(inputs)  # forced ids는 config에 이미 반영됨
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        pieces.append(hyp)

    # 7) 스티칭(중복 제거)
    stitched = ""
    for seg in pieces:
        seg = seg.strip()
        if not seg:
            continue
        if not stitched:
            stitched = seg
            continue
        k = longest_suffix_prefix(stitched, seg, max_len=args.prefix_keep_chars)
        stitched = stitched + seg[k:]

    hyp_norm = normalizer(stitched)

    print("=" * 60)
    print("TRANSCRIPT (stitched)")
    print(stitched)
    print("=" * 60)
    print("TRANSCRIPT (normalized for CER/WER)")
    print(hyp_norm)

    # 8) CER(optional)
    if args.ref_txt and os.path.exists(args.ref_txt):
        with open(args.ref_txt, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        ref_norm = normalizer(ref)
        import jiwer
        cer = jiwer.cer([ref_norm], [hyp_norm])
        print("=" * 60)
        print(f"CER against reference: {cer*100:.2f}%")
        print("=" * 60)

    # 9) 결과 저장 (정규화 텍스트)
    os.makedirs("results", exist_ok=True)
    model_tag = sanitize_model_tag(args.model_path or args.base_model)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join("results", f"{model_tag}_{timestamp}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(hyp_norm + "\n")
    print(f"[SAVED] normalized transcript → {out_path}")


if __name__ == "__main__":
    run()
