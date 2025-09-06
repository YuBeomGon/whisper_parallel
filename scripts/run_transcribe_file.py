# ==============================================================================
# 파일: scripts/run_transcribe_file.py
# 역할: 단일 오디오 파일 전사(긴 오디오 슬라이딩, parallel/γ 지원)
# 사용 예:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.run_transcribe_file \
#   --audio_path data/insurerance_data/insurerance_1.wav \
#   --ref_txt   data/insurerance_data/insurerance_1.txt \
#   --arch parallel --gamma 0.0 \
#   --base_model openai/whisper-large-v3-turbo \
#   --model_path saved/merged-parallel-ko-a0_1 \
#   --language ko
# ==============================================================================
import os, time, glob
import argparse
from datetime import datetime

import numpy as np
import soundfile as sf

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer

from safetensors.torch import safe_open, load_file

try:
    from models.parallel_decoder import ParallelWhisperDecoderLayer
    HAS_PARALLEL = True
except Exception:
    HAS_PARALLEL = False


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


def longest_suffix_prefix(a: str, b: str, max_len=80):
    a_tail = a[-max_len:]
    for k in range(len(a_tail), 0, -1):
        if b.startswith(a_tail[-k:]):
            return k
    return 0


def build_model(base_model_id: str, arch: str, torch_dtype):
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
    )
    if arch == "parallel":
        if not HAS_PARALLEL:
            raise RuntimeError("Parallel decoder not found. Add models.parallel_decoder or use --arch original.")
        for i in range(model.config.decoder_layers):
            vanilla = model.model.decoder.layers[i]
            model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(vanilla, layer_idx=i)
    return model


def retie_output_proj(model: WhisperForConditionalGeneration):
    if hasattr(model, "proj_out"):
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
    elif hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.decoder.embed_tokens.weight


def set_decoder_gamma(model, gamma: float):
    for layer in model.model.decoder.layers:
        if hasattr(layer, "gamma"):
            layer.gamma.fill_(float(gamma))


def main():
    ap = argparse.ArgumentParser("Single long-audio transcription (sliding window)")
    ap.add_argument("--audio_path", type=str, required=True)
    ap.add_argument("--ref_txt", type=str, default="")
    ap.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--arch", choices=["original", "parallel"], default="parallel")
    ap.add_argument("--gamma", type=float, default=0., help="디코더 병렬 전환 계수. 1.0=원본(순차), 0.0=완전 병렬. 미지정 시 체크포인트값 사용")

    ap.add_argument("--language", type=str, default="ko")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=225)

    ap.add_argument("--window_sec", type=float, default=30.0)
    ap.add_argument("--hop_sec", type=float, default=25.0)
    ap.add_argument("--prefix_keep_chars", type=int, default=80)

    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

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

    # 언어 프롬프트를 config에 반영
    forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = forced

    # 로드/슬라이스/전사
    wav, sr = load_audio(args.audio_path, target_sr=16000)
    win = int(args.window_sec * sr)
    hop = int(args.hop_sec * sr)
    assert 0 < hop <= win, "hop_sec must be >0 and <= window_sec"

    stops = max(1, len(wav) - win + 1)
    starts = list(range(0, stops, hop))
    if starts and (len(wav) - starts[-1]) > int(0.5 * win) and (len(wav) - starts[-1]) < win:
        starts.append(len(wav) - win)

    pieces = []
    for start in starts:
        chunk = wav[start: start + win]
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
        inputs = inputs.to(device=device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            ids = model.generate(inputs)
        txt = processor.batch_decode(ids, skip_special_tokens=True)[0]
        pieces.append(txt)

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

    hyp_raw = stitched
    hyp_norm = normalizer(hyp_raw)

    print("=" * 60)
    print("TRANSCRIPT (stitched)")
    print(hyp_raw)
    print("=" * 60)
    print("TRANSCRIPT (normalized)")
    print(hyp_norm)

    # 저장
    model_tag = sanitize_model_tag(args.model_path or args.base_model)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = os.path.splitext(os.path.basename(args.audio_path))[0]
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{model_tag}_{timestamp}_{stem}_raw.txt"), "w", encoding="utf-8") as f:
        f.write(hyp_raw + "\n")
    with open(os.path.join("results", f"{model_tag}_{timestamp}_{stem}_norm.txt"), "w", encoding="utf-8") as f:
        f.write(hyp_norm + "\n")
    print(f"[SAVED] results → results/{model_tag}_{timestamp}_{stem}_*.txt")

    # CER
    if args.ref_txt and os.path.exists(args.ref_txt):
        ref = open(args.ref_txt, "r", encoding="utf-8").read().strip()
        ref_norm = normalizer(ref)
        import jiwer
        cer = jiwer.cer([ref_norm], [hyp_norm])
        print("=" * 60)
        print(f"CER: {cer*100:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()
