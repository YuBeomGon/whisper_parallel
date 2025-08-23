# ==============================================================================
# 파일: scripts/merge_lm_back_to_asr.py
# 역할: 텍스트 LM(WhisperDecoderLM) 가중치를 Parallel ASR 모델에 보간(interpolate) 병합
#       - ASR: ParallelWhisperDecoderLayer로 아키텍처 교체 후, 학습 체크포인트 로드
#       - LM : LoRA 어댑터면 merge_and_unload, 아니면 그대로 state_dict 로드
#       - 지정된 하위 모듈만 alpha로 보간(교차어텐션은 제외)
# 사용법:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.merge_lm_back_to_asr \
#   --asr_dir saved/whisper-parallel-zeroth_ko/checkpoint-78000 \
#   --lm_dir  saved/whisper-decoder-lm-ko \
#   --base_model openai/whisper-large-v3-turbo \
#   --alpha 0.3 \
#   --include_self_attn --include_ffn --include_ln \
#   --gamma_after 0.0 \
#   --out_dir saved/merged-parallel-ko-a0_3
# ==============================================================================

import os, glob, argparse
from typing import Dict, Tuple
import torch
from safetensors.torch import safe_open, load_file

from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Parallel 디코더
from models.parallel_decoder import ParallelWhisperDecoderLayer
# LM 래퍼 (학습 때 썼던 것)
from models.whisper_decoder_lm import WhisperDecoderLM

# (선택) LoRA 병합
try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


# ---------- 유틸: safetensors/binary state dict 로드 ----------
def load_checkpoint_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        from collections import OrderedDict
        sd = OrderedDict()
        for shard in sorted(glob.glob(os.path.join(model_path, "model-*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        return sd
    st = os.path.join(model_path, "model.safetensors")
    if os.path.exists(st):
        return load_file(st, device="cpu")
    pt = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu")
    raise FileNotFoundError(f"No weights in: {model_path}")


# ---------- ASR(Parallel) 모델 구성 ----------
def build_parallel_asr(asr_ckpt_dir: str, base_model_id: str, torch_dtype) -> WhisperForConditionalGeneration:
    """
    1) 베이스 Whisper 로드
    2) 디코더 레이어를 ParallelWhisperDecoderLayer로 교체
    3) 학습된 ASR 체크포인트 가중치 로드(strict=False)
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
    )
    # 레이어 교체
    for i in range(model.config.decoder_layers):
        model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(model.config, layer_idx=i)

    # 체크포인트 가중치 덮어쓰기
    sd = load_checkpoint_state_dict(asr_ckpt_dir)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ASR] load_state: missing={len(missing)} unexpected={len(unexpected)}")
    return model


# ---------- LM(학생) 가중치 가져오기 ----------
def load_student_decoder_state(
    lm_dir: str,
    base_model_id: str,
    torch_dtype,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    WhisperDecoderLM을 재구성하고:
      - LoRA 어댑터면 PeftModel.from_pretrained(...).merge_and_unload()
      - 아니면 lm_dir의 state_dict를 그대로 로드
    반환:
      dec_sd: 'model.decoder.*' 네임스페이스로 맞춘 디코더 파라미터 dict
      proj_sd: { 'proj_out.weight' 또는 'lm_head.weight': tensor }
    """
    base = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch_dtype, attn_implementation="sdpa"
    )
    student = WhisperDecoderLM(base).to(device)

    # 우선 LoRA 어댑터 시도
    dec_model = None
    if HAS_PEFT:
        try:
            dec_model = PeftModel.from_pretrained(student, lm_dir, device_map=None)
            dec_model = dec_model.merge_and_unload()
            print("[LM] Loaded PEFT adapter and merged into base.")
        except Exception:
            dec_model = None

    if dec_model is None:
        # 일반 state dict (학습 시 전체 저장한 경우)
        sd = load_checkpoint_state_dict(lm_dir)
        missing, unexpected = student.load_state_dict(sd, strict=False)
        print(f"[LM] loaded raw state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        dec_model = student

    # state dict 정리: decoder.* → model.decoder.*, output_proj → proj_out/lm_head
    raw = dec_model.state_dict()

    # 디코더 파트
    dec_sd = {}
    for k, v in raw.items():
        if k.startswith("decoder."):
            dec_sd["model." + k] = v  # model.decoder....
    # 출력 프로젝션 (우리 래퍼는 output_proj* 이름)
    proj_sd = {}
    for name in ("output_proj.weight", "proj_out.weight", "lm_head.weight"):
        if name in raw:
            # WhisperForConditionalGeneration은 대개 proj_out.weight
            proj_sd[name] = raw[name]
            break

    return dec_sd, proj_sd


# ---------- 보간 머지 ----------
def should_keep(name: str,
                include_self_attn: bool,
                include_ffn: bool,
                include_ln: bool,
                include_embeddings: bool,
                include_output_proj: bool) -> bool:
    if "encoder_attn" in name or "cross_attn" in name:   # cross는 항상 제외
        return False
    if "embed_tokens" in name or "embed_positions" in name:
        return include_embeddings
    if "layer_norm" in name:
        return include_ln
    if "self_attn" in name:
        return include_self_attn
    if (".fc1." in name) or (".fc2." in name):
        return include_ffn
    if name in ("proj_out.weight", "lm_head.weight"):
        return include_output_proj
    # 그 외 잔여 파라미터는 보통 디코더 내부 공용 파트 → 기본은 제외
    return False


def interpolate_into_asr(
    asr_model: WhisperForConditionalGeneration,
    student_dec_sd: Dict[str, torch.Tensor],
    student_proj_sd: Dict[str, torch.Tensor],
    alpha: float,
    include_self_attn: bool,
    include_ffn: bool,
    include_ln: bool,
    include_embeddings: bool,
    include_output_proj: bool,
) -> None:
    """
    asr_model 파라미터를 in-place로 업데이트
    dst <- (1-α)*dst + α*src
    """
    asr_params = dict(asr_model.named_parameters())
    merged_cnt = 0

    with torch.no_grad():
        # 디코더 파라미터
        for n, src in student_dec_sd.items():
            if n not in asr_params:
                continue
            if not should_keep(n, include_self_attn, include_ffn, include_ln,
                               include_embeddings, include_output_proj):
                continue
            dst = asr_params[n]
            if dst.shape != src.shape:
                continue
            merged = ((1.0 - alpha) * dst.float() + alpha * src.float()).to(dst.dtype)
            dst.copy_(merged)
            merged_cnt += 1

        # 출력 proj
        # 학생 키가 output_proj.weight/ proj_out.weight / lm_head.weight 중 하나일 수 있음
        if include_output_proj:
            # ASR 쪽 실제 키 찾기
            asr_key = "proj_out.weight" if "proj_out" in asr_params else ("lm_head.weight" if "lm_head.weight" in asr_params else None)
            if asr_key:
                # 학생 sd에서 가중치 하나 골라오기
                src = None
                for cand in ("proj_out.weight", "lm_head.weight", "output_proj.weight"):
                    if cand in student_proj_sd:
                        src = student_proj_sd[cand]
                        break
                if src is not None:
                    dst = asr_params[asr_key]
                    merged = ((1.0 - alpha) * dst.float() + alpha * src.float()).to(dst.dtype)
                    dst.copy_(merged)
                    merged_cnt += 1

    print(f"[MERGE] parameters interpolated: {merged_cnt}")


def set_parallel_gamma(model, gamma: float):
    for layer in model.model.decoder.layers:
        if hasattr(layer, "gamma"):
            layer.gamma.fill_(float(gamma))


def main():
    ap = argparse.ArgumentParser("Merge LM (text) into Parallel ASR via interpolation")
    ap.add_argument("--asr_dir", type=str, required=True, help="학습된 Parallel ASR 체크포인트 디렉토리 (saved/whisper-parallel-.../checkpoint-XXXXX)")
    ap.add_argument("--lm_dir", type=str, required=True, help="텍스트 LM 체크포인트 디렉토리 (saved/whisper-decoder-lm-...)")
    ap.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo", help="Whisper 베이스 ID")
    ap.add_argument("--alpha", type=float, default=0.3, help="보간 계수 α (0:ASR 그대로, 1:LM 그대로)")
    ap.add_argument("--include_self_attn", action="store_true")
    ap.add_argument("--include_ffn", action="store_true")
    ap.add_argument("--include_ln", action="store_true")
    ap.add_argument("--include_embeddings", action="store_true")
    ap.add_argument("--include_output_proj", action="store_true")
    ap.add_argument("--gamma_after", type=float, default=0.0, help="병합 후 Parallel γ 고정값 (완전 병렬=0.0)")
    ap.add_argument("--out_dir", type=str, required=True, help="병합 결과 저장 디렉토리")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 1) Parallel ASR 로드
    asr = build_parallel_asr(args.asr_dir, args.base_model, torch_dtype=torch_dtype)
    asr.to(device)

    # 2) 학생(LM) 디코더 가중치
    dec_sd, proj_sd = load_student_decoder_state(
        args.lm_dir, args.base_model, torch_dtype=torch_dtype, device=device
    )

    # 3) 보간 병합
    interpolate_into_asr(
        asr,
        student_dec_sd=dec_sd,
        student_proj_sd=proj_sd,
        alpha=args.alpha,
        include_self_attn=args.include_self_attn,
        include_ffn=args.include_ffn,
        include_ln=args.include_ln,
        include_embeddings=args.include_embeddings,
        include_output_proj=args.include_output_proj,
    )

    # 4) γ 고정 (항상 parallel로 평가)
    set_parallel_gamma(asr, args.gamma_after)

    # 5) 저장
    os.makedirs(args.out_dir, exist_ok=True)
    asr.save_pretrained(args.out_dir, safe_serialization=True)
    # 편의상 processor도 같이 복사
    try:
        proc = WhisperProcessor.from_pretrained(args.base_model)
        proc.save_pretrained(args.out_dir)
    except Exception:
        pass

    print(f"[DONE] merged model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
