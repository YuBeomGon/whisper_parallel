#!/usr/bin/env bash
# =====================================================================================
# Script: run_train_zeroth.sh
# Role  : Whisper 병렬 디코더(γ-annealing) 학습 실행 스크립트
# Note  : CUDA 디바이스, 하이퍼파라미터, γ 스케줄, Freeze 옵션을 한 곳에서 관리
# Usage : ./run_train_zeroth.sh
# =====================================================================================


CUDA_VISIBLE_DEVICES=7 python -m scripts.run_training   --dataset zeroth_ko   --model_name openai/whisper-large-v3-turbo   --learning_rate 1e-4 --warmup_steps 1000   --max_steps 20000 --eval_steps 2000 --save_steps 2000   --gamma_start 1.0 --gamma_end 0.0 --gamma_start_frac 0.0 --gamma_end_frac 0.6
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_training   --dataset librispeech_en --model_name openai/whisper-large-v3-turbo   --learning_rate 1e-4 --warmup_steps 1000   --max_steps 100000 --eval_steps 2000 --save_steps 2000   --gamma_start 1.0 --gamma_end 0.0 --gamma_start_frac 0.0 --gamma_end_frac 0.6


set -euo pipefail

# -----------------------------
# [1] GPU 선택
#   - 쉼표로 구분해 여러 GPU를 노출 (예: "1,7")
#   - HuggingFace Trainer는 가시 GPU 수를 보고 자동 병렬화/데이터패럴렐을 사용
# -----------------------------
export CUDA_VISIBLE_DEVICES="1,7"

# -----------------------------
# [2] 데이터/모델 설정
#   - dataset: zeroth_ko | librispeech_en
#   - model_name: 사전학습 Whisper 체크포인트
# -----------------------------
DATASET="zeroth_ko"
MODEL_NAME="openai/whisper-large-v3-turbo"

# -----------------------------
# [3] 러닝 하이퍼파라미터
#   - learning_rate, warmup_steps: 1e-4 / 1000 권장
#   - max_steps: 총 스텝 수
#   - eval_steps/save_steps: 평가/저장 주기
#   - 배치는 스크립트 내부 기본값(8) 사용; 변경하려면 run_training.py 인자 사용
# -----------------------------
LR="1e-4"
WARMUP_STEPS="1000"
MAX_STEPS="20000"
EVAL_STEPS="2000"
SAVE_STEPS="2000"

# -----------------------------
# [4] γ-annealing 스케줄
#   - gamma_start: 1.0 → 원본(순차) 입력
#   - gamma_end  : 0.0 → 완전 병렬 입력
#   - gamma_start_frac / gamma_end_frac: 전환 구간(비율)
#     예) 0.0~0.4 구간에서 1.0 → 0.0로 변화
# -----------------------------
GAMMA_START="1.0"
GAMMA_END="0.0"
GAMMA_START_FRAC="0.0"
GAMMA_END_FRAC="0.4"

# -----------------------------
# [5] Freeze 옵션
#   - --freeze_cross_attn:
#       디코더 Cross-Attn(Wqkv/Wo) 고정 → 음성 정렬(align) 파괴 방지
#   - --freeze_cross_ln:
#       Cross-Attn 앞 LayerNorm 고정 → 초기 분포 보존
#   - --unfreeze_cross_ln_at_gamma 0.6:
#       γ가 0.6 이하로 떨어지면 LN만 해제(optimizer에 즉시 편입)
# -----------------------------
FREEZE_FLAGS="--freeze_cross_attn --freeze_cross_ln --unfreeze_cross_ln_at_gamma 0.6"

# -----------------------------
# [6] 로깅/출력 경로
#   - 타임스탬프 기반 디렉토리 생성
#   - 표준출력을 tee로 로그 파일에 저장
# -----------------------------
STAMP="$(date +'%Y%m%d_%H%M%S')"
OUTDIR="./runs/zeroth_${STAMP}"
LOGFILE="${OUTDIR}/train.log"
mkdir -p "${OUTDIR}"

# -----------------------------
# [7] 실행 커맨드 구성
#   - 필요 시 추가 인자:
#     --per_device_train_batch_size 8
#     --gradient_accumulation_steps 2
#     --early_stop_min_step  (기본: gamma_end_frac*max_steps 이후에 EarlyStop 활성)
# -----------------------------
CMD=(
  python -m scripts.run_training
  --dataset "${DATASET}"
  --model_name "${MODEL_NAME}"
  --learning_rate "${LR}"
  --warmup_steps "${WARMUP_STEPS}"
  --max_steps "${MAX_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --gamma_start "${GAMMA_START}"
  --gamma_end "${GAMMA_END}"
  --gamma_start_frac "${GAMMA_START_FRAC}"
  --gamma_end_frac "${GAMMA_END_FRAC}"
  ${FREEZE_FLAGS}
  --output_dir "${OUTDIR}"
)

# -----------------------------
# [8] 정보 출력 & 실행
# -----------------------------
echo "====================================================================================="
echo "[Whisper Parallel-Decoder Training]"
echo "  CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES}"
echo "  Dataset               : ${DATASET}"
echo "  Model                 : ${MODEL_NAME}"
echo "  LR / Warmup           : ${LR} / ${WARMUP_STEPS}"
echo "  Steps (max/eval/save) : ${MAX_STEPS} / ${EVAL_STEPS} / ${SAVE_STEPS}"
echo "  Gamma (start→end)     : ${GAMMA_START} → ${GAMMA_END}  (frac: ${GAMMA_START_FRAC}→${GAMMA_END_FRAC})"
echo "  Freeze flags          : ${FREEZE_FLAGS}"
echo "  Output dir            : ${OUTDIR}"
echo "  Log file              : ${LOGFILE}"
echo "====================================================================================="

# 커맨드 에코(가독성용)
printf "CMD: "; printf "%q " "${CMD[@]}"; echo

# 실행 (stdout/stderr를 로그로 저장)
"${CMD[@]}" 2>&1 | tee -a "${LOGFILE}"




# 병렬 디코더 LM 학습
CUDA_VISIBLE_DEVICES=1 python -m scripts.run_lm_training \
  --base_model openai/whisper-large-v3-turbo \
  --language ko \
  --text_path "data/text_corpus/insurance_terms.txt,data/text_corpus/zeroth_corpus.txt,data/text_corpus/zeroth_corpus.txt" \
  --output_dir ./whisper-decoder-lm-ko \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --lora \
  --use_kd --kd_alpha 0.5 --kd_temp 2.0 \
  --use_anchor --anchor_weight 5e-4 \
  --anchor_mask "self_attn.=1.0,fc=1.0,embed_tokens=0.2,proj_out=0.2" \
  --eval_ratio 0.1


  CUDA_VISIBLE_DEVICES=0 \
python -m scripts.run_lm_training \
  --base_model openai/whisper-large-v3-turbo \
  --language ko \
  --text_paths data/text_corpus/insurance_terms.txt \
  --dataset_mode packed \
  --max_length 448 \
  --use_lora \
  --output_dir ./lm_ko_insurance_lora_packed

