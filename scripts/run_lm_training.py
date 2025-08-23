# ==============================================================================
# 파일: scripts/run_lm_training.py
# 역할: Whisper 디코더 LM(A안) 텍스트 학습
#       - 베이스 whisper 모델에서 디코더만 래핑하여 LM 학습 (Cross-Attn/Encoder 미사용)
#       - 한 줄=한 문장 텍스트 파일(여러 개 쉼표로 연결) 바로 토크나이즈
#       - 옵션: LoRA, 지식증류(KD), 앵커 정규화(L2-SP), 일부 모듈 동결
# 설명
# ASR(Whisper)의 디코더만 떼어 “GPT처럼” 텍스트 전용 Causal LM으로 미세학습 → 나중에 ASR 풀모델로 다시 이식해서 도메인 텍스트 적응 효과를 가져오자.
# 왜 디코더만?
# Whisper의 토크나이저/사전/포지션 임베딩/LN 통계를 그대로 활용해야 ASR과 합칠 때 충돌이 적고 효과가 바로 납니다.
# GPT 계열 새 모델로 가면 토큰 체계/포지션/LN이 달라져 이식 비용↑\
# 학습 원칙(지금 코드 설계의 핵심)
# Cross-Attn/Encoder 완전 제거: 디코더를 decode-only로 사용 (LM 학습에 오디오가 개입하지 않음).
# Whisper 디코더의 길이 제약 유지: max_length≈448 권장(Whisper의 pos-embed 학습 범위).
# **언어/태스크 프롬프트(prefix)**를 매 샘플 앞에 부착 → 디코더 입력 분포 보존.
# 가중치 변화 제한:
# LoRA(q,k,v,out,fc1,fc2)로 소수 파라미터만 학습
# KD(teacher=원본 디코더)로 student 로짓을 부드럽게 견인
# **앵커 정규화(L2-SP)**로 원본 파라미터에서 너무 멀어지지 않도록 유도
# 이식 용이성: 훈련 후 디코더와 출력층만 ASR 모델에 덮어쓰기(or LoRA merge)
# corpus가 매우 적기 때문에 ASR훈련에 사용된 zeroth text data를 lm 훈련에 사용 (일종의 continual learning)
# 학습 예시:
# CUDA_VISIBLE_DEVICES=1 python -m scripts.run_lm_training \
#   --base_model openai/whisper-large-v3-turbo \
#   --language ko \
#   --text_path "data/text_corpus/insurance_terms.txt,data/text_corpus/zeroth_corpus.txt,data/text_corpus/zeroth_corpus.txt" \
#   --output_dir ./whisper-decoder-lm-ko \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --lora \
#   --use_kd --kd_alpha 0.5 --kd_temp 2.0 \
#   --use_anchor --anchor_weight 5e-4 \
#   --anchor_mask "self_attn.=1.0,fc=1.0,embed_tokens=0.2,proj_out=0.2" \
#   --eval_ratio 0.1
# ==============================================================================

import os
import argparse
from copy import deepcopy
from typing import Dict

import torch
from torch import nn
from datasets import load_dataset

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments,
)

# 모델/콜레이터
from models.whisper_decoder_lm import WhisperDecoderLM
from utils.lm_collator import DataCollatorWhisperLM

# (선택) LoRA
try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


# ------------------------------
# KD + 앵커 손실을 Trainer에 주입
# ------------------------------
class KDAnchoredTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher=None,
        kd_alpha: float = 0.5,
        kd_temp: float = 2.0,
        use_kd: bool = False,
        use_anchor: bool = False,
        anchor_weight: float = 0.0,
        anchor_mask: dict | None = None,
        anchor_ref_state: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval() if teacher is not None else None
        for p in (self.teacher.parameters() if self.teacher is not None else []):
            p.requires_grad_(False)

        self.use_kd = use_kd
        self.kd_alpha = kd_alpha
        self.kd_temp = kd_temp

        self.use_anchor = use_anchor
        self.anchor_weight = anchor_weight
        self.anchor_mask = anchor_mask or {}       # 예: {"self_attn.":1.0, "fc":1.0, "embed_tokens":0.2, "proj_out":0.2}
        self.anchor_ref_state = anchor_ref_state or {}  # {param_name: tensor(고정 스냅샷)}

    def _anchor_coef(self, name: str) -> float:
        # 부분 문자열 매칭(간단/빠름): anchor_mask의 key가 name에 포함되면 coef로 사용
        for k, v in self.anchor_mask.items():
            if k in name:
                return float(v)
        return 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        HF Trainer가 넘기는 추가 인자(num_items_in_batch 등)를 **kwargs로 흡수해야 합니다.
        """
        labels = inputs.get("labels", None)

        # 1) 학생(LoRA 적용된 디코더 LM) 기본 loss
        outputs = model(**inputs)
        # 모델이 loss를 반환하지 않는 경우 대비(일반적으로는 반환함)
        student_loss = outputs.get("loss", None)
        if student_loss is None and labels is not None:
            # label_smoother는 Trainer가 가지고 있음
            student_loss = self.label_smoother(outputs, labels)
        loss = student_loss

        # 2) KD(선택)
        if self.use_kd and self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    labels=None,
                )
                t_logits = t_out["logits"]
            s_logits = outputs["logits"]

            T = self.kd_temp
            kd = torch.nn.functional.kl_div(
                torch.log_softmax(s_logits / T, dim=-1),
                torch.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            loss = (1.0 - self.kd_alpha) * loss + self.kd_alpha * kd

        # 3) 앵커(선택): 파라미터 편차 L2
        if self.use_anchor and self.anchor_weight > 0.0 and self.anchor_ref_state:
            reg = 0.0
            for n, p in model.named_parameters():
                coef = self._anchor_coef(n)
                if coef == 0.0 or n not in self.anchor_ref_state:
                    continue
                ref = self.anchor_ref_state[n].to(p.device, dtype=p.dtype)
                reg = reg + coef * torch.sum((p - ref) ** 2)
            loss = loss + self.anchor_weight * reg

        return (loss, outputs) if return_outputs else loss


def parse_args():
    p = argparse.ArgumentParser("Whisper Decoder LM training (text-only, KD+Anchor)")

    # 모델/언어
    p.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    p.add_argument("--language", type=str, default="ko", help="ko|en|…")

    # 데이터
    p.add_argument("--text_path", type=str, required=True,
                   help="텍스트 코퍼스 경로(한 줄=한 문장). 여러 개면 쉼표로 구분")
    p.add_argument("--max_length", type=int, default=448,
                   help="Whisper 디코더 포지션 한계(≈448) 내 권장")

    # 학습/스케줄
    p.add_argument("--output_dir", type=str, default="./whisper-decoder-lm")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)   # ← 소형/보수적
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_clip", type=float, default=1.0)

    # 동결
    p.add_argument("--freeze_embeddings", action="store_true",
                   help="임베딩/포지션 임베딩 동결")
    p.add_argument("--freeze_norms", action="store_true",
                   help="LayerNorm 동결(초기 안정화)")

    # LoRA
    p.add_argument("--lora", action="store_true", help="LoRA 적용")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # KD
    p.add_argument("--use_kd", action="store_true", help="지식증류 활성화")
    p.add_argument("--kd_alpha", type=float, default=0.5, help="KD 가중치 α")
    p.add_argument("--kd_temp", type=float, default=2.0, help="KD 온도 T")

    # 앵커(L2-SP)
    p.add_argument("--use_anchor", action="store_true", help="앵커 정규화 활성화")
    p.add_argument("--anchor_weight", type=float, default=5e-4,
                   help="앵커 전역 계수 λ")
    # 모듈별 가중치(쉼표로): "self_attn.=1.0,fc=1.0,embed_tokens=0.1,lm_head=0.1"
    p.add_argument("--anchor_mask", type=str,
                   default="self_attn.=1.0,fc=1.0,embed_tokens=0.2,proj_out=0.2")
    
    p.add_argument("--eval_ratio", type=float, default=0.1,
                   help="train에서 eval로 떼어낼 비율 (0~1). 예: 0.1 → 10%를 eval로 사용")    

    return p.parse_args()


def parse_anchor_mask(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        k, v = token.split("=")
        out[k.strip()] = float(v)
    return out


def run():
    args = parse_args()

    # 성능 스위치
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8

    # 1) 베이스 Whisper + Processor
    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="sdpa",
    )
    processor = WhisperProcessor.from_pretrained(args.base_model)
    tok = processor.tokenizer
    tok.pad_token = tok.eos_token   # Whisper 관례

    # 2) Student: 디코더 LM 래퍼
    student = WhisperDecoderLM(base).to(device)

    # 3) Teacher (KD용): 동일 베이스에서 디코더만 래핑, 파라미터 고정
    teacher = None
    if args.use_kd:
        teacher = WhisperDecoderLM(deepcopy(base)).to(device)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

    # 4) 동결(선택)
    if args.freeze_embeddings:
        student.get_input_embeddings().weight.requires_grad = False
        if hasattr(student.decoder, "embed_positions"):
            for p in student.decoder.embed_positions.parameters():
                p.requires_grad = False

    if args.freeze_norms:
        for n, p in student.named_parameters():
            if "layer_norm" in n:
                p.requires_grad = False

    # 5) LoRA(선택)
    if args.lora:
        if not HAS_PEFT:
            raise RuntimeError("peft 패키지가 필요합니다. (pip install peft)")
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student, peft_cfg)
        student.print_trainable_parameters()
    
    # ★ LoRA 적용을 포함한 "현재" 파라미터를 앵커 기준으로 스냅샷    
    anchor_ref_state = None
    if args.use_anchor and args.anchor_weight > 0.0:
        anchor_ref_state = {n: p.detach().clone().cpu() for n, p in student.named_parameters()}     
        
    # 앵커 마스크 파싱 (기본값/커맨드에서 proj_out 키 사용)
    def parse_anchor_mask(s: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not s:
            return out
        for token in s.split(","):
            k, v = token.strip().split("=")
            out[k.strip()] = float(v)
        return out

    anchor_mask = parse_anchor_mask(args.anchor_mask)           

    # 6) 데이터셋: 한 줄=한 문장 (여러 파일 쉼표 구분)
    paths = [p.strip() for p in args.text_path.split(",")]
    ds_dict = load_dataset("text", data_files={"train": paths})
    eval_ratio = getattr(args, "eval_ratio", 0.1)
    ds_all = ds_dict["train"].train_test_split(test_size=eval_ratio, seed=args.seed)
    ds_train, ds_eval = ds_all["train"], ds_all["test"]

    # 언어 프롬프트(prefix) → 디코더 분포 유지에 도움
    forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    prefix_ids = [tid for _, tid in forced]  # [(role,id)] → [id,...]

    def tokenize_fn(ex):
        text = ex["text"].strip()
        ids = tok(text, add_special_tokens=False).input_ids
        full = prefix_ids + ids + [tok.eos_token_id]
        full = full[: args.max_length]
        return {"input_ids": full}

    ds_train = ds_all["train"].map(tokenize_fn, remove_columns=["text"])
    ds_eval = ds_all["test"].map(tokenize_fn, remove_columns=["text"])

    collator = DataCollatorWhisperLM(tokenizer=tok)

    # 7) 학습 인자
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",     # ← eval_strategy -> evaluation_strategy
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        save_total_limit=3,
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        max_grad_norm=args.gradient_clip,
    )

    # 8) 앵커 마스크 파싱
    anchor_mask = parse_anchor_mask(args.anchor_mask)

    # 9) Trainer 실행 (KD + 앵커)
    trainer = KDAnchoredTrainer(
        model=student,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
        processing_class=processor,         # 또는 tokenizer=tok
        # label_names=["labels"],           # 선택(필수 아님)
        teacher=teacher if args.use_kd else None,   # ← 인자명 수정!
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        use_kd=args.use_kd,                 # ← 빠졌다면 명시
        use_anchor=args.use_anchor,
        anchor_weight=args.anchor_weight,
        anchor_mask=parse_anchor_mask(args.anchor_mask),
        anchor_ref_state=anchor_ref_state,  # ← 스냅샷 전달
        )
    
    # 학습 파라미터 출력
    print("학습 파라미터:")
    for n, p in student.named_parameters():
        if p.requires_grad:
            print(n)    

    trainer.train()
    trainer.save_model(args.output_dir)  # LoRA면 어댑터 저장, 풀파라미터면 전체 저장


if __name__ == "__main__":
    run()
