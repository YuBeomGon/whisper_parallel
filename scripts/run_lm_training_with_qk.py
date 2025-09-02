# ==============================================================================
# 파일: scripts/run_lm_training_with_qk.py
# 역할: Whisper 디코더를 텍스트 전용 Causal LM으로 학습 (Cross-Attn/Encoder 미사용)
#       - LoRA를 self-attn의 Q/K에만 적용(QK-only)하여 도메인 텍스트 패턴 적응
#       - KD(지식증류) + 앵커 정규화(L2-SP) 옵션 지원
#       - 임베딩/포지션/출력(proj_out/lm_head)은 변경하지 않음(동결 권장)
# 사용법(예시):
# CUDA_VISIBLE_DEVICES=0 python -m scripts.run_lm_training_with_qk \
#   --base_model openai/whisper-large-v3-turbo \
#   --language ko \
#   --text_path "data/text_corpus/insurance_terms.txt,data/text_corpus/zeroth_corpus.txt" \
#   --output_dir ./whisper-decoder-lm-qk \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --use_kd --kd_alpha 0.5 --kd_temp 2.0 \
#   --use_anchor --anchor_weight 5e-4 \
#   --anchor_mask "self_attn.q_proj=1.0,self_attn.k_proj=1.0,embed_tokens=0.2,proj_out=0.2"
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

from models.whisper_decoder_lm import WhisperDecoderLM
from utils.lm_collator import DataCollatorWhisperLM

# LoRA
try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


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
        self.anchor_mask = anchor_mask or {}
        self.anchor_ref_state = anchor_ref_state or {}

    def _anchor_coef(self, name: str) -> float:
        for k, v in self.anchor_mask.items():
            if k in name:
                return float(v)
        return 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        student_loss = outputs.get("loss", None)
        if student_loss is None and labels is not None:
            student_loss = self.label_smoother(outputs, labels)
        loss = student_loss

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
    p = argparse.ArgumentParser("Whisper Decoder LM training (QK-only LoRA, KD+Anchor)")
    p.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo")
    p.add_argument("--language", type=str, default="ko")
    p.add_argument("--text_path", type=str, required=True)
    p.add_argument("--max_length", type=int, default=448)

    p.add_argument("--output_dir", type=str, default="./whisper-decoder-lm-qk")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_clip", type=float, default=1.0)

    p.add_argument("--freeze_embeddings", action="store_true")
    p.add_argument("--freeze_norms", action="store_true")

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--use_kd", action="store_true")
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_temp", type=float, default=2.0)

    p.add_argument("--use_anchor", action="store_true")
    p.add_argument("--anchor_weight", type=float, default=5e-4)
    p.add_argument("--anchor_mask", type=str,
                   default="self_attn.q_proj=1.0,self_attn.k_proj=1.0,embed_tokens=0.2,proj_out=0.2")
    p.add_argument("--eval_ratio", type=float, default=0.1)
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8

    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="sdpa",
    )
    processor = WhisperProcessor.from_pretrained(args.base_model)
    tok = processor.tokenizer
    tok.pad_token = tok.eos_token

    student = WhisperDecoderLM(base).to(device)

    teacher = None
    if args.use_kd:
        teacher = WhisperDecoderLM(deepcopy(base)).to(device)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

    if args.freeze_embeddings:
        student.get_input_embeddings().weight.requires_grad = False
        if hasattr(student.decoder, "embed_positions"):
            for p in student.decoder.embed_positions.parameters():
                p.requires_grad = False

    if args.freeze_norms:
        for n, p in student.named_parameters():
            if "layer_norm" in n:
                p.requires_grad = False

    if not HAS_PEFT:
        raise RuntimeError("peft 패키지가 필요합니다. (pip install peft)")
    target_modules = ["q_proj", "k_proj"]
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

    anchor_ref_state = None
    if args.use_anchor and args.anchor_weight > 0.0:
        anchor_ref_state = {n: p.detach().clone().cpu() for n, p in student.named_parameters()}

    paths = [p.strip() for p in args.text_path.split(",")]
    ds_dict = load_dataset("text", data_files={"train": paths})
    ds_all = ds_dict["train"].train_test_split(test_size=args.eval_ratio, seed=args.seed)

    forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    prefix_ids = [tid for _, tid in forced]

    def tokenize_fn(ex):
        text = ex["text"].strip()
        ids = tok(text, add_special_tokens=False).input_ids
        full = prefix_ids + ids + [tok.eos_token_id]
        full = full[: args.max_length]
        return {"input_ids": full}

    ds_train = ds_all["train"].map(tokenize_fn, remove_columns=["text"])
    ds_eval = ds_all["test"].map(tokenize_fn, remove_columns=["text"])
    collator = DataCollatorWhisperLM(tokenizer=tok)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",  # ← fix
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
        seed=args.seed,
    )

    trainer = KDAnchoredTrainer(
        model=student,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
        processing_class=processor,
        teacher=teacher if args.use_kd else None,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        use_kd=args.use_kd,
        use_anchor=args.use_anchor,
        anchor_weight=args.anchor_weight,
        anchor_mask=parse_anchor_mask(args.anchor_mask),
        anchor_ref_state=anchor_ref_state,
    )

    print("=== Trainable params (expect only q_proj/k_proj & LoRA) ===")
    bad = []
    for n, p in student.named_parameters():
        if p.requires_grad and (("q_proj" not in n) and ("k_proj" not in n)):
            bad.append(n)
        if p.requires_grad:
            print(n)
    assert len(bad) == 0, f"Q/K-only가 아님. 열린 모듈: {bad[:5]}"

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    run()
