# ==============================================================================
# 파일: scripts/run_training.py
# 역할: 병렬 디코더 학습(인코더 freeze + γ-annealing + CER Early Stopping)
#       └ 데이터셋/모델/하이퍼파라미터를 CLI 인자로 받음
# ==============================================================================

import os
import math
import argparse
import torch
import jiwer
# ↓↓↓ 정규화기: KO/EN 둘 다 import
from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback

from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from data.prepare_zeroth_korean import load_and_prepare_dataset as load_ko
from models.parallel_decoder import ParallelWhisperDecoderLayer

# (선택) LibriSpeech도 지원하려면 파일을 추가하고 import 하세요.
try:
    from data.prepare_librispeech import load_and_prepare_dataset as load_en
    HAS_LIBRISPEECH = True
except Exception:
    HAS_LIBRISPEECH = False


def set_cross_attn_requires_grad(model, requires_grad: bool):
    # 디코더 cross-attn 파라미터 전부 동결/해제
    for layer in model.model.decoder.layers:
        for p in layer.cross_attn.parameters():
            p.requires_grad = requires_grad

def set_cross_ln_requires_grad(model, requires_grad: bool):
    # cross-attn 앞 LayerNorm 동결/해제
    for layer in model.model.decoder.layers:
        for p in layer.encoder_attn_layer_norm.parameters():
            p.requires_grad = requires_grad
            
class UnfreezeCrossLNAtGamma(TrainerCallback):
    """gamma가 threshold 이하로 내려가면 cross-attn LN만 해제"""
    def __init__(self, threshold: float, lr: float):
        self.threshold = threshold
        self.lr = lr
        self.done = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.done:
            return control

        model = kwargs["model"]
        optimizer = kwargs.get("optimizer", None)

        # gamma 읽기(첫 레이어 기준)
        gamma = None
        for layer in model.model.decoder.layers:
            if hasattr(layer, "gamma"):
                g = layer.gamma
                gamma = float(g.item() if hasattr(g, "item") else g)
                break

        # 아직 임계값 못 내려오면 패스
        if gamma is None or gamma > self.threshold:
            return control

        # 1) LN requires_grad 켜기
        new_params = []
        for layer in model.model.decoder.layers:
            for p in layer.encoder_attn_layer_norm.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)

        # 2) 옵티마이저 param group에 없으면 추가
        if optimizer is not None and new_params:
            existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
            to_add = [p for p in new_params if id(p) not in existing]
            if to_add:
                optimizer.add_param_group({"params": to_add, "lr": self.lr})
                print(f"[UnfreezeCrossLNAtGamma] Added {len(to_add)} LN params to optimizer")

        print(f"[UnfreezeCrossLNAtGamma] gamma={gamma:.3f} ≤ {self.threshold:.3f} → cross-attn LN unfrozen")
        self.done = True
        return control
    
class DelayedEarlyStoppingCallback(TrainerCallback):
    """
    min_step 이전에는 '기다리기'만 하고 patience 카운트/스톱을 하지 않음.
    min_step 이후부터 일반 Early Stopping 로직 적용.
    """
    def __init__(self, metric_name: str = "cer", greater_is_better: bool = False,
                 patience: int = 20, threshold: float = 5e-4, min_step: int = 0):
        self.metric_key = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
        self.sign = 1.0 if greater_is_better else -1.0
        self.patience = patience
        self.threshold = threshold
        self.min_step = min_step
        self.best = None
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.metric_key not in metrics:
            return control

        step = state.global_step
        value = metrics[self.metric_key]

        def improved(new, ref):
            # improvement if sign*(new - ref) > threshold
            return self.sign * (new - ref) < -self.threshold

        # 항상 best 추적은 하지만,
        # min_step 이전에는 wait/patience를 올리지 않음
        if self.best is None or improved(value, self.best):
            self.best = value
            self.wait = 0
            return control

        if step < self.min_step:
            # 아직은 기다리기만
            return control

        # min_step 이후부터 patience 카운트
        self.wait += 1
        if self.wait >= self.patience:
            print(f"[DelayedEarlyStopping] step {step}: no improvement in {self.patience} evals → STOP")
            control.should_training_stop = True
        return control             
            
            


# ---------- γ-annealing 콜백 ----------
class ParallelTransitionCallback(TrainerCallback):
    """gamma: 1.0(원본) -> 0.0(완전 병렬)로 코사인 전개"""
    def __init__(self, total_steps: int, start: float, end: float,
                 start_frac: float, end_frac: float):
        self.total_steps = max(1, int(total_steps))
        self.start = start
        self.end = end
        self.s_begin = int(self.total_steps * start_frac)
        self.s_end = int(self.total_steps * end_frac)

    def _gamma_at(self, step: int) -> float:
        if step <= self.s_begin:
            return self.start
        if step >= self.s_end:
            return self.end
        t = (step - self.s_begin) / max(1, (self.s_end - self.s_begin))
        # cosine from start->end
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * t))

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        gamma = self._gamma_at(step)
        model = kwargs["model"]
        for layer in model.model.decoder.layers:
            if hasattr(layer, "gamma"):
                layer.gamma.fill_(gamma)
        if step % 1000 == 0:  # 가끔 찍기
            print(f"[Gamma] step={step} gamma={gamma:.4f}")                
        return control


def build_compute_metrics(processor: WhisperProcessor, normalizer):
    """Seq2SeqTrainer용 CER 계산 (언어별 정규화기 사용)"""
    def _compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        pred_str = [normalizer(s) for s in pred_str]
        label_str = [normalizer(s) for s in label_str]

        cer = jiwer.cer(label_str, pred_str)
        return {"cer": cer}
    return _compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Whisper 병렬 디코더 학습 스크립트")
    # --- 핵심: 모델/데이터셋 선택 ---
    p.add_argument("--model_name", type=str, default="openai/whisper-large-v3-turbo",
                   help="기반 사전학습 모델 ID")
    p.add_argument("--dataset", choices=["zeroth_ko", "librispeech_en"], default="librispeech_en",
                   help="학습/평가 데이터셋 선택")
    p.add_argument("--language", type=str, default=None,
                   help="강제 언어 프롬프트(기본: zeroth_ko=ko, librispeech_en=en)")

    # LibriSpeech 옵션
    p.add_argument("--ls_train_split", type=str, default="train.clean.360",
                   help="LibriSpeech train split (예: train.clean.360)")
    p.add_argument("--ls_eval_split", type=str, default="validation.clean",
                   help="LibriSpeech eval split (예: validation)")

    # --- 러닝 하이퍼파라미터 ---
    p.add_argument("--output_dir", type=str, default=None, help="출력 디렉토리(기본: 자동)")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--save_steps", type=int, default=2000)

    # --- γ 스케줄 ---
    p.add_argument("--gamma_start", type=float, default=1.0)
    p.add_argument("--gamma_end", type=float, default=0.0)
    p.add_argument("--gamma_start_frac", type=float, default=0.0)
    p.add_argument("--gamma_end_frac", type=float, default=0.4)

    # --- 기타 ---
    p.add_argument("--no_freeze_encoder", action="store_true", help="인코더 동결 해제")
    
    p.add_argument("--freeze_cross_attn", action="store_true",
                help="디코더 cross-attn(Wqkv/wo) 동결")
    p.add_argument("--freeze_cross_ln", action="store_true",
                help="디코더 cross-attn 앞 LayerNorm 동결")
    p.add_argument("--unfreeze_cross_ln_at_gamma", type=float, default=None,
                help="gamma가 이 값 이하로 내려가면 cross-attn LN 해제")

    # Early Stopping을 언제부터 켤지
    p.add_argument("--early_stop_min_step", type=int, default=None,
                help="이 스텝 이후부터 Early Stopping 적용 (미지정 시 gamma_end_frac*max_steps)")    
    
    p.add_argument("--hf_cache_base", type=str, default="./hf_cache",
                help="HF datasets cache base dir (데이터셋별 하위폴더로 분리)")
    p.add_argument("--hf_timeout", type=int, default=90,
                help="HuggingFace Hub HTTP timeout (sec)")
    p.add_argument("--hf_enable_transfer", action="store_true",
                help="HF 전송 가속(hf_transfer) 활성화")
    
    return p.parse_args()


def run_training():
    args = parse_args()

    # set cache_dir and hf_timeout:
    os.environ["HUGGINGFACE_HUB_HTTP_TIMEOUT"] = str(args.hf_timeout)
    if args.hf_enable_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # 데이터셋별 분리 캐시 디렉토리
    ds_sub = "zeroth" if args.dataset == "zeroth_ko" else "librispeech"
    dataset_cache_dir = os.path.join(args.hf_cache_base, ds_sub)
    os.makedirs(dataset_cache_dir, exist_ok=True)
    print(f"[HF cache] {dataset_cache_dir}")    

    # 성능 스위치
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    # 1) 모델/프로세서
    print(f"'{args.model_name}' 모델을 로드합니다...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    processor = WhisperProcessor.from_pretrained(args.model_name)

    # 언어 프롬프트 (dataset에 기반한 기본값, --language로 덮어쓰기 가능)
    if args.language is None:
        language = "ko" if args.dataset == "zeroth_ko" else "en"
    else:
        language = args.language

    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.max_new_tokens = 225
    model.generation_config.num_beams = 1
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # === 여기서 정규화기 선택 ===
    if language.lower().startswith("en"):
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    # 2) 병렬 디코더 교체 + 인코더 freeze
    print("디코더 레이어를 병렬 구조로 교체합니다...")
    for i in range(model.config.decoder_layers):
        model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(model.config, layer_idx=i)

    if not args.no_freeze_encoder:
        print("인코더 파라미터를 동결합니다...")
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    else:
        print("인코더 동결 해제(--no_freeze_encoder).")
        
    # ★ cross-attn / LN 동결 옵션
    if args.freeze_cross_attn:
        print("Freeze: decoder cross-attn weights")
        set_cross_attn_requires_grad(model, False)
    if args.freeze_cross_ln:
        print("Freeze: cross-attn LayerNorm")
        set_cross_ln_requires_grad(model, False)        

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    model.config.use_cache = False
    # model.gradient_checkpointing_enable()  # 느려지면 OFF 권장

    # 3) 데이터
    if args.dataset == "zeroth_ko":
        datasets = load_ko(
            processor,
            train_split="train",
            eval_split="test",
            cache_dir=dataset_cache_dir,
            streaming_train=True,
            hf_revision="main",
    )
    else:
        if not HAS_LIBRISPEECH:
            raise RuntimeError("librispeech_en을 선택했지만 'data/prepare_librispeech.py'가 없습니다.")
        datasets = load_en(
            processor,
            train_split=args.ls_train_split,      # e.g., train.clean.100 / train.360
            eval_split=args.ls_eval_split,        # e.g., validation.clean / test.clean
            cache_dir=dataset_cache_dir,
            streaming_train=True,
            hf_revision="main",
            )

    model_dtype = next(model.parameters()).dtype
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, model_dtype=model_dtype)

    # 4) 학습 인자
    out_dir = args.output_dir or f"./whisper-parallel-{args.dataset}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=max(50, args.eval_steps // 2),
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch_fused",
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
    )

    # 5) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor, normalizer),
    )

    # 6) 콜백: γ-anneal + Early Stopping
    gamma_cb = ParallelTransitionCallback(
        total_steps=training_args.max_steps,
        start=args.gamma_start, end=args.gamma_end,
        start_frac=args.gamma_start_frac, end_frac=args.gamma_end_frac,
    )
    trainer.add_callback(gamma_cb)
    
    # LN 지연 해제: 예) --unfreeze_cross_ln_at_gamma 0.6
    if args.freeze_cross_ln and args.unfreeze_cross_ln_at_gamma is not None:
        trainer.add_callback(UnfreezeCrossLNAtGamma(args.unfreeze_cross_ln_at_gamma, lr=args.learning_rate))
    
    
    # Early Stopping: min_step 이후부터만 적용
    min_step = args.early_stop_min_step
    if min_step is None:
        # 기본값: γ 전환이 끝나는 시점 이후(= gamma_end_frac * max_steps)
        min_step = int(training_args.max_steps * args.gamma_end_frac)
        
    delayed_early = DelayedEarlyStoppingCallback(
        metric_name="cer",
        greater_is_better=False,
        patience=10,
        threshold=5e-4,
        min_step=min_step,          # ★ 예: 100000 * 0.4 = 40000
    )
    
    trainer.add_callback(delayed_early)

    print("모델 학습을 시작합니다...")
    trainer.train()
    print("학습 완료!")


if __name__ == "__main__":
    run_training()
