# ==============================================================================
# 파일: scripts/run_training_kspon.py
# 역할: KsponSpeech 폴더(원본 .pcm/.txt)에서 직접 스트리밍하여 병렬 디코더 학습
#       - 대용량 HF Dataset 저장 없이 IterableDataset로 학습, MapDataset로 평가
#       - γ-annealing, cross-attn/LN 동결, 지연 EarlyStopping, CER 메트릭 지원
#
# 사용 예시:
# CUDA_VISIBLE_DEVICES=0 python -m scripts.run_training_kspon \
#   --base_dir /home/voice/project/stt/dataset/10.한국어음성 \
#   --model_name openai/whisper-large-v3-turbo \
#   --decoder_mode parallel \
#   --learning_rate 1e-4 --warmup_steps 1000 \
#   --max_steps 200000 --eval_steps 2000 --save_steps 2000 \
#   --gamma_start 1.0 --gamma_end 0.0 --gamma_start_frac 0.0 --gamma_end_frac 0.6 \
#   --output_dir ./saved/whisper-parallel-kspon


# ==============================================================================

import os
import re
import glob
import math
import random
import argparse
import numpy as np
from typing import List, Tuple, Iterator, Dict

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, get_worker_info

import jiwer
from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding

# 병렬 디코더
try:
    from models.parallel_decoder import ParallelWhisperDecoderLayer
    HAS_PARALLEL = True
except Exception:
    HAS_PARALLEL = False
    

os.environ.setdefault("ACCELERATE_SPLIT_BATCHES", "true")
os.environ.setdefault("ACCELERATE_DISPATCH_BATCHES", "false")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")  # bf16 환경이면    


# ----------------------------- 텍스트 정규화 규칙 -----------------------------
def clean_sentence(sentence: str) -> str:
    # 1) (표준어/발음) → 표준어만
    sentence = re.sub(r'\((.*?)\/(.*?)\)', r'\1', sentence)
    # 2) 기타 괄호 내용 제거(안전장치)
    sentence = re.sub(r'\([^)]*\)', '', sentence)
    # 3) 노이즈/발화 태그 제거 (b/, o/, n/, l/ 등)
    sentence = re.sub(r'([a-zA-Z]|\d)/', '', sentence)
    # 4) 잘림(*) 제거
    sentence = sentence.replace('*', '')
    # 5) 구두점 정제: ',', '.', '?'만 허용
    sentence = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,?]', '', sentence)
    # 6) 외래어 표기 샘플 맵핑(필요시 확장)
    foreign_word_map = {'에이아이': 'AI', '오케이': 'OK'}
    for kor, eng in foreign_word_map.items():
        sentence = sentence.replace(kor, eng)
    # 7) 공백 정규화
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


# ----------------------------- 파일 검색/분할 -----------------------------
def find_kspon_files(base_dir: str) -> Tuple[List[str], List[str]]:
    """KsponSpeech_01~05 → train, KsponSpeech_eval → test(있으면)"""
    train_glob = os.path.join(base_dir, 'KsponSpeech_0[1-5]', '**', '*.pcm')
    eval_glob  = os.path.join(base_dir, 'KsponSpeech_eval', '**', '*.pcm')

    train_files = sorted(glob.glob(train_glob, recursive=True))
    eval_files  = sorted(glob.glob(eval_glob,  recursive=True))

    if not train_files:
        raise FileNotFoundError(f"No train .pcm files under {train_glob}")

    # eval 없으면 train의 1% 샘플을 eval로 전용
    if not eval_files:
        n = max(500, int(0.01 * len(train_files)))  # 최소 500개 또는 1%
        eval_files = random.sample(train_files, n)
        # 학습에서 제외(중복 방지)
        train_set = set(train_files)
        for p in eval_files:
            if p in train_set:
                train_set.remove(p)
        train_files = sorted(list(train_set))
        print(f"[KSpon] No 'KsponSpeech_eval' found. Using {len(eval_files)} files from train as eval.")

    print(f"[KSpon] train files: {len(train_files)} | eval files: {len(eval_files)}")
    return train_files, eval_files


def txt_from_pcm(pcm_path: str) -> str:
    return os.path.splitext(pcm_path)[0] + ".txt"


def read_text_with_fallback(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="cp949", errors="ignore") as f:
            return f.read()


def load_pcm_mono16k(pcm_path: str) -> Tuple[np.ndarray, int]:
    """Kspon PCM은 보통 16kHz s16le mono."""
    arr = np.fromfile(pcm_path, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, 16000


# ----------------------------- Dataset 구현 -----------------------------
def make_labels(processor: WhisperProcessor, text: str, language: str) -> List[int]:
    tok = processor.tokenizer
    prefix = [tid for _, tid in processor.get_decoder_prompt_ids(language=language, task="transcribe")]
    ids = tok(text, add_special_tokens=False).input_ids
    return prefix + ids + [tok.eos_token_id]


class KsponIterableDataset(IterableDataset):
    """대용량 학습용: __iter__에서 즉시 로드/전처리."""
    def __init__(self, file_paths: List[str], processor: WhisperProcessor, language: str, shuffle: bool = True, seed: int = 42):
        self.files = list(file_paths)
        self.processor = processor
        self.language = language
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Dict]:
        files = self.files
        # 워커별 샤딩
        info = get_worker_info()
        if info is not None:
            files = files[info.id::info.num_workers]

        # 에폭마다 섞기(Trainer가 DataLoader를 다시 생성할 때 호출됨)
        if self.shuffle:
            rng = random.Random()
            rng.seed(self.seed + (info.id if info is not None else 0))
            files = files[:]  # copy
            rng.shuffle(files)

        for pcm_path in files:
            txt_path = txt_from_pcm(pcm_path)
            try:
                raw_text = read_text_with_fallback(txt_path)
                cleaned = clean_sentence(raw_text)
                if not cleaned:
                    continue

                audio, sr = load_pcm_mono16k(pcm_path)
                feats = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features[0]
                labels = make_labels(self.processor, cleaned, self.language)

                yield {"input_features": feats, "labels": torch.tensor(labels, dtype=torch.long)}
            except Exception as e:
                # 파일 오류는 건너뜀
                continue


class KsponMapDataset(Dataset):
    """평가용: 길이를 알아야 해서 Map-style로 구현(지연 로딩)."""
    def __init__(self, file_paths: List[str], processor: WhisperProcessor, language: str):
        self.files = list(file_paths)
        self.processor = processor
        self.language = language

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pcm_path = self.files[idx]
        txt_path = txt_from_pcm(pcm_path)
        raw_text = read_text_with_fallback(txt_path)
        cleaned = clean_sentence(raw_text)

        audio, sr = load_pcm_mono16k(pcm_path)
        feats = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features[0]
        labels = make_labels(self.processor, cleaned, self.language)
        return {"input_features": feats, "labels": torch.tensor(labels, dtype=torch.long)}
    
    
def _cast_all_layer_norm_to_fp32(model: nn.Module):
    """
    DataParallel/autocast에서 hidden_states가 float32로 승격되는 경우가 있어
    LN 가중치/입력 dtype이 mismatch 날 수 있음.
    모든 LayerNorm을 FP32로 고정해 dtype mismatch를 차단.
    """
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.to(torch.float32)    


def _sync_layernorm_dtype_with_input(model: nn.Module):
    """
    입력 텐서 dtype과 LayerNorm(weight/bias) dtype이 다르면
    forward 이전에 파라미터 dtype을 입력에 맞춰 즉시 변환.
    (generate/eval 경로에서 FP32 <-> BF16 섞이는 상황을 안전하게 처리)
    """
    def _pre_hook(module, args):
        # args: (hidden_states, ...)
        if not args:
            return
        x = args[0]
        if not hasattr(module, "weight") or module.weight is None:
            return
        if module.weight.dtype != x.dtype:
            module.weight.data = module.weight.data.to(x.dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(x.dtype)
        return

    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            # 동일 hook 중복 등록 방지하고 싶으면 조건 체크 추가 가능
            m.register_forward_pre_hook(_pre_hook, with_kwargs=False)

# ----------------------------- 콜백/메트릭 -----------------------------
class ParallelTransitionCallback(TrainerCallback):
    """gamma: 1.0 → 0.0 코사인 스케줄"""
    def __init__(self, total_steps: int, start: float, end: float, start_frac: float, end_frac: float):
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
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * t))

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        gamma = self._gamma_at(step)
        model = kwargs["model"]
        for layer in model.model.decoder.layers:
            if hasattr(layer, "gamma"):
                layer.gamma.fill_(gamma)
        if step % 1000 == 0:
            print(f"[Gamma] step={step} gamma={gamma:.4f}")
        return control


class UnfreezeCrossLNAtGamma(TrainerCallback):
    """gamma가 임계 이하가 되면 cross-attn 앞 LN만 해제"""
    def __init__(self, threshold: float, lr: float):
        self.threshold = threshold
        self.lr = lr
        self.done = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.done:
            return control
        model = kwargs["model"]
        optimizer = kwargs.get("optimizer")

        gamma = None
        for layer in model.model.decoder.layers:
            if hasattr(layer, "gamma"):
                g = layer.gamma
                gamma = float(g.item() if hasattr(g, "item") else g)
                break
        if gamma is None or gamma > self.threshold:
            return control

        # LN requires_grad 켜기
        new_params = []
        for layer in model.model.decoder.layers:
            for p in layer.encoder_attn_layer_norm.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    new_params.append(p)
        if optimizer is not None and new_params:
            existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
            to_add = [p for p in new_params if id(p) not in existing]
            if to_add:
                optimizer.add_param_group({"params": to_add, "lr": self.lr})
                print(f"[UnfreezeCrossLN] Added {len(to_add)} params")

        print(f"[UnfreezeCrossLN] gamma={gamma:.3f} ≤ {self.threshold:.3f} → cross-attn LN unfrozen")
        self.done = True
        return control


class DelayedEarlyStoppingCallback(TrainerCallback):
    """min_step 이전엔 기다렸다가, 이후부터 patience 카운트."""
    def __init__(self, metric_name="cer", greater_is_better=False, patience=10, threshold=5e-4, min_step=0):
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
            return self.sign * (new - ref) < -self.threshold

        if self.best is None or improved(value, self.best):
            self.best = value
            self.wait = 0
            return control

        if step < self.min_step:
            return control

        self.wait += 1
        if self.wait >= self.patience:
            print(f"[EarlyStop(delayed)] step {step}: no improvement in {self.patience} evals → STOP")
            control.should_training_stop = True
        return control


def build_compute_metrics(processor, normalizer_like, lang_hint: str | None = None):
    """
    normalizer_like: callable(정규화기) 또는 문자열('ko'/'en').
    문자열이면 여기서 인스턴스화해 callable로 만든다.
    """
    if callable(normalizer_like):
        norm = normalizer_like
    else:
        key = (normalizer_like or lang_hint or "ko").lower()
        norm = EnglishTextNormalizer() if key.startswith("en") else BasicTextNormalizer()

    def _compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(preds,  skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        # 정규화 (항상 callable)
        pred_norm  = [norm(s) if isinstance(s, str) else "" for s in pred_str]
        label_norm = [norm(s) if isinstance(s, str) else "" for s in label_str]

        # 빈 ref 제거
        keep = [(r.strip(), p.strip()) for r, p in zip(label_norm, pred_norm) if r and r.strip()]
        if not keep:
            return {"cer": 0.0, "skipped_empty_refs": len(label_norm)}

        refs, hyps = zip(*keep)
        try:
            cer = jiwer.cer(list(refs), list(hyps))
        except ValueError:
            cer = 0.0
        return {"cer": cer, "skipped_empty_refs": len(label_norm) - len(refs)}

    return _compute_metrics

# ----------------------------- 파서/메인 -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Train Whisper on KsponSpeech via folder streaming")
    p.add_argument("--base_dir", type=str, required=True, help="KsponSpeech 상위 폴더(예: 10.한국어음성)")
    p.add_argument("--model_name", type=str, default="openai/whisper-large-v3-turbo")
    p.add_argument("--decoder_mode", choices=["parallel", "vanilla"], default="parallel")

    # 러닝 설정
    p.add_argument("--output_dir", type=str, default="./saved/whisper-parallel-kspon")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--save_steps", type=int, default=2000)

    # γ 스케줄
    p.add_argument("--gamma_start", type=float, default=1.0)
    p.add_argument("--gamma_end", type=float, default=0.0)
    p.add_argument("--gamma_start_frac", type=float, default=0.0)
    p.add_argument("--gamma_end_frac", type=float, default=0.6)

    # 동결 옵션
    p.add_argument("--no_freeze_encoder", action="store_true")
    p.add_argument("--freeze_cross_attn", action="store_true")
    p.add_argument("--freeze_cross_ln", action="store_true")
    p.add_argument("--unfreeze_cross_ln_at_gamma", type=float, default=None)

    # 기타
    p.add_argument("--language", type=str, default="ko")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 성능 스위치
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    # 1) 파일 목록 수집
    train_files, eval_files = find_kspon_files(args.base_dir)

    # 2) 모델/프로세서
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    processor = WhisperProcessor.from_pretrained(args.model_name)

    # 언어 프롬프트 & retie
    forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.max_new_tokens = 225
    model.generation_config.num_beams = 1
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # proj_out ↔ embed_tokens weight tie 보장
    if hasattr(model, "proj_out"):
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
    elif hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.decoder.embed_tokens.weight

    # 3) 병렬 디코더 교체/동결
    if args.decoder_mode == "parallel":
        if not HAS_PARALLEL:
            raise RuntimeError("models.parallel_decoder가 필요합니다.")
        for i in range(model.config.decoder_layers):
            vanilla = model.model.decoder.layers[i]  # ← 프리트레인 가중치 포함
            model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(vanilla, layer_idx=i)
        _cast_all_layer_norm_to_fp32(model)
        _sync_layernorm_dtype_with_input(model)
    else:
        print("바닐라(원복) Whisper 디코더를 그대로 사용합니다. (γ 스케줄 비활성)")        
    if not args.no_freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    if args.freeze_cross_attn:
        for layer in model.model.decoder.layers:
            for p in layer.cross_attn.parameters():
                p.requires_grad = False
    if args.freeze_cross_ln:
        for layer in model.model.decoder.layers:
            for p in layer.encoder_attn_layer_norm.parameters():
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Params] Trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    model.config.use_cache = False

    # 4) Dataset/Collator
    train_ds = KsponIterableDataset(train_files, processor, args.language, shuffle=True, seed=args.seed)
    eval_ds  = KsponMapDataset(eval_files, processor, args.language)

    model_dtype = next(model.parameters()).dtype
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, model_dtype=model_dtype)
    
    compute_metrics = build_compute_metrics(processor, normalizer_like=args.language, lang_hint=args.language)


    # 5) Trainer 세팅
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
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
        dataloader_drop_last=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    # 6) 콜백: γ 스케줄 + LN 지연 해제 + 지연 EarlyStopping
    if args.decoder_mode == "parallel":
        trainer.add_callback(ParallelTransitionCallback(
            total_steps=training_args.max_steps,
            start=args.gamma_start, end=args.gamma_end,
            start_frac=args.gamma_start_frac, end_frac=args.gamma_end_frac,
        ))
        if args.freeze_cross_ln and args.unfreeze_cross_ln_at_gamma is not None:
            trainer.add_callback(UnfreezeCrossLNAtGamma(args.unfreeze_cross_ln_at_gamma, lr=args.learning_rate))

    min_step = int(training_args.max_steps * (args.gamma_end_frac if args.decoder_mode == "parallel" else 0.6))
    trainer.add_callback(DelayedEarlyStoppingCallback(
        metric_name="cer", greater_is_better=False, patience=5, threshold=5e-4, min_step=min_step
    ))

    # 7) 학습
    print("[Train] Start training on KsponSpeech (streaming from folder)")
    trainer.train()
    print("[Train] Done.")

    # 최종 저장은 Seq2SeqTrainingArguments(load_best_model_at_end=True)로 best도 함께 보관됨.


if __name__ == "__main__":
    main()
