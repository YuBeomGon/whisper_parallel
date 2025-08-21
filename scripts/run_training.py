# ==============================================================================
# 파일: scripts/run_training.py
# 역할: 1단계 학습(병렬 구조 성능 검증)을 실행하는 메인 스크립트
# ==============================================================================
import torch, os
#torch.backends.cudnn.enabled = False
#os.environ["CUDNN_DISABLE"] = "1"
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from data.prepare_zeroth_korean import load_and_prepare_dataset
from models.parallel_decoder import ParallelWhisperDecoderLayer

def run_training():
    """1단계 모델 학습을 위한 메인 함수"""
    # --- 1. 모델 및 프로세서 설정 ---
    # 학습/평가 모두 동일하게 사용할 베이스 모델 ID (turbo 사용 중)
    model_name = "openai/whisper-large-v3-turbo"
    print(f"'{model_name}' 모델을 로드합니다...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # --- 2. 모델 구조 변경 ---
    print("디코더 레이어를 병렬 구조로 교체합니다...")
    # layer_idx를 반드시 전달해야 캐시(use_cache) 경로가 안정적으로 동작
    for i in range(model.config.decoder_layers):
        model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(model.config, layer_idx=i)

    # 인코더 동결
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    # 학습 시에는 캐시 비활성화 + 그래디언트 체크포인팅 권장
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # --- 3. 데이터 준비 ---
    datasets = load_and_prepare_dataset(processor)  # {"train": ..., "test": ...}
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # --- 4. 학습 인자(Arguments) 설정 ---
    # NOTE: evaluation_strategy 가 정식 파라미터명입니다. (eval_strategy 아님)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-large-v3-parallel-zeroth-streaming",
        per_device_train_batch_size=1,          # OOM 나면 1로
        gradient_accumulation_steps=8,          # effective batch 8
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,                         # 스트리밍 데이터셋 → 스텝 베이스
        eval_strategy="steps",            # ← 수정: evaluation_strategy -> eval_strategy
        eval_steps=1000,
        save_steps=1000,
        logging_steps=25,
        report_to="none",                       # wandb 사용 시 ["wandb"]
        fp16=True,                              # 혼합 정밀도 학습
        save_safetensors=True,                  # 평가 스크립트의 로더와 포맷 일치
        dataloader_num_workers=4,               # 시스템에 맞춰 조정
    )

    # --- 5. Trainer 생성 및 학습 시작 ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
        # Whisper 예제에서는 tokenizer=processor.feature_extractor 사용
        # tokenizer=processor.feature_extractor,
        processing_class=processor,
        # compute_metrics는 2단계(정식 평가)에서 CER/WER 계산용으로 추가 권장
    )

    print("모델 학습을 시작합니다...")
    trainer.train()
    print("학습 완료!")

if __name__ == "__main__":
    run_training()
