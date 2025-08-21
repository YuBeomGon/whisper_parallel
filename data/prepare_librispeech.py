# ==============================================================================
# 파일: data/prepare_librispeech.py
# 역할: Hugging Face 'librispeech_asr' 로드/전처리
#      - train_split / eval_split 문자열 그대로 사용
#      - 각 split별로 'clean'/'other' 구성 자동 감지
# 예) train_split='train.clean.100', eval_split='validation.clean' (또는 'test.other')
# ==============================================================================
from typing import Dict
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from whisper.normalizers.english import EnglishTextNormalizer


def _cfg_from_split(split: str) -> str:
    # 스플릿 문자열에 'other' 포함되면 other, 아니면 clean
    return "other" if ".other" in split else "clean"

def load_and_prepare_dataset(
    processor: WhisperProcessor,
    train_split: str = "train.360",
    eval_split: str = "validation.clean",
) -> Dict:
    normalizer = EnglishTextNormalizer()

    def process_function(batch):
        audio = batch["audio"]
        input_features = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        normalized_text = normalizer(batch["text"])
        labels = processor(text=normalized_text).input_ids
        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch

    # --- Train (streaming) ---
    train_cfg = _cfg_from_split(train_split)
    print(f"학습(train) split='{train_split}' (cfg='{train_cfg}', streaming) 로드 중...")
    train_ds = load_dataset("librispeech_asr", train_cfg, split=train_split, streaming=True)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    train_ds = train_ds.map(process_function)
    print("학습 데이터셋 준비 완료.")

    # --- Eval (standard) ---
    eval_cfg = _cfg_from_split(eval_split)
    print(f"평가(eval) split='{eval_split}' (cfg='{eval_cfg}') 로드 중...")
    eval_ds = load_dataset("librispeech_asr", eval_cfg, split=eval_split)
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.map(
        process_function,
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )
    print("평가 데이터셋 준비 완료.")
    return DatasetDict({"train": train_ds, "test": eval_ds})
