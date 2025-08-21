# ==============================================================================
# 파일: data/prepare_zeroth_korean.py
# 역할: 'Bingsu/zeroth-korean' 로드/전처리 (train/test 스플릿 지원)
# ==============================================================================
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from whisper.normalizers.basic import BasicTextNormalizer

def load_and_prepare_dataset(processor: WhisperProcessor,
                             train_split: str = "train",
                             eval_split: str = "test") -> DatasetDict:
    """
    Zeroth-Korean 데이터셋을 로드하고 전처리합니다.
    - train_split: 기본 'train'
    - eval_split : 기본 'test'
    """
    normalizer = BasicTextNormalizer()

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
    print(f"학습(train) split='{train_split}' (streaming) 로드 중...")
    train_ds = load_dataset("Bingsu/zeroth-korean", split=train_split, streaming=True)
    train_ds = train_ds.map(process_function)
    print("학습 데이터셋 준비 완료.")

    # --- Eval (standard) ---
    print(f"평가(eval) split='{eval_split}' 로드 중...")
    eval_ds = load_dataset("Bingsu/zeroth-korean", split=eval_split)
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.map(
        process_function,
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )
    print("평가 데이터셋 준비 완료.")
    return DatasetDict({"train": train_ds, "test": eval_ds})
