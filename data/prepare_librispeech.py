# ==============================================================================
# data/prepare_librispeech.py
# 목적: LibriSpeech 데이터셋 준비
# 사용 예:
#   python -m data.prepare_librispeech --out data/librispeech_asr.py
# ==============================================================================
from datasets import DatasetDict, Audio
from transformers import WhisperProcessor
from whisper.normalizers.english import EnglishTextNormalizer
from utils.hf_io import load_with_retry

def load_and_prepare_dataset(
    processor: WhisperProcessor,
    train_split: str = "train.clean.100",
    eval_split: str = "validation.clean",
    cache_dir: str | None = None,
    streaming_train: bool = True,
    hf_revision: str = "main",
):
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

    # train (streaming)
    print(f"학습(train) split='{train_split}' (streaming={streaming_train}) 로드 중...")
    train_ds = load_with_retry(
        "librispeech_asr",
        split=train_split,
        streaming=streaming_train,
        revision=hf_revision,
        cache_dir=cache_dir,
    )
    train_ds = train_ds.map(process_function)
    print("학습 데이터셋 준비 완료.")

    # eval (non-streaming)
    print(f"평가(eval) split='{eval_split}' 로드 중...")
    eval_ds = load_with_retry(
        "librispeech_asr",
        split=eval_split,
        revision=hf_revision,
        cache_dir=cache_dir,
    )
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.map(
        process_function,
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )
    print("평가 데이터셋 준비 완료.")

    return DatasetDict({"train": train_ds, "test": eval_ds})
