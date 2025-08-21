# ==============================================================================
# 파일: data_prep/prepare_zeroth_korean.py
# 역할: 'Bingsu/zeroth-korean' 데이터셋을 학습 및 평가용으로 로드하고 전처리합니다.
#      - 학습(train): 스트리밍 방식으로 로드하여 디스크 공간과 로딩 시간을 절약합니다.
#      - 평가(test): 일반 방식으로 로드하여 검증을 수행합니다.
# ==============================================================================
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from whisper.normalizers.basic import BasicTextNormalizer

def load_and_prepare_dataset(processor: WhisperProcessor):
    """
    Bingsu/zeroth-korean 데이터셋을 로드하고 전처리합니다.
    학습은 스트리밍, 평가는 일반 방식으로 로드합니다.
    """
    # --- 1. 텍스트 정규화 및 전처리 함수 정의 ---
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 수정 지점: process_function의 정의를 파일 상단으로 이동
    # --------------------------------------------------------
    normalizer = BasicTextNormalizer()

    def process_function(batch):
        """데이터셋의 각 샘플에 대한 전처리 함수"""
        # 오디오를 16kHz로 리샘플링하고 Mel Spectrogram으로 변환
        audio = batch["audio"]
        # 스트리밍 데이터는 decode_example을 직접 호출할 필요가 없습니다.
        # processor가 내부적으로 처리합니다.
        input_features = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # 텍스트를 정규화하고 토큰 ID로 변환
        normalized_text = normalizer(batch["text"])
        labels = processor(text=normalized_text).input_ids
        
        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch


    # --- 2. 학습(train) 데이터셋 준비 (스트리밍 방식) ---
    print("학습(train) 데이터셋을 스트리밍 방식으로 로드합니다...")
    train_dataset = load_dataset("Bingsu/zeroth-korean", split="train", streaming=True)
    # 이제 process_function이 정의되었으므로 에러 없이 호출 가능
    train_dataset = train_dataset.map(process_function)
    print("학습 데이터셋 준비 완료.")

    # --- 3. 평가(test) 데이터셋 준비 (일반 방식) ---
    print("평가(test) 데이터셋을 일반 방식으로 로드합니다...")
    test_dataset = load_dataset("Bingsu/zeroth-korean", split="test")
    
    print("평가 데이터셋의 오디오 데이터를 메모리에 미리 로드합니다...")
    test_dataset = test_dataset.map(lambda example: {"audio": example["audio"]})

    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.map(
        process_function, 
        remove_columns=test_dataset.column_names,
        num_proc=1 
    )
    print("평가 데이터셋 준비 완료.")

    return DatasetDict({"train": train_dataset, "test": test_dataset})

# ==============================================================================
#       ↓↓↓ 기능 단위 검증 코드 (수정됨) ↓↓↓
# ==============================================================================
if __name__ == "__main__":
    import time

    print("="*50)
    print("데이터 전처리 모듈 기능 단위 검증을 시작합니다.")
    print("="*50)

    # --- 프로세서 로드 ---
    model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_name)
    print(f"✅ '{model_name}' 프로세서 로드 완료.")

    # --- 학습 데이터셋(스트리밍) 검증 ---
    try:
        print("\n--- 학습 데이터셋 (스트리밍) 테스트 ---")
        start_time = time.time()
        streaming_train_ds = load_dataset("Bingsu/zeroth-korean", split="train", streaming=True)
        sample = next(iter(streaming_train_ds))
        duration = time.time() - start_time
        print(f"✅ 학습 데이터 스트리밍 로드 성공! (첫 샘플 로딩 시간: {duration:.2f}초)")
        print(f"   샘플 데이터 확인: {sample['text']}")
    except Exception as e:
        print(f"❌ 학습 데이터셋 로드 실패: {e}")
        exit()

    # --- 평가 데이터셋(일반) 검증 ---
    try:
        print("\n--- 평가 데이터셋 (일반) 테스트 ---")
        start_time = time.time()
        # .select()를 이용해 작은 부분만 테스트
        test_ds = load_dataset("Bingsu/zeroth-korean", split="test").select(range(100))
        duration = time.time() - start_time
        print(f"✅ 평가 데이터 로드 성공! (100개 샘플 로딩 시간: {duration:.2f}초)")
        print(f"   데이터셋 정보: {test_ds}")
    except Exception as e:
        print(f"❌ 평가 데이터셋 로드 실패: {e}")
        exit()
        
    print("\n\n🎉 모든 기능 단위 검증을 통과했습니다!")