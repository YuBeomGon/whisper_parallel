# ==============================================================================
# 파일: utils/data_collator.py
# 역할: 음성-텍스트 Seq2Seq 모델 학습을 위한 데이터 콜레이터를 정의합니다.
#      - 배치 내의 오디오 피처(input_features)와 텍스트 레이블(labels)을
#        가장 긴 샘플 길이에 맞춰 동적으로 패딩합니다.
#      - 레이블의 패딩 토큰은 loss 계산에서 제외되도록 -100으로 마스킹합니다.
# ==============================================================================
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Speech-to-text Seq2Seq 모델을 위한 데이터 콜레이터 클래스.
    Hugging Face의 WhisperProcessor 객체를 사용하여 오디오와 텍스트를 처리합니다.
    """
    processor: Any
    model_dtype: Optional[torch.dtype] = None   # ★ 추가: 모델 dtype(f16/bf16)로 캐스팅용

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        데이터 샘플 리스트를 하나의 배치로 묶고 패딩을 적용합니다.
        이 함수는 Trainer에 의해 호출됩니다.

        Args:
            features (List[Dict[str, Any]]): 
                각각 'input_features'와 'labels' 키를 포함하는 딕셔너리 리스트.
                예: [{'input_features': [...], 'labels': [...]}, ...]

        Returns:
            Dict[str, torch.Tensor]: 모델 학습에 바로 사용할 수 있는 패딩된 배치.
        """
        # 1. 오디오 입력(input_features) 분리 및 패딩
        # 배치 내 모든 input_features를 동일한 길이로 맞추기 위해 패딩합니다.
        # features에서 input_features만 추출
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # feature_extractor를 사용하여 패딩 수행
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        
        # ★ 모델 dtype으로 캐스팅
        if self.model_dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.model_dtype)        

        # 2. 텍스트 라벨(labels) 분리 및 패딩
        # 배치 내 모든 라벨 시퀀스를 동일한 길이로 맞추기 위해 패딩합니다.
        # features에서 labels만 추출
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # tokenizer를 사용하여 패딩 수행
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # 3. 라벨의 패딩 토큰을 -100으로 교체 (Loss 계산 시 무시)
        # PyTorch CrossEntropyLoss는 라벨 값이 -100인 위치의 loss를 계산하지 않습니다.
        # tokenizer.pad_token_id로 패딩된 부분을 -100으로 마스킹합니다.
        # attention_mask가 0인 부분(패딩된 부분)을 찾아 -100으로 채웁니다.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 4. Whisper 모델은 디코더 입력을 내부적으로 처리하므로,
        #    BOS 토큰을 수동으로 시프트할 필요가 없습니다.
        #    최종적으로 패딩된 labels를 배치에 추가합니다.
        batch["labels"] = labels

        return batch