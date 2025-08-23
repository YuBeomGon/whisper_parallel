# ==============================================================================
# 파일: utils/lm_collator.py
# 역할: 텍스트용 LM 데이터 콜레이터 (Whisper 토크나이저 사용)
#       - longest padding
#       - labels에서 pad→-100 마스킹
# ==============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class DataCollatorWhisperLM:
    tokenizer: Any  # processor.tokenizer
    pad_to_multiple_of: int | None = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # features: [{"input_ids": [...]}]
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        # labels는 input_ids 복사 후 pad 위치를 -100으로
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
