# ==============================================================================
# data/text_lm_dataset.py
# 목적: 여러 txt 파일을 읽어 하나로 이어붙이고 block_size 단위로 잘라 샘플을 만드는 데이터셋
# 사용 예: 
#   python -m data.text_lm_dataset --files data/text_corpus/*.txt --out data/text_lm_dataset.py
# ==============================================================================
from typing import List, Iterator
import pathlib
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

class PackedWhisperTextDataset(Dataset):
    """
    여러 txt 파일(UTF-8, 한 줄 = 한 문장)을 읽어
    [ ... 문장1 + EOS, 문장2 + EOS, ... ]를 하나로 이어붙이고
    block_size 단위로 잘라 샘플을 만드는 데이터셋.
    """
    def __init__(self, processor: WhisperProcessor, files: List[str], block_size: int = 448):
        self.processor = processor
        self.tok = processor.tokenizer
        self.block_size = block_size

        # 읽기
        corpus: List[str] = []
        for p in files:
            text = pathlib.Path(p).read_text(encoding="utf-8")
            # 빈 줄/공백 라인 제거, 좌우 공백 정리
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            corpus.extend(lines)

        # 토큰으로 이어붙이기: 문장마다 EOS 추가
        all_ids: List[int] = []
        eos = self.tok.eos_token_id
        for sent in corpus:
            ids = self.tok.encode(sent, add_special_tokens=False)
            all_ids.extend(ids + [eos])

        # block 단위로 자르기
        self.blocks: List[List[int]] = []
        for i in range(0, len(all_ids), block_size):
            chunk = all_ids[i : i + block_size]
            self.blocks.append(chunk)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = self.blocks[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        # pad 없음(마지막 블록이 짧다면 collator에서 pad)
        return {"input_ids": input_ids}
