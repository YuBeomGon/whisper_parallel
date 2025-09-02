# ==============================================================================
# data/extract_zeroth_text.py
# 목적: Bingsu/zeroth-korean 전사 텍스트를 추출해 한 줄=한 문장 txt로 저장
# 사용 예:
#   python -m data.extract_zeroth_text --out data/text_corpus/zeroth_corpus.txt \
#       --splits train test --dedup --min_chars 3
#
# 옵션:
#   --splits: 사용할 split 목록 (train, test 등)
#   --dedup:  중복 문장 제거
#   --min_chars: 너무 짧은 문장 제거
#   --hf_cache: HF_DATASETS_CACHE 지정(네트워크 이슈/캐시 분리용)
# ==============================================================================
import os
import argparse
from datasets import load_dataset
from whisper.normalizers.basic import BasicTextNormalizer

def parse_args():
    p = argparse.ArgumentParser("Extract zeroth-korean transcripts to a plain txt")
    p.add_argument("--out", type=str, required=True,
                   help="출력 파일 경로 (예: data/text_corpus/zeroth_corpus.txt)")
    p.add_argument("--splits", nargs="+", default=["train"],
                   help="사용할 split들 (예: train test)")
    p.add_argument("--dedup", action="store_true", help="중복 문장 제거")
    p.add_argument("--min_chars", type=int, default=3, help="최소 문자수 필터")
    p.add_argument("--hf_cache", type=str, default=None,
                   help="HF_DATASETS_CACHE 경로 (선택)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.hf_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_cache

    norm = BasicTextNormalizer()
    lines = []

    for sp in args.splits:
        print(f"[extract] loading zeroth split='{sp}'")
        ds = load_dataset("Bingsu/zeroth-korean", split=sp)  # 전부 메모리에 올려도 되는 크기
        for t in ds["text"]:
            s = norm(t.strip())
            if not s:
                continue
            if len(s) < args.min_chars:
                continue
            lines.append(s)

    if args.dedup:
        print("[extract] deduplication on")
        before = len(lines)
        lines = list(dict.fromkeys(lines))  # 순서보존 dedup
        print(f"[extract] {before} -> {len(lines)} after dedup")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")

    print(f"[extract] wrote {len(lines):,} lines to {args.out}")

if __name__ == "__main__":
    main()