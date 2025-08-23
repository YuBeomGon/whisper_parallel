import re, sys, pathlib

def clean_script(text: str) -> str:
    # 1) 각주/메모 라인 제거(※, <...>, “예” 인 경우 … 등)
    lines = text.splitlines()
    keep = []
    for ln in lines:
        if ln.strip().startswith("※"): continue
        if re.match(r"^\s*<.*?>\s*$", ln): continue
        if "“예” 인 경우" in ln or '"예" 인 경우' in ln: continue
        keep.append(ln)
    text = "\n".join(keep)

    # 2) 괄호() 안 지시문 제거(문장부호와 공백 정돈)
    text = re.sub(r"\([^)]*\)", "", text)

    # 3) 별표 앞뒤 공백 정리 및 남은 별표 삭제
    text = text.replace("*", "")

    # 4) 고유번호 안내 대괄호 → 변수 자리
    text = re.sub(r"\[설계사 본인의 고유번호 14자리\]", "{AGENT_ID14}", text)

    # 5) ○○ 시각 → 변수 자리(원문 패턴 유지)
    text = text.replace("○○○○년 ○○월 ○○일 ○○시 ○○분", "{YYYY}년 {MM}월 {DD}일 {hh}시 {mm}분")

    # 6) 남은 이중 공백/빈 줄 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()

    # 7) 문장 끝 공백/마침표 정돈
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text

if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".clean.txt")
    dst.write_text(clean_script(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"saved: {dst}")
