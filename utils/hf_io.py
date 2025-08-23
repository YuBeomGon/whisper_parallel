# utils/hf_io.py
import time
from datasets import load_dataset

def load_with_retry(*args, max_tries=6, sleep_base=2, **kwargs):
    """
    HF Hub 50x/타임아웃을 지수 백오프로 재시도.
    sleep_base^i 초 대기 (2, 4, 8, 16, …)
    """
    last = None
    for i in range(max_tries):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            last = e
            wait = sleep_base ** i
            print(f"[HF retry] {i+1}/{max_tries} failed: {type(e).__name__}: {e} → retry in {wait}s")
            time.sleep(wait)
    raise last
