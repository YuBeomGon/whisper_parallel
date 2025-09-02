# ==============================================================================
# 파일: utils/logger.py
# 역할: 학습 로그 초기화 및 설정
# ==============================================================================
import os
import sys
from datetime import datetime
from pathlib import Path
import shutil
import subprocess


def _init_run_logger(decoder_mode: str) -> str:
    """
    STDOUT/STDERR를 콘솔 + 파일 동시 기록(tee)로 설정하고
    실행/환경 스냅샷을 로그 헤더에 남긴다.
    반환: 로그 파일 절대 경로(str)
    """
    # 날짜/시간 + prefix
    now = datetime.now()
    date = now.strftime("%y%m%d")     # 예: 250824
    tstr = now.strftime("%H%M%S")     # 예: 134522
    prefix = "whisper_origin" if decoder_mode == "vanilla" else "whisper_parallel"

    # logs/<YYMMDD>/<prefix>_<HHMMSS>.log
    log_dir = Path("logs") / date
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_{tstr}.log"

    # tee 구현(콘솔 + 파일)
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    f = open(log_file, "a", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)

    # 헤더(환경 스냅샷)
    print("=" * 60)
    print(f" LAUNCH @ {now.isoformat(timespec='seconds')}")
    print(f" CWD       : {os.getcwd()}")
    print(f" CMD ARGS  : {' '.join(sys.argv)}")
    try:
        import transformers
        print(f" TRANSFORMERS: {transformers.__version__}")
    except Exception:
        pass
    try:
        import torch as _torch
        print(f" TORCH       : {_torch.__version__}")
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        print(" GPU:\n" + out)
    except Exception:
        print(" GPU: n/a")

    # 현재 파일 스냅샷 저장
    try:
        src_path = Path(__file__).resolve()
        snap_path = log_dir / f"run_training_{tstr}.py"
        shutil.copyfile(src_path, snap_path)
        print(f" SNAPSHOT  : saved {src_path.name} -> {snap_path.name}")
    except Exception as e:
        print(f" SNAPSHOT  : failed ({e})")

    # latest 심볼릭 링크
    try:
        latest = log_dir / f"latest_{prefix}.log"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_file.name)  # 상대 링크
    except Exception:
        pass

    print("=" * 60)
    return str(log_file.resolve())
