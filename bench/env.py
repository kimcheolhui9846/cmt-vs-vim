"""결과 파일에 붙일 환경 정보. 재현이 안 되는 수치는 논문에 쓸 수 없다."""
import platform
import subprocess

import torch


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def snapshot() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
    }
