"""peak VRAM. OOM은 실패가 아니라 기록할 결과다."""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MemoryResult:
    peak_bytes: int | None
    status: str


def measure_peak_memory(fn: Callable[[], object]) -> MemoryResult:
    if not torch.cuda.is_available():
        try:
            fn()
        except torch.cuda.OutOfMemoryError:
            return MemoryResult(peak_bytes=None, status="oom")
        return MemoryResult(peak_bytes=None, status="no_cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return MemoryResult(peak_bytes=None, status="oom")

    torch.cuda.synchronize()
    return MemoryResult(peak_bytes=torch.cuda.max_memory_allocated(), status="ok")
