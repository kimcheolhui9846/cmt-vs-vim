"""peak VRAM. OOM은 실패가 아니라 기록할 결과다."""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MemoryResult:
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    status: str


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM이 오는 두 가지 형태를 모두 인정한다.

    `torch.cuda.OutOfMemoryError`가 표준 경로지만, cuDNN workspace 할당 실패
    등에서는 메시지에 'out of memory'가 든 평범한 RuntimeError로 샌다. 후자를
    놓치면 sweep이 고해상도 셀에서 죽고, 하필 그 셀이 논문의 메모리 주장이
    걸린 자리다. 반대로 RuntimeError를 전부 삼키면 진짜 버그가 가짜 메모리
    한계로 결과표에 실리므로, 메시지를 확인해 구분한다.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def measure_peak_memory(fn: Callable[[], object]) -> MemoryResult:
    if not torch.cuda.is_available():
        try:
            fn()
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            return MemoryResult(None, None, "oom")
        return MemoryResult(None, None, "no_cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        fn()
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return MemoryResult(None, None, "oom")

    torch.cuda.synchronize()
    return MemoryResult(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        status="ok",
    )
