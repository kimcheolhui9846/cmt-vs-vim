"""8GB에 들어가는 최대 배치와 그때의 처리량.

배치를 2배씩 올려 실패 지점을 잡고, 그 구간을 이분 탐색한다. 선형 증가는
큰 모델에서 너무 느리고, 2배만 쓰면 6과 8을 구분하지 못한다.
"""
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from bench.memory import is_oom


@dataclass(frozen=True)
class ThroughputResult:
    batch: int
    images_per_sec: float | None


def _fits(model: nn.Module, input_shape: tuple[int, ...], batch: int, device: str) -> bool:
    """`bench.memory`와 같은 OOM 판정을 쓴다. 두 모듈이 기준을 달리하면,
    메모리 측정에서는 oom으로 기록되는 상황이 배치 탐색에서는 예외로 터진다."""
    try:
        with torch.no_grad():
            model(torch.zeros(batch, *input_shape, device=device))
    except RuntimeError as exc:
        if not is_oom(exc):
            raise
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return False
    return True


def find_max_batch(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
    limit: int = 512,
) -> int:
    model = model.to(device).eval()

    if not _fits(model, input_shape, 1, device):
        return 0

    low = 1
    high = 2
    while high <= limit and _fits(model, input_shape, high, device):
        low = high
        high *= 2

    high = min(high, limit + 1)
    while low + 1 < high:
        mid = (low + high) // 2
        if _fits(model, input_shape, mid, device):
            low = mid
        else:
            high = mid
    return min(low, limit)


def measure_throughput(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
    limit: int = 512,
    warmup: int = 5,
    iters: int = 20,
) -> ThroughputResult:
    batch = find_max_batch(model, input_shape, device=device, limit=limit)
    if batch == 0:
        return ThroughputResult(batch=0, images_per_sec=None)

    model = model.to(device).eval()
    x = torch.zeros(batch, *input_shape, device=device)
    use_cuda = device.startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if use_cuda:
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)

    return ThroughputResult(
        batch=batch, images_per_sec=batch / statistics.median(samples)
    )
