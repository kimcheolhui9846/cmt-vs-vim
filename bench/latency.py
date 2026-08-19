"""추론 latency. CUDA는 비동기라 벽시계로 재면 커널 큐잉 시간만 재게 된다.

배치 1 latency는 커널 실행 자체보다 실행 오버헤드에 좌우돼서, 같은 조건을 다시
재도 값이 달라진다 — vim_s@224가 sweep에서는 30.00 ms, 격리 측정에서는 16.67 ms로
1.8배 차이가 났다. 그래서 한 번 재고 끝내지 않고 측정 블록을 통째로 여러 번 돌려
반복 간 편차를 결과에 남긴다. 중앙값 하나만 남기면 재현되지 않는다는 사실이
숫자에서 사라진다.
"""
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LatencyResult:
    """반복별 중앙값 목록. 요약값은 전부 여기서 파생된다."""

    repeats_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.repeats_ms)

    @property
    def min_ms(self) -> float:
        return min(self.repeats_ms)

    @property
    def max_ms(self) -> float:
        return max(self.repeats_ms)

    @property
    def spread(self) -> float:
        """최대/최소 비. 1.0이면 반복이 일치했다는 뜻이다."""
        return self.max_ms / self.min_ms


def _measure_once(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    use_cuda: bool,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if use_cuda:
            torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(x)
                samples.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(samples)


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    warmup: int = 50,
    iters: int = 100,
    device: str = "cuda",
    repeats: int = 3,
) -> LatencyResult:
    if repeats < 1:
        raise ValueError(f"repeats는 1 이상이어야 한다 (받은 값: {repeats})")

    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)
    use_cuda = device.startswith("cuda")

    # 반복마다 워밍업을 다시 돈다. 워밍업을 밖으로 빼면 두 번째 반복부터는 첫
    # 블록이 만들어 둔 상태를 물려받아, 실행 간 차이가 측정에서 지워진다.
    return LatencyResult(
        repeats_ms=tuple(
            _measure_once(model, x, warmup, iters, use_cuda) for _ in range(repeats)
        )
    )
