"""추론 latency. CUDA는 비동기라 벽시계로 재면 커널 큐잉 시간만 재게 된다."""
import statistics
import time

import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    warmup: int = 50,
    iters: int = 100,
    device: str = "cuda",
) -> float:
    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)
    use_cuda = device.startswith("cuda")

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
