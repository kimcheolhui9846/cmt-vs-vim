import torch
import torch.nn as nn

from bench.throughput import find_max_batch, measure_throughput


class FitsUpTo(nn.Module):
    """batch가 threshold를 넘으면 OOM을 던지는 가짜 모델."""

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        if x.shape[0] > self.threshold:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return self.fc(x)


def test_finds_the_largest_batch_that_fits():
    assert find_max_batch(FitsUpTo(6), input_shape=(4,), device="cpu") == 6


def test_finds_exact_power_of_two_boundary():
    assert find_max_batch(FitsUpTo(8), input_shape=(4,), device="cpu") == 8


def test_returns_zero_when_even_batch_one_fails():
    assert find_max_batch(FitsUpTo(0), input_shape=(4,), device="cpu") == 0


def test_respects_the_upper_limit():
    """상한이 없으면 큰 모델에서 탐색이 끝없이 늘어난다."""
    assert find_max_batch(FitsUpTo(9999), input_shape=(4,), device="cpu", limit=32) == 32


def test_throughput_reports_batch_and_rate():
    result = measure_throughput(FitsUpTo(4), input_shape=(4,), device="cpu")
    assert result.batch == 4
    assert result.images_per_sec > 0


def test_throughput_is_none_when_nothing_fits():
    result = measure_throughput(FitsUpTo(0), input_shape=(4,), device="cpu")
    assert result.batch == 0
    assert result.images_per_sec is None


class RaisesOomWordedRuntimeError(nn.Module):
    """OOM이 평범한 RuntimeError로 새는 형태. bench.memory와 판정이 일치해야 한다."""

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        if x.shape[0] > self.threshold:
            raise RuntimeError("CUDA error: out of memory allocating workspace")
        return self.fc(x)


def test_batch_search_treats_oom_worded_runtime_error_as_not_fitting():
    """배치 탐색은 일부러 OOM을 유발한다. 여기서 판정이 좁으면 탐색이 죽는다."""
    assert find_max_batch(
        RaisesOomWordedRuntimeError(6), input_shape=(4,), device="cpu"
    ) == 6


class HasABug(nn.Module):
    def forward(self, x):
        raise RuntimeError("shape '[2, 3]' is invalid for input of size 7")


def test_batch_search_does_not_swallow_unrelated_runtime_errors():
    """진짜 버그를 '안 들어감'으로 처리하면 max_batch가 조용히 0이 된다."""
    import pytest

    with pytest.raises(RuntimeError, match="invalid for input"):
        find_max_batch(HasABug(), input_shape=(4,), device="cpu")
