import pytest
import torch

from bench.memory import measure_peak_memory


def test_oom_is_recorded_as_data_not_raised():
    def blows_up():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    result = measure_peak_memory(blows_up)
    assert result.status == "oom"
    assert result.peak_bytes is None


def test_other_exceptions_still_propagate():
    """OOM만 삼킨다. 진짜 버그를 'oom'으로 기록하면 결과가 오염된다."""

    def has_a_bug():
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        measure_peak_memory(has_a_bug)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_successful_run_reports_positive_peak():
    def allocates():
        torch.zeros(1024, 1024, device="cuda")

    result = measure_peak_memory(allocates)
    assert result.status == "ok"
    assert result.peak_bytes > 0
