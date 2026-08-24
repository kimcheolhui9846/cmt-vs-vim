import pytest
import torch

from bench.memory import measure_peak_memory


def test_oom_is_recorded_as_data_not_raised():
    def blows_up():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    result = measure_peak_memory(blows_up)
    assert result.status == "oom"
    assert result.peak_allocated_bytes is None
    assert result.peak_reserved_bytes is None


def test_runtime_error_worded_as_oom_is_also_recorded_as_oom():
    """CUDA OOM은 OutOfMemoryError로만 오지 않는다. cuDNN 실패 등에서는 메시지에
    'out of memory'가 든 평범한 RuntimeError로 샌다. 이걸 놓치면 sweep이
    고해상도 셀에서 통째로 죽는데, 하필 거기가 논문 주장이 걸린 자리다."""

    def blows_up():
        raise RuntimeError("CUDA error: out of memory when allocating workspace")

    result = measure_peak_memory(blows_up)
    assert result.status == "oom"


def test_unrelated_runtime_error_still_propagates():
    """RuntimeError를 전부 삼키면 진짜 버그가 가짜 메모리 한계로 논문에 실린다."""

    def has_a_bug():
        raise RuntimeError("shape '[2, 3]' is invalid for input of size 7")

    with pytest.raises(RuntimeError, match="invalid for input"):
        measure_peak_memory(has_a_bug)


def test_other_exceptions_still_propagate():
    """OOM만 삼킨다. 진짜 버그를 'oom'으로 기록하면 결과가 오염된다."""

    def has_a_bug():
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        measure_peak_memory(has_a_bug)


def test_no_cuda_is_reported_distinctly_from_oom(monkeypatch):
    """세 상태 중 하나가 커버리지 0으로 나가면 안 된다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = measure_peak_memory(lambda: None)
    assert result.status == "no_cuda"
    assert result.peak_allocated_bytes is None
    assert result.peak_reserved_bytes is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_successful_run_reports_both_statistics():
    def allocates():
        torch.zeros(1024, 1024, device="cuda")

    result = measure_peak_memory(allocates)
    assert result.status == "ok"
    assert result.peak_allocated_bytes > 0
    assert result.peak_reserved_bytes >= result.peak_allocated_bytes
