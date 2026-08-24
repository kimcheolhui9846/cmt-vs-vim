import pytest
import torch
import torch.nn as nn

from bench.latency import LatencyResult, measure_latency


class CountingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        self.calls += 1
        return self.fc(x)


def _measure(model, **kwargs):
    kwargs.setdefault("input_shape", (4,))
    kwargs.setdefault("warmup", 2)
    kwargs.setdefault("iters", 5)
    kwargs.setdefault("device", "cpu")
    return measure_latency(model, **kwargs)


def test_runs_warmup_plus_measured_iterations():
    model = CountingModule()
    _measure(model, warmup=3, iters=7, repeats=1)
    assert model.calls == 10


def test_returns_a_positive_median():
    result = _measure(CountingModule())
    assert isinstance(result, LatencyResult)
    assert isinstance(result.median_ms, float)
    assert result.median_ms > 0


def test_does_not_track_gradients():
    """grad를 켜고 재면 학습 경로 비용이 섞여 추론 latency가 아니게 된다."""

    class GradProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.saw_grad_enabled = None
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            self.saw_grad_enabled = torch.is_grad_enabled()
            return self.fc(x)

    model = GradProbe()
    _measure(model, warmup=1, iters=1)
    assert model.saw_grad_enabled is False


# --- 반복 측정 ---------------------------------------------------------------


def test_each_repeat_warms_up_again():
    """반복은 같은 워밍업 상태를 다시 표본화하는 게 아니라, 측정 블록 자체를 다시
    돈다. 워밍업을 한 번만 하면 첫 블록이 만들어 놓은 클럭·할당자 상태를 물려받아
    실행 간 차이가 그대로 감춰진다 — 이 실험이 잡으려는 게 바로 그 차이다."""
    model = CountingModule()
    _measure(model, warmup=3, iters=7, repeats=2)
    assert model.calls == 20


def test_keeps_every_repeat_not_just_the_summary():
    """중앙값만 남기면 30 ms와 16 ms가 하나의 숫자로 합쳐져, 재현되지 않는다는
    사실이 CSV에서 사라진다."""
    result = _measure(CountingModule(), repeats=3)
    assert len(result.repeats_ms) == 3
    assert all(sample > 0 for sample in result.repeats_ms)


def test_median_is_the_median_of_the_repeats():
    result = LatencyResult(repeats_ms=(10.0, 30.0, 20.0))
    assert result.median_ms == 20.0


def test_exposes_the_range_across_repeats():
    result = LatencyResult(repeats_ms=(16.67, 30.0, 18.0))
    assert result.min_ms == 16.67
    assert result.max_ms == 30.0
    assert result.spread == pytest.approx(30.0 / 16.67)


def test_a_single_repeat_is_still_a_result():
    result = _measure(CountingModule(), repeats=1)
    assert len(result.repeats_ms) == 1
    assert result.median_ms == result.repeats_ms[0]
    assert result.spread == 1.0


def test_zero_repeats_is_refused_not_silently_empty():
    """0회는 '측정하지 않았다'인데, 조용히 빈 결과를 내면 sweep이 그걸 값처럼
    CSV에 적는다."""
    with pytest.raises(ValueError, match="repeats"):
        _measure(CountingModule(), repeats=0)
