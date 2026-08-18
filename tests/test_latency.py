import torch
import torch.nn as nn

from bench.latency import measure_latency


class CountingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        self.calls += 1
        return self.fc(x)


def test_runs_warmup_plus_measured_iterations():
    model = CountingModule()
    measure_latency(model, input_shape=(4,), warmup=3, iters=7, device="cpu")
    assert model.calls == 10


def test_returns_a_positive_float():
    latency = measure_latency(
        CountingModule(), input_shape=(4,), warmup=2, iters=5, device="cpu"
    )
    assert isinstance(latency, float)
    assert latency > 0


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
    measure_latency(model, input_shape=(4,), warmup=1, iters=1, device="cpu")
    assert model.saw_grad_enabled is False
