import torch
import torch.nn as nn

from bench.flops import count_flops


class TinyLinear(nn.Module):
    """Linear(10, 20) 한 겹. MAC = 10 * 20 = 200."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc(x)


def test_counts_a_known_linear_exactly():
    result = count_flops(TinyLinear(), input_shape=(10,))
    assert result.traced == 200


def test_reports_nothing_uncounted_for_supported_ops():
    result = count_flops(TinyLinear(), input_shape=(10,))
    assert result.uncounted_ops == ()


class CustomOpModule(nn.Module):
    """fvcore가 모르는 연산을 부르는 모듈."""

    def forward(self, x):
        return torch._C._nn.gelu(x) * x.sum()


def test_uncounted_ops_are_surfaced_not_silently_zero():
    """미등록 연산이 조용히 0으로 세어지면 Vim FLOPs가 통째로 사라진다."""
    result = count_flops(CustomOpModule(), input_shape=(10,))
    assert result.uncounted_ops != (), "미등록 연산이 보고되지 않았다"
