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


def test_op_handler_is_applied_to_traced_count():
    """A registered handler's FLOPs count is included in traced total."""
    def sum_handler(inputs, outputs):
        return 5  # Analytically: sum reduction over 10 elements = ~5 ops

    result = count_flops(
        CustomOpModule(),
        input_shape=(10,),
        op_handlers={'aten::sum': sum_handler}
    )
    # The handler must contribute its return value to traced
    assert result.traced == 5


def test_handler_registration_does_not_hide_unsupported_ops():
    """Registering a handler for one op must not remove other unsupported ops."""
    def sum_handler(inputs, outputs):
        return 5

    result = count_flops(
        CustomOpModule(),
        input_shape=(10,),
        op_handlers={'aten::sum': sum_handler}
    )
    # Even with sum handled, gelu and mul must remain in uncounted_ops
    assert 'aten::gelu' in result.uncounted_ops
    assert 'aten::mul' in result.uncounted_ops
