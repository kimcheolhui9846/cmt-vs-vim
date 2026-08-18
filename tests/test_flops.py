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


def test_op_handler_result_is_reported_as_analytic_not_traced():
    """핸들러 값은 그래프에서 센 값이 아니라 공식이다.

    합계에는 들어가되 traced와 섞이면 안 된다. 섞이는 순간 "이 숫자가 어디서
    왔는가"에 답할 수 없게 되고, 그게 이 저장소가 없애려는 문제 그 자체다.
    """
    def sum_handler(inputs, outputs):
        return 5  # Analytically: sum reduction over 10 elements = ~5 ops

    result = count_flops(
        CustomOpModule(),
        input_shape=(10,),
        op_handlers={'aten::sum': sum_handler}
    )
    assert result.analytic == 5
    assert result.traced == 0
    assert result.total == 5


class LinearPlusCustomOp(nn.Module):
    """세어지는 연산과 핸들러가 채우는 연산이 한 모델에 같이 있는 경우."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc(x) * x.sum()


def test_traced_and_analytic_are_separated_within_one_model():
    """Vim 행이 정확히 이 모양이다 — 절반은 측정, 절반은 공식."""
    result = count_flops(
        LinearPlusCustomOp(),
        input_shape=(10,),
        op_handlers={'aten::sum': lambda inputs, outputs: 5},
    )
    assert result.traced == 200, "Linear의 MAC이 analytic으로 새어 들어갔다"
    assert result.analytic == 5
    assert result.total == 205


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
