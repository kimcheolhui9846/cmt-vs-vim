"""FLOPs 계측.

fvcore는 핸들러가 없는 연산을 0으로 센다. 그 침묵이 가장 위험한 실패 모드라
(Vim의 fused op이 통째로 사라진다) 미등록 연산을 항상 함께 반환한다.

핸들러가 채운 값은 그래프에서 센 값이 아니라 공식으로 계산한 값이다. 출처가 다른
두 수를 하나로 합치면 "이 숫자가 어디서 왔는지" 가 다시 사라지므로 — 그게 이
저장소가 존재하는 이유다 — traced 와 analytic 을 분리해 반환한다.
"""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


# fvcore가 규약상 세지 않는 연산들 — elementwise · 정규화 · 형상 변경. 세 모델에
# 같은 규약이 적용되므로 이것들이 빠져도 비교는 공정하다.
#
# 이 목록의 존재 이유는 "미등록 = 항상 위험"이 아니기 때문이다. 진짜 위험한 것은
# **연산량을 실제로 지닌** 연산이 미등록으로 남는 경우다. 실제로 두 번 겪었다:
#   - aten::scaled_dot_product_attention — DeiT의 attention matmul을 통째로 삼켰다
#     (4.25G로 측정, 공개값 4.6G). timm이 fused SDPA를 쓰기 때문이다.
#   - prim::PythonOp.MambaInnerFnNoOutProj — Vim의 conv1d·x_proj·dt_proj·scan 전부.
# 둘 다 이 목록에 없다. 목록에 없는 연산이 미등록으로 남으면 실패해야 한다.
#
# 여기에 연산을 추가할 때는 "그 연산이 정말 FLOPs를 지니지 않는가"를 먼저 답할 것.
# 통과시키려고 추가하는 순간 이 가드는 아무것도 막지 않는다.
FLOP_FREE_OPS = frozenset(
    {
        "aten::add",
        "aten::add_",
        "aten::div",
        "aten::exp",
        "aten::flip",
        "aten::gelu",
        "aten::mean",
        "aten::mul",
        "aten::mul_",
        "aten::neg",
        "aten::pow",
        "aten::rsqrt",
        "aten::silu",
        "aten::softmax",
        "aten::sub",
        "prim::PythonOp.SwishImplementation",
    }
)


@dataclass(frozen=True)
class FlopResult:
    traced: int  # fvcore가 그래프에서 직접 센 값
    analytic: int  # 등록된 핸들러가 공식으로 채운 값
    uncounted_ops: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.traced + self.analytic

    @property
    def unexpected_uncounted_ops(self) -> tuple[str, ...]:
        """규약상 0이라고 볼 수 없는데도 세어지지 않은 연산.

        비어 있지 않으면 그 행의 FLOPs는 과소 계상된 값이다.
        """
        return tuple(op for op in self.uncounted_ops if op not in FLOP_FREE_OPS)


def _strip_namespace(op: str) -> str:
    return op.split("::", 1)[-1]


def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
    op_handlers: dict[str, Callable] | None = None,
    device: str = "cpu",
) -> FlopResult:
    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)

    analysis = FlopCountAnalysis(model, x)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)

    handlers = op_handlers or {}
    for name, handler in handlers.items():
        analysis.set_op_handle(name, handler)

    # fvcore의 by_operator()는 네임스페이스를 떼고 보고한다("aten::sum" -> "sum",
    # "prim::PythonOp.X" -> "PythonOp.X"). 핸들러는 붙은 이름으로 등록하므로 양쪽을
    # 같은 형태로 맞춰야 한다. 안 맞추면 analytic이 조용히 0이 되고, 공식으로 채운
    # 값이 traced로 둔갑해 출처가 사라진다.
    handler_names = {_strip_namespace(name) for name in handlers}
    by_operator = analysis.by_operator()
    analytic = int(
        sum(
            count
            for op, count in by_operator.items()
            if _strip_namespace(op) in handler_names
        )
    )
    total = int(analysis.total())
    uncounted = tuple(sorted(analysis.unsupported_ops().keys()))
    return FlopResult(
        traced=total - analytic, analytic=analytic, uncounted_ops=uncounted
    )
