"""FLOPs 계측.

fvcore는 핸들러가 없는 연산을 0으로 센다. 그 침묵이 가장 위험한 실패 모드라
(Vim의 selective scan이 통째로 사라진다) 미등록 연산을 항상 함께 반환한다.
"""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


@dataclass(frozen=True)
class FlopResult:
    traced: int
    uncounted_ops: tuple[str, ...]


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

    for name, handler in (op_handlers or {}).items():
        analysis.set_op_handle(name, handler)

    total = analysis.total()
    uncounted = tuple(sorted(analysis.unsupported_ops().keys()))
    return FlopResult(traced=total, uncounted_ops=uncounted)
