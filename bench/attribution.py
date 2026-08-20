"""한 장에 대한 입력 gradient 크기 지도. E2와 E3가 공유하는 유일한 계산이다.

두 실험이 이 위에서 갈린다 — E2는 이미지마다 peak로 나눠 평균하고(`bench/erf.py`),
E3는 정규화 없이 객체마다 한 장을 그대로 쓴다. 그 차이가 각자의 파일에 있어야
어느 쪽을 고쳐도 다른 쪽이 조용히 따라 바뀌지 않는다.
"""
from typing import Callable

import numpy as np
import torch


def gradient_map(
    scalar_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """|d scalar_fn(x) / dx|를 채널에 걸쳐 합한 (H, W) 지도. 정규화하지 않는다.

    호출자의 텐서를 건드리지 않기 위해 device로 옮긴 뒤 clone한다 — 그러지
    않으면 grad가 호출자 텐서에 남아 다음 인스턴스의 측정에 섞인다.
    """
    x = image.unsqueeze(0).to(device).clone().requires_grad_(True)
    scalar_fn(x).sum().backward()
    return x.grad.detach().abs().sum(dim=1)[0].cpu().numpy().astype(np.float64)
