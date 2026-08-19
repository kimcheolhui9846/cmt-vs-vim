"""Luo et al.(2016) 방식의 ERF 측정.

한 장씩 backward를 돌린다. 배치로 묶으면 중심 토큰 스칼라의 합에 대한 gradient가
되어 이미지별 정규화를 할 수 없고, 정규화가 없으면 gradient가 큰 한 장이 평균을
삼킨다.
"""
import numpy as np
import torch
import torch.nn as nn

from models.probes import center_token_scalar


def accumulate_erf(
    model_name: str,
    model: nn.Module,
    images: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    model = model.to(device).eval()
    total = np.zeros(images.shape[-2:], dtype=np.float64)

    for image in images:
        x = image.unsqueeze(0).to(device).clone().requires_grad_(True)
        center_token_scalar(model_name, model, x).sum().backward()
        grad = x.grad.detach().abs().sum(dim=1)[0].cpu().numpy().astype(np.float64)
        peak = grad.max()
        if peak > 0:
            grad = grad / peak
        total += grad

    return total / len(images)
