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


def _covariance(erf: np.ndarray) -> np.ndarray:
    """ERF를 합이 1인 2D 분포로 보고 2차 모먼트 공분산을 구한다."""
    weights = erf / erf.sum()
    rows, cols = np.indices(erf.shape)
    mean_row = float((weights * rows).sum())
    mean_col = float((weights * cols).sum())
    d_row = rows - mean_row
    d_col = cols - mean_col
    return np.array([
        [(weights * d_col * d_col).sum(), (weights * d_col * d_row).sum()],
        [(weights * d_col * d_row).sum(), (weights * d_row * d_row).sum()],
    ])


def anisotropy_index(erf: np.ndarray) -> float:
    """√(λ_max / λ_min). 등방이면 1.0."""
    eigenvalues = np.linalg.eigvalsh(_covariance(erf))
    return float(np.sqrt(eigenvalues.max() / eigenvalues.min()))


def principal_angle_deg(erf: np.ndarray) -> float:
    """주축과 수평축 사이 각도, [0, 90]. 지수가 1에 가까우면 의미가 없다."""
    values, vectors = np.linalg.eigh(_covariance(erf))
    x, y = vectors[:, int(values.argmax())]
    return float(np.degrees(np.arctan2(abs(y), abs(x))))


def _log_slope(profile: np.ndarray) -> float:
    positive = profile > 0
    distances = np.arange(1, len(profile) + 1)[positive]
    return float(np.polyfit(distances, np.log(profile[positive]), 1)[0])


def decay_ratio(erf: np.ndarray, max_distance: int = 64) -> float:
    """수직 감쇠 기울기 / 수평 감쇠 기울기. 1보다 크면 수직이 더 가파르다."""
    row, col = np.unravel_index(erf.argmax(), erf.shape)
    offsets = np.arange(1, max_distance + 1)
    horizontal = (erf[row, col + offsets] + erf[row, col - offsets]) / 2
    vertical = (erf[row + offsets, col] + erf[row - offsets, col]) / 2
    return float(_log_slope(vertical) / _log_slope(horizontal))


def has_converged(values: list[float], tolerance: float = 0.05) -> bool:
    """마지막 두 점의 상대 변화가 tolerance 이내인가.

    점이 하나뿐이면 판단할 근거가 없으므로 False다. True를 돌려주면 N=16으로 잰
    값이 '수렴한 값'으로 논문에 실린다.
    """
    if len(values) < 2:
        return False
    previous, latest = values[-2], values[-1]
    if previous == 0:
        return latest == 0
    return abs(latest - previous) / abs(previous) <= tolerance
