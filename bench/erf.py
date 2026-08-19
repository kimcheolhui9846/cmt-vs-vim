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
    """√(λ_max / λ_min). 등방이면 1.0.

    2차 모먼트는 거리 제곱으로 가중하므로 far-field 꼬리에 지배된다 — 중심의
    좁은 능선이 비등방이어도 넓고 등방적인 배경(pedestal)이 있으면 전체 지수는
    등방 쪽으로 끌려간다. `central_crop`으로 꼬리를 자른 맵에 이 함수를 다시
    적용하면(`anisotropy_central`) 그 효과를 분리해 볼 수 있다.
    """
    eigenvalues = np.linalg.eigvalsh(_covariance(erf))
    return float(np.sqrt(eigenvalues.max() / eigenvalues.min()))


def central_crop(erf: np.ndarray, size: int = 128) -> np.ndarray:
    """배열 중심에서 size×size만큼 잘라낸다.

    새 수학이 아니다 — 기존 `anisotropy_index`를 꼬리 없는 맵에 그대로 적용해
    "지수가 far-field에 얼마나 좌우되는가"를 데이터로 드러내려는 목적이다.
    """
    n_rows, n_cols = erf.shape
    row0 = (n_rows - size) // 2
    col0 = (n_cols - size) // 2
    return erf[row0 : row0 + size, col0 : col0 + size]


def principal_angle_deg(erf: np.ndarray) -> float:
    """주축과 수평축 사이 각도, [0, 90]. 지수가 1에 가까우면 의미가 없다."""
    values, vectors = np.linalg.eigh(_covariance(erf))
    x, y = vectors[:, int(values.argmax())]
    return float(np.degrees(np.arctan2(abs(y), abs(x))))


def _log_slope(profile: np.ndarray) -> float:
    positive = profile > 0
    distances = np.arange(1, len(profile) + 1)[positive]
    return float(np.polyfit(distances, np.log(profile[positive]), 1)[0])


MIN_DECAY_WINDOW = 8
"""log-log 기울기를 적합하는 데 필요한 최소 반경.

이보다 좁은 창에서 낸 기울기 비는 몇 개 점의 잡음에 지배되어 모델 간 비교에 쓸 수
없다. 8은 임의의 하한이 아니다 — 이 실험이 쓰는 224² 입력의 ViT 계열 patch 크기
16의 절반으로, 그보다 좁으면 "감쇠 프로파일"이라 부를 만한 표본 자체가 없다.
"""


def _peak(erf: np.ndarray) -> tuple[int, int]:
    return np.unravel_index(erf.argmax(), erf.shape)


def decay_window(erf: np.ndarray, max_distance: int = 64) -> int:
    """피크에서 상하좌우 네 방향 모두로 배열 밖을 넘지 않는 최대 반경과
    max_distance 중 작은 값.

    피크가 경계에 가까우면 `decay_ratio`가 `erf[row ± offsets, col]`로 배열 밖을
    읽는다(IndexError). 이 함수가 실제로 쓸 수 있는 반경을 먼저 계산해 그 사고를
    막는다. 중심 피크(row=col=112, 224² 배열)에서는 상하좌우 여유가 전부
    max_distance(64)보다 크므로 그대로 64를 돌려준다 — 기존 동작과 동일하다.
    """
    row, col = _peak(erf)
    n_rows, n_cols = erf.shape
    up, down = row, n_rows - 1 - row
    left, right = col, n_cols - 1 - col
    return int(min(max_distance, up, down, left, right))


def decay_ratio(erf: np.ndarray, max_distance: int = 64) -> tuple[float, int]:
    """(수직 감쇠 기울기 / 수평 감쇠 기울기, 실제로 쓴 반경)을 함께 돌려준다.
    비율이 1보다 크면 수직이 더 가파르다.

    반경을 `decay_window()`로 따로 조회해 호출자가 별도로 다시 계산하게 두면,
    기록된 반경이 실제로 이 비율을 낸 반경과 같다는 보장이 코드 구조상 없다 —
    두 번의 호출이 어긋날 수 있기 때문이다. 그래서 한 번의 호출에서 값과
    반경을 쌍으로 돌려준다.

    피크가 경계에서 `MIN_DECAY_WINDOW`보다 가까우면 조용히 더 좁은 창으로
    계산하지 않고 ValueError를 던진다 — 반경이 셀마다 다르게 좁아지면 같은 열에
    반경이 다른 값이 섞여 비교 불가능한 숫자가 된다.
    """
    row, col = _peak(erf)
    window = decay_window(erf, max_distance)
    if window < MIN_DECAY_WINDOW:
        raise ValueError(
            f"피크 ({row}, {col})가 경계에서 반경 {window}만큼만 떨어져 있다 "
            f"(최소 {MIN_DECAY_WINDOW} 필요) — 감쇠 기울기를 신뢰할 수 없다."
        )
    offsets = np.arange(1, window + 1)
    horizontal = (erf[row, col + offsets] + erf[row, col - offsets]) / 2
    vertical = (erf[row + offsets, col] + erf[row - offsets, col]) / 2
    return float(_log_slope(vertical) / _log_slope(horizontal)), window


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
