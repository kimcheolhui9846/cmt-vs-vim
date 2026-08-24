"""Luo et al.(2016) 방식의 ERF 측정.

한 장씩 backward를 돌린다. 배치로 묶으면 중심 토큰 스칼라의 합에 대한 gradient가
되어 이미지별 정규화를 할 수 없고, 정규화가 없으면 gradient가 큰 한 장이 평균을
삼킨다.
"""
import numpy as np
import torch
import torch.nn as nn

from bench.attribution import gradient_map
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
        grad = gradient_map(
            lambda x: center_token_scalar(model_name, model, x), image, device=device
        )
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


def peak_location(erf: np.ndarray) -> tuple[int, int]:
    """가장 밝은 픽셀의 (행, 열).

    프로브가 실제로 중심 토큰을 읽고 있는지 판정하는 정직성 게이트의 입력이고
    (설계 문서의 가드 2), `decay_ratio`가 감쇠 프로파일을 뽑는 원점이기도 하다.
    결과에 좌표를 그대로 남겨야 "피크가 어디로 튀었는가"를 산문이 아니라
    데이터로 확인할 수 있다.
    """
    row, col = np.unravel_index(erf.argmax(), erf.shape)
    return int(row), int(col)


def _require_mass(erf: np.ndarray) -> None:
    """질량이 0인 맵을 반경 계산에 넣지 못하게 막는다.

    전부 0인 ERF는 이 저장소가 이미 한 번 물린 실패 모드다(Task 4: DeiT의 `norm`
    캡처에서 16장 중 11장이 정확히 0). 가드가 없으면 예외도 NaN도 아닌 그럴듯한
    숫자가 나온다 — `erf / erf.sum()`이 0/0 = NaN 배열이 되고, NaN 비교가 전부
    False라 `np.searchsorted`가 0을 돌려주며, `mass_radius`가 "중심에서 가장 가까운
    픽셀까지의 거리" 0.7071을 반환한다. 하필 그 0.71이 cls 토큰 게이트
    (random_init 반경 < natural 반경)를 **통과한다.** 측정이 통째로 비었는데
    게이트가 초록불을 주는 셈이라, 조용히 넘기지 않고 여기서 터뜨린다.
    """
    total = erf.sum()
    if not total > 0:
        raise ValueError(
            f"ERF 질량이 0이다(sum={total}) — 반경이 정의되지 않는다. "
            "캡처 지점이 gradient를 전혀 받지 못했는지 확인할 것."
        )


def _distance_from_center(erf: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(erf.shape)
    center_row = (erf.shape[0] - 1) / 2
    center_col = (erf.shape[1] - 1) / 2
    return np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)


def mass_radius(erf: np.ndarray, fraction: float = 0.5) -> float:
    """배열 중심에서 재어 전체 질량의 `fraction`을 담는 최소 반경.

    설계 문서가 cls 토큰 오인 가드로 지정한 "질량 50% 반경"이 이것이다. 학습되지
    않은 모델의 ERF는 좁아야 하고(Luo et al. 2016), cls 토큰을 잡고 있었다면
    랜덤 초기화에서도 넓게 나온다 — 절대 임계값이 아니라 `random_init` vs
    `natural` 비교로 건다.

    `rms_radius`와 달리 거리 제곱 가중이 아니라 분위수라, 넓고 옅은 배경이
    있어도 코어 크기를 그대로 보고한다(`anisotropy_index`의 꼬리 지배와 같은
    문제를 피한다).
    """
    _require_mass(erf)
    weights = (erf / erf.sum()).ravel()
    distances = _distance_from_center(erf).ravel()
    order = np.argsort(distances)
    cumulative = np.cumsum(weights[order])
    return float(distances[order][int(np.searchsorted(cumulative, fraction))])


def rms_radius(erf: np.ndarray) -> float:
    """배열 중심 기준 RMS 반경.

    Task 9의 게이트 스크립트가 실제로 쓴 정의이므로, 이미 커밋된 보고서·인계
    문서의 숫자(64.3 / 47.4 / 14.9 등)를 계속 대조할 수 있도록 `mass_radius`와
    함께 기록한다. 거리 제곱 가중이라 `mass_radius`보다 far-field 꼬리에
    훨씬 민감하다.
    """
    _require_mass(erf)
    weights = erf / erf.sum()
    return float(np.sqrt((weights * _distance_from_center(erf) ** 2).sum()))


def decay_window(erf: np.ndarray, max_distance: int = 64) -> int:
    """피크에서 상하좌우 네 방향 모두로 배열 밖을 넘지 않는 최대 반경과
    max_distance 중 작은 값.

    피크가 경계에 가까우면 `decay_ratio`가 `erf[row ± offsets, col]`로 배열 밖을
    읽는다(IndexError). 이 함수가 실제로 쓸 수 있는 반경을 먼저 계산해 그 사고를
    막는다. 중심 피크(row=col=112, 224² 배열)에서는 상하좌우 여유가 전부
    max_distance(64)보다 크므로 그대로 64를 돌려준다 — 기존 동작과 동일하다.
    """
    row, col = peak_location(erf)
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

    쓸 수 있는 반경이 `max_distance`에 못 미치면 조용히 더 좁은 창으로 계산하지
    않고 ValueError를 던진다 — 반경이 셀마다 다르게 좁아지면 같은 열에 반경이
    다른 값이 섞여 비교 불가능한 숫자가 된다.

    예전 가드는 `MIN_DECAY_WINDOW = 8` 하한이었는데, 그 하한은 방금 말한
    비교 가능성을 집행하지 못했다. [8, 64] 구간의 어떤 반경이든 같은 열에
    들어갈 수 있었기 때문이다. 커밋된 맵으로 창을 바꿔 가며 재보면 그 구간
    안에서 값이 순위째로 바뀐다 — `natural`/N=512에서 vim_s는 창 16에서 2.391,
    64에서 1.345, 96에서 0.949이고, cmt_s는 창 8에서 -0.014(감쇠비가 음수)다.
    그래서 하한이 아니라 "창 전체"를 요구한다.
    """
    row, col = peak_location(erf)
    window = decay_window(erf, max_distance)
    if window < max_distance:
        raise ValueError(
            f"피크 ({row}, {col})가 경계에서 반경 {window}만큼만 떨어져 있다 "
            f"(max_distance {max_distance} 전체가 필요) — 더 좁은 창으로 낸 "
            f"기울기 비는 같은 열의 다른 셀과 비교할 수 없다."
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


ANGLE_TOLERANCE_DEG = 1.0
"""주축 각도의 수렴 허용오차(도).

임의의 값이 아니라 이 실험의 격자에서 유도한 값이다. 감쇠비를 재는 창이 피크에서
64px이므로, 주축이 1° 기울면 그 창 끝에서 축이 tan(1°)×64 ≈ 1.1px 움직인다 —
224² 맵이 표현할 수 있는 가장 작은 변위인 1픽셀 수준이다. 그보다 작은 각도 변화는
맵이 구분할 수 있는 방향 차이가 아니다.
"""


def has_converged_deg(
    values: list[float], tolerance_deg: float = ANGLE_TOLERANCE_DEG
) -> bool:
    """마지막 두 점의 **절대** 변화가 tolerance_deg 이내인가.

    각도에는 상대 기준을 쓸 수 없다. `principal_angle_deg`는 [0, 90]의 각도이고
    0에 가까운 값은 '작은 값'이 아니라 '수평에 정렬됨'이라는 가장 강한 결론인데,
    상대 기준은 하필 그 구간에서만 터무니없이 빡빡해진다. 실측 vim_s/random_init이
    정확히 그 사례다 — 여섯 개 N에서 각도가 0.0697°~0.1088°로 붙어 있는데도
    0.008°의 절대 흔들림이 상대 10% 변화로 읽혀 여섯 N 전부 '미수렴'으로 찍혔고,
    논문 3.1절의 문자 그대로의 주장(scan 방향에 정렬된 ERF 타원)을 직접 잰 값이
    지표 버그로 폐기됐다.

    나머지 세 지표(비등방 지수·중심 비등방 지수·감쇠비)는 1 근방의 스케일 없는
    비이므로 상대 기준(`has_converged`)이 옳다 — 그쪽은 그대로 둔다.

    `principal_angle_deg`가 [0, 90]으로 접힌 표현이라 0°와 90° 양 끝에서 모두
    연속이므로(축은 부호가 없다), 단순 절대차로 충분하다.
    """
    if len(values) < 2:
        return False
    return abs(values[-1] - values[-2]) <= tolerance_deg
