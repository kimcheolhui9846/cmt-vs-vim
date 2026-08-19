import numpy as np
import pytest

from bench.erf import (
    anisotropy_index,
    decay_ratio,
    decay_window,
    has_converged,
    principal_angle_deg,
)


def _gaussian(sigma_x: float, sigma_y: float, size: int = 224) -> np.ndarray:
    axis = np.arange(size) - size // 2
    x, y = np.meshgrid(axis, axis)  # x는 열(수평), y는 행(수직)
    return np.exp(-(x**2) / (2 * sigma_x**2) - (y**2) / (2 * sigma_y**2))


def _gaussian_at(
    center_row: int, center_col: int, sigma_x: float, sigma_y: float, size: int = 224
) -> np.ndarray:
    """피크 위치를 임의로 옮긴 가우시안. 경계 근처 피크를 흉내내는 데 쓴다."""
    rows, cols = np.indices((size, size))
    return np.exp(
        -((cols - center_col) ** 2) / (2 * sigma_x**2)
        - ((rows - center_row) ** 2) / (2 * sigma_y**2)
    )


def test_a_round_blob_is_isotropic():
    assert anisotropy_index(_gaussian(20, 20)) == pytest.approx(1.0, abs=0.02)


def test_a_stretched_blob_reports_its_stretch():
    """수평으로 3배 늘인 가우시안의 비등방 지수는 3이다."""
    assert anisotropy_index(_gaussian(30, 10)) == pytest.approx(3.0, rel=0.05)


def test_the_principal_axis_of_a_horizontal_blob_is_horizontal():
    assert principal_angle_deg(_gaussian(30, 10)) == pytest.approx(0.0, abs=2.0)


def test_the_principal_axis_of_a_vertical_blob_is_vertical():
    assert principal_angle_deg(_gaussian(10, 30)) == pytest.approx(90.0, abs=2.0)


def test_decay_ratio_is_one_when_both_axes_fall_alike():
    assert decay_ratio(_gaussian(20, 20)) == pytest.approx(1.0, rel=0.05)


def test_decay_ratio_exceeds_one_when_the_vertical_falls_faster():
    """논문 3.1절이 Vim에 대해 예측하는 방향이다 — 수직이 더 가파르다."""
    assert decay_ratio(_gaussian(30, 10)) > 1.5


def test_convergence_looks_at_the_last_two_points():
    assert has_converged([5.0, 2.0, 1.30, 1.31]) is True
    assert has_converged([5.0, 2.0, 1.30, 1.60]) is False


def test_a_single_measurement_has_not_converged():
    """점이 하나면 변화를 볼 수 없다. True를 돌려주면 N=16짜리 값이 수렴한
    값으로 논문에 실린다."""
    assert has_converged([1.3]) is False


def test_decay_window_uses_the_full_max_distance_for_a_center_peak():
    """중심 피크(112,112)에서는 상하좌우 여유가 전부 64보다 크므로 그대로 64다.
    기존 decay_ratio 8건이 값 변화 없이 통과해야 하는 이유가 이것이다."""
    assert decay_window(_gaussian(20, 20)) == 64


def test_decay_window_respects_a_smaller_max_distance():
    assert decay_window(_gaussian(20, 20), max_distance=20) == 20


def test_decay_window_shrinks_when_the_peak_is_near_an_edge():
    """피크가 위쪽 경계에서 10칸 떨어져 있으면(다른 세 방향은 더 넓으므로)
    반경은 min(64, 10, 213, 112, 111) = 10이어야 한다."""
    erf = _gaussian_at(center_row=10, center_col=112, sigma_x=20, sigma_y=20)
    assert decay_window(erf) == 10


def test_decay_ratio_raises_when_the_peak_is_too_close_to_the_edge():
    """반경이 MIN_DECAY_WINDOW(8)보다 좁으면 조용히 좁은 창으로 계산하지 않고
    터진다 — cmt_s/noise 조건에서 실측 중 재현된 실패 모드다(피크가 이미지
    모서리 근처에 찍혀 반경이 0에 가까웠다)."""
    erf = _gaussian_at(center_row=3, center_col=112, sigma_x=20, sigma_y=20)
    with pytest.raises(ValueError):
        decay_ratio(erf)


def test_decay_ratio_still_works_exactly_at_the_minimum_window():
    """경계값(반경 정확히 8)에서는 터지지 않아야 한다 — '미만'이지 '이하'가
    아니다."""
    erf = _gaussian_at(center_row=8, center_col=112, sigma_x=20, sigma_y=20)
    assert isinstance(decay_ratio(erf), float)
