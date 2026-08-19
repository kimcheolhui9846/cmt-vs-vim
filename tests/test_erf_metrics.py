import numpy as np
import pytest

from bench.erf import anisotropy_index, decay_ratio, has_converged, principal_angle_deg


def _gaussian(sigma_x: float, sigma_y: float, size: int = 224) -> np.ndarray:
    axis = np.arange(size) - size // 2
    x, y = np.meshgrid(axis, axis)  # x는 열(수평), y는 행(수직)
    return np.exp(-(x**2) / (2 * sigma_x**2) - (y**2) / (2 * sigma_y**2))


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
