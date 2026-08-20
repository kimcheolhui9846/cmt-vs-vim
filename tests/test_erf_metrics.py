import numpy as np
import pytest

from bench.erf import (
    ANGLE_TOLERANCE_DEG,
    anisotropy_index,
    central_crop,
    decay_ratio,
    decay_window,
    has_converged,
    has_converged_deg,
    mass_radius,
    peak_location,
    principal_angle_deg,
    rms_radius,
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
    ratio, window = decay_ratio(_gaussian(20, 20))
    assert ratio == pytest.approx(1.0, rel=0.05)
    assert window == 64


def test_decay_ratio_exceeds_one_when_the_vertical_falls_faster():
    """논문 3.1절이 Vim에 대해 예측하는 방향이다 — 수직이 더 가파르다."""
    ratio, _window = decay_ratio(_gaussian(30, 10))
    assert ratio > 1.5


def test_convergence_looks_at_the_last_two_points():
    assert has_converged([5.0, 2.0, 1.30, 1.31]) is True
    assert has_converged([5.0, 2.0, 1.30, 1.60]) is False


def test_a_single_measurement_has_not_converged():
    """점이 하나면 변화를 볼 수 없다. True를 돌려주면 N=16짜리 값이 수렴한
    값으로 논문에 실린다."""
    assert has_converged([1.3]) is False


# --- 각도 전용 절대 허용오차 ------------------------------------------------
#
# 상대 기준을 각도에 그대로 쓰면 0에 가까운 각도에서만 기준이 터무니없이
# 빡빡해진다 — 이것은 값이 작아서 생기는 인공물이지 측정이 불안정하다는
# 뜻이 아니다. 실측 vim_s/random_init이 정확히 그 사례다.


def test_an_angle_that_barely_moves_in_degrees_has_converged():
    """실측 vim_s/random_init의 주축 각도 (N=128, 256, 512)다. 여섯 개 N 전부
    0.07°~0.11° 사이에 있고 마지막 변화는 0.008°인데, 상대 기준은 이를
    10% 변동으로 읽어 '미수렴'으로 버렸다."""
    assert has_converged_deg([0.0902773896263949, 0.0776279886352549,
                              0.0697775086330904]) is True


def test_the_relative_rule_is_what_discarded_that_angle():
    """이 테스트가 버그 자체를 고정한다 — 같은 계열을 상대 5% 기준에 넣으면
    False가 나온다. 절대 경로가 없으면 이 계열은 영원히 인용 불가로 남는다."""
    assert has_converged([0.0902773896263949, 0.0776279886352549,
                          0.0697775086330904]) is False


def test_an_angle_that_genuinely_wanders_has_not_converged():
    """실측 deit_s/natural의 마지막 두 점(31.37° -> 34.06°). 절대 기준이
    '전부 통과시키는 무른 기준'이 되면 안 된다는 것을 이 케이스가 막는다."""
    assert has_converged_deg([31.371855369015933, 34.05553292955484]) is False


def test_a_large_angle_that_is_stable_in_degrees_also_converges():
    """절대 기준은 값의 크기와 무관해야 한다 — 큰 각도라고 더 관대해지거나
    더 빡빡해지지 않는다."""
    assert has_converged_deg([80.4, 80.9]) is True


def test_the_angle_tolerance_is_inclusive_at_the_boundary():
    exactly = [10.0, 10.0 + ANGLE_TOLERANCE_DEG]
    just_over = [10.0, 10.0 + ANGLE_TOLERANCE_DEG * 1.001]
    assert has_converged_deg(exactly) is True
    assert has_converged_deg(just_over) is False


def test_a_single_angle_has_not_converged():
    assert has_converged_deg([0.07]) is False


# --- 피크 위치와 질량 반경 ---------------------------------------------------


def test_peak_location_finds_the_brightest_pixel():
    erf = _gaussian_at(center_row=40, center_col=90, sigma_x=20, sigma_y=20)
    assert peak_location(erf) == (40, 90)


def test_mass_radius_matches_the_analytic_half_mass_radius_of_a_gaussian():
    """중심 대칭 2D 가우시안에서 질량 50%를 담는 반경은 σ·√(2 ln 2) =
    1.1774σ다. σ=20이면 23.55."""
    assert mass_radius(_gaussian(20, 20)) == pytest.approx(23.55, abs=0.6)


def test_rms_radius_matches_the_analytic_rms_radius_of_a_gaussian():
    """같은 가우시안의 RMS 반경은 √2·σ = 28.28이다. 두 반경이 서로 다른
    수라는 것 자체가 설계 문서('질량 50% 반경')와 구현(RMS)이 갈렸던
    지점이다."""
    assert rms_radius(_gaussian(20, 20)) == pytest.approx(28.28, abs=0.6)


def test_the_rms_radius_is_pulled_by_a_far_field_tail_and_the_half_mass_one_is_not():
    """왜 설계 문서의 정의를 기록값으로 삼는가에 대한 근거다. RMS는 거리
    제곱 가중이라 anisotropy_index와 같은 꼬리 지배 문제를 그대로 갖는다 —
    좁은 코어에 넓고 옅은 받침을 얹으면 RMS는 크게 늘어나지만 질량 50%
    반경은 코어 안에 머문다.

    받침의 진폭 0.003은 아무 값이나 고른 게 아니다. 가우시안의 질량은 σ²에
    비례하므로 이 받침이 갖는 질량은 코어의 0.003×(90/10)² = 0.24배, 즉 전체의
    약 20%다 — 질량의 과반이 여전히 코어에 있어야 "50% 반경이 코어 안에
    머문다"가 의미 있는 주장이 된다. 받침이 과반을 가지면 50% 반경이 밖으로
    나가는 게 맞는 동작이지 결함이 아니다."""
    core = _gaussian(10, 10)
    with_tail = core + 0.003 * _gaussian(90, 90)

    assert rms_radius(with_tail) > 2 * rms_radius(core)
    assert mass_radius(with_tail) < 1.5 * mass_radius(core)


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
    """반경이 max_distance에 못 미치면 조용히 좁은 창으로 계산하지 않고
    터진다 — cmt_s/noise 조건에서 실측 중 재현된 실패 모드다(피크가 이미지
    모서리 근처에 찍혀 반경이 0에 가까웠다)."""
    erf = _gaussian_at(center_row=3, center_col=112, sigma_x=20, sigma_y=20)
    with pytest.raises(ValueError):
        decay_ratio(erf)


def test_a_window_that_would_fit_a_slope_is_still_rejected_if_it_is_short():
    """이것이 하한 8과 '창 전체 요구'를 가르는 테스트다.

    반경 16은 옛 하한(8)을 넉넉히 넘으므로 옛 가드는 통과시켰다. 그러나 그
    셀의 값은 max_distance=64로 잰 다른 셀들과 같은 열에 들어가는데, 실측
    맵에서 창을 16으로 좁히면 vim_s/natural의 감쇠비가 1.35에서 2.39로,
    cmt_s는 1.02에서 0.87로 움직인다 — 순위까지 바뀐다. docstring이 주장하는
    '같은 열의 숫자는 서로 비교 가능하다'를 실제로 집행하려면 창이 조금이라도
    좁으면 거절해야 한다."""
    erf = _gaussian_at(center_row=16, center_col=112, sigma_x=20, sigma_y=20)
    assert decay_window(erf) == 16  # 옛 하한 8은 넘는다
    with pytest.raises(ValueError):
        decay_ratio(erf)


def test_decay_ratio_works_when_the_window_is_exactly_max_distance():
    """경계값(여유가 정확히 max_distance)에서는 터지지 않아야 한다 —
    '미만'이지 '이하'가 아니다."""
    erf = _gaussian_at(center_row=64, center_col=112, sigma_x=20, sigma_y=20)
    ratio, window = decay_ratio(erf)
    assert isinstance(ratio, float)
    assert window == 64


def test_decay_ratio_returns_the_window_it_actually_used():
    """반경을 decay_window()로 따로 다시 계산하면 두 호출이 어긋날 여지가
    생긴다 — decay_ratio가 자기가 실제로 쓴 반경을 값과 함께 돌려줘야
    호출자가 별도 계산 없이 그대로 CSV에 남길 수 있다."""
    erf = _gaussian_at(center_row=10, center_col=112, sigma_x=20, sigma_y=20)
    _ratio, window = decay_ratio(erf, max_distance=10)
    assert window == decay_window(erf, max_distance=10) == 10


def test_the_rejection_message_names_the_full_window_it_required():
    """CSV의 error 열에 그대로 남는 문장이다. '최소 8 필요'처럼 없어진 하한을
    가리키면, 커밋된 데이터가 존재하지 않는 규칙을 근거로 대게 된다."""
    erf = _gaussian_at(center_row=2, center_col=112, sigma_x=20, sigma_y=20)
    with pytest.raises(ValueError, match="64"):
        decay_ratio(erf)


def test_central_crop_keeps_the_middle_of_the_array():
    erf = _gaussian(20, 20)
    cropped = central_crop(erf, size=128)

    assert cropped.shape == (128, 128)
    # 224² 배열의 (112,112)는 128² 잘라낸 배열의 (64,64)로 옮겨간다
    # (row0 = (224-128)//2 = 48; 112-48 = 64).
    assert cropped[64, 64] == erf[112, 112]


def test_central_crop_removes_the_far_field_tail():
    """중심 128²는 96px 반경의 원반이다 — 224² 전체보다 훨씬 좁으므로 넓게
    퍼진 등방적 배경(pedestal)이 있는 맵에서는 자른 쪽의 지수가 더 커야 한다.
    새 수학이 아니라 anisotropy_index를 잘라낸 맵에 그대로 적용한 것임을
    확인한다."""
    # 중심의 좁고 수평으로 늘어난 blob 위에, 넓고 둥근(등방) 배경을 얹는다.
    core = _gaussian(30, 10)
    pedestal = 0.5 * _gaussian(90, 90)
    erf = core + pedestal

    whole = anisotropy_index(erf)
    central = anisotropy_index(central_crop(erf, size=128))

    assert central > whole


def test_an_all_zero_map_is_refused_by_both_radii():
    """전부 0인 ERF는 이 저장소가 이미 한 번 물린 실패 모드다 — Task 4에서
    DeiT의 `norm` 캡처가 16장 중 11장을 정확히 0으로 냈다.

    가드가 없으면 조용하고 **그럴듯한** 오답이 나온다. 0/0 = NaN이 채워진
    배열에서 NaN 비교가 전부 False라 `searchsorted`가 0을 돌려주고,
    `mass_radius`가 '중심에서 가장 가까운 픽셀까지의 거리' 0.7071을 반환한다.
    0.71은 cls 토큰 게이트(random_init 반경 < natural 반경)를 **통과한다** —
    측정이 통째로 비었는데 게이트는 초록불을 준다. 예외 없이 넘어가면 안 된다."""
    zero = np.zeros((224, 224))

    with pytest.raises(ValueError, match="0"):
        mass_radius(zero)
    with pytest.raises(ValueError, match="0"):
        rms_radius(zero)


def test_a_map_with_any_mass_is_still_accepted():
    """가드가 정상 맵까지 막으면 안 된다. 커밋된 맵 중 가장 작은 합도 106이다."""
    almost_empty = np.zeros((224, 224))
    almost_empty[100, 100] = 1e-12

    assert mass_radius(almost_empty) > 0
    assert rms_radius(almost_empty) > 0
