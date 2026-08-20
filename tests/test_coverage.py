"""질의 패치와 적중률 계산. 모델도 GPU도 거치지 않는 순수 함수만 다룬다."""
import numpy as np
import pytest

from bench.coverage import MIN_PATCH_COVERAGE, patch_coverage, query_patch

SIZE = 224
GRIDS = (7, 14)   # CMT는 7x7, DeiT와 Vim은 14x14 — models/probes.py가 출처다


def _u_shaped_mask() -> np.ndarray:
    """무게중심이 마스크 바깥에 떨어지는 U자 객체.

    왼팔 rows 32~159 cols 0~63, 오른팔 rows 32~159 cols 160~223,
    바닥 rows 160~191 cols 0~223. 무게중심은 약 (119.9, 111.5)로 U의 빈 속이다.
    """
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[32:160, 0:64] = True
    mask[32:160, 160:224] = True
    mask[160:192, 0:224] = True
    return mask


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    return float(rows.mean()), float(cols.mean())


def test_the_u_shape_really_has_its_centroid_outside():
    """테스트 도형 자체를 먼저 검증한다. 무게중심이 마스크 안이면 이 태스크의
    테스트는 아무것도 막지 못하면서 통과한다."""
    mask = _u_shaped_mask()
    row, col = _centroid(mask)
    assert not mask[int(round(row)), int(round(col))]


def test_query_patch_stays_inside_the_mask():
    """정직성 장치 1. 무게중심을 그대로 쓰는 구현은 여기서 실패해야 한다."""
    mask = _u_shaped_mask()
    for grid in GRIDS:
        patch = query_patch(mask, grid)
        assert patch is not None
        row, col = patch
        assert patch_coverage(mask, grid)[row, col] > MIN_PATCH_COVERAGE


def test_query_patch_is_not_the_patch_holding_the_centroid():
    """무게중심이 든 패치는 U의 빈 속이라 마스크를 전혀 덮지 않는다."""
    mask = _u_shaped_mask()
    grid = 14
    cell = SIZE // grid
    row, col = _centroid(mask)
    centroid_patch = (int(row) // cell, int(col) // cell)

    assert patch_coverage(mask, grid)[centroid_patch] == 0.0
    assert query_patch(mask, grid) != centroid_patch


def test_query_patch_picks_the_candidate_nearest_the_centroid():
    """후보가 여럿이면 무게중심에 가장 가까운 것을 고른다.

    큰 덩어리 하나와 멀리 떨어진 작은 덩어리 하나를 만든다. 무게중심(105.1,
    105.1)은 큰 덩어리 안이므로 답은 큰 덩어리 쪽 패치다. 후보 목록의 첫
    원소를 그냥 돌려주는 구현은 argwhere 순서상 작은 덩어리의 (0, 0)을
    돌려주므로 여기서 실패한다.
    """
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[96:160, 96:160] = True   # 큰 덩어리
    mask[0:32, 0:32] = True       # 멀리 있는 작은 덩어리

    assert query_patch(mask, grid=14) == (6, 6)
    assert query_patch(mask, grid=7) == (3, 3)


def test_query_patch_returns_none_when_no_patch_is_mostly_covered():
    """어떤 패치도 과반으로 덮지 못하는 가느다란 객체.

    None을 돌려주지 않고 아무 패치나 고르면 질의가 배경에 놓인 채 측정이 진행된다.
    """
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[100:101, 20:200] = True  # 두께 1px 가로선

    assert query_patch(mask, grid=14) is None
    assert query_patch(mask, grid=7) is None


def test_query_patch_returns_none_for_an_empty_mask():
    assert query_patch(np.zeros((SIZE, SIZE), dtype=bool), grid=14) is None


def test_patch_coverage_is_the_fraction_of_each_cell():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[0:8, 0:16] = True  # 14x14 격자에서 (0,0) 셀(16x16)의 정확히 절반

    coverage = patch_coverage(mask, grid=14)

    assert coverage.shape == (14, 14)
    assert coverage[0, 0] == pytest.approx(0.5)
    assert coverage[0, 1] == 0.0


def test_patch_coverage_rejects_a_grid_that_does_not_divide_the_image():
    with pytest.raises(ValueError, match="나누어떨어지지"):
        patch_coverage(np.zeros((SIZE, SIZE), dtype=bool), grid=15)


def test_a_half_covered_patch_is_not_a_candidate():
    """경계는 '초과'다. 정확히 절반 덮인 패치는 후보가 아니다 — 그 패치의
    질의 토큰은 절반의 확률로 배경을 본다."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[0:8, 0:16] = True

    assert query_patch(mask, grid=14) is None
