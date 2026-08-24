"""질의 패치와 적중률 계산. 모델도 GPU도 거치지 않는 순수 함수만 다룬다."""
import numpy as np
import pytest

from bench.coverage import (
    MIN_PATCH_COVERAGE,
    mass_fraction,
    object_pixels,
    patch_coverage,
    population_size,
    precision_at_k,
    query_patch,
    random_baseline,
)

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


def _square_mask(top: int, left: int, side: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def test_precision_is_one_when_attribution_sits_exactly_on_the_object():
    obj = _square_mask(96, 96, 32)
    void = np.zeros((SIZE, SIZE), dtype=bool)
    attribution = np.where(obj, 1.0, 0.0)

    assert precision_at_k(attribution, obj, void) == pytest.approx(1.0)


def test_precision_is_zero_when_attribution_avoids_the_object():
    obj = _square_mask(96, 96, 32)
    void = np.zeros((SIZE, SIZE), dtype=bool)
    attribution = np.where(obj, 0.0, 1.0)
    # 동점을 없앤다 — 배경에만 서로 다른 값을 준다
    attribution[~obj] = np.linspace(1.0, 2.0, int((~obj).sum()))

    assert precision_at_k(attribution, obj, void) == pytest.approx(0.0)


def test_void_pixels_never_enter_the_ranking():
    """void가 모집단에 남으면 상위 K개 자리를 차지하면서 객체도 배경도 아닌
    것으로 집계돼, 경계가 두꺼운 객체에서 precision이 체계적으로 낮아진다."""
    obj = _square_mask(96, 96, 32)
    void = _square_mask(90, 90, 44) & ~obj
    attribution = np.zeros((SIZE, SIZE))
    attribution[void] = 100.0   # void가 가장 높은 기여도를 갖는다
    attribution[obj] = 1.0

    assert precision_at_k(attribution, obj, void) == pytest.approx(1.0)
    assert mass_fraction(attribution, obj, void) == pytest.approx(1.0)


def test_population_size_and_k_exclude_void():
    obj = _square_mask(96, 96, 32)
    void = _square_mask(0, 0, 10)

    assert object_pixels(obj, void) == 32 * 32
    assert population_size(void) == SIZE * SIZE - 100


def test_random_baseline_is_k_over_n():
    assert random_baseline(k=4096, n=50176) == pytest.approx(4096 / 50176)


def test_random_orderings_converge_to_the_analytic_baseline():
    """기준선이 해석적으로 맞는지 시뮬레이션으로 확인한다.

    조정 가능한 임계값이 아니라 계산된 값이라는 것이 요점이다 — 게이트를
    통과시키려고 손댈 여지가 없다.
    """
    obj = _square_mask(80, 80, 64)
    void = np.zeros((SIZE, SIZE), dtype=bool)
    k = object_pixels(obj, void)
    expected = random_baseline(k, population_size(void))

    rng = np.random.default_rng(0)
    scores = [
        precision_at_k(rng.random((SIZE, SIZE)), obj, void) for _ in range(100)
    ]

    assert np.mean(scores) == pytest.approx(expected, abs=0.005)


def test_mass_fraction_uses_magnitude_not_rank():
    """precision@K는 순위만 보고 질량 비율은 크기를 본다.

    기여도를 거의 균일하게 두고 객체 쪽만 0.1% 높이면 순위는 완벽하지만
    질량은 면적 비율만큼밖에 안 온다. 두 지표가 같은 값을 내면 한쪽은
    잉여이므로, 갈리는 입력에서 갈리는지를 단언한다.
    """
    obj = _square_mask(96, 96, 32)
    void = np.zeros((SIZE, SIZE), dtype=bool)
    attribution = np.full((SIZE, SIZE), 1.0)
    attribution[obj] = 1.001

    area_fraction = obj.sum() / (SIZE * SIZE)   # 0.0204
    assert precision_at_k(attribution, obj, void) == pytest.approx(1.0)
    assert mass_fraction(attribution, obj, void) == pytest.approx(
        area_fraction, rel=0.01
    )


def test_metrics_refuse_an_empty_object():
    void = np.zeros((SIZE, SIZE), dtype=bool)
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    with pytest.raises(ValueError, match="K가 0"):
        precision_at_k(np.ones((SIZE, SIZE)), empty, void)


def test_mass_fraction_refuses_an_all_zero_attribution():
    """전부 0인 기여도 지도는 E2가 이미 한 번 물린 실패 모드다. 나누기 0을
    조용히 NaN으로 흘리면 그럴듯한 숫자가 집계에 섞인다."""
    obj = _square_mask(96, 96, 32)
    void = np.zeros((SIZE, SIZE), dtype=bool)
    with pytest.raises(ValueError, match="질량이 0"):
        mass_fraction(np.zeros((SIZE, SIZE)), obj, void)


import pandas as pd

from bench.coverage import (
    LOW_SAMPLE_MIN,
    aggregate,
    area_bin,
    aspect_class,
    aspect_ratio,
    bounding_box,
    common_subset,
    expected_cells,
)


def test_area_bins_cover_every_fraction_without_gaps():
    for fraction in (0.0, 0.019, 0.02, 0.049, 0.05, 0.099, 0.1, 0.199, 0.2,
                     0.399, 0.4, 0.999, 1.0):
        assert isinstance(area_bin(fraction), str)


def test_area_bin_boundaries_belong_to_the_bin_they_open():
    """경계값은 아래를 포함하고 위를 제외한다. 0.02는 "2-5%"이지 "<2%"가 아니다."""
    assert area_bin(0.019) == "<2%"
    assert area_bin(0.02) == "2-5%"
    assert area_bin(0.05) == "5-10%"
    assert area_bin(0.10) == "10-20%"
    assert area_bin(0.20) == "20-40%"
    assert area_bin(0.40) == ">=40%"


def test_area_bin_rejects_a_fraction_outside_zero_to_one():
    with pytest.raises(ValueError):
        area_bin(1.5)


def test_bounding_box_is_inclusive():
    mask = _square_mask(10, 20, 5)
    assert bounding_box(mask) == (10, 20, 14, 24)


def test_aspect_ratio_is_width_over_height():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[10:20, 40:70] = True   # 높이 10, 너비 30
    assert aspect_ratio(mask) == pytest.approx(3.0)


def test_aspect_class_splits_wide_square_and_tall():
    assert aspect_class(3.0) == "wide"
    assert aspect_class(1.5) == "wide"
    assert aspect_class(1.0) == "square"
    assert aspect_class(1 / 1.5) == "tall"
    assert aspect_class(0.25) == "tall"


def test_aspect_class_is_symmetric_under_inversion():
    """가로형 기준과 세로형 기준이 서로의 역수여야 한다. 아니면 E2와의 교차
    검증(Vim은 세로형에서 나빠야 한다)이 기준의 비대칭을 재게 된다."""
    for ratio in (1.2, 1.6, 2.0, 5.0):
        assert aspect_class(ratio) != "tall"
        assert aspect_class(1 / ratio) != "wide"
        if aspect_class(ratio) == "wide":
            assert aspect_class(1 / ratio) == "tall"


def _measurement_frame() -> pd.DataFrame:
    rows = []
    for model in ("deit_s", "vim_s"):
        for index in range(40):
            rows.append({
                "model": model,
                "condition": "pretrained",
                "image": f"img_{index:03d}",
                "instance_id": 1,
                "area_bin": "<2%" if index < 35 else "20-40%",
                "aspect_class": "square",
                "precision_at_k": 0.5 if model == "deit_s" else 0.8,
                "mass_fraction": 0.3,
                "random_baseline": 0.01,
                "status": "ok",
            })
    return pd.DataFrame(rows)


def _two_model_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "deit_s", "condition": "pretrained", "image": "a",
         "instance_id": 1, "status": "ok"},
        {"model": "cmt_s", "condition": "pretrained", "image": "a",
         "instance_id": 1, "status": "ok"},
        {"model": "deit_s", "condition": "pretrained", "image": "b",
         "instance_id": 1, "status": "ok"},
        {"model": "cmt_s", "condition": "pretrained", "image": "b",
         "instance_id": 1, "status": "no_query_patch"},
    ])


def test_aggregate_reports_mean_sem_and_count():
    out = aggregate(_measurement_frame(), ("model", "condition", "area_bin"))

    cell = out.query("model == 'deit_s' and area_bin == '<2%'").iloc[0]
    assert cell["precision_mean"] == pytest.approx(0.5)
    assert cell["n"] == 35
    assert cell["baseline_mean"] == pytest.approx(0.01)


def test_aggregate_flags_bins_below_the_low_sample_threshold():
    """표본 30 미만 구간은 값을 싣되 인용하지 않는다. 표시가 없으면 그
    구분이 결과 파일에서 사라진다."""
    out = aggregate(_measurement_frame(), ("model", "condition", "area_bin"))

    small = out.query("model == 'deit_s' and area_bin == '20-40%'").iloc[0]
    large = out.query("model == 'deit_s' and area_bin == '<2%'").iloc[0]
    assert small["n"] < LOW_SAMPLE_MIN and bool(small["low_sample"])
    assert large["n"] >= LOW_SAMPLE_MIN and not bool(large["low_sample"])


def test_aggregate_ignores_rows_that_did_not_measure():
    df = _measurement_frame()
    df.loc[df.index[:10], "status"] = "no_query_patch"
    df.loc[df.index[:10], "precision_at_k"] = np.nan

    out = aggregate(df, ("model", "condition", "area_bin"))

    assert out["n"].sum() == 70


def test_common_subset_keeps_only_instances_every_cell_measured():
    """CMT의 격자는 7x7이라 작은 객체에서 질의 후보가 없다. 모델마다 다른
    부분집합으로 평균을 내면 그 차이가 곧 모델 차이로 읽힌다."""
    cells = expected_cells(("deit_s", "cmt_s"), ("pretrained",))

    kept = common_subset(_two_model_frame(), cells)

    assert set(kept["image"]) == {"a"}
    assert len(kept) == 2


def test_common_subset_fails_when_a_model_produced_no_rows_at_all():
    """모델 하나가 통째로 빠진 실행에서 조용히 두 모델만 비교하지 않도록 막는다.

    status로 표시된 실패와 달리 '행이 아예 없는' 실패는 df 안에 흔적이 없다.
    기대 셀을 df에서 유추하면 기준 개수가 함께 줄어, 빠진 모델이 있었다는
    사실 자체가 결과에서 사라진다.
    """
    cells = expected_cells(("deit_s", "cmt_s", "vim_s"), ("pretrained",))

    with pytest.raises(ValueError, match="vim_s"):
        common_subset(_two_model_frame(), cells)


def test_aggregate_keeps_a_single_sample_sem_as_nan():
    """표본이 1이면 표준오차는 정의되지 않는다(ddof=1). 0으로 채우면 오차 막대가
    없는 점이 그림에서 가장 정밀한 값처럼 보인다."""
    out = aggregate(_measurement_frame().head(1), ("model", "condition", "area_bin"))

    assert out.iloc[0]["n"] == 1
    assert pd.isna(out.iloc[0]["precision_sem"])
