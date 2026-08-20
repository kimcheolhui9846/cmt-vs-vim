"""인스턴스 목록과 샘플링. 실제 모델을 돌리지 않는다."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from experiments.e3_dilution import (
    INSTANCES_PER_BIN,
    MIN_MASK_PIXELS,
    build_catalog,
    instance_rows,
    sample_instances,
)
from tests.vocfixtures import write_mask_png


def _pair(tmp_path: Path, name: str, boxes: dict[int, tuple[int, int, int, int]]) -> Path:
    """boxes = {instance_id: (top, left, height, width)}. 마스크와 짝 이미지를 만든다."""
    mask = np.zeros((224, 224), dtype=np.uint8)
    for instance_id, (top, left, height, width) in boxes.items():
        mask[top : top + height, left : left + width] = instance_id
    mask_path = write_mask_png(tmp_path / f"{name}.png", mask)
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(
        tmp_path / f"{name}.jpg"
    )
    return mask_path


def test_instance_rows_describe_each_object(tmp_path):
    mask_path = _pair(tmp_path, "a", {1: (0, 0, 40, 80), 2: (100, 100, 60, 60)})

    rows = instance_rows(mask_path, tmp_path)

    assert [row["instance_id"] for row in rows] == [1, 2]
    first = rows[0]
    assert first["area_px"] == 40 * 80
    assert first["area_fraction"] == pytest.approx(40 * 80 / (224 * 224))
    assert first["aspect_ratio"] == pytest.approx(2.0)
    assert first["aspect_class"] == "wide"
    assert first["image"] == "a"


def test_instances_below_the_minimum_area_are_dropped(tmp_path):
    """상위 K개가 너무 작으면 precision이 잡음이 된다. 512px는 패치 두 개쯤이다."""
    small = int(MIN_MASK_PIXELS ** 0.5) - 2
    mask_path = _pair(tmp_path, "a", {1: (0, 0, small, small), 2: (100, 0, 40, 40)})

    rows = instance_rows(mask_path, tmp_path)

    assert small * small < MIN_MASK_PIXELS <= 40 * 40
    assert [row["instance_id"] for row in rows] == [2]


def test_population_excludes_void(tmp_path):
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[0:40, 0:40] = 1
    mask[200:224, :] = 255
    mask_path = write_mask_png(tmp_path / "v.png", mask)
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(tmp_path / "v.jpg")

    row = instance_rows(mask_path, tmp_path)[0]

    assert row["void_px"] == 24 * 224
    assert row["population_n"] == 224 * 224 - 24 * 224
    assert row["random_baseline"] == pytest.approx(row["k"] / row["population_n"])


def test_catalog_is_sorted_and_stable(tmp_path):
    paths = [
        _pair(tmp_path, "b", {1: (0, 0, 40, 40)}),
        _pair(tmp_path, "a", {1: (0, 0, 40, 40), 2: (60, 60, 40, 40)}),
    ]

    catalog = build_catalog(paths, tmp_path)
    again = build_catalog(list(reversed(paths)), tmp_path)

    assert list(catalog["image"]) == ["a", "a", "b"]
    assert catalog.equals(again), "파일시스템 순회 순서가 목록을 바꾸면 재현이 깨진다"


def _catalog(per_bin: int, measurable: bool = True) -> pd.DataFrame:
    """여섯 구간에 per_bin개씩 든 합성 목록."""
    from bench.coverage import AREA_BINS

    rows = []
    for bin_index, (low, _high, label) in enumerate(AREA_BINS):
        for i in range(per_bin):
            rows.append({
                "image": f"img_{bin_index:02d}_{i:04d}",
                "instance_id": 1,
                "area_fraction": low + 0.001,
                "area_bin": label,
                "measurable_by_all": measurable,
            })
    return pd.DataFrame(rows)


def test_same_seed_gives_the_same_instances():
    first = sample_instances(_catalog(50), per_bin=10, seed=0)
    second = sample_instances(_catalog(50), per_bin=10, seed=0)
    assert first.equals(second)


def test_different_seeds_give_different_instances():
    first = sample_instances(_catalog(50), per_bin=10, seed=0)
    second = sample_instances(_catalog(50), per_bin=10, seed=1)
    assert not first.equals(second)


def test_every_area_bin_gets_the_same_number_of_instances():
    """층화의 요점. 균등 추출은 <2% 구간을 10개까지 떨어뜨려 인용 불가로 만든다
    (고정 환경 실측) — 하필 논문 3.2절의 희석이 가장 세게 걸릴 구간이다."""
    from bench.coverage import AREA_BINS

    sample = sample_instances(_catalog(50), per_bin=10, seed=0)

    counts = sample.groupby("area_bin").size()
    assert set(counts.index) == {label for _, _, label in AREA_BINS}
    assert set(counts) == {10}
    assert len(sample) == 10 * len(AREA_BINS)


def test_instances_no_model_can_query_are_never_sampled():
    """세 모델이 서로 다른 부분집합을 재면 그 차이가 곧 모델 차이로 읽힌다."""
    catalog = _catalog(50, measurable=True)
    catalog.loc[catalog.index[::2], "measurable_by_all"] = False

    sample = sample_instances(catalog, per_bin=10, seed=0)

    assert sample["measurable_by_all"].all()


def test_a_bin_without_enough_instances_fails_loudly():
    """조용히 적게 반환하면 구간당 100개로 잰 줄 알았던 값이 실은 3개가 된다."""
    with pytest.raises(ValueError, match="구간"):
        sample_instances(_catalog(3), per_bin=INSTANCES_PER_BIN, seed=0)


from experiments.e3_dilution import MEASUREMENT_COLUMNS, measure_instance  # noqa: E402


class _FakeModel:
    """격자만 흉내 내는 가짜. 실제 모델을 돌리지 않고 배선을 검증한다."""


def test_measure_instance_records_no_query_patch_instead_of_guessing(
    tmp_path, monkeypatch
):
    """질의 후보가 없는 인스턴스는 아무 패치나 고르지 않고 그 사실을 남긴다.

    아무 패치나 고르면 질의가 배경에 놓인 채 낮은 precision이 나오고, 그건
    "이 모델은 객체를 통합하지 못한다"로 읽힌다.
    """
    import experiments.e3_dilution as e3

    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[100:101, 20:200] = 1   # 두께 1px — 어떤 패치도 과반으로 덮지 못한다
    write_mask_png(tmp_path / "thin.png", mask)
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(
        tmp_path / "thin.jpg"
    )

    def _never_called(*args, **kwargs):
        raise AssertionError("질의 후보가 없는데 gradient를 계산했다")

    monkeypatch.setattr(e3, "gradient_map", _never_called)

    row = e3.measure_instance(
        "deit_s", _FakeModel(), {"image": "thin", "instance_id": 1},
        image_dir=tmp_path, masks_dir=tmp_path, device="cpu",
    )

    assert row["status"] == "no_query_patch"
    assert row["precision_at_k"] is None
    assert row["query_row"] is None


def test_measure_instance_scores_a_perfect_attribution(tmp_path, monkeypatch):
    """기여도가 마스크에 정확히 얹히면 precision@K가 1이어야 한다.

    가짜 gradient를 주입해 배선만 본다 — 실제 모델은 Task 9의 고정 환경
    실행이 확인한다.
    """
    import experiments.e3_dilution as e3

    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[80:144, 80:144] = 1
    write_mask_png(tmp_path / "box.png", mask)
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(tmp_path / "box.jpg")

    monkeypatch.setattr(
        e3, "gradient_map",
        lambda scalar_fn, image, device: np.where(mask == 1, 1.0, 0.0),
    )

    row = e3.measure_instance(
        "deit_s", _FakeModel(), {"image": "box", "instance_id": 1},
        image_dir=tmp_path, masks_dir=tmp_path, device="cpu",
    )

    assert row["status"] == "ok"
    assert row["precision_at_k"] == pytest.approx(1.0)
    assert row["mass_fraction"] == pytest.approx(1.0)
    assert row["k"] == 64 * 64
    assert row["random_baseline"] == pytest.approx(64 * 64 / (224 * 224))
    assert row["query_row"] is not None


def test_measure_instance_query_patch_is_inside_the_mask(tmp_path, monkeypatch):
    """질의 좌표가 실제로 마스크를 과반으로 덮는 패치여야 한다."""
    import experiments.e3_dilution as e3
    from bench.coverage import MIN_PATCH_COVERAGE, patch_coverage
    from models.probes import PATCH_GRID_AT_224

    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[32:160, 0:64] = 1
    mask[32:160, 160:224] = 1
    mask[160:192, 0:224] = 1   # U자
    write_mask_png(tmp_path / "u.png", mask)
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(tmp_path / "u.jpg")

    monkeypatch.setattr(
        e3, "gradient_map", lambda scalar_fn, image, device: np.ones((224, 224))
    )

    row = e3.measure_instance(
        "vim_s", _FakeModel(), {"image": "u", "instance_id": 1},
        image_dir=tmp_path, masks_dir=tmp_path, device="cpu",
    )

    coverage = patch_coverage(mask == 1, PATCH_GRID_AT_224["vim_s"])
    assert coverage[row["query_row"], row["query_col"]] > MIN_PATCH_COVERAGE


def test_measurement_columns_carry_every_grouping_key():
    """면적 구간과 종횡비가 행에 없으면 집계를 다시 계산할 수 없다.
    개별 기여도 지도는 커밋하지 않으므로 CSV가 유일한 원본이다."""
    for column in ("area_bin", "aspect_class", "area_fraction", "aspect_ratio",
                   "random_baseline", "k", "population_n", "status"):
        assert column in MEASUREMENT_COLUMNS


def test_a_failing_metric_does_not_lose_the_row(tmp_path, monkeypatch):
    """기여도가 전부 0이면 질량 비율은 정의되지 않지만 그 행은 남아야 한다.
    행이 사라지면 '측정하지 않은 셀'과 구분할 수 없다."""
    import experiments.e3_dilution as e3

    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[80:144, 80:144] = 1
    write_mask_png(tmp_path / "z.png", mask)
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(tmp_path / "z.jpg")

    monkeypatch.setattr(
        e3, "gradient_map", lambda scalar_fn, image, device: np.zeros((224, 224))
    )

    row = e3.measure_instance(
        "deit_s", _FakeModel(), {"image": "z", "instance_id": 1},
        image_dir=tmp_path, masks_dir=tmp_path, device="cpu",
    )

    assert row["status"] == "error"
    assert "질량이 0" in row["error"]
