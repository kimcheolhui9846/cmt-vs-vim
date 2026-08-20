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
