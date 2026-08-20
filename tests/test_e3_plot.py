"""그림이 무엇을 그릴지 정하는 판단만 검증한다. matplotlib은 거치지 않는다."""
import numpy as np
import pandas as pd
import pytest

from figures.e3_plot import (
    AREA_ORDER,
    ASPECT_ORDER,
    bin_positions,
    bin_series,
    ordered_bins,
)

# plot_dilution은 세 모델 x 두 조건이 모두 있는 CSV를 요구한다 — 한 모델이
# 통째로 빠진 결과를 그림이 조용히 그려 주면 안 되기 때문이다.
ALL_MODELS = ("deit_s", "cmt_s", "vim_s")


def _summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "vim_s", "condition": "pretrained", "area_bin": "5-10%",
         "precision_mean": 0.8, "precision_sem": 0.01, "baseline_mean": 0.07,
         "n": 40, "low_sample": False},
        {"model": "vim_s", "condition": "pretrained", "area_bin": "<2%",
         "precision_mean": 0.6, "precision_sem": 0.02, "baseline_mean": 0.01,
         "n": 50, "low_sample": False},
        {"model": "deit_s", "condition": "pretrained", "area_bin": "<2%",
         "precision_mean": 0.5, "precision_sem": 0.03, "baseline_mean": 0.01,
         "n": 50, "low_sample": False},
    ])


def test_bins_are_ordered_by_area_not_alphabetically():
    """문자열 정렬은 '10-20%'를 '2-5%'보다 앞에 놓는다. 그러면 x축이 면적
    순서가 아니게 되고, 이 그림의 요점인 '면적이 커질수록'이 사라진다."""
    shuffled = ["20-40%", "<2%", "10-20%", "2-5%"]
    assert ordered_bins(shuffled) == ["<2%", "2-5%", "10-20%", "20-40%"]


def test_bin_positions_are_absolute_not_relative_to_the_series():
    """모델마다 가진 구간이 달라도 같은 라벨은 같은 x 좌표에 놓여야 한다.

    문자열을 matplotlib에 그대로 넘기면 범주형 축이 카테고리를 만나는 순서대로
    쌓는다. 실측: 한 모델이 `<2%`/`20-40%`만, 다른 모델이 `2-5%`/`5-10%`만 가지면
    축이 `['<2%', '20-40%', '2-5%', '5-10%']`이 되고, 곡선이 뒤섞인 축에 그려지는데
    그림은 멀쩡해 보인다.
    """
    assert bin_positions(["<2%", "20-40%"]) == [0, 4]
    assert bin_positions(["2-5%", "5-10%"]) == [1, 2]
    assert bin_positions(list(AREA_ORDER)) == list(range(len(AREA_ORDER)))


def test_area_order_matches_the_bin_labels():
    from bench.coverage import AREA_BINS

    assert AREA_ORDER == tuple(label for _, _, label in AREA_BINS)


def test_aspect_order_puts_tall_and_wide_at_the_ends():
    assert ASPECT_ORDER == ("wide", "square", "tall")


def test_bin_series_returns_only_that_models_rows_in_bin_order():
    series = bin_series(_summary(), "vim_s")

    assert list(series["area_bin"]) == ["<2%", "5-10%"]
    assert list(series["precision_mean"]) == [0.6, 0.8]


def test_bin_series_keeps_the_baseline_alongside_each_point():
    """기준선이 없으면 0.6이 좋은 값인지 알 수 없다. 인스턴스마다 K/N이
    달라서 구간마다 바닥이 다르다."""
    series = bin_series(_summary(), "vim_s")

    assert list(series["baseline_mean"]) == [0.01, 0.07]


def test_bin_series_fails_on_an_unknown_bin_label():
    """구간 라벨이 바뀌면 조용히 빈 그림이 나온다. 크게 실패시킨다."""
    bad = _summary()
    bad.loc[0, "area_bin"] = "huge"

    with pytest.raises(ValueError, match="huge"):
        bin_series(bad, "vim_s")


def test_plot_writes_a_file(tmp_path):
    from bench.coverage import LOW_SAMPLE_MIN
    from figures.e3_plot import plot_dilution

    rows = []
    for model in ("deit_s", "cmt_s", "vim_s"):
        for condition in ("pretrained", "random_init"):
            for index in range(LOW_SAMPLE_MIN + 5):
                fraction = 0.01 + 0.01 * (index % 6)
                rows.append({
                    "model": model, "condition": condition,
                    "image": f"img_{index:03d}", "instance_id": 1,
                    "area_bin": "<2%" if index % 2 else "2-5%",
                    "aspect_class": ASPECT_ORDER[index % 3],
                    "area_fraction": fraction,
                    "precision_at_k": 0.4 + 0.01 * index,
                    "mass_fraction": 0.2,
                    "random_baseline": fraction,
                    "status": "ok",
                })
    csv_path = tmp_path / "coverage.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out = plot_dilution(csv_path, tmp_path / "e3_coverage.png")

    assert out.exists() and out.stat().st_size > 0


def test_plot_refuses_an_empty_csv(tmp_path):
    from figures.e3_plot import plot_dilution

    csv_path = tmp_path / "coverage.csv"
    pd.DataFrame(columns=["model", "status"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="비어"):
        plot_dilution(csv_path, tmp_path / "out.png")
