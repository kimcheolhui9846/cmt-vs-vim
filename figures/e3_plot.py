"""E3 그림. results/e3/ 만 읽는다.

무엇을 그릴지 정하는 판단은 `ordered_bins`와 `bin_series`에 있다. 둘 다 순수
함수라 matplotlib을 거치지 않고 직접 검증할 수 있다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import style as figstyle
import pandas as pd

FIGSIZE = {"repo": (14.0, 10.0), "paper": (7.0, 5.0)}

from bench.coverage import (
    AREA_BINS,
    LOW_SAMPLE_MIN,
    aggregate,
    common_subset,
    expected_cells,
)
from experiments.e3_dilution import CONDITIONS
from models.registry import MODEL_NAMES

AREA_ORDER = tuple(label for _, _, label in AREA_BINS)
"""x축 순서. 문자열 정렬은 '10-20%'를 '2-5%' 앞에 놓아 면적 순서를 깬다 —
그러면 이 그림의 요점인 "면적이 커질수록"이 사라진다."""

ASPECT_ORDER = ("wide", "square", "tall")
"""E2와의 교차 검증 축. 가로형과 세로형을 양 끝에 둬야 차이가 눈에 들어온다.

E2는 Vim의 수직 감쇠가 더 가파름(감쇠비 1.345)을 측정했다. 그렇다면 Vim은
세로형에서 뚜렷이 낮아야 한다.
"""

MODEL_COLOURS = {"deit_s": "tab:blue", "cmt_s": "tab:green", "vim_s": "tab:orange"}


def ordered_bins(labels) -> list[str]:
    unknown = set(labels) - set(AREA_ORDER)
    if unknown:
        raise ValueError(f"모르는 면적 구간 라벨: {sorted(unknown)}")
    return [label for label in AREA_ORDER if label in set(labels)]


def bin_positions(labels) -> list[int]:
    """각 라벨이 x축에서 놓일 자리. `AREA_ORDER`의 인덱스다.

    문자열을 그대로 matplotlib에 넘기면 안 된다. 범주형 축은 카테고리를 **만나는
    순서대로** 쌓으므로, 모델마다 가진 구간이 다르면 축이 면적 순서를 잃는다.
    실측: 한 모델이 `<2%`와 `20-40%`만, 다른 모델이 `2-5%`와 `5-10%`만 가지면 축이
    `['<2%', '20-40%', '2-5%', '5-10%']`이 된다. 곡선은 그 뒤섞인 축에 그려지는데
    **그림은 멀쩡해 보인다.**

    `ordered_bins`는 이것을 막지 못한다 — 그것은 한 모델의 계열 안에서만 순서를
    잡는다. 축의 순서는 호출 사이에 누적되는 별개의 문제다.
    """
    return [AREA_ORDER.index(label) for label in labels]


def bin_series(summary: pd.DataFrame, model: str) -> pd.DataFrame:
    """한 모델의 구간별 요약을 면적 순서로."""
    rows = summary[summary["model"] == model]
    order = ordered_bins(rows["area_bin"])
    indexed = rows.set_index("area_bin")
    return indexed.loc[order].reset_index()


BASELINE_TOLERANCE = 1e-9
"""구간별 기준선이 모델 간에 벌어져도 되는 한계.

같아야 하는 값이므로 실질적으로 0이다. 부동소수 합산 순서 때문에 생기는
마지막 자리 차이만 허용한다.
"""


def baseline_series(summary: pd.DataFrame) -> pd.DataFrame:
    """구간별 무작위 기준선. 모델과 무관한 **하나의** 계열이다.

    공통 부분집합에서는 세 모델이 같은 인스턴스를 재므로 구간별 K/N 평균이
    모델과 무관하게 같아야 한다. 다르다면 모델마다 다른 인스턴스를 잰 것이고,
    그것이 바로 이 실험이 막으려는 함정이다 — 조용히 평균내지 않고 터뜨린다.

    모델마다 한 번씩 그리던 예전 방식은 같은 선을 세 겹으로 겹쳐 색을 뭉개고,
    범례 없는 네 번째 계열처럼 보이게 만들었다.
    """
    spread = summary.groupby("area_bin")["baseline_mean"].agg(
        lambda values: values.max() - values.min()
    )
    disagreeing = spread[spread > BASELINE_TOLERANCE]
    if not disagreeing.empty:
        raise ValueError(
            f"구간별 기준선이 모델마다 다르다: {disagreeing.to_dict()}. "
            "세 모델이 같은 인스턴스를 재지 않았다는 뜻이다 — common_subset을 확인할 것."
        )
    order = ordered_bins(summary["area_bin"])
    means = summary.groupby("area_bin")["baseline_mean"].mean().reindex(order)
    return means.reset_index()


def _draw_bins(ax, summary: pd.DataFrame, title: str) -> None:
    for model, colour in MODEL_COLOURS.items():
        if model not in set(summary["model"]):
            continue
        series = bin_series(summary, model)
        x = bin_positions(series["area_bin"])
        ax.errorbar(
            x, series["precision_mean"],
            yerr=series["precision_sem"], marker="o", color=colour, label=model,
            capsize=3,
        )
        low = series[series["low_sample"]]
        if not low.empty:
            ax.scatter(bin_positions(low["area_bin"]), low["precision_mean"],
                       marker="x", color="black", zorder=5,
                       label=f"n < {LOW_SAMPLE_MIN}")

    # 기준선은 인스턴스마다 K/N이라 구간마다 다르지만 **모델 간에는 같다** —
    # 공통 부분집합이 세 모델에 같은 인스턴스를 주기 때문이다. 그래서 모델
    # 색이 아니라 중립색으로 한 번만 그리고 범례에 이름을 준다.
    baseline = baseline_series(summary)
    ax.plot(bin_positions(baseline["area_bin"]), baseline["baseline_mean"],
            linestyle=":", color="dimgray", label="random baseline K/N")

    # 눈금은 어느 모델이 무엇을 가졌든 여섯 구간 전부를 면적 순서로 고정한다.
    # 문자열 x에 맡기면 축이 만나는 순서대로 쌓여 순서를 잃는다.
    ax.set_xticks(range(len(AREA_ORDER)))
    ax.set_xticklabels(AREA_ORDER)
    ax.set_xlabel("object area (fraction of the image)")
    ax.set_ylabel("precision@K")
    ax.set_title(title)
    # 라벨 중복(저표본 x 표시가 모델마다 붙는다)을 제거한다
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)


def _draw_aspect(ax, summary: pd.DataFrame, title: str) -> None:
    classes = [c for c in ASPECT_ORDER if c in set(summary["aspect_class"])]
    width = 0.25
    for offset, (model, colour) in enumerate(MODEL_COLOURS.items()):
        rows = summary[summary["model"] == model].set_index("aspect_class")
        rows = rows.reindex(classes)
        positions = [index + (offset - 1) * width for index in range(len(classes))]
        ax.bar(positions, rows["precision_mean"], width=width, color=colour,
               yerr=rows["precision_sem"], capsize=3, label=model)
        for position, low, n in zip(positions, rows["low_sample"], rows["n"]):
            if bool(low):
                ax.text(position, 0.01, f"n={int(n)}", ha="center", fontsize=7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("precision@K")
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_dilution(csv_path: Path | str, out_path: Path | str,
                  style: str = "repo") -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path}가 비어 있다 — 먼저 실험을 실행할 것")

    # 세 모델 전부에서 질의를 찾은 인스턴스만 쓴다. 모델마다 다른 부분집합으로
    # 평균을 내면 그 차이가 곧 모델 차이로 읽힌다 — 하필 이 실험이 재려는 축이
    # 객체 크기인데, CMT의 7x7 격자가 작은 객체를 빼놓는다.
    kept = common_subset(df, expected_cells(MODEL_NAMES, CONDITIONS))
    if kept.empty:
        raise ValueError(f"{csv_path}에 세 모델 모두 측정한 인스턴스가 없다")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE[figstyle.check(style)],
                             constrained_layout=True)

    for column, condition in enumerate(("pretrained", "random_init")):
        cell = kept[kept["condition"] == condition]
        if cell.empty:
            axes[0][column].text(0.5, 0.5, "not measured",
                                 transform=axes[0][column].transAxes,
                                 ha="center", va="center")
            continue
        _draw_bins(
            axes[0][column],
            aggregate(cell, ("model", "condition", "area_bin")),
            f"{condition} - precision@K by object area",
        )
        _draw_aspect(
            axes[1][column],
            aggregate(cell, ("model", "condition", "aspect_class")),
            f"{condition} - by bounding-box aspect",
        )

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=figstyle.dpi(style))
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(plot_dilution("results/e3/coverage.csv", "results/e3/e3_coverage.png"))
