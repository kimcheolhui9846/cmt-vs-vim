"""E1 그림. CSV만 읽는다 — 손으로 넣은 숫자가 그림에 들어가지 않게.

무엇을 그릴지 정하는 판단은 `plotted_series`와 `missing_cells`에 있다. 둘 다
순수 함수라 matplotlib을 거치지 않고 직접 검증할 수 있다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("flops_total", "FLOPs", 1e9, "GFLOPs"),
    ("latency_ms", "Latency", 1.0, "ms"),
    ("peak_allocated_bytes", "Peak VRAM (allocated)", 1024**3, "GiB"),
    ("peak_reserved_bytes", "Peak VRAM (reserved)", 1024**3, "GiB"),
]

# 측정되지 않은 셀은 색과 문구로 이유를 말한다. 0으로 그리지 않는다.
MISSING_STATUSES = {
    "oom": ("tab:red", "OOM"),
    "error": ("tab:orange", "ERROR"),
    "no_cuda": ("tab:gray", "no CUDA"),
}


def plotted_series(df: pd.DataFrame, column: str) -> dict[str, list[tuple[int, float]]]:
    """모델별로 선에 올릴 (해상도, 값). status가 "ok"이고 값이 있는 셀만."""
    series: dict[str, list[tuple[int, float]]] = {}

    for model, group in df.groupby("model"):
        usable = group[(group["status"] == "ok") & group[column].notna()]
        points = [
            (int(row.resolution), float(getattr(row, column)))
            for row in usable.sort_values("resolution").itertuples()
        ]
        if points:
            series[str(model)] = points

    return series


def missing_cells(df: pd.DataFrame) -> list[tuple[int, str, str]]:
    """측정되지 않은 셀 (해상도, 모델, 상태). 해상도 오름차순."""
    unmeasured = df[df["status"].isin(MISSING_STATUSES)]
    return [
        (int(row.resolution), str(row.model), str(row.status))
        for row in unmeasured.sort_values(["resolution", "model"]).itertuples()
    ]


def plot_sweep(csv_path: Path | str, out_path: Path | str) -> Path:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{csv_path}가 비어 있다 — 먼저 sweep을 실행할 것")

    if df.empty:
        raise ValueError(f"{csv_path}가 비어 있다 — 먼저 sweep을 실행할 것")

    out_path = Path(out_path)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4))
    unmeasured = missing_cells(df)

    for ax, (column, title, scale, unit) in zip(axes, PANELS):
        series = plotted_series(df, column)

        for model, points in series.items():
            ax.plot(
                [resolution for resolution, _ in points],
                [value / scale for _, value in points],
                marker="o",
                label=model,
            )

        _mark_unmeasured(ax, unmeasured)

        ax.set_xlabel("input resolution (px)")
        ax.set_ylabel(unit)
        ax.set_title(title)

        if series:
            ax.set_yscale("log")
            ax.legend()
        else:
            # 범례를 부르면 "No artists with labels" 경고가 난다.
            ax.text(
                0.5, 0.5, "no values measured",
                transform=ax.transAxes, ha="center", va="center",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _mark_unmeasured(ax, unmeasured: list[tuple[int, str, str]]) -> None:
    """같은 해상도에서 여러 모델이 실패해도 라벨이 겹치지 않게 쌓아 올린다.

    y 위치를 axes fraction으로 잡아, 성공한 행이 하나도 없어 축 범위가
    기본값인 패널에서도 라벨이 화면 안에 남는다.
    """
    seen_at: dict[int, int] = {}

    for resolution, model, status in unmeasured:
        color, label = MISSING_STATUSES[status]
        offset = seen_at.get(resolution, 0)
        seen_at[resolution] = offset + 1

        ax.axvline(resolution, color=color, linestyle=":", alpha=0.6)
        ax.annotate(
            f"{model} {label}",
            xy=(resolution, 0.95 - 0.09 * offset),
            xycoords=("data", "axes fraction"),
            rotation=90,
            fontsize=7,
            color=color,
            ha="right",
            va="top",
        )


if __name__ == "__main__":
    print(plot_sweep("results/e1/sweep.csv", "results/e1/e1_sweep.png"))
