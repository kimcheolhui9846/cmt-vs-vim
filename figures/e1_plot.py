"""E1 그림. CSV만 읽는다 — 손으로 넣은 숫자가 그림에 들어가지 않게.

무엇을 그릴지 정하는 판단은 `plotted_series`와 `missing_cells`에 있다. 둘 다
순수 함수라 matplotlib을 거치지 않고 직접 검증할 수 있다.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ANALYTIC_SHARE_COLUMN = "flops_analytic_share"

# (열, 제목, 나눌 값, 단위, y축). y축을 패널마다 선언하는 이유는 analytic 비중이
# 정확히 0인 모델(DeiT·CMT)이 있기 때문이다 — log 축은 0을 그리지 못해서, 그 패널만
# log로 두면 "이 모델은 전부 측정값"이라는 사실이 그림에서 사라진다.
PANELS = [
    ("flops_total", "FLOPs", 1e9, "GFLOPs", "log"),
    ("latency_ms", "Latency", 1.0, "ms", "log"),
    ("peak_allocated_bytes", "Peak VRAM (allocated)", 1024**3, "GiB", "log"),
    ("peak_reserved_bytes", "Peak VRAM (reserved)", 1024**3, "GiB", "log"),
    (ANALYTIC_SHARE_COLUMN, "Analytic share of FLOPs", 0.01, "%", "linear"),
]

# 측정되지 않은 셀은 색과 문구로 이유를 말한다. 0으로 그리지 않는다.
MISSING_STATUSES = {
    "oom": ("tab:red", "OOM"),
    "error": ("tab:orange", "ERROR"),
    "no_cuda": ("tab:gray", "no CUDA"),
}


def with_analytic_share(df: pd.DataFrame) -> pd.DataFrame:
    """FLOPs 중 공식으로 채운 비중을 열로 덧붙인다.

    Vim의 FLOPs는 절반 가까이가 fused op 핸들러가 공식으로 채운 값이다. 합계만
    그리면 그 사실이 그림에서 보이지 않으므로 비중을 따로 드러낸다. DeiT·CMT는
    0이 나오고, 그 0이 "이 값은 전부 fvcore가 직접 센 것"이라는 뜻이다.

    FLOPs를 못 잰 셀은 0이 아니라 NaN이다. 0으로 채우면 전부 측정된 모델과
    똑같아 보이는데, 측정 실패를 0으로 그리지 않는다는 이 그림의 원칙에 어긋난다.
    """
    missing = [c for c in ("flops_analytic", "flops_total") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} 열이 없다 — 이 열이 생기기 전에 만들어진 CSV다. "
            "sweep을 다시 돌릴 것."
        )

    total = pd.to_numeric(df["flops_total"], errors="coerce")
    analytic = pd.to_numeric(df["flops_analytic"], errors="coerce")
    return df.assign(**{ANALYTIC_SHARE_COLUMN: analytic / total.where(total > 0)})


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

    df = with_analytic_share(df)

    out_path = Path(out_path)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4))
    unmeasured = missing_cells(df)

    for ax, (column, title, scale, unit, yscale) in zip(axes, PANELS):
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
            ax.set_yscale(yscale)
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
