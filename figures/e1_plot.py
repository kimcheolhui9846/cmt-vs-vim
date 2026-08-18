"""E1 그림. CSV만 읽는다 — 손으로 넣은 숫자가 그림에 들어가지 않게."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("flops_traced", "FLOPs", 1e9, "GFLOPs"),
    ("latency_ms", "Latency", 1.0, "ms"),
    ("peak_allocated_bytes", "Peak VRAM (allocated)", 1024**3, "GiB"),
    ("peak_reserved_bytes", "Peak VRAM (reserved)", 1024**3, "GiB"),
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

    for ax, (column, title, scale, unit) in zip(axes, PANELS):
        for model, group in df.groupby("model"):
            ok = group[group["status"] == "ok"].sort_values("resolution")
            if not ok.empty:
                ax.plot(
                    ok["resolution"], ok[column] / scale, marker="o", label=model
                )
            # OOM은 0이 아니라 표식으로 남긴다.
            for _, row in group[group["status"] == "oom"].iterrows():
                ax.axvline(row["resolution"], linestyle=":", alpha=0.4)
                ax.annotate(
                    f"{row['model']} OOM",
                    xy=(row["resolution"], ax.get_ylim()[1] * 0.5),
                    rotation=90,
                    fontsize=7,
                    ha="right",
                )
        ax.set_xlabel("input resolution (px)")
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(plot_sweep("results/e1/sweep.csv", "results/e1/e1_sweep.png"))
