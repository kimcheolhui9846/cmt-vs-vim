"""E4의 2x2 표와 학습곡선.

캔버스에 그려지는 문자열은 전부 영어다 — matplotlib 기본 폰트에 한글 글리프가 없어
PNG에 네모 상자로 찍힌다. 주석과 docstring은 한글 그대로다.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.factorial import CELLS, cell_means, incomplete_seeds, summarize  # noqa: E402

MISSING_STATUSES = {
    "error": ("tab:orange", "ERROR"),
    "oom": ("tab:red", "OOM"),
}

CELL_LABELS = {
    "a_deit_ti": "A DeiT-Ti (flat, attn)",
    "b_vim_ti": "B Vim-Ti (flat, SSM)",
    "c_cmt_ti": "C CMT-Ti (hier, attn)",
    "d_hvim": "D H-Vim (hier, SSM)",
}


def _read(csv_path) -> list[dict]:
    with Path(csv_path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def missing_cells(rows: list[dict]) -> list[tuple[str, int, str]]:
    """측정되지 않은 (칸, seed, 상태). 빈칸이 왜 비었는지를 그림에 남기기 위한 것."""
    return [
        (row["cell"], int(row["seed"]), row["status"])
        for row in rows
        if row.get("status") in MISSING_STATUSES
    ]


def plot_e4(runs_csv, curves_dir, out_path) -> Path:
    rows = _read(runs_csv)
    means = cell_means(rows)
    effects = summarize(rows)

    fig, (bar_ax, curve_ax) = plt.subplots(1, 2, figsize=(13, 5))

    labels = [CELL_LABELS[c] for c in CELLS]
    values = [means[c][0] * 100 for c in CELLS]
    errors = [means[c][1] * 100 for c in CELLS]
    bar_ax.bar(range(len(CELLS)), values, yerr=errors, capsize=4,
               color=["tab:blue", "tab:cyan", "tab:green", "tab:olive"])
    bar_ax.set_xticks(range(len(CELLS)))
    bar_ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    bar_ax.set_ylabel("Tiny-ImageNet top-1 (%)")
    bar_ax.set_title("E4 2x2 factorial (mean +- std over seeds)")

    caption = "  ".join(
        f"{name}: {mean * 100:+.2f} +- {std * 100:.2f}"
        for name, (mean, std) in effects.items()
    )
    bar_ax.annotate(caption, xy=(0.5, -0.32), xycoords="axes fraction",
                    ha="center", fontsize=8)

    for cell in CELLS:
        for curve in sorted(Path(curves_dir).glob(f"{cell}_seed*.csv")):
            points = _read(curve)
            curve_ax.plot(
                [int(p["epoch"]) for p in points],
                [float(p["val_top1"]) * 100 for p in points],
                label=curve.stem, linewidth=1,
            )
    curve_ax.set_xlabel("epoch")
    curve_ax.set_ylabel("val top-1 (%)")
    curve_ax.set_title("Validation curves")
    if curve_ax.get_legend_handles_labels()[0]:
        curve_ax.legend(fontsize=6, ncol=2)

    for i, (cell, seed, status) in enumerate(missing_cells(rows)):
        colour, label = MISSING_STATUSES[status]
        bar_ax.annotate(f"{cell} seed{seed} {label}", xy=(0.02, 0.95 - 0.06 * i),
                        xycoords="axes fraction", color=colour, fontsize=7)

    for seed in incomplete_seeds(rows):
        bar_ax.annotate(f"seed {seed} incomplete", xy=(0.02, 0.05),
                        xycoords="axes fraction", color="tab:gray", fontsize=7)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_e4("results/e4/runs.csv", "results/e4/curves",
                  "results/e4/e4_factorial.png"))
