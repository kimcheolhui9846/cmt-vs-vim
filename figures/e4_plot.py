"""E4의 2x2 표와 학습곡선.

캔버스에 그려지는 문자열은 전부 영어다 — matplotlib 기본 폰트에 한글 글리프가 없어
PNG에 네모 상자로 찍힌다. 주석과 docstring은 한글 그대로다.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import style as figstyle  # noqa: E402

FIGSIZE = {"repo": (13.0, 5.0), "paper": (7.0, 2.8)}

from bench.factorial import (  # noqa: E402
    CELLS,
    cell_means,
    complete_seed_count,
    incomplete_seeds,
    summarize,
)

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


def incomplete_seed_positions(rows: list[dict]) -> list[tuple[int, float]]:
    """미완성 seed마다 서로 다른 y 좌표를 배정한다.

    Fix round 1, finding 1: 전부 같은 자리(0.02, 0.05)에 찍으면 seed가 둘 이상
    미완성일 때 라벨이 겹쳐서 안 보인다. missing_cells 루프가 enumerate로 이미
    피하는 문제를 여기서도 똑같이 피한다.
    """
    return [(seed, 0.05 + 0.06 * i) for i, seed in enumerate(incomplete_seeds(rows))]


def effect_caption(effects: dict) -> str:
    """효과 캡션 문자열. 잴 수 없는 효과는 숫자가 아니라 n/a로 적는다.

    summarize가 (None, None)을 주는 상태 — 네 칸이 다 찬 seed가 하나도 없는 상태 —
    를 "+0.00 +- 0.00"으로 찍으면, 아무것도 재지 않은 그림이 이 실험의 헤드라인
    수치를 0으로 주장한다. 캔버스에 그려지므로 문자열은 전부 ASCII다.
    """
    parts = []
    for name, (mean, std) in effects.items():
        if mean is None:
            parts.append(f"{name}: n/a")
        else:
            parts.append(f"{name}: {mean * 100:+.2f} +- {std * 100:.2f}")
    return "  ".join(parts)


def bar_title(complete_seeds: int) -> str:
    """막대와 캡션이 서로 다른 표본에서 나온다는 사실을 제목에 적는다.

    막대는 `bench.factorial.cell_means`에서 나오고 그 함수는 칸별로 `status == "ok"`인
    행을 전부 평균낸다 — 네 칸이 다 차지 않은 seed의 행도 포함한다. 반면 캡션의 세
    효과는 `summarize`가 네 칸이 모두 찬 seed에 대해서만 계산한다. 그래서 A가 성공
    seed 3개, D가 2개인 중간 상태에서 두 표본 크기가 실제로 다르다.

    이전 제목 "mean +- std over n=<k> complete seeds"는 그 n을 막대에 붙여, A 막대가
    3-seed 평균인데 n=2라고 주장했다. 막대 쪽 표본을 completed seed로 좁히지 않고
    문구를 고치는 쪽을 택한 이유는, 좁히면 실제로 측정된 run이 그림에서 사라지고
    완성된 seed가 0인 동안 네 막대가 전부 높이 0으로 그려지기 때문이다 — 방금
    `summarize`에서 걷어낸 "재지 않은 것을 측정된 0으로 그리는" 실패와 같은 모양이다.

    두 줄로 돌려준다. 첫 줄이 막대의 표본을, 둘째 줄이 캡션 효과의 n을 밝힌다.
    캔버스에 그려지므로 전부 ASCII다.
    """
    return (
        "E4 2x2 factorial -- bars: mean +- std of all ok runs per cell\n"
        f"effects below: n={complete_seeds} seeds with all four cells"
    )


def plot_e4(runs_csv, curves_dir, out_path, style: str = "repo") -> Path:
    rows = _read(runs_csv)
    means = cell_means(rows)
    effects = summarize(rows)

    fig, (bar_ax, curve_ax) = plt.subplots(
        1, 2, figsize=FIGSIZE[figstyle.check(style)])

    labels = [CELL_LABELS[c] for c in CELLS]
    values = [means[c][0] * 100 for c in CELLS]
    errors = [means[c][1] * 100 for c in CELLS]
    bar_ax.bar(range(len(CELLS)), values, yerr=errors, capsize=4,
               color=["tab:blue", "tab:cyan", "tab:green", "tab:olive"])
    bar_ax.set_xticks(range(len(CELLS)))
    bar_ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    bar_ax.set_ylabel("Tiny-ImageNet top-1 (%)")
    # 두 줄짜리 제목이라 기본 크기로는 패널 폭을 넘어간다.
    bar_ax.set_title(bar_title(complete_seed_count(rows)), fontsize=9)

    bar_ax.annotate(effect_caption(effects), xy=(0.5, -0.32), xycoords="axes fraction",
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

    for seed, y in incomplete_seed_positions(rows):
        bar_ax.annotate(f"seed {seed} incomplete", xy=(0.02, y),
                        xycoords="axes fraction", color="tab:gray", fontsize=7)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=figstyle.dpi(style))
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_e4("results/e4/runs.csv", "results/e4/curves",
                  "results/e4/e4_factorial.png"))
