"""results/*/의 CSV에서 논문의 표와 매크로를 생성한다.

논문에 손으로 옮겨 적은 숫자가 남지 않게 하는 장치다. 본문은 여기서 나온
\\input과 매크로로만 수치를 받는다.

효과값은 bench.factorial의 함수를 그대로 부른다 — 논문과 저장소가 같은 계산을
쓰게 하려는 것이다. 여기서 다시 계산하면 두 값이 갈라질 수 있고, 갈라져도
아무 표시가 남지 않는다.
"""
import csv
import math
import statistics
from pathlib import Path

from bench.factorial import CELLS, cell_means, summarize

RESULTS = Path("results")

CELL_LABEL = {
    "a_deit_ti": "A DeiT-Ti (flat, attn)",
    "b_vim_ti": "B Vim-Ti (flat, SSM)",
    "c_cmt_ti": "C CMT-Ti (hier, attn)",
    "d_hvim": "D H-Vim (hier, SSM)",
}
CELL_SUFFIX = {"a_deit_ti": "A", "b_vim_ti": "B", "c_cmt_ti": "C", "d_hvim": "D"}
EFFECT_LABEL = {
    "structure": r"Structure $(C{+}D)/2-(A{+}B)/2$",
    "operator": r"Operator $(A{+}C)/2-(B{+}D)/2$",
    "interaction": r"Interaction $(D{-}B)-(C{-}A)$",
}
AREA_BINS = ("<2%", "2-5%", "5-10%", "10-20%", "20-40%", ">=40%")
MODELS = ("deit_s", "cmt_s", "vim_s")
MODEL_LABEL = {"deit_s": "DeiT-S", "cmt_s": "CMT-S", "vim_s": "Vim-S"}


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _e1_rows():
    return _read(RESULTS / "e1" / "sweep.csv")


def _e4_rows():
    return _read(RESULTS / "e4" / "runs.csv")


def _e2_rows(n_images="512"):
    return [r for r in _read(RESULTS / "e2" / "erf_metrics.csv")
            if r["n_images"] == n_images]


def _e3_rows():
    return [r for r in _read(RESULTS / "e3" / "coverage.csv")
            if r["status"] == "ok"]


def _table(caption, label, colspec, header, body_rows):
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\",
             r"\midrule"]
    lines += [row + r" \\" for row in body_rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_e1_sweep():
    body = []
    for row in _e1_rows():
        ips = row["images_per_sec"]
        body.append(" & ".join([
            MODEL_LABEL[row["model"]],
            row["resolution"],
            f"{float(row['params']) / 1e6:.2f}",
            f"{float(row['flops_total']) / 1e9:.2f}",
            f"{float(row['latency_ms']):.2f}",
            f"{float(row['peak_allocated_bytes']) / 2 ** 20:.1f}",
            f"{float(ips):.1f}" if ips else r"\textemdash",
        ]))
    return _table(
        "Measured resolution sweep in the pinned environment. "
        "\\textemdash{} marks a cell where the throughput search hit CUDA "
        "OOM; those runs still completed every other measurement, and the "
        "reason is recorded in the \\texttt{error} column of "
        "\\texttt{results/e1/sweep.csv}.",
        "tab:e1", "llrrrrr",
        "Model & Res. & Params (M) & FLOPs (G) & Latency (ms) & "
        "Peak alloc (MiB) & img/s", body)


def tab_e2_erf():
    body = []
    for row in _e2_rows():
        if row["condition"] not in ("natural", "random_init"):
            continue
        body.append(" & ".join([
            MODEL_LABEL[row["model"]],
            row["condition"].replace("_", r"\_"),
            f"{float(row['anisotropy']):.3f}",
            f"{float(row['principal_angle_deg']):.2f}",
            f"{float(row['decay_ratio']):.3f}",
        ]))
    return _table(
        "ERF anisotropy at $n=512$ images. The covariance index separates no "
        "model; the axis-wise decay ratio and the untrained condition do.",
        "tab:e2", "llrrr",
        "Model & Condition & Anisotropy & Principal angle (deg) & Decay ratio",
        body)


def tab_e3_excess():
    rows = _e3_rows()
    body = []
    for area_bin in AREA_BINS:
        cells = []
        for model in MODELS:
            values = [float(r["precision_at_k"]) - float(r["random_baseline"])
                      for r in rows
                      if r["model"] == model and r["condition"] == "pretrained"
                      and r["area_bin"] == area_bin]
            cells.append(f"{statistics.fmean(values):.4f}" if values
                         else r"\textemdash")
        body.append(" & ".join([area_bin.replace("%", r"\%"), *cells]))
    return _table(
        "Above-baseline precision@K by object area, pretrained weights. "
        "Vim-S is the lowest in every bin. Raw precision@K rises with object "
        "area for all three models because the random baseline rises faster; "
        "only the excess over that baseline is interpretable.",
        "tab:e3", "lrrr",
        "Area bin & " + " & ".join(MODEL_LABEL[m] for m in MODELS), body)


def tab_e4_cells():
    means = cell_means(_e4_rows())
    body = [" & ".join([CELL_LABEL[cell],
                        f"{means[cell][0] * 100:.2f}",
                        f"{means[cell][1] * 100:.2f}"]) for cell in CELLS]
    return _table(
        "E4 cell means, Tiny-ImageNet top-1 (\\%), $n=3$ seeds.",
        "tab:e4cells", "lrr", "Cell & Top-1 & Std", body)


def tab_e4_effects():
    effects = summarize(_e4_rows())
    body = [" & ".join([EFFECT_LABEL[name],
                        f"{effects[name][0] * 100:+.2f}",
                        f"{effects[name][1] * 100:.2f}"])
            for name in ("structure", "operator", "interaction")]
    return _table(
        "E4 factorial effects in percentage points, mean and standard "
        "deviation over $n=3$ seeds. Only the structure effect is large "
        "relative to its spread. The interaction sign is positive in every "
        "seed but its magnitude varies by a factor of 4.7, so we report the "
        "sign and not the size.",
        "tab:e4effects", "lrr", "Effect & Mean & Std", body)


def _macro(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _e3_tall_wide(model="vim_s", condition="pretrained"):
    """세로형과 가로형 객체의 precision 차이와 Welch z를 돌려준다.

    HANDOFF와 results/e3/README.md가 인용하는 값과 같은 계산이다.
    """
    rows = _e3_rows()
    groups = {
        aspect: [float(r["precision_at_k"]) for r in rows
                 if r["model"] == model and r["condition"] == condition
                 and r["aspect_class"] == aspect]
        for aspect in ("tall", "wide")
    }
    tall, wide = groups["tall"], groups["wide"]
    diff = statistics.fmean(tall) - statistics.fmean(wide)
    se = math.sqrt(statistics.variance(tall) / len(tall)
                   + statistics.variance(wide) / len(wide))
    return diff, diff / se


def _latency_spread():
    """한 실행 안에서 잰 latency 반복의 최대 비(max/min).

    실행 **사이**의 1.66배 변동은 두 커밋의 CSV를 대조해야 나오므로 여기서
    계산하지 않는다. 이 값은 한 실행 안의 편차이며, 둘은 다른 양이다.
    """
    ratios = []
    for row in _e1_rows():
        low, high = float(row["latency_min_ms"]), float(row["latency_max_ms"])
        if low > 0:
            ratios.append(high / low)
    return max(ratios)


def macros():
    e1 = {(r["model"], r["resolution"]): r for r in _e1_rows()}
    e2 = {(r["model"], r["condition"]): r for r in _e2_rows()}
    e4 = _e4_rows()
    effects, means = summarize(e4), cell_means(e4)
    tall_wide, tall_wide_z = _e3_tall_wide()

    lines = ["% paper/make_tables.py가 생성한다. 손으로 고치지 않는다."]
    for name in ("structure", "operator", "interaction"):
        key = name.capitalize()
        lines.append(_macro(f"{key}Effect", f"{effects[name][0] * 100:.2f}"))
        lines.append(_macro(f"{key}Std", f"{effects[name][1] * 100:.2f}"))
    for cell in CELLS:
        suffix = CELL_SUFFIX[cell]
        hours = [float(r["hours"]) for r in e4 if r["cell"] == cell]
        lines.append(_macro(f"Cell{suffix}", f"{means[cell][0] * 100:.2f}"))
        lines.append(_macro(f"Hours{suffix}", f"{statistics.fmean(hours):.3f}"))
    lines += [
        _macro("HoursTotal", f"{sum(float(r['hours']) for r in e4):.2f}"),
        _macro("CmtParamsLow",
               f"{float(e1[('cmt_s', '224')]['params']) / 1e6:.2f}"),
        _macro("CmtParamsHigh",
               f"{float(e1[('cmt_s', '1024')]['params']) / 1e6:.2f}"),
        _macro("VimFlopsHigh",
               f"{float(e1[('vim_s', '1024')]['flops_total']) / 1e9:.2f}"),
        _macro("DeitFlopsHigh",
               f"{float(e1[('deit_s', '1024')]['flops_total']) / 1e9:.2f}"),
        _macro("VimImgsHigh",
               f"{float(e1[('vim_s', '1024')]['images_per_sec']):.1f}"),
        _macro("DeitImgsHigh",
               f"{float(e1[('deit_s', '1024')]['images_per_sec']):.1f}"),
        _macro("VimPeakHigh",
               f"{float(e1[('vim_s', '1024')]['peak_allocated_bytes']) / 2 ** 20:.1f}"),
        _macro("DeitPeakHigh",
               f"{float(e1[('deit_s', '1024')]['peak_allocated_bytes']) / 2 ** 20:.1f}"),
        _macro("VimMaxBatchHigh", e1[("vim_s", "1024")]["max_batch"].split(".")[0]),
        _macro("DeitMaxBatchHigh", e1[("deit_s", "1024")]["max_batch"].split(".")[0]),
        _macro("LatencySpreadHigh", f"{_latency_spread():.2f}"),
        _macro("VimAnisoRandom",
               f"{float(e2[('vim_s', 'random_init')]['anisotropy']):.3f}"),
        _macro("VimAngleRandom",
               f"{float(e2[('vim_s', 'random_init')]['principal_angle_deg']):.3f}"),
        _macro("VimDecayNatural",
               f"{float(e2[('vim_s', 'natural')]['decay_ratio']):.2f}"),
        _macro("VimDecayRandom",
               f"{float(e2[('vim_s', 'random_init')]['decay_ratio']):.2f}"),
        _macro("VimTallWide", f"{tall_wide:+.4f}"),
        _macro("VimTallWideZ", f"{tall_wide_z:+.2f}"),
    ]
    return "\n".join(lines) + "\n"


BUILDERS = {
    "tab_e1_sweep.tex": tab_e1_sweep,
    "tab_e2_erf.tex": tab_e2_erf,
    "tab_e3_excess.tex": tab_e3_excess,
    "tab_e4_cells.tex": tab_e4_cells,
    "tab_e4_effects.tex": tab_e4_effects,
    "macros.tex": macros,
}


def build(out_dir="paper/generated"):
    """표와 매크로를 전부 다시 만든다.

    남아 있던 옛 파일을 지운다. 안 지우면 이름이 바뀐 표의 옛 판이 남아
    본문이 조용히 옛 숫자를 \\input한다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.tex"):
        if stale.name not in BUILDERS:
            stale.unlink()
    written = []
    for name, builder in BUILDERS.items():
        path = out / name
        path.write_text(builder(), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in build():
        print(path)
