"""results/*/의 CSV에서 논문의 표와 매크로를 생성한다.

논문에 손으로 옮겨 적은 숫자가 남지 않게 하는 장치다. 본문은 여기서 나온
\\input과 매크로로만 수치를 받는다.

효과값은 bench.factorial의 함수를 그대로 부른다 — 논문과 저장소가 같은 계산을
쓰게 하려는 것이다. 여기서 다시 계산하면 두 값이 갈라질 수 있고, 갈라져도
아무 표시가 남지 않는다.
"""
import csv
import json
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


# 논문 §3이 "하나의 고정 환경"을 주장한다. 네 실험이 실제로 같은 환경에서
# 나왔는지 여기서 확인하고, 다르면 매크로를 만들지 않고 죽는다. 손으로 쓴
# 버전 문자열은 환경이 바뀌어도 조용히 옛 값으로 남는다.
ENV_KEYS = ("python", "torch", "cuda", "gpu", "driver")
EXPERIMENTS = ("e1", "e2", "e3", "e4")


def shared_environment():
    """네 실험의 env.json이 합의하는 환경. 어긋나면 ValueError."""
    envs = {}
    for name in EXPERIMENTS:
        path = RESULTS / name / "env.json"
        envs[name] = json.loads(path.read_text(encoding="utf-8"))

    common = {}
    for key in ENV_KEYS:
        values = {name: env.get(key) for name, env in envs.items()}
        distinct = set(values.values())
        if len(distinct) != 1:
            raise ValueError(
                f"env.json의 '{key}'가 실험마다 다르다: {values}. "
                f"논문 3절은 하나의 고정 환경을 주장하므로 이대로 쓸 수 없다."
            )
        common[key] = distinct.pop()
    return common


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
    lines = [r"\begin{table}[tbp]", r"\centering",
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


def tab_e4_runs():
    """열두 run을 그대로 싣는다.

    본문 표는 칸 평균만 보여 준다. 요인 효과를 독자가 직접 다시 계산하려면
    run 단위 값이 있어야 하므로 부록에 전부 싣는다.
    """
    body = []
    for row in _e4_rows():
        body.append(" & ".join([
            CELL_SUFFIX[row["cell"]],
            row["seed"],
            f"{float(row['top1']) * 100:.2f}",
            f"{float(row['top5']) * 100:.2f}",
            f"{float(row['params']) / 1e6:.3f}",
            f"{float(row['hours']):.3f}",
        ]))
    return _table(
        "Every E4 run. Cells are A flat/attention, B flat/SSM, "
        "C hierarchical/attention, D hierarchical/SSM; all twelve ran "
        "$300$ epochs to completion under one recipe. The effects in "
        r"Table~\ref{tab:e4effects} are computed from these rows.",
        "tab:e4runs", "llrrrr",
        "Cell & Seed & Top-1 & Top-5 & Params (M) & Hours", body)


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


# E1의 sweep은 같은 고정 환경에서 두 번 돌았다. 실행 A는 이 커밋이 커밋했고
# 실행 B가 지금 results/e1/sweep.csv에 있다. 실행 **사이**의 변동은 두 CSV를
# 대조해야만 나오므로 옛 판을 git에서 꺼내 읽는다.
RUN_A_COMMIT = "fa35e51"
RUN_A_PATH = "results/e1/sweep.csv"


def _rows_at_commit(commit, path):
    """어느 커밋 시점의 CSV를 읽는다.

    실행 **사이**의 변동은 옛 판과 대조해야만 나온다. 파일로 중복 커밋하지
    않는 이유는 이미 이력 안에 있기 때문이고, 해시를 박아 두었으므로 결과는
    결정적이다.
    """
    import io as _io
    import subprocess

    try:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"{commit}:{path}를 꺼내지 못했다. 이 저장소의 이력이 있어야 "
            f"실행 사이 변동을 계산할 수 있다."
        ) from exc
    return list(csv.DictReader(_io.StringIO(blob)))


def _run_a_rows():
    return _rows_at_commit(RUN_A_COMMIT, RUN_A_PATH)


# E2도 두 번 돌았다. 이 커밋의 판은 서로 다른 이미지 집합을 썼고 N=256까지만
# 갔다. 두 실행을 비교할 때는 **같은 N에서** 비교해야 한다 - 표본 크기가 다른
# 두 값을 나란히 놓으면 표본 효과가 실행 간 변동으로 읽힌다.
ERF_RUN_A_COMMIT = "d9b45a2"
ERF_RUN_A_PATH = "results/e2/erf_metrics.csv"
ERF_COMPARE_N = "256"


def between_run_decay_drift(model="vim_s", condition="natural"):
    """두 독립 E2 실행 사이의 감쇠비 차이(%). 같은 N에서 잰다."""
    def pick(rows):
        for row in rows:
            if (row["model"] == model and row["condition"] == condition
                    and row["n_images"] == ERF_COMPARE_N):
                return float(row["decay_ratio"])
        raise ValueError(
            f"{model}/{condition}/N={ERF_COMPARE_N} 행이 없다")

    old_value = pick(_rows_at_commit(ERF_RUN_A_COMMIT, ERF_RUN_A_PATH))
    new_value = pick(_read(RESULTS / "e2" / "erf_metrics.csv"))
    return abs(old_value - new_value) / old_value * 100.0


def between_run_latency_spread():
    """두 독립 실행 사이의 latency 최대 비와 그것이 나온 셀.

    한 실행 안의 반복 편차(_latency_spread)와는 다른 양이다. 이 저장소가
    반복 측정으로 잡으려 한 것은 앞의 것인데, 실제로 크게 갈린 것은 뒤의
    것이다 - 배치 1 latency의 불확실성은 프로세스 경계에 있다.
    """
    run_a = {(r["model"], r["resolution"]): r for r in _run_a_rows()}
    worst_ratio, worst_cell = 0.0, None
    for row in _e1_rows():
        key = (row["model"], row["resolution"])
        if key not in run_a:
            continue
        a = float(run_a[key]["latency_ms"])
        b = float(row["latency_ms"])
        ratio = max(a, b) / min(a, b)
        if ratio > worst_ratio:
            worst_ratio, worst_cell = ratio, key
    return worst_ratio, worst_cell


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
        _macro("LatencyBetweenRunSpread", f"{between_run_latency_spread()[0]:.2f}"),
        _macro("VimDecayRunDrift", f"{between_run_decay_drift():.2f}"),
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
    # 종횡비 결과는 세 모델을 나란히 놓아야 뜻이 산다 - vim만 다르다는 것이
    # 주장이므로 나머지 둘의 값도 논문에 실린다.
    for model, suffix in (("deit_s", "Deit"), ("cmt_s", "Cmt")):
        diff, z = _e3_tall_wide(model=model)
        lines.append(_macro(f"{suffix}TallWide", f"{diff:+.4f}"))
        lines.append(_macro(f"{suffix}TallWideZ", f"{z:+.2f}"))
    diff, z = _e3_tall_wide(condition="random_init")
    lines.append(_macro("VimTallWideRandom", f"{diff:+.4f}"))
    lines.append(_macro("VimTallWideRandomZ", f"{z:+.2f}"))
    # 파라미터 예산은 측정값이 아니라 사전 등록한 설계 상수다. 그래도 손으로
    # 쓰지 않는 이유는 같다 - 탐색 코드가 쓰는 값과 논문이 말하는 값이 갈라지면
    # "네 칸 모두 대역 안"이라는 판정이 무의미해진다.
    from experiments.e4_widths import PARAM_TARGET, PARAM_TOLERANCE

    lines += [
        _macro("ParamBudget", f"{PARAM_TARGET / 1e6:.2f}"),
        _macro("ParamTolerance", f"{PARAM_TOLERANCE * 100:.0f}"),
    ]
    env = shared_environment()
    lines += [
        _macro("PyVersion", env["python"]),
        _macro("TorchVersion", env["torch"]),
        _macro("CudaVersion", env["cuda"]),
        _macro("GpuName", env["gpu"]),
        _macro("DriverVersion", env["driver"]),
    ]
    return "\n".join(lines) + "\n"


BUILDERS = {
    "tab_e1_sweep.tex": tab_e1_sweep,
    "tab_e2_erf.tex": tab_e2_erf,
    "tab_e3_excess.tex": tab_e3_excess,
    "tab_e4_cells.tex": tab_e4_cells,
    "tab_e4_effects.tex": tab_e4_effects,
    "tab_e4_runs.tex": tab_e4_runs,
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
