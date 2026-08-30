import pandas as pd
import pytest

from figures.e1_plot import (
    ANALYTIC_SHARE_COLUMN,
    PANELS,
    error_bar_offsets,
    error_spans,
    missing_cells,
    plot_sweep,
    plotted_series,
    with_analytic_share,
)

BASE_ROW = {
    "model": "deit_s",
    "resolution": 224,
    "params": 22_000_000,
    "flops_traced": 4.6e9,
    "flops_analytic": 0,
    "flops_total": 4.6e9,
    "flops_uncounted_ops": "",
    "flops_unexpected_ops": "",
    "latency_ms": 5.0,
    "latency_min_ms": 4.5,
    "latency_max_ms": 6.0,
    "latency_repeats_ms": "4.5000;6.0000;5.0000",
    "peak_allocated_bytes": 1_000_000,
    "peak_reserved_bytes": 2_000_000,
    "max_batch": 32,
    "images_per_sec": 100.0,
    "status": "ok",
    "error": None,
}

UNMEASURED = {
    "latency_ms": None,
    "latency_min_ms": None,
    "latency_max_ms": None,
    "latency_repeats_ms": "",
    "peak_allocated_bytes": None,
    "peak_reserved_bytes": None,
    "max_batch": None,
    "images_per_sec": None,
}


def _row(**overrides):
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def _frame(rows):
    return pd.DataFrame(rows)


def _write_csv(path, rows):
    _frame(rows).to_csv(path, index=False)
    return path


def test_oom_cells_are_absent_from_the_series_not_zero():
    """OOM을 0으로 그리면 '메모리를 안 썼다'로 읽혀 논문 주장이 뒤집힌다."""
    df = _frame(
        [
            _row(resolution=224),
            _row(resolution=1024, status="oom", **UNMEASURED),
        ]
    )

    series = plotted_series(df, "peak_allocated_bytes")

    assert series["deit_s"] == [(224, 1_000_000)]
    assert all(resolution != 1024 for resolution, _ in series["deit_s"])


def test_error_and_no_cuda_cells_are_also_absent_from_the_series():
    df = _frame(
        [
            _row(resolution=512, status="error", error="RuntimeError: boom"),
            _row(resolution=768, status="no_cuda", **UNMEASURED),
        ]
    )

    assert plotted_series(df, "latency_ms") == {}


def test_a_row_missing_only_one_column_does_not_poison_the_others():
    """FLOPs는 OOM 셀에도 남는다. latency가 없다고 FLOPs까지 버리면 안 된다."""
    df = _frame([_row(resolution=1024, status="oom", flops_total=90e9, **UNMEASURED)])

    assert plotted_series(df, "flops_total") == {}
    assert missing_cells(df) == [(1024, "deit_s", "oom")]


def test_missing_cells_reports_every_unmeasured_status():
    """실패한 측정과 아직 재지 않은 셀이 그림에서 같아 보이면 안 된다."""
    df = _frame(
        [
            _row(resolution=224),
            _row(resolution=512, status="error", error="RuntimeError: boom"),
            _row(resolution=768, status="no_cuda", **UNMEASURED),
            _row(resolution=1024, status="oom", **UNMEASURED),
        ]
    )

    assert missing_cells(df) == [
        (512, "deit_s", "error"),
        (768, "deit_s", "no_cuda"),
        (1024, "deit_s", "oom"),
    ]


def test_writes_a_png_from_the_csv(tmp_path):
    csv = _write_csv(tmp_path / "sweep.csv", [_row()])

    out = plot_sweep(csv, tmp_path / "e1.png")

    assert out.exists()
    assert out.stat().st_size > 0


def test_refuses_an_empty_csv(tmp_path):
    """빈 입력에서 조용히 빈 그림을 내면 논문에 빈 그림이 실린다."""
    csv = _write_csv(tmp_path / "sweep.csv", [])

    with pytest.raises(ValueError, match="비어"):
        plot_sweep(csv, tmp_path / "e1.png")


def test_renders_when_nothing_succeeded(tmp_path):
    """전 해상도가 OOM인 모델도 그림이 나와야 한다. 성공 행이 하나도 없는
    패널에서 축 범위 계산과 범례가 깨지기 쉽다."""
    csv = _write_csv(
        tmp_path / "sweep.csv",
        [_row(resolution=1024, status="oom", **UNMEASURED)],
    )

    out = plot_sweep(csv, tmp_path / "e1.png")

    assert out.exists()
    assert out.stat().st_size > 0


# --- analytic 비중 패널 -------------------------------------------------------


def test_analytic_share_is_the_formula_filled_fraction():
    df = _frame([
        _row(model="vim_s", flops_traced=4.0e9, flops_analytic=1.0e9, flops_total=5.0e9),
    ])
    shares = with_analytic_share(df)[ANALYTIC_SHARE_COLUMN]
    assert shares.iloc[0] == pytest.approx(0.2)


def test_fully_measured_models_get_a_zero_share_not_a_blank():
    """DeiT·CMT는 0이어야 한다. 빈칸이면 '측정 실패'와 구분되지 않는다."""
    df = _frame([_row(flops_traced=4.6e9, flops_analytic=0, flops_total=4.6e9)])
    shares = with_analytic_share(df)[ANALYTIC_SHARE_COLUMN]
    assert shares.iloc[0] == 0.0
    assert shares.notna().all()


def test_share_is_undefined_rather_than_zero_when_flops_are_missing():
    """FLOPs를 못 잰 셀의 비중은 0이 아니라 '알 수 없음'이다.

    0으로 채우면 '전부 측정된 모델'과 똑같이 보인다 — 측정 실패를 0으로 그리지
    않는다는 이 그림의 원칙과 정면으로 어긋난다.
    """
    df = _frame([_row(status="oom", flops_traced=None, flops_analytic=None,
                      flops_total=None, **UNMEASURED)])
    shares = with_analytic_share(df)[ANALYTIC_SHARE_COLUMN]
    assert shares.isna().all()


def test_missing_analytic_column_fails_loudly():
    """열 자체가 없으면 스키마가 어긋난 것이다. 빈 패널로 넘어가면 안 된다."""
    df = _frame([_row()]).drop(columns=["flops_analytic"])
    with pytest.raises(ValueError, match="flops_analytic"):
        with_analytic_share(df)


def test_analytic_share_panel_is_linear_not_log():
    """log 축은 0을 그리지 못한다.

    DeiT·CMT의 비중이 정확히 0이므로, 이 패널이 log면 '전부 측정값'이라는 사실이
    그림에서 사라진다. 다른 패널은 자릿수가 커서 log가 맞다.
    """
    panels = {column: yscale for column, _, _, _, yscale, _ in PANELS}
    assert panels[ANALYTIC_SHARE_COLUMN] == "linear"
    assert panels["flops_total"] == "log"


def test_analytic_share_panel_is_drawn(tmp_path):
    csv = tmp_path / "sweep.csv"
    _frame([
        _row(model="deit_s", flops_traced=4.6e9, flops_analytic=0, flops_total=4.6e9),
        _row(model="vim_s", flops_traced=4.2e9, flops_analytic=1.5e9, flops_total=5.7e9),
    ]).to_csv(csv, index=False)

    out = plot_sweep(csv, tmp_path / "e1.png")

    assert out.exists()


# --- latency 반복 편차 --------------------------------------------------------


def test_error_spans_carry_the_repeat_range():
    df = _frame([_row(resolution=224, latency_min_ms=16.67, latency_max_ms=30.0)])

    spans = error_spans(df, "latency_min_ms", "latency_max_ms")

    assert spans == {"deit_s": {224: (16.67, 30.0)}}


def test_error_spans_skip_cells_that_were_not_measured():
    """편차가 없는 셀을 (0, 0)으로 채우면 그림에서 '완벽히 재현됨'으로 읽힌다."""
    df = _frame([_row(resolution=1024, status="oom", **UNMEASURED)])

    assert error_spans(df, "latency_min_ms", "latency_max_ms") == {}


def test_offsets_are_distances_from_the_plotted_value():
    """matplotlib의 yerr는 값에서의 거리다. 절대 좌표를 넘기면 막대가 엉뚱한
    곳에 그려지는데, 그림만 봐서는 틀린 줄 모른다."""
    points = [(224, 18.0)]
    spans = {224: (16.67, 30.0)}

    lower, upper = error_bar_offsets(points, spans, scale=1.0)

    assert lower == pytest.approx([18.0 - 16.67])
    assert upper == pytest.approx([30.0 - 18.0])


def test_offsets_follow_the_panel_scale():
    lower, upper = error_bar_offsets([(224, 2.0)], {224: (1.0, 4.0)}, scale=2.0)

    assert lower == pytest.approx([0.5])
    assert upper == pytest.approx([1.0])


def test_a_point_without_a_span_gets_no_bar():
    """반복 열이 없던 시절의 CSV도 그림은 나와야 한다 — 막대만 없으면 된다."""
    lower, upper = error_bar_offsets([(224, 5.0)], {}, scale=1.0)

    assert lower == [0.0]
    assert upper == [0.0]


def test_only_the_latency_panel_declares_bounds():
    bounds = {column: bound for column, _, _, _, _, bound in PANELS}

    assert bounds["latency_ms"] == ("latency_min_ms", "latency_max_ms")
    assert bounds["flops_total"] is None


def test_draws_a_figure_with_latency_bars(tmp_path):
    csv = _write_csv(
        tmp_path / "sweep.csv",
        [
            _row(resolution=224, latency_min_ms=16.67, latency_max_ms=30.0),
            _row(resolution=384, latency_min_ms=25.0, latency_max_ms=26.0),
        ],
    )

    out = plot_sweep(csv, tmp_path / "e1.png")

    assert out.exists()
    assert out.stat().st_size > 0


def test_column_gaps_finds_cells_that_are_ok_but_missing_this_column():
    """status가 "ok"인데 한 열만 비어 있는 셀을 찾는다.

    E1의 throughput OOM 세 셀이 그렇다. missing_cells는 status로만 판단하므로
    이 셀들을 잡지 못하고, 잡지 못하면 선이 빈 자리를 가로질러 이어져 재지
    않은 해상도에 측정값이 있는 것처럼 보인다.
    """
    import pandas as pd

    from figures.e1_plot import column_gaps

    df = pd.DataFrame([
        {"model": "vim_s", "resolution": 512, "status": "ok",
         "images_per_sec": None},
        {"model": "vim_s", "resolution": 768, "status": "ok",
         "images_per_sec": 25.7},
        {"model": "cmt_s", "resolution": 512, "status": "oom",
         "images_per_sec": None},
    ])
    # status가 ok인 결측만 나온다. oom 행은 missing_cells가 따로 표시한다.
    assert column_gaps(df, "images_per_sec") == [(512, "vim_s")]


def test_column_gaps_is_empty_when_the_column_is_complete():
    import pandas as pd

    from figures.e1_plot import column_gaps

    df = pd.DataFrame([
        {"model": "vim_s", "resolution": 512, "status": "ok",
         "flops_total": 3.0e10},
    ])
    assert column_gaps(df, "flops_total") == []


def test_column_gaps_matches_the_committed_sweep():
    """커밋된 CSV에서 실제로 세 셀이 잡히는지 본다."""
    import pandas as pd

    from figures.e1_plot import column_gaps

    df = pd.read_csv("results/e1/sweep.csv")
    assert set(column_gaps(df, "images_per_sec")) == {
        (512, "cmt_s"), (1024, "cmt_s"), (512, "vim_s")}
