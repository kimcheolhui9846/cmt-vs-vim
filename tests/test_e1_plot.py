import pandas as pd
import pytest

from figures.e1_plot import (
    ANALYTIC_SHARE_COLUMN,
    PANELS,
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
    "peak_allocated_bytes": 1_000_000,
    "peak_reserved_bytes": 2_000_000,
    "max_batch": 32,
    "images_per_sec": 100.0,
    "status": "ok",
    "error": None,
}

UNMEASURED = {
    "latency_ms": None,
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
    panels = {column: yscale for column, _, _, _, yscale in PANELS}
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
