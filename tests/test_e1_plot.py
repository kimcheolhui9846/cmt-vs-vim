import pandas as pd
import pytest

from figures.e1_plot import missing_cells, plot_sweep, plotted_series

BASE_ROW = {
    "model": "deit_s",
    "resolution": 224,
    "params": 22_000_000,
    "flops_traced": 4.6e9,
    "flops_uncounted_ops": "",
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
    df = _frame([_row(resolution=1024, status="oom", flops_traced=90e9, **UNMEASURED)])

    assert plotted_series(df, "flops_traced") == {}
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
