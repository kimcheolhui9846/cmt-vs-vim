import pandas as pd
import pytest

from figures.e1_plot import plot_sweep


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_writes_a_png_from_the_csv(tmp_path):
    csv = _write_csv(
        tmp_path / "sweep.csv",
        [
            {
                "model": "deit_s",
                "resolution": 224,
                "params": 22_000_000,
                "flops_traced": 4.6e9,
                "flops_uncounted_ops": "",
                "latency_ms": 5.0,
                "peak_allocated_bytes": 1_000_000,
                "peak_reserved_bytes": 2_000_000,
                "status": "ok",
            }
        ],
    )
    out = plot_sweep(csv, tmp_path / "e1.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_refuses_an_empty_csv(tmp_path):
    """빈 입력에서 조용히 빈 그림을 내면 논문에 빈 그림이 실린다."""
    csv = _write_csv(tmp_path / "sweep.csv", [])
    with pytest.raises(ValueError, match="비어"):
        plot_sweep(csv, tmp_path / "e1.png")


def test_oom_rows_do_not_become_zero_points(tmp_path):
    """OOM을 0으로 그리면 '메모리를 안 쓴다'로 읽힌다."""
    csv = _write_csv(
        tmp_path / "sweep.csv",
        [
            {
                "model": "deit_s",
                "resolution": 1024,
                "params": 22_000_000,
                "flops_traced": None,
                "flops_uncounted_ops": "",
                "latency_ms": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "status": "oom",
            }
        ],
    )
    out = plot_sweep(csv, tmp_path / "e1.png")
    assert out.exists()
