import pandas as pd

from experiments.e1_resolution_sweep import RESOLUTIONS, run_sweep

EXPECTED_COLUMNS = [
    "model",
    "resolution",
    "params",
    "flops_traced",
    "flops_uncounted_ops",
    "latency_ms",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "max_batch",
    "images_per_sec",
    "status",
]


def test_sweeps_the_five_resolutions_from_the_spec():
    assert RESOLUTIONS == (224, 384, 512, 768, 1024)


def test_writes_one_row_per_model_resolution_pair(tmp_path):
    df = run_sweep(
        model_names=("deit_s",), resolutions=(224, 384), out_dir=tmp_path
    )
    assert len(df) == 2
    assert list(df.columns) == EXPECTED_COLUMNS


def test_persists_a_csv_and_an_env_snapshot(tmp_path):
    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)
    assert (tmp_path / "sweep.csv").exists()
    assert (tmp_path / "env.json").exists()


def test_oom_rows_are_kept_with_status_not_dropped(tmp_path, monkeypatch):
    """OOM은 결과다. 행이 사라지면 메모리 주장의 증거가 사라진다."""
    import experiments.e1_resolution_sweep as e1
    from bench.memory import MemoryResult

    monkeypatch.setattr(
        e1, "measure_peak_memory", lambda fn: MemoryResult(None, None, "oom")
    )
    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "oom"
