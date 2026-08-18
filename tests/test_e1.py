import pandas as pd

import experiments.e1_resolution_sweep as e1
from bench.memory import MemoryResult
from experiments.e1_resolution_sweep import COLUMNS, RESOLUTIONS, run_sweep

EXPECTED_COLUMNS = [
    "model",
    "resolution",
    "params",
    "flops_traced",
    "flops_analytic",
    "flops_total",
    "flops_uncounted_ops",
    "flops_unexpected_ops",
    "latency_ms",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "max_batch",
    "images_per_sec",
    "status",
    "error",
]


def _stub_row(model_name: str, resolution: int) -> dict:
    row = {column: None for column in COLUMNS}
    row.update(
        model=model_name,
        resolution=resolution,
        flops_uncounted_ops="",
        flops_unexpected_ops="",
        status="ok",
    )
    return row


def test_sweeps_the_five_resolutions_from_the_spec():
    assert RESOLUTIONS == (224, 384, 512, 768, 1024)


def test_writes_one_row_per_model_resolution_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "_measure_one", _stub_row)

    df = run_sweep(
        model_names=("deit_s", "cmt_s"), resolutions=(224, 384), out_dir=tmp_path
    )

    assert len(df) == 4
    assert list(df.columns) == EXPECTED_COLUMNS


def test_persists_a_csv_and_an_env_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "_measure_one", _stub_row)

    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert (tmp_path / "sweep.csv").exists()
    assert (tmp_path / "env.json").exists()


def test_a_failing_cell_does_not_lose_the_cells_already_measured(tmp_path, monkeypatch):
    """15셀 한 시간짜리 실행이 중간에 죽어도 앞의 결과는 남아야 한다.
    이 방어가 없으면 GPU 한 시간이 예외 하나로 사라진다."""
    calls = {"n": 0}

    def flaky(model_name, resolution):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("이 셀만 터진다")
        return _stub_row(model_name, resolution)

    monkeypatch.setattr(e1, "_measure_one", flaky)

    df = run_sweep(
        model_names=("deit_s",), resolutions=(224, 384, 512), out_dir=tmp_path
    )

    assert len(df) == 3
    assert list(df["status"]) == ["ok", "error", "ok"]
    assert "이 셀만 터진다" in df.iloc[1]["error"]

    on_disk = pd.read_csv(tmp_path / "sweep.csv")
    assert len(on_disk) == 3


def test_csv_exists_before_the_sweep_finishes(tmp_path, monkeypatch):
    """마지막에 한 번만 쓰면 중간 실패 시 파일 자체가 없다."""
    seen = []

    def record_then_stub(model_name, resolution):
        seen.append((tmp_path / "sweep.csv").exists())
        return _stub_row(model_name, resolution)

    monkeypatch.setattr(e1, "_measure_one", record_then_stub)
    run_sweep(model_names=("deit_s",), resolutions=(224, 384), out_dir=tmp_path)

    assert seen[1] is True, "두 번째 셀을 재기 전에 첫 셀이 이미 디스크에 있어야 한다"


def test_oom_rows_are_kept_with_status_not_dropped(tmp_path, monkeypatch):
    """OOM은 결과다. 행이 사라지면 메모리 주장의 증거가 사라진다."""
    monkeypatch.setattr(
        e1, "measure_peak_memory", lambda fn: MemoryResult(None, None, "oom")
    )

    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["status"] == "oom"


def test_flops_survive_an_oom_row(tmp_path, monkeypatch):
    """FLOPs는 CPU 트레이스라 OOM과 무관하다. 메모리에 안 들어가는 해상도에서도
    연산량은 논문 표에 들어갈 값이므로 함께 잃으면 안 된다."""
    monkeypatch.setattr(
        e1, "measure_peak_memory", lambda fn: MemoryResult(None, None, "oom")
    )

    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert df.iloc[0]["status"] == "oom"
    assert df.iloc[0]["flops_total"] > 0
    assert df.iloc[0]["params"] > 0


def test_env_json_records_the_memory_budget(tmp_path, monkeypatch):
    """어떤 예산에서 잰 OOM인지 기록되지 않으면 결과를 해석할 수 없다.

    WSL2는 VRAM이 모자라면 시스템 메모리로 넘겨서 OOM을 내지 않는다. 그래서 이
    sweep의 OOM 경계는 "8GB"가 아니라 "할당자에 건 예산"의 경계다. 그 값이 결과
    파일에 없으면 재현도, 해석도 불가능하다.
    """
    import json

    monkeypatch.setattr(e1, "_measure_one", _stub_row)
    monkeypatch.setattr(e1, "apply_memory_budget", lambda: 1234)

    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    env = json.loads((tmp_path / "env.json").read_text())
    assert env["gpu_memory_budget_bytes"] == 1234
