import pandas as pd

import pytest

import experiments.e1_resolution_sweep as e1
from bench.latency import LatencyResult
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
    "latency_min_ms",
    "latency_max_ms",
    "latency_repeats_ms",
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


def test_throughput_oom_keeps_what_was_already_measured(tmp_path, monkeypatch):
    """처리량 단계의 OOM이 셀 전체를 지우면 안 된다.

    배치 탐색 안의 OOM은 `_fits`가 잡지만, 탐색이 끝난 뒤 그 배치로 실제 측정할 때
    단편화로 나는 OOM은 아무도 잡지 않았다. 그래서 첫 실행에서 cmt_s@512와
    vim_s@512가 status="error"로 떨어지면서, 이미 잰 FLOPs·latency·메모리까지
    전부 NaN이 됐다. OOM은 기록할 결과지 셀을 버릴 이유가 아니다.
    """
    def _oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.19 GiB")

    monkeypatch.setattr(e1, "measure_throughput", _oom)

    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)
    row = df.iloc[0]

    assert row["status"] != "error", "OOM이 셀 전체를 error로 만들었다"
    assert row["flops_total"] > 0, "이미 잰 FLOPs가 사라졌다"
    assert row["params"] > 0
    assert pd.isna(row["max_batch"]), "재지 못한 처리량이 값처럼 남았다"
    assert "out of memory" in str(row["error"])


# --- latency 반복 측정 --------------------------------------------------------


def test_records_every_latency_repeat_not_just_the_median(tmp_path, monkeypatch):
    """배치 1 latency는 다시 재면 값이 달라진다 — vim_s@224가 30.00 ms와 16.67 ms로
    갈렸다. 중앙값만 남기면 그 사실이 CSV에서 사라져서, 표를 읽는 사람이 재현되지
    않는 숫자를 재현되는 숫자로 읽는다."""
    monkeypatch.setattr(e1, "apply_memory_budget", lambda: 1234)
    monkeypatch.setattr(
        e1, "measure_latency", lambda *a, **k: LatencyResult((16.67, 30.0, 18.0))
    )

    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)
    row = df.iloc[0]

    assert row["latency_ms"] == pytest.approx(18.0)
    assert row["latency_min_ms"] == pytest.approx(16.67)
    assert row["latency_max_ms"] == pytest.approx(30.0)

    recorded = [float(v) for v in str(row["latency_repeats_ms"]).split(";")]
    assert recorded == pytest.approx([16.67, 30.0, 18.0])


def test_the_sweep_actually_repeats(tmp_path, monkeypatch):
    """열만 만들어 두고 1회만 재면 편차가 항상 0으로 나온다 — 측정하지 않은 것을
    '편차 없음'으로 보고하는 셈이다."""
    seen = {}

    def spy(*args, **kwargs):
        seen["repeats"] = kwargs.get("repeats")
        return LatencyResult((1.0,))

    monkeypatch.setattr(e1, "apply_memory_budget", lambda: 1234)
    monkeypatch.setattr(e1, "measure_latency", spy)
    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert e1.LATENCY_REPEATS > 1
    assert seen["repeats"] == e1.LATENCY_REPEATS


def test_warns_when_the_repeats_disagree(tmp_path, monkeypatch, capsys):
    """편차를 열에만 적어 두면 아무도 안 본다. 실행이 끝날 때 어느 셀이 재현되지
    않았는지 말해야, 그 숫자가 표에 단일 값으로 실리는 걸 막을 수 있다."""
    monkeypatch.setattr(e1, "apply_memory_budget", lambda: 1234)
    monkeypatch.setattr(
        e1, "measure_latency", lambda *a, **k: LatencyResult((16.67, 30.0, 18.0))
    )

    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)
    out = capsys.readouterr().out

    assert "deit_s@224" in out
    assert "1.80" in out


def test_reproducible_cells_are_not_flagged(tmp_path, monkeypatch, capsys):
    """전부 경고하면 경고가 의미를 잃는다."""
    monkeypatch.setattr(e1, "apply_memory_budget", lambda: 1234)
    monkeypatch.setattr(
        e1, "measure_latency", lambda *a, **k: LatencyResult((10.0, 10.1, 10.05))
    )

    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert "deit_s@224" not in capsys.readouterr().out
