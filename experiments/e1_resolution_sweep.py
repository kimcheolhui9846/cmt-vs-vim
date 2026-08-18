"""E1 — 해상도 sweep 실측.

논문 표 1은 손으로 계산한 추정값이었다. 이 스크립트가 그 자리를 대체할 실측
CSV를 만든다. 그림과 표는 이 CSV만 읽는다.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from bench.env import snapshot
from bench.flops import count_flops
from bench.latency import measure_latency
from bench.memory import measure_peak_memory
from bench.throughput import measure_throughput
from models.registry import MODEL_NAMES, build_model

RESOLUTIONS = (224, 384, 512, 768, 1024)

COLUMNS = [
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


def _op_handlers(model_name: str) -> dict:
    if model_name == "vim_s":
        from models.vim import VIM_OP_HANDLERS

        return VIM_OP_HANDLERS
    return {}


def _measure_one(model_name: str, resolution: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, pretrained=False, img_size=resolution)
    shape = (3, resolution, resolution)

    row = {
        "model": model_name,
        "resolution": resolution,
        "params": sum(p.numel() for p in model.parameters()),
        "flops_traced": None,
        "flops_uncounted_ops": "",
        "latency_ms": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "max_batch": None,
        "images_per_sec": None,
        "status": "ok",
    }

    # latency 측정을 peak memory 측정 안에서 한 번만 돌린다. 밖에서 또 부르면
    # 1024²에서 150 iteration을 두 번 돌게 된다.
    captured: dict[str, float] = {}

    def _timed_run() -> None:
        captured["latency_ms"] = measure_latency(model, shape, device=device)

    memory = measure_peak_memory(_timed_run)
    row["peak_allocated_bytes"] = memory.peak_allocated_bytes
    row["peak_reserved_bytes"] = memory.peak_reserved_bytes
    row["status"] = memory.status
    row["latency_ms"] = captured.get("latency_ms")

    if memory.status == "oom":
        return row

    flops = count_flops(model, shape, op_handlers=_op_handlers(model_name))
    row["flops_traced"] = flops.traced
    row["flops_uncounted_ops"] = ";".join(flops.uncounted_ops)

    throughput = measure_throughput(model, shape, device=device)
    row["max_batch"] = throughput.batch
    row["images_per_sec"] = throughput.images_per_sec
    return row


def run_sweep(
    model_names: tuple[str, ...] = MODEL_NAMES,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    out_dir: Path | str = "results/e1",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _measure_one(name, res) for name in model_names for res in resolutions
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)

    df.to_csv(out_dir / "sweep.csv", index=False)
    (out_dir / "env.json").write_text(json.dumps(snapshot(), indent=2))
    return df


if __name__ == "__main__":
    print(run_sweep().to_string(index=False))
