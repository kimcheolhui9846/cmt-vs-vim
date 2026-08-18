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
from bench.memory import is_oom, measure_peak_memory
from bench.throughput import measure_throughput
from models.registry import MODEL_NAMES, build_model, traceable

RESOLUTIONS = (224, 384, 512, 768, 1024)

COLUMNS = [
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


def _op_handlers(model_name: str) -> dict:
    if model_name == "vim_s":
        from models.vim import VIM_OP_HANDLERS

        return VIM_OP_HANDLERS
    return {}


def _flops_device(model_name: str, device: str) -> str:
    """Vim만 CUDA에서 센다.

    selective scan과 causal conv1d가 CUDA 전용 커널이라 CPU에서는 트레이스 자체가
    불가능하다. 나머지 둘은 CPU에서 세는 편이 낫다 — 그래프가 같아 값이 동일하고,
    메모리에 들어가지 않는 셀에서도 연산량이 남는다.
    """
    return device if model_name == "vim_s" else "cpu"


def _blank_row(model_name: str, resolution: int) -> dict:
    row = {column: None for column in COLUMNS}
    row.update(
        model=model_name,
        resolution=resolution,
        flops_uncounted_ops="",
        flops_unexpected_ops="",
    )
    return row


def _measure_one(model_name: str, resolution: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (3, resolution, resolution)

    row = _blank_row(model_name, resolution)
    row["status"] = "ok"

    model = build_model(model_name, pretrained=False, img_size=resolution)
    row["params"] = sum(p.numel() for p in model.parameters())

    # FLOPs를 먼저 잰다. DeiT·CMT는 CPU 트레이스라 OOM이 날 수 없고, 그래서
    # 메모리에 들어가지 않는 셀에서도 연산량은 남는다. Vim만 예외다 — 커널이 CUDA
    # 전용이라 이 모델은 FLOPs 측정도 OOM 날 수 있다. 그때는 연산량 칸만 비우고
    # 계속 간다. 셀 전체를 error로 날리면 그 해상도에서 OOM이 났다는 사실까지 잃는다.
    try:
        with traceable(model_name, model):
            flops = count_flops(
                model,
                shape,
                op_handlers=_op_handlers(model_name),
                device=_flops_device(model_name, device),
            )
    except RuntimeError as exc:
        if not is_oom(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        row["flops_traced"] = flops.traced
        row["flops_analytic"] = flops.analytic
        row["flops_total"] = flops.total
        row["flops_uncounted_ops"] = ";".join(flops.uncounted_ops)
        row["flops_unexpected_ops"] = ";".join(flops.unexpected_uncounted_ops)

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

    throughput = measure_throughput(model, shape, device=device)
    row["max_batch"] = throughput.batch
    row["images_per_sec"] = throughput.images_per_sec
    return row


def _error_row(model_name: str, resolution: int, exc: BaseException) -> dict:
    row = _blank_row(model_name, resolution)
    row["status"] = "error"
    row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_sweep(
    model_names: tuple[str, ...] = MODEL_NAMES,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    out_dir: Path | str = "results/e1",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    (out_dir / "env.json").write_text(json.dumps(snapshot(), indent=2))

    rows: list[dict] = []
    for name in model_names:
        for resolution in resolutions:
            try:
                rows.append(_measure_one(name, resolution))
            except Exception as exc:  # 한 셀의 실패로 전체를 잃지 않는다
                rows.append(_error_row(name, resolution, exc))
            # 셀마다 다시 쓴다. 한 시간짜리 실행이 도중에 죽어도 앞의 결과는 남는다.
            pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows, columns=COLUMNS)

    failed = df[df["status"] == "error"]
    if not failed.empty:
        cells = ", ".join(f"{r.model}@{r.resolution}" for r in failed.itertuples())
        print(f"경고: {len(failed)}개 셀 실패 — {cells}. 숫자를 쓰기 전에 확인할 것.")

    return df


if __name__ == "__main__":
    print(run_sweep().to_string(index=False))
