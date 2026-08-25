"""E4의 12 run을 돌린다.

seed 1의 네 칸을 먼저 전부 돈다. 2x2 표가 한 번 완성되어 조기에 신호를 보기
위해서다 — 칸 우선으로 돌면 43시간짜리 b_vim_ti 세 개를 끝낼 때까지 표가 비어 있다.

run마다 CSV를 다시 쓴다. E1에서 배운 것이다 — 마지막에 한 번만 쓰면 중간 실패가
앞선 결과를 전부 지운다.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from bench.env import snapshot
from bench.memory import is_oom
from bench.train import TrainConfig, train
from data.tiny_imagenet import build_loaders, build_mixup, ensure_tiny_imagenet
from experiments.e4_widths import (
    count_params,
    load_cell_config,
    load_common_config,
)
from models.registry import E4_CELLS, build_e4_model

RUN_COLUMNS = (
    "cell", "model", "seed", "epochs_done", "top1", "top5",
    "params", "hours", "status", "error",
)


def run_order(seeds: list[int]) -> list[tuple[str, int]]:
    return [(cell, seed) for seed in seeds for cell in E4_CELLS]


def completed_runs(csv_path: Path) -> set[tuple[str, int]]:
    path = Path(csv_path)
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            (row["cell"], int(row["seed"]))
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _status_for(exc: Exception) -> str:
    """OOM은 error와 다른 지시다 — 배치를 줄이라는 뜻이지 코드를 고치라는 뜻이 아니다.

    bench.memory.is_oom이 CUDA OOM의 두 형태(torch.cuda.OutOfMemoryError, 그리고
    cuDNN workspace 할당 실패 등에서 새는 평범한 RuntimeError)를 모두 인정한다.
    """
    return "oom" if is_oom(exc) else "error"


def _existing_rows(csv_path: Path) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(out_dir: str = "results/e4") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")

    common = load_common_config()
    cfg = TrainConfig(
        epochs=common["epochs"], lr=common["lr"], min_lr=common["min_lr"],
        warmup_epochs=common["warmup_epochs"], weight_decay=common["weight_decay"],
        label_smoothing=common["label_smoothing"], drop_path=common["drop_path"],
    )
    root = ensure_tiny_imagenet()
    runs_csv = out / "runs.csv"
    rows = _existing_rows(runs_csv)
    done = completed_runs(runs_csv)

    for cell, seed in run_order(common["seeds"]):
        if (cell, seed) in done:
            continue
        _seed_everything(seed)
        cell_cfg = load_cell_config(cell)
        row = {c: "" for c in RUN_COLUMNS}
        row.update({"cell": cell, "model": cell, "seed": seed})
        try:
            model = build_e4_model(cell, cell_cfg,
                                   num_classes=common["num_classes"],
                                   img_size=common["img_size"],
                                   drop_path=common["drop_path"])
            row["params"] = count_params(model)
            train_loader, val_loader = build_loaders(
                root, common["batch_size"], common["workers"], common["img_size"]
            )
            result = train(
                model, train_loader, val_loader, cfg,
                ckpt_path=Path("checkpoints") / f"e4_{cell}_seed{seed}.pt",
                curve_path=out / "curves" / f"{cell}_seed{seed}.csv",
                device="cuda", mixup=build_mixup(common["num_classes"]),
            )
            row.update({
                "epochs_done": result["epochs_done"],
                "top1": round(result["top1"], 6),
                "top5": round(result["top5"], 6),
                "hours": round(result["hours"], 3),
                "status": "ok",
            })
        except Exception as exc:  # OOM(RuntimeError)도 데이터로 남긴다.
            # BaseException을 잡지 않는다 — 그러면 오퍼레이터가 99시간짜리 job
            # 도중에 Ctrl-C로 멈춘 것(KeyboardInterrupt)까지 이 run의 실패로
            # 기록하고 다음 run으로 넘어가 버린다. KeyboardInterrupt·SystemExit는
            # 여기를 지나쳐 그대로 올라가 job을 멈춘다.
            row.update({
                "status": _status_for(exc),
                "error": f"{type(exc).__name__}: {exc}",
            })

        rows = [r for r in rows if not (r["cell"] == cell and int(r["seed"]) == seed)]
        rows.append(row)
        write_rows(runs_csv, rows)


if __name__ == "__main__":
    main()
