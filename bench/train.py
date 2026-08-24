"""E4의 학습 루프. 네 칸이 같은 코드를 지난다.

lr 스케줄을 상태 있는 스케줄러가 아니라 순수 함수로 두는 이유는 재개다. 최장 run이
약 15시간이라 중간에 끊긴다는 전제로 짜야 하는데, 상태 있는 스케줄러는 재시작할 때
warmup을 다시 돌아 조용히 다른 레시피가 된다.
"""
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CURVE_COLUMNS = ("epoch", "train_loss", "val_top1", "val_top5", "lr")


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    min_lr: float
    warmup_epochs: int
    weight_decay: float
    label_smoothing: float
    drop_path: float


def lr_at(epoch: int, cfg: TrainConfig) -> float:
    """warmup 선형 상승 후 cosine 감쇠. epoch만 보고 답한다."""
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / (cfg.warmup_epochs + 1)
    span = max(cfg.epochs - cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct1 = correct5 = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        k = min(5, logits.shape[1])
        top = logits.topk(k, dim=1).indices
        correct1 += (top[:, 0] == y).sum().item()
        correct5 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    return correct1 / total, correct5 / total


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int) -> None:
    tmp = Path(path).with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        # 세 난수 스트림을 전부 저장한다. drop_path·dropout은 CUDA RNG에서,
        # timm.data.Mixup은 NumPy RNG에서 뽑는다 — 하나라도 빠지면 재개한 run이
        # 끊기지 않은 run과 다른 난수 궤적을 타고, 그 사실이 어디에도 남지 않는다.
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }, tmp)
    tmp.replace(path)  # 쓰다 죽어도 이전 체크포인트가 남는다


def load_checkpoint(path: Path, model, optimizer, scaler) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    return state["epoch"] + 1


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    lr: float, mixup=None) -> float:
    model.train()
    for group in optimizer.param_groups:
        group["lr"] = lr

    total_loss = batches = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if mixup is not None:
            x, y = mixup(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


def _append_curve(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _drop_curve_rows_from(path: Path, start: int) -> None:
    """재개 직전에 epoch >= start인 곡선 행을 지운다.

    루프 안에서 `_append_curve`가 `save_checkpoint`보다 먼저 실행되므로, 그 사이에
    죽으면 체크포인트에는 반영되지 않은 epoch의 행이 CSV에 먼저 남는다. 재개하면
    그 epoch을 다시 돌면서 같은 epoch에 대해 또 한 행을 적어 중복이 생긴다. 재개
    시작(`start`) 이전에 그런 미확정 행을 지워, 다시 돌 때 epoch당 정확히 한 행만
    남게 한다.
    """
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) < start]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def train(model, train_loader, val_loader, cfg: TrainConfig, ckpt_path: Path,
          curve_path: Path, device: str, mixup=None) -> dict:
    """한 run을 끝까지 돌린다. 체크포인트가 있으면 이어서 돈다."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    from timm.loss import SoftTargetCrossEntropy

    # mixup이 켜지면 라벨이 soft target이 되므로 손실도 함께 바뀌어야 한다.
    # CrossEntropyLoss에 soft target을 먹이면 조용히 다른 것을 최적화한다.
    criterion = (
        SoftTargetCrossEntropy() if mixup is not None
        else nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    )

    ckpt_path = Path(ckpt_path)
    curve_path = Path(curve_path)
    start = load_checkpoint(ckpt_path, model, optimizer, scaler) if ckpt_path.exists() else 0
    # 체크포인트가 아예 없는 최초 run도 미확정 곡선 행을 가질 수 있다 — epoch 0을
    # 다 돌아 곡선 행은 썼지만 save_checkpoint가 한 번도 실행되기 전에 죽으면,
    # 재시작 시점에는 체크포인트 파일 자체가 없다(start=0). 이 정리를
    # `if ckpt_path.exists():` 안에 가둬두면 그 경우를 놓쳐 epoch 0이 중복된다.
    # start=0이면 이 호출이 곡선 파일을 통째로 비우므로, 최초 run은 항상 빈
    # 곡선에서 시작한다.
    _drop_curve_rows_from(curve_path, start)

    started = time.perf_counter()
    top1 = top5 = 0.0
    for epoch in range(start, cfg.epochs):
        lr = lr_at(epoch, cfg)
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                               device, lr, mixup)
        top1, top5 = evaluate(model, val_loader, device)
        _append_curve(curve_path, {
            "epoch": epoch, "train_loss": round(loss, 6),
            "val_top1": round(top1, 6), "val_top5": round(top5, 6),
            "lr": round(lr, 8),
        })
        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch)

    return {
        "epochs_done": cfg.epochs,
        "top1": top1,
        "top5": top5,
        "hours": (time.perf_counter() - started) / 3600,
    }
