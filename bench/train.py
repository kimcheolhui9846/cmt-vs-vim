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
        "torch_rng": torch.get_rng_state(),
    }, tmp)
    tmp.replace(path)  # 쓰다 죽어도 이전 체크포인트가 남는다


def load_checkpoint(path: Path, model, optimizer, scaler) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
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
    start = load_checkpoint(ckpt_path, model, optimizer, scaler) if ckpt_path.exists() else 0

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
