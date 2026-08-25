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


def param_groups(model, weight_decay: float) -> list[dict]:
    """weight decay를 rank >= 2인 파라미터에만 건다.

    DeiT 원 레시피는 bias와 norm 가중치를 weight decay에서 뺀다. 파라미터 그룹 없이
    `AdamW(model.parameters(), weight_decay=0.05)`로 두면 norm의 gamma·beta와 모든
    bias까지 0으로 끌려가, 코드가 문서에 적힌 레시피와 다른 것을 돌게 된다.

    네 칸에 같은 규칙으로 걸리므로 요인 대비에는 영향이 없다. rank로 가르는 이유는
    프레임워크를 가리지 않기 때문이다 — 이름 규칙(`.bias`, `norm.`)은 timm·
    VisionMamba·CMT가 서로 다르지만, "행렬이면 decay, 벡터/스칼라면 no-decay"는
    셋 모두에서 같은 집합을 고른다.

    남는 편차: `pos_embed`·`cls_token`은 rank 3이라 이 규칙에서 decay를 받는다.
    DeiT는 둘을 `no_weight_decay()`로 빼므로 여기가 원 레시피와 다르다. 이름 기반
    예외로 빼지 않는 이유는 그 두 파라미터가 A·B에만 있어 칸마다 다른 규칙이
    되기 때문이다. configs와 설계 문서에 편차로 적어 둔다.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int,
                    elapsed_seconds: float = 0.0) -> None:
    path = Path(path)
    # curve 파일과 달리 체크포인트는 부모를 만들지 않고 있었다. checkpoints/가
    # gitignore되어 있으므로, 새로 clone한 곳에서는 12 run이 전부 첫 epoch 끝에서
    # 죽는다. _append_curve와 같은 규칙으로 맞춘다.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        # runs.csv의 hours는 누적이어야 한다. 이 값을 체크포인트에 넣지 않으면
        # 재개한 run이 마지막 호출분만 보고해, epoch 290에서 끊긴 run이 0.3h로
        # 적힌다 — 공표하는 열이므로 그대로 논문에 실린다.
        "elapsed_seconds": elapsed_seconds,
        # 세 난수 스트림을 전부 저장한다. drop_path·dropout은 CUDA RNG에서,
        # timm.data.Mixup은 NumPy RNG에서 뽑는다 — 하나라도 빠지면 재개한 run이
        # 끊기지 않은 run과 다른 난수 궤적을 타고, 그 사실이 어디에도 남지 않는다.
        #
        # 네 번째 스트림은 알면서 범위 밖에 둔다: DataLoader 워커의 base_seed다.
        # persistent_workers=True라 이 seed는 iterator를 만들 때 한 번 뽑히고 이후
        # epoch마다 다시 뽑지 않으므로, 재개하면 복원된 스트림의 다른 위치에서
        # 새로 뽑힌다. 그 결과 RandAugment와 random erasing이 중단 이후 다른 궤적을
        # 탄다. 셔플 순서는 복원되고 라벨 정합성은 영향을 받지 않는다 — 즉 재개한
        # run은 끊기지 않은 run과 비트 단위로 같지 않으며, 이 파일은 그 차이를
        # 없애지 않고 알려진 한계로 적어 둔다.
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }, tmp)
    tmp.replace(path)  # 쓰다 죽어도 이전 체크포인트가 남는다


def load_checkpoint(path: Path, model, optimizer, scaler) -> tuple[int, float]:
    """(다음 epoch, 지금까지 누적된 학습 초)를 돌려준다."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    # 이 키가 없는 옛 체크포인트도 읽을 수 있어야 한다 — 없으면 0부터 센다.
    return state["epoch"] + 1, float(state.get("elapsed_seconds", 0.0))


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
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
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
    start, carried_seconds = (
        load_checkpoint(ckpt_path, model, optimizer, scaler)
        if ckpt_path.exists() else (0, 0.0)
    )
    # 체크포인트가 아예 없는 최초 run도 미확정 곡선 행을 가질 수 있다 — epoch 0을
    # 다 돌아 곡선 행은 썼지만 save_checkpoint가 한 번도 실행되기 전에 죽으면,
    # 재시작 시점에는 체크포인트 파일 자체가 없다(start=0). 이 정리를
    # `if ckpt_path.exists():` 안에 가둬두면 그 경우를 놓쳐 epoch 0이 중복된다.
    # start=0이면 이 호출이 곡선 파일을 통째로 비우므로, 최초 run은 항상 빈
    # 곡선에서 시작한다.
    _drop_curve_rows_from(curve_path, start)

    started = time.perf_counter()
    top1 = top5 = 0.0
    epochs_done = start
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
        epochs_done = epoch + 1
        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch,
                        elapsed_seconds=carried_seconds
                        + (time.perf_counter() - started))

    if epochs_done == start:
        # 루프가 한 번도 돌지 않은 경우다. 마지막 epoch까지 끝낸 체크포인트가 있는데
        # runs.csv 행은 아직 없는 창이 실제로 존재한다 — save_checkpoint(epoch=299)가
        # write_rows보다 먼저 커밋되므로, 그 사이에 죽으면 이 run은 done에 없고 다시
        # 들어와 여기로 온다. top1의 초기값 0.0을 그대로 돌려주면 status="ok"인 0점이
        # CSV에 남고, 그 0이 상호작용 항을 통째로 무너뜨린다. 점수를 지어내지 말고
        # 여기서 한 번 재어서 돌려준다.
        top1, top5 = evaluate(model, val_loader, device)

    return {
        "epochs_done": epochs_done,
        "top1": top1,
        "top5": top5,
        "hours": (carried_seconds + time.perf_counter() - started) / 3600,
    }
