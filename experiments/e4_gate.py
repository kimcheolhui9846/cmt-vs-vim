"""관문 1 — 네 칸이 200장을 외우는지 확인한다.

상위 설계는 E4의 관문을 "ImageNet 체크포인트 재평가"로 정했으나 이 저장소는 그것을
쓰지 않는다. ImageNet val 6.7GB가 없고, 더 중요하게는 그 관문이 평가 경로만 검증하고
학습 루프는 건드리지 않기 때문이다. E4에서 조용히 틀릴 곳은 학습 쪽이다.

augmentation과 mixup을 끄고 돈다. 켜 두면 200장을 외우는 것 자체가 불가능해져
관문이 "학습이 되는가"가 아니라 "정규화가 센가"를 재게 된다.
"""
import sys

import torch
import torch.nn as nn

from bench.train import TrainConfig, lr_at
from data.tiny_imagenet import (
    NUM_CLASSES,
    build_eval_transform,
    ensure_tiny_imagenet,
)
from experiments.e4_widths import load_cell_config
from models.registry import E4_CELLS, build_e4_model

GATE_THRESHOLD = 0.95


def gate_verdict(scores: dict[str, float]) -> list[str]:
    """임계값 미달인 칸 이름을 순서대로 돌려준다. 빈 목록이면 통과다."""
    return [cell for cell, score in scores.items() if score < GATE_THRESHOLD]


def overfit_subset(cell: str, n: int = 200, steps: int = 300,
                   device: str = "cuda") -> float:
    """고정된 n장을 반복 학습하고 그 n장에 대한 학습 정확도를 돌려준다."""
    from torchvision.datasets import ImageFolder

    root = ensure_tiny_imagenet()
    dataset = ImageFolder(str(root / "train"), transform=build_eval_transform(64))
    indices = list(range(0, len(dataset), max(len(dataset) // n, 1)))[:n]
    xs = torch.stack([dataset[i][0] for i in indices]).to(device)
    ys = torch.tensor([dataset[i][1] for i in indices]).to(device)

    # drop path도 끈다 — augmentation·mixup을 끄는 것과 같은 이유다. 정규화를
    # 켜 둔 채로는 200장을 외우는 것 자체가 불가능해 관문이 "학습이 되는가"가
    # 아니라 "정규화가 센가"를 재게 된다.
    model = build_e4_model(cell, load_cell_config(cell), num_classes=NUM_CLASSES,
                           img_size=64, drop_path=0.0).to(device).train()
    cfg = TrainConfig(epochs=steps, lr=1e-3, min_lr=1e-5, warmup_epochs=5,
                      weight_decay=0.0, label_smoothing=0.0, drop_path=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = criterion(model(xs), ys)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    with torch.no_grad():
        predicted = model(xs).argmax(dim=1)
    return (predicted == ys).float().mean().item()


def main() -> None:
    scores = {cell: overfit_subset(cell) for cell in E4_CELLS}
    for cell, score in scores.items():
        print(f"{cell:<12} train acc {score:.3f}")

    failing = gate_verdict(scores)
    if failing:
        print(f"관문 1 실패: {', '.join(failing)} — 본 학습을 시작하지 말 것")
        sys.exit(1)
    print("관문 1 통과")


if __name__ == "__main__":
    main()
