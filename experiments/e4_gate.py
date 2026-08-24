"""관문 1 — 네 칸이 200장을 외우는지 확인한다.

상위 설계는 E4의 관문을 "ImageNet 체크포인트 재평가"로 정했으나 이 저장소는 그것을
쓰지 않는다. ImageNet val 6.7GB가 없고, 더 중요하게는 그 관문이 평가 경로만 검증하고
학습 루프는 건드리지 않기 때문이다.

**이 관문이 실제로 잡는 것**: lr 폭주(발산해서 200장도 못 외움), 얼어붙었거나
배선이 빠진 파라미터(gradient가 흐르지 않아 못 외움), D칸(Hierarchical Vim)의
배선 오류, 그리고 이 함수 내부에서 자기모순인 라벨링(`xs`/`ys`를 만드는 방식
자체가 어긋나면 그 결과로도 못 외움) — 전부 200장/300 step 안에 못 외우는
것으로 드러난다. `overfit_subset`은 99시간짜리 본 학습이 실제로 지날
`bench.train.train_one_epoch`를 그대로 태우므로, 그 함수 안의 로더 순회·손실
평균·mixup 분기에 있는 버그도 이 경로로 실행된다.

**이 관문이 잡지 못하는 것**: (1) 검증(val) 분할의 라벨 매핑 어긋남 — 이 함수는
`train` 분할에서 `ImageFolder`로 이미지와 라벨을 같은 호출로 얻으므로, 그 매핑이
의미상 틀려도 내부적으로는 일관돼 200장을 그대로 외운다. val 라벨 매핑은
`tests/test_tiny_imagenet.py`의 몫이다. (2) augmentation이 라벨과 어긋나는 경우 —
augmentation과 mixup을 의도적으로 꺼 두므로(아래) 그 경로 자체가 이 관문에서
전혀 실행되지 않는다.

augmentation과 mixup을 끄고 돈다. 켜 두면 200장을 외우는 것 자체가 불가능해져
관문이 "학습이 되는가"가 아니라 "정규화가 센가"를 재게 된다.
"""
import sys

import torch
import torch.nn as nn

from bench.train import TrainConfig, lr_at, train_one_epoch
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


def select_subset_indices(dataset_len: int, n: int) -> list[int]:
    """데이터셋에서 균등 stride로 n개 인덱스를 고른다.

    `ImageFolder`는 클래스별로 인덱스가 연속으로 정렬돼 있으므로, stride가
    클래스당 이미지 수와 맞아떨어지면 이 선택이 클래스마다 한 장씩 건너뛰며
    뽑는다. 현재 Tiny-ImageNet train 분할(클래스당 500장)에서 n=200이면 그
    우연이 성립해 200개 클래스를 전부 건드린다 — 하지만 이 함수 자체는 그
    우연에 기대지 않고 dataset_len과 n만으로 stride를 계산한다. n이나
    데이터셋 크기가 바뀌면 이 우연이 깨져 클래스 몇 개만 반복해서 뽑을 수
    있으므로, `tests/test_e4_gate.py`가 이 함수만 따로 단위 테스트한다.
    """
    stride = max(dataset_len // n, 1)
    return list(range(0, dataset_len, stride))[:n]


def overfit_subset(cell: str, n: int = 200, steps: int = 300,
                   device: str = "cuda") -> float:
    """고정된 n장을 반복 학습하고 그 n장에 대한 학습 정확도를 돌려준다."""
    from torchvision.datasets import ImageFolder

    root = ensure_tiny_imagenet()
    dataset = ImageFolder(str(root / "train"), transform=build_eval_transform(64))
    indices = select_subset_indices(len(dataset), n)
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

    # 99시간짜리 본 학습이 실제로 지날 함수(train_one_epoch)를 그대로 태운다.
    # 고정 배치 하나를 원소 하나짜리 "로더" [(xs, ys)]로 감싸면, 이 함수를
    # 부를 때마다 그 배치를 정확히 한 번 학습한다 — 로더 순회·손실 평균·mixup
    # 분기가 전부 본 학습과 같은 코드 경로를 지난다. mixup은 본 학습에서도
    # 끌 수 있는 옵션이므로 여기서는 None을 그대로 넘긴다.
    for step in range(steps):
        train_one_epoch(model, [(xs, ys)], optimizer, scaler, criterion, device,
                        lr_at(step, cfg), mixup=None)

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
