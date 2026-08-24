"""학습 루프의 순수 부분과 재개를 검증한다.

lr 스케줄을 순수 함수로 뺀 이유가 여기 있다 — 재개가 스케줄을 이어받는지를
optimizer 내부를 뒤지지 않고 직접 확인할 수 있다.
"""
import csv

import numpy as np
import torch
import torch.nn as nn

from bench.train import (
    TrainConfig,
    _append_curve,
    evaluate,
    load_checkpoint,
    lr_at,
    save_checkpoint,
    train,
)

CFG = TrainConfig(epochs=300, lr=2.5e-4, min_lr=1e-5, warmup_epochs=5,
                  weight_decay=0.05, label_smoothing=0.1, drop_path=0.1)


def test_warmup_starts_near_zero_and_reaches_base_lr():
    assert lr_at(0, CFG) < CFG.lr
    assert lr_at(CFG.warmup_epochs, CFG) == CFG.lr


def test_cosine_decays_to_min_lr_at_the_end():
    assert lr_at(CFG.epochs - 1, CFG) < CFG.lr
    assert lr_at(CFG.epochs - 1, CFG) >= CFG.min_lr


def test_lr_is_a_pure_function_of_epoch():
    """재개가 스케줄을 이어받는다는 것은 이 함수가 epoch만 보고 답한다는 뜻이다.

    상태를 들고 있는 스케줄러였다면 재시작이 warmup을 다시 돌아 다른 레시피가 된다.
    `lr_at(137, CFG) == lr_at(137, CFG)`처럼 같은 호출을 연달아 반복하는 것만으로는
    아무것도 판별하지 못한다 — epoch을 무시하고 외부 상태(예: 전역 스케줄러의
    호출 횟수)를 읽는 구현도 그 assert는 항상 통과시킨다. 그래서 여기서는 호출
    이력(먼저 다른 epoch들을 얼마나 물었는지)과 호출 순서를 바꿔도 같은 epoch은
    같은 값을 내야 한다는 것을 직접 확인한다.
    """
    fresh = lr_at(137, CFG)
    for epoch in range(137):
        lr_at(epoch, CFG)  # 재개 시 이전 epoch들을 다시 묻는 상황을 흉내낸다
    assert lr_at(137, CFG) == fresh

    epochs = [0, 5, 137, 299]
    forward = [lr_at(e, CFG) for e in epochs]
    backward = [lr_at(e, CFG) for e in reversed(epochs)]
    assert forward == list(reversed(backward))


def test_resume_returns_the_next_epoch(tmp_path):
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, scaler, epoch=41)
    assert load_checkpoint(path, model, optimizer, scaler) == 42


def test_resume_restores_optimizer_state(tmp_path):
    """optimizer 상태를 빼면 Adam의 모멘텀이 리셋되어 재개가 다른 궤적을 탄다."""
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    before = optimizer.state_dict()["state"][0]["exp_avg"].clone()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, scaler, epoch=0)

    fresh_model = nn.Linear(4, 3)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    load_checkpoint(path, fresh_model, fresh_opt, scaler)
    assert torch.equal(fresh_opt.state_dict()["state"][0]["exp_avg"], before)


def test_resume_restores_all_rng_streams(tmp_path):
    """CPU torch, CUDA, NumPy 세 스트림을 전부 저장·복원해야 한다.

    GPU 드롭아웃·drop_path는 CUDA RNG에서, timm.data.Mixup은 np.random에서만
    뽑는다. 셋 중 하나라도 체크포인트에서 빠지면 재개한 run이 끊기지 않은 run과
    다른 난수 궤적을 타고, 그 사실이 결과 파일 어디에도 남지 않는다. CUDA 스트림은
    이 테스트 환경에서 실제 값 비교로 검증할 수 없는 경우(CUDA 없음)에도 최소한
    체크포인트에 그 키가 존재하는지는 확인한다.
    """
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    path = tmp_path / "ckpt.pt"

    torch.manual_seed(0)
    np.random.seed(0)
    save_checkpoint(path, model, optimizer, scaler, epoch=0)

    state = torch.load(path, map_location="cpu")
    assert "cuda_rng" in state
    assert "numpy_rng" in state
    assert "torch_rng" in state

    # 저장 시점 바로 다음에 뽑혔어야 할 값들 (끊기지 않은 run이 냈을 값)
    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)

    # 재시작 전에 다른 코드가 두 스트림을 어지럽혔다고 가정한다
    torch.rand(100)
    np.random.rand(100)

    fresh_model = nn.Linear(4, 3)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    load_checkpoint(path, fresh_model, fresh_opt, scaler)

    assert torch.equal(torch.rand(4), expected_torch)
    assert (np.random.rand(4) == expected_numpy).all()


def test_fresh_run_does_not_duplicate_curve_rows(tmp_path):
    """체크포인트를 한 번도 못 쓰고 죽은 뒤 재시작하는, 최초 run에서만 나는
    경우를 재현한다 — 논문이 인용하는 결과 파일이 오염되는 경로다.

    epoch 0을 다 돌아 곡선 행은 썼지만 `save_checkpoint`가 한 번도 실행되기
    전에 죽으면, 재시작 시점에는 체크포인트 파일 자체가 없다(즉 곡선에는
    epoch 0 행이 있는데 체크포인트는 없음 — 위 테스트의 "체크포인트가 한
    epoch 뒤처진" 상황과 다르다). 재개 정리를 "체크포인트가 있을 때만" 돌리면
    이 경우를 놓쳐 epoch 0이 중복된다.
    """
    ckpt_path = tmp_path / "ckpt.pt"
    curve_path = tmp_path / "curve.csv"

    _append_curve(curve_path, {
        "epoch": 0, "train_loss": 9.9, "val_top1": 0.0, "val_top5": 0.0, "lr": 1e-4,
    })
    assert not ckpt_path.exists()  # save_checkpoint가 한 번도 실행되지 못한 상황

    model = nn.Linear(4, 3)
    xs = torch.randn(4, 4, dtype=torch.float32)
    ys = torch.randint(0, 3, (4,))
    loader = [(xs, ys)]
    cfg = TrainConfig(epochs=2, lr=1e-3, min_lr=1e-4, warmup_epochs=1,
                      weight_decay=0.0, label_smoothing=0.0, drop_path=0.0)

    train(model, loader, loader, cfg, ckpt_path, curve_path, device="cuda")

    with curve_path.open(newline="", encoding="utf-8") as handle:
        epochs = [int(row["epoch"]) for row in csv.DictReader(handle)]

    assert epochs == [0, 1]


def test_evaluate_returns_top1_and_top5_as_fractions():
    class RankedLogits(nn.Module):
        """정답 클래스(3)의 순위를 샘플마다 다르게 둬서 top-1과 top-5 판정이 진짜로
        다른 로직을 쓰는지 구분한다. 정답이 매번 top-1이자 top-5이기만 하면, top-5를
        top-1과 같은 로직(1등만 확인)으로 계산하는 회귀도 이 테스트를 속인다."""
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros(batch, 10)
            for i in range(batch):
                if i % 2 == 0:
                    order = [3, 0, 1, 2, 4, 5, 6, 7, 8, 9]  # 정답이 1등: top-1·top-5 모두 정답
                else:
                    order = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]  # 정답이 5등: top-5만 정답
                for rank, cls in enumerate(order):
                    logits[i, cls] = 9 - rank
            return logits

    xs = torch.randn(8, 3, 4, 4)
    ys = torch.full((8,), 3)
    loader = [(xs, ys)]
    top1, top5 = evaluate(RankedLogits(), loader, device="cpu")
    assert top1 == 0.5
    assert top5 == 1.0
    assert top1 != top5
