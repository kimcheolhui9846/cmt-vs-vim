"""학습 루프의 순수 부분과 재개를 검증한다.

lr 스케줄을 순수 함수로 뺀 이유가 여기 있다 — 재개가 스케줄을 이어받는지를
optimizer 내부를 뒤지지 않고 직접 확인할 수 있다.
"""
import torch
import torch.nn as nn

from bench.train import TrainConfig, evaluate, load_checkpoint, lr_at, save_checkpoint

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
    """
    assert lr_at(137, CFG) == lr_at(137, CFG)


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


def test_evaluate_returns_top1_and_top5_as_fractions():
    class Constant(nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 10)
            logits[:, 3] = 1.0
            return logits

    xs = torch.randn(8, 3, 4, 4)
    ys = torch.full((8,), 3)
    loader = [(xs, ys)]
    top1, top5 = evaluate(Constant(), loader, device="cpu")
    assert top1 == 1.0
    assert top5 == 1.0
