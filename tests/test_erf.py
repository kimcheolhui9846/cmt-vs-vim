import numpy as np
import torch
import torch.nn as nn

from bench.erf import accumulate_erf


class CenterPixel(nn.Module):
    """중심 픽셀만 보는 모델. ERF가 정확히 한 점이어야 한다."""

    def forward(self, x):
        return x[:, :, 112, 112].sum(dim=1)


def test_a_center_only_model_gives_a_single_bright_pixel(monkeypatch):
    monkeypatch.setattr(
        "bench.erf.center_token_scalar", lambda name, model, x: model(x)
    )
    images = torch.zeros(3, 3, 224, 224)

    erf = accumulate_erf("toy", CenterPixel(), images, device="cpu")

    assert erf.shape == (224, 224)
    assert np.unravel_index(erf.argmax(), erf.shape) == (112, 112)
    assert erf.sum() == erf[112, 112]


def test_one_loud_image_does_not_dominate_the_average(monkeypatch):
    """이미지별 정규화가 빠지면 gradient가 큰 한 장이 평균을 삼킨다."""
    scales = iter([1.0, 1000.0])
    monkeypatch.setattr(
        "bench.erf.center_token_scalar",
        lambda name, model, x: next(scales) * model(x),
    )
    erf = accumulate_erf("toy", CenterPixel(), torch.zeros(2, 3, 224, 224), device="cpu")

    assert erf[112, 112] == 1.0


class BatchNormCenter(nn.Module):
    """BatchNorm을 거친 뒤 중심 픽셀을 보는 토이 모델.

    train 모드의 BatchNorm은 배치와 모든 공간 위치에 걸쳐 통계를 내므로 한 픽셀의
    출력이 다른 이미지·다른 위치에도 의존한다. eval 모드는 running 통계를 쓰므로
    이미지 하나만의 함수다. 두 모드의 gradient가 실제로 갈리도록 running 통계를
    입력 분포와 다르게 박아 둔다.
    """

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.bn.running_mean.fill_(0.0)
        self.bn.running_var.fill_(4.0)

    def forward(self, x):
        return self.bn(x)[:, :, 112, 112].sum(dim=1)


def test_accumulate_erf_puts_the_model_in_eval_mode(monkeypatch):
    """CMT의 캡처 지점(`_swish`)은 BatchNorm 바로 뒤다. train 모드로 재면 BN이
    배치와 49개 공간 위치에 걸쳐 정규화해 버려서, 예외도 NaN도 없이 '그럴듯한
    등방' 오답이 나온다 — 이 저장소가 잡으려는 조용한 실패의 전형이다.

    eval 모드에서는 중심 픽셀의 출력이 그 픽셀만의 함수라 gradient가 (112,112)
    한 점에만 실린다. train 모드에서는 배치 통계가 모든 픽셀을 통과하므로
    gradient가 맵 전체로 새어 나가고 erf.sum()이 peak보다 커진다 — 그래서
    아래 두 단언은 .eval()이 빠지면 실제로 깨진다."""
    monkeypatch.setattr(
        "bench.erf.center_token_scalar", lambda name, model, x: model(x)
    )
    model = BatchNormCenter()
    model.train()  # 호출자가 실수로 train 모드인 모델을 넘긴 상황

    erf = accumulate_erf("toy", model, torch.randn(4, 3, 224, 224), device="cpu")

    assert model.training is False, "accumulate_erf가 .eval()을 걸지 않았다"
    assert erf.sum() == erf[112, 112]


def test_the_input_batch_is_not_modified(monkeypatch):
    """requires_grad를 원본에 걸면 호출한 쪽의 텐서가 오염된다."""
    monkeypatch.setattr(
        "bench.erf.center_token_scalar", lambda name, model, x: model(x)
    )
    images = torch.zeros(1, 3, 224, 224)
    accumulate_erf("toy", CenterPixel(), images, device="cpu")

    assert images.requires_grad is False
    assert images.grad is None
