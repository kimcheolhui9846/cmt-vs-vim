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


def test_the_input_batch_is_not_modified(monkeypatch):
    """requires_grad를 원본에 걸면 호출한 쪽의 텐서가 오염된다."""
    monkeypatch.setattr(
        "bench.erf.center_token_scalar", lambda name, model, x: model(x)
    )
    images = torch.zeros(1, 3, 224, 224)
    accumulate_erf("toy", CenterPixel(), images, device="cpu")

    assert images.requires_grad is False
    assert images.grad is None
