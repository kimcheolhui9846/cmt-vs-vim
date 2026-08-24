"""gradient 지도가 실제로 그 입력에 대한 미분인지 확인한다.

모델을 거치지 않는다. 미분값을 손으로 알 수 있는 스칼라 함수를 넣어 지도가
해석적 정답과 일치하는지 본다 — 모델을 넣으면 정답을 모르므로 이 검증이 성립하지
않는다.
"""
import numpy as np
import pytest
import torch

from bench.attribution import gradient_map


def test_gradient_map_matches_the_analytic_derivative():
    """스칼라를 sum(w * x)로 두면 |d/dx|는 채널에 걸쳐 합한 |w|다."""
    torch.manual_seed(0)
    weights = torch.randn(1, 3, 8, 8)
    image = torch.zeros(3, 8, 8)

    grad = gradient_map(lambda x: (weights.to(x.device) * x).sum(), image, device="cpu")

    expected = weights.abs().sum(dim=1)[0].numpy()
    assert grad.shape == (8, 8)
    assert np.allclose(grad, expected, atol=1e-6)


def test_gradient_map_is_zero_where_the_scalar_does_not_depend_on_the_input():
    """중심 토큰 스칼라의 수용영역 밖은 정확히 0이어야 한다."""
    image = torch.zeros(3, 8, 8)

    grad = gradient_map(lambda x: x[:, :, 2:4, 2:4].sum(), image, device="cpu")

    assert grad[2:4, 2:4].min() > 0
    assert grad[0, 0] == 0.0


def test_gradient_map_does_not_normalise():
    """E2는 이미지마다 peak로 나누지만 E3는 크기를 그대로 써야 한다.
    (질량 비율이 크기를 쓰는 지표다.) 여기서 정규화하면 두 실험이 조용히
    다른 양을 재게 된다."""
    image = torch.zeros(3, 4, 4)

    grad = gradient_map(lambda x: (5.0 * x).sum(), image, device="cpu")

    assert grad.max() == pytest.approx(15.0)   # 채널 3개 x 5.0


def test_gradient_map_leaves_the_caller_tensor_untouched():
    """호출자가 넘긴 텐서에 grad가 붙어 남으면 다음 인스턴스의 측정에 섞인다."""
    image = torch.zeros(3, 4, 4)

    gradient_map(lambda x: x.sum(), image, device="cpu")

    assert image.grad is None
    assert not image.requires_grad
