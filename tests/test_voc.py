from pathlib import Path

import pytest
import torch
from PIL import Image

from data.voc import IMAGENET_MEAN, IMAGENET_STD, load_images, sample_image_paths


def _paths(n: int) -> list[Path]:
    return [Path(f"img_{i:04d}.jpg") for i in range(n)]


def test_same_seed_gives_the_same_sample():
    """이미지가 달라지면 ERF도 달라진다. 재현되지 않는 샘플은 측정을 무효로 만든다."""
    first = sample_image_paths(_paths(100), n=10, seed=0)
    second = sample_image_paths(_paths(100), n=10, seed=0)
    assert first == second


def test_different_seeds_give_different_samples():
    assert sample_image_paths(_paths(100), 10, seed=0) != sample_image_paths(
        _paths(100), 10, seed=1
    )


def test_sample_order_does_not_depend_on_input_order():
    """파일시스템 순회 순서는 OS마다 다르다. 정렬하지 않으면 같은 seed로도
    다른 이미지가 뽑힌다."""
    forward = sample_image_paths(_paths(100), 10, seed=0)
    backward = sample_image_paths(list(reversed(_paths(100))), 10, seed=0)
    assert forward == backward


def test_asking_for_more_than_exists_fails_loudly():
    """조용히 적게 반환하면 N=256으로 잰 줄 알았던 값이 실은 N=40이 된다."""
    with pytest.raises(ValueError, match="256"):
        sample_image_paths(_paths(40), n=256, seed=0)


def _solid_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _expected_pixel(color: tuple[int, int, int]) -> torch.Tensor:
    """ImageNet 정규화를 수식대로 직접 계산한다: (c/255 - mean) / std."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = torch.tensor(color, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
    return (rgb - mean) / std


def test_load_images_returns_the_right_shape(tmp_path):
    paths = [
        _solid_image(tmp_path / f"img_{i}.png", (50, 50), (10 * i, 20 * i, 30 * i))
        for i in range(3)
    ]
    x = load_images(paths)
    assert x.shape == (3, 3, 224, 224)


def test_load_images_applies_imagenet_normalization(tmp_path):
    """정규화를 빼면 이 테스트가 깨진다. 단색 이미지라 기대값을 수식으로 계산할 수 있다."""
    color = (200, 100, 50)
    path = _solid_image(tmp_path / "solid.png", (50, 50), color)

    x = load_images([path])

    expected = _expected_pixel(color).expand_as(x)
    assert torch.allclose(x, expected, atol=1e-4)


def test_load_images_resizes_before_cropping(tmp_path):
    """Resize가 CenterCrop보다 먼저여야 한다.

    400x100 단색 이미지를 넣는다. 순서가 맞으면 Resize(224)가 짧은 변(높이 100)을
    224로 먼저 키워 896x224가 되고, 그 다음 CenterCrop(224)이 폭만 잘라 224x224 전부가
    원본 색이다. 순서가 뒤바뀌면 CenterCrop(224)이 높이 100짜리 이미지에 먼저 적용돼
    위아래를 0(검정)으로 패딩하므로, 출력에 원본 색이 아닌 픽셀이 섞인다.
    """
    color = (30, 200, 90)
    path = _solid_image(tmp_path / "wide.png", (400, 100), color)

    x = load_images([path])

    expected = _expected_pixel(color).expand_as(x)
    assert torch.allclose(x, expected, atol=1e-4)
