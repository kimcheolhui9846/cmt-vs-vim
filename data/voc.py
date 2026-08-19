"""VOC 2012 이미지 준비. E2와 E3가 같은 이미지를 쓴다."""
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_image_paths(paths: list[Path], n: int, seed: int = 0) -> list[Path]:
    """고정 seed로 n장을 뽑는다.

    정렬을 먼저 하는 이유는 파일시스템 순회 순서가 OS·파일시스템마다 다르기
    때문이다. 정렬하지 않으면 같은 seed로도 다른 이미지가 뽑혀 재현이 깨진다.
    """
    if n > len(paths):
        raise ValueError(f"{n}장을 요청했는데 {len(paths)}장뿐이다")
    ordered = sorted(paths)
    return sorted(random.Random(seed).sample(ordered, n))


def load_images(paths: list[Path], size: int = 224) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return torch.stack([
        transform(Image.open(path).convert("RGB")) for path in paths
    ])


def ensure_voc(root: str | Path = "data") -> Path:
    """VOC2012 trainval(약 1.9GB)을 받고 JPEG 디렉터리를 돌려준다.

    torchvision이 체크섬까지 확인한다. 이미 받았으면 다시 받지 않는다.
    """
    from torchvision.datasets import VOCSegmentation

    root = Path(root)
    images = root / "VOCdevkit" / "VOC2012" / "JPEGImages"
    if not images.is_dir():
        VOCSegmentation(root=str(root), year="2012", image_set="val", download=True)
    if not images.is_dir():
        raise RuntimeError(f"VOC 다운로드 후에도 {images}가 없다")
    return images
