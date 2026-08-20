"""VOC 2012 인스턴스 마스크 준비. E3가 객체 단위로 재기 위한 입력이다.

이 파일의 유일한 책임은 **이미지와 마스크가 같은 자리에 놓이게 하는 것**이다.
어긋나면 모든 숫자가 무의미한데 예외도 경고도 나지 않고, 결과는 "이 모델은
객체를 통합하지 못한다"로 읽힌다.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data.voc import IMAGENET_MEAN, IMAGENET_STD

VOID_LABEL = 255
"""VOC가 객체 경계에 칠하는 값. 객체도 배경도 아니다."""

BACKGROUND_LABEL = 0


def mask_dir(root: str | Path = "data") -> Path:
    """SegmentationObject 디렉터리. 없으면 하드 실패한다.

    `data.voc.ensure_voc`가 받는 것과 같은 아카이브 안에 들어 있으므로 별도
    다운로드 경로를 두지 않는다 — 두면 두 경로가 서로 다른 판본을 가리킬 수 있다.
    """
    from data.voc import ensure_voc

    ensure_voc(root)
    path = Path(root) / "VOCdevkit" / "VOC2012" / "SegmentationObject"
    if not path.is_dir():
        raise RuntimeError(f"{path}가 없다 — VOC 아카이브가 완전한지 확인할 것")
    return path


def image_path_for(mask_path: Path, image_dir: Path) -> Path:
    return Path(image_dir) / (Path(mask_path).stem + ".jpg")


def _geometry(size: int, interpolation: InterpolationMode) -> transforms.Compose:
    """이미지와 마스크가 공유하는 기하 변환.

    두 갈래가 각자 Resize/CenterCrop을 적어 두면 한쪽만 고쳐도 아무 데서도
    터지지 않는다. 한 함수에서 만들어 보간 방식만 다르게 준다.
    """
    return transforms.Compose([
        transforms.Resize(size, interpolation=interpolation),
        transforms.CenterCrop(size),
    ])


def load_mask(path: Path, size: int = 224) -> np.ndarray:
    """인스턴스 마스크를 (size, size) uint8 배열로. 값은 인스턴스 id다.

    nearest 보간을 쓰는 이유는 값이 밝기가 아니라 id이기 때문이다. bilinear로
    줄이면 id 1과 3 사이에 2가 생기고, 그 2는 존재하지 않는 객체다.
    """
    image = Image.open(path)
    if image.mode not in ("P", "L"):
        raise ValueError(
            f"{path}의 모드가 {image.mode!r}다 — 팔레트(P) 또는 그레이스케일(L)이어야 "
            "한다. RGB로 변환하면 인스턴스 id가 팔레트 색으로 바뀐다."
        )
    resized = _geometry(size, InterpolationMode.NEAREST)(image)
    return np.array(resized, dtype=np.uint8)


def load_image_and_mask(
    image_path: Path, mask_path: Path, size: int = 224
) -> tuple[torch.Tensor, np.ndarray]:
    """같은 기하 변환을 통과한 (이미지 텐서, 마스크 배열)."""
    pixels = _geometry(size, InterpolationMode.BILINEAR)(
        Image.open(image_path).convert("RGB")
    )
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return to_tensor(pixels), load_mask(mask_path, size=size)


def instance_ids(mask: np.ndarray) -> list[int]:
    values = set(np.unique(mask).tolist()) - {BACKGROUND_LABEL, VOID_LABEL}
    return sorted(int(value) for value in values)


def instance_mask(mask: np.ndarray, instance_id: int) -> np.ndarray:
    if instance_id in (BACKGROUND_LABEL, VOID_LABEL):
        raise ValueError(f"{instance_id}는 인스턴스 id가 아니다")
    return mask == instance_id


def void_mask(mask: np.ndarray) -> np.ndarray:
    return mask == VOID_LABEL
