"""Tiny-ImageNet 200 준비. E4의 네 칸이 같은 데이터를 쓴다.

val이 ImageFolder 구조가 아니라는 점이 이 데이터셋에서 가장 조용한 함정이다.
10000장이 val/images/에 평평하게 있고 라벨은 val_annotations.txt에 있다.
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_DIRNAME = "tiny-imagenet-200"
NUM_CLASSES = 200


def ensure_tiny_imagenet(root: str | Path = "data") -> Path:
    """압축을 풀어 데이터셋 루트를 돌려준다. 이미 있으면 다시 받지 않는다."""
    from torchvision.datasets.utils import download_and_extract_archive

    root = Path(root)
    base = root / TINY_DIRNAME
    if not (base / "val" / "val_annotations.txt").is_file():
        download_and_extract_archive(TINY_URL, download_root=str(root))
    if not (base / "val" / "val_annotations.txt").is_file():
        raise RuntimeError(f"다운로드 후에도 {base}가 올바른 구조가 아니다")
    return base


def class_to_index(root: Path) -> dict[str, int]:
    """train 하위 디렉터리 이름을 정렬해 인덱스를 매긴다.

    정렬하는 이유는 파일시스템 순회 순서가 OS·파일시스템마다 다르기 때문이다.
    torchvision의 ImageFolder도 같은 규약(sorted)을 쓰므로 두 쪽이 일치한다.
    """
    wnids = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    return {wnid: i for i, wnid in enumerate(wnids)}


def val_items(root: Path) -> list[tuple[Path, int]]:
    """val 이미지 경로와 라벨. 라벨은 train의 클래스 인덱스를 그대로 쓴다."""
    annotations = root / "val" / "val_annotations.txt"
    if not annotations.is_file():
        raise FileNotFoundError(f"{annotations}가 없다 — val 라벨을 만들 수 없다")

    index = class_to_index(root)
    items = []
    for line in annotations.read_text().splitlines():
        if not line.strip():
            continue
        name, wnid = line.split("\t")[:2]
        items.append((root / "val" / "images" / name, index[wnid]))
    return items


class TinyImageNetVal(Dataset):
    def __init__(self, root: Path, transform):
        self.items = val_items(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.voc import IMAGENET_MEAN, IMAGENET_STD

# DeiT 원 레시피는 (0.08, 1.0)이다. 64px에서 8%까지 잘라내면 남는 것이 5x5 픽셀이라
# 라벨이 무의미해진다. 네 칸에 동일 적용하므로 요인 비교에는 영향이 없다.
TRAIN_CROP_SCALE = (0.6, 1.0)


def build_train_transform(size: int = 64, crop_scale=TRAIN_CROP_SCALE):
    """crop_scale은 configs/e4_common.yaml에서 들어온다.

    기본값을 이 파일에만 두면 yaml을 고쳐도 아무 일이 일어나지 않는다 — 값이 우연히
    같아 지금은 티가 나지 않지만, 레시피를 바꾸려는 사람에게는 조용한 no-op이다.
    """
    from timm.data import create_transform

    return create_transform(
        input_size=size,
        is_training=True,
        scale=tuple(crop_scale),
        ratio=(3 / 4, 4 / 3),
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def build_eval_transform(size: int = 64):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_mixup(num_classes: int = NUM_CLASSES, mixup_alpha: float = 0.8,
                cutmix_alpha: float = 1.0, label_smoothing: float = 0.1):
    """세 값 모두 configs/e4_common.yaml에서 들어온다.

    label_smoothing이 여기 있는 이유: mixup을 켜면 손실이 SoftTargetCrossEntropy로
    바뀌어 nn.CrossEntropyLoss(label_smoothing=...)가 실행되지 않는다. 실제 학습에
    적용되는 smoothing은 이 Mixup이 soft target을 만들 때 넣는 값 하나뿐이다.
    """
    from timm.data import Mixup

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )


def build_loaders(
    root: Path, batch_size: int, workers: int, size: int = 64,
    crop_scale=TRAIN_CROP_SCALE,
) -> tuple[DataLoader, DataLoader]:
    train = ImageFolder(str(root / "train"),
                        transform=build_train_transform(size, crop_scale))
    if train.class_to_idx != class_to_index(root):
        raise RuntimeError(
            "ImageFolder의 클래스 인덱스가 val 매핑과 다르다 — 라벨이 어긋난다"
        )
    val = TinyImageNetVal(root, transform=build_eval_transform(size))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers,
                   pin_memory=True, drop_last=True, persistent_workers=workers > 0),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers,
                   pin_memory=True, persistent_workers=workers > 0),
    )
