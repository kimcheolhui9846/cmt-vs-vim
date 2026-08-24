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
