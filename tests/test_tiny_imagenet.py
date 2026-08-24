"""Tiny-ImageNet 라벨 매핑 검증.

val이 ImageFolder 구조가 아니라는 것이 이 데이터셋의 함정이다. 매핑이 train의
클래스 순서와 어긋나면 네 칸 전부 같은 크기로 틀려서 요인 대비는 정상으로 보인다.
"""
from pathlib import Path

import pytest
from PIL import Image

from data.tiny_imagenet import class_to_index, val_items


def _make_tree(root: Path, wnids: list[str]) -> Path:
    """실제 Tiny-ImageNet과 같은 구조의 최소 트리를 만든다."""
    base = root / "tiny-imagenet-200"
    for wnid in wnids:
        images = base / "train" / wnid / "images"
        images.mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(images / f"{wnid}_0.JPEG")

    val_images = base / "val" / "images"
    val_images.mkdir(parents=True)
    lines = []
    for i, wnid in enumerate(reversed(wnids)):
        name = f"val_{i}.JPEG"
        Image.new("RGB", (64, 64)).save(val_images / name)
        lines.append(f"{name}\t{wnid}\t0\t0\t63\t63")
    (base / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")
    return base


def test_class_index_is_sorted_wnid_order(tmp_path):
    """정렬된 wnid 순서여야 한다. 파일시스템 순회 순서는 OS마다 다르다."""
    base = _make_tree(tmp_path, ["n02", "n01", "n03"])
    assert class_to_index(base) == {"n01": 0, "n02": 1, "n03": 2}


def test_val_labels_follow_the_train_class_order(tmp_path):
    """val_annotations의 wnid가 train과 같은 인덱스로 매핑돼야 한다.

    픽스처는 val을 train과 뒤집힌 순서로 넣는다 — 파일 순서를 그대로 라벨로 쓰는
    구현이라면 여기서 죽는다.
    """
    base = _make_tree(tmp_path, ["n01", "n02", "n03"])
    index = class_to_index(base)

    for path, label in val_items(base):
        assert label == index[_wnid_of(base, path.name)]


def _wnid_of(base: Path, filename: str) -> str:
    for line in (base / "val" / "val_annotations.txt").read_text().splitlines():
        name, wnid = line.split("\t")[:2]
        if name == filename:
            return wnid
    raise AssertionError(f"{filename}이 val_annotations에 없다")


def test_val_items_covers_every_annotated_image(tmp_path):
    base = _make_tree(tmp_path, ["n01", "n02", "n03"])
    assert len(val_items(base)) == 3


def test_missing_annotation_file_fails_loudly(tmp_path):
    """조용히 빈 목록을 돌려주면 정확도가 계산되지 않은 채 0으로 남는다."""
    base = _make_tree(tmp_path, ["n01"])
    (base / "val" / "val_annotations.txt").unlink()
    with pytest.raises(FileNotFoundError):
        val_items(base)


import torch

from data.tiny_imagenet import (
    TRAIN_CROP_SCALE,
    build_eval_transform,
    build_mixup,
    build_train_transform,
)


def test_train_crop_scale_is_the_documented_deviation():
    """DeiT 원 레시피는 (0.08, 1.0)이다. 64px에서 8%는 5x5 픽셀이라 라벨이 무의미해진다.

    이 상수가 바뀌면 네 칸의 비교 자체는 유지되지만 문서와 어긋난다.
    """
    assert TRAIN_CROP_SCALE == (0.6, 1.0)


def test_train_transform_outputs_the_training_resolution():
    out = build_train_transform(size=64)(Image.new("RGB", (64, 64)))
    assert out.shape == (3, 64, 64)


def test_eval_transform_is_deterministic():
    """평가 경로에 무작위가 남아 있으면 같은 체크포인트가 매번 다른 top-1을 낸다."""
    transform = build_eval_transform(size=64)
    image = Image.new("RGB", (64, 64), color=(31, 63, 127))
    assert torch.equal(transform(image), transform(image))


def test_mixup_produces_soft_targets():
    """mixup이 꺼져 있으면 라벨이 정수로 남는다 — 레시피가 적용되지 않은 것이다."""
    mixup = build_mixup(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([1, 2, 3, 4])
    _, soft = mixup(x, y)
    assert soft.shape == (4, 200)
    assert not torch.equal(soft, torch.nn.functional.one_hot(y, 200).float())
