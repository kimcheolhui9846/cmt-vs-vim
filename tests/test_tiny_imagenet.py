"""Tiny-ImageNet 라벨 매핑 검증.

val이 ImageFolder 구조가 아니라는 것이 이 데이터셋의 함정이다. 매핑이 train의
클래스 순서와 어긋나면 네 칸 전부 같은 크기로 틀려서 요인 대비는 정상으로 보인다.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.tiny_imagenet import (
    TRAIN_CROP_SCALE,
    build_eval_transform,
    build_mixup,
    build_train_transform,
    class_to_index,
    val_items,
)


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



def test_train_crop_scale_is_the_documented_deviation():
    """DeiT 원 레시피는 (0.08, 1.0)이다. 64px에서 8%는 5x5 픽셀이라 라벨이 무의미해진다.

    이 상수가 바뀌면 네 칸의 비교 자체는 유지되지만 문서와 어긋난다.
    """
    assert TRAIN_CROP_SCALE == (0.6, 1.0)


def test_train_transform_outputs_the_training_resolution():
    out = build_train_transform(size=64)(Image.new("RGB", (64, 64)))
    assert out.shape == (3, 64, 64)


def test_eval_transform_is_deterministic():
    """평가 경로에 무작위가 남아 있으면 같은 체크포인트가 매번 다른 top-1을 낸다.

    p=0.5 이항 변환(예: RandomHorizontalFlip)을 추가하는 회귀를 탐지하려면 같은 이미지를
    여러 번 변환해서 모두 동일한지 확인해야 한다. 대칭 이미지는 뒤집혀도 같아 보이므로,
    비대칭 이미지를 쓴다. 한 번 비교는 50% 확률로 통과하지만, 충분한 반복으로 이항 무작위성을
    본질적으로 항상 탐지한다.
    """
    transform = build_eval_transform(size=64)
    # 비대칭 이미지 생성: 좌반부 빨강, 우반부 녹색 (뒤집으면 다름)
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img_array[:, :32, 0] = 200  # 좌반부 빨강
    img_array[:, 32:, 1] = 100  # 우반부 녹색
    image = Image.fromarray(img_array, "RGB")
    # 같은 이미지를 여러 번 변환해서 모두 동일한지 확인 (p=0.5 이항: 0.5^20 << 0.001%)
    results = [transform(image) for _ in range(20)]
    first = results[0]
    for result in results[1:]:
        assert torch.equal(first, result), "무작위가 감지됨 — eval 경로가 결정론적이지 않음"


def test_mixup_produces_soft_targets():
    """mixup이 꺼져 있으면 라벨이 정수로 남는다 — 레시피가 적용되지 않은 것이다.

    label_smoothing만으로도 one-hot과 달라지지만, 각 샘플의 부드러운 라벨은 자신의 클래스에만
    주로 몰려 있다. mixup이 켜져 있으면 적어도 일부 샘플은 여러 클래스 간에 실제로 혼합되어
    다양한 클래스에서 상당한 확률을 가진다.
    """
    mixup = build_mixup(num_classes=200)
    x = torch.randn(4, 3, 64, 64)
    # 다양한 클래스 라벨을 써서 혼합이 일어나면 여러 클래스에 확률이 퍼져 있어야 함
    y = torch.tensor([0, 50, 100, 150])
    _, soft = mixup(x, y)
    assert soft.shape == (4, 200)
    # 적어도 하나의 샘플이 두 개 이상의 클래스에 상당한 확률(0.01 이상)을 가져야 함 (혼합 증거)
    has_mixed = False
    for sample_soft in soft:
        # 값이 0.01 이상인 클래스 개수 세기
        num_classes_above_threshold = (sample_soft >= 0.01).sum().item()
        if num_classes_above_threshold >= 2:
            has_mixed = True
            break
    assert has_mixed, "혼합이 실제로 일어났다면 적어도 하나 샘플이 여러 클래스에 확률을 가져야 함"


def test_mixup_alphas_and_smoothing_come_from_the_caller():
    """세 값이 configs/e4_common.yaml에서 들어와야 한다.

    yaml에 적어 두고 코드에 같은 값을 박아 두면, yaml을 고치는 것이 조용한 no-op이
    된다 — 값이 우연히 일치하는 동안에는 아무 증상도 없다.
    """
    mixup = build_mixup(num_classes=10, mixup_alpha=0.3, cutmix_alpha=0.7,
                        label_smoothing=0.2)
    assert mixup.mixup_alpha == pytest.approx(0.3)
    assert mixup.cutmix_alpha == pytest.approx(0.7)
    assert mixup.label_smoothing == pytest.approx(0.2)


def _crop_scale_of(transform):
    for step in transform.transforms:
        if hasattr(step, "scale"):
            return tuple(step.scale)
    raise AssertionError("RandomResizedCrop을 찾지 못했다")


def test_train_transform_honours_the_crop_scale_argument():
    assert _crop_scale_of(build_train_transform(size=64, crop_scale=(0.2, 0.9))) == (0.2, 0.9)
    assert _crop_scale_of(build_train_transform(size=64)) == TRAIN_CROP_SCALE
