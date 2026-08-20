"""마스크가 이미지와 같은 자리에 놓이는지 확인한다.

변환이 어긋나면 precision@K가 낮게 나올 뿐이고, 그건 "모델이 객체를 통합하지
못한다"와 구분되지 않는다. 예외도 경고도 나지 않는다.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.voc_masks import (
    BACKGROUND_LABEL,
    VOID_LABEL,
    image_path_for,
    instance_ids,
    instance_mask,
    load_image_and_mask,
    load_mask,
    void_mask,
)

RECT = (120, 140, 280, 260)  # (left, top, right, bottom) — 원본 좌표계, 우/하 제외
"""두 테스트 이미지(400x300, 400x600) 어느 쪽에서도 CenterCrop에 살아남는 자리.

Resize(224)가 짧은 변만 224로 맞추므로 배율이 이미지마다 다르다. 400x600에서는
배율이 0.56이고 CenterCrop이 행 [56, 280)만 남기므로, 원본 행이 [100, 500] 밖이면
사각형이 통째로 잘려 나간다. 400x300에서는 이미지 높이가 300이라 행이 그보다
작아야 한다. 두 조건이 겹치는 구간에 void 테두리 2px 여유를 둔 값이다.

좌표를 바꾸면 두 크기 모두에서 다시 확인할 것 — 한쪽에서만 맞는 값은 정렬을
검증하지 않으면서 통과한다.
"""


def _synthetic_pair(tmp_path: Path, size: tuple[int, int]) -> tuple[Path, Path]:
    """알려진 자리에 사각형이 있는 이미지와 그에 대응하는 인스턴스 마스크.

    이미지의 사각형은 순수한 빨강, 배경은 검정. 마스크의 같은 사각형은 id 1,
    배경은 0이고, 사각형을 두른 두께 2px 테두리는 255(void)다 — VOC의 실제
    구조를 그대로 흉내 낸다.
    """
    left, top, right, bottom = RECT
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[top:bottom, left:right, 0] = 255
    image_path = tmp_path / "sample.png"
    Image.fromarray(image).save(image_path)

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[top - 2 : bottom + 2, left - 2 : right + 2] = VOID_LABEL
    mask[top:bottom, left:right] = 1
    mask_path = tmp_path / "sample_mask.png"
    palette_image = Image.fromarray(mask, mode="P")
    # 팔레트를 명시해야 한다. 주지 않으면 Pillow의 PNG 인코더가 실제로 쓰인
    # 색만 남기고 인덱스를 다시 매기면서 255(void)가 사라진다 — 실측: 저장 전
    # {0, 1, 255}가 다시 읽으면 {0, 1}이 된다. 실제 VOC 파일은 768바이트
    # 팔레트를 갖고 있어 멀쩡하므로, 여기서 그 구조를 재현하지 않으면 테스트만
    # 조용히 다른 입력을 보게 된다.
    palette_image.putpalette([value // 3 for value in range(768)])
    palette_image.save(mask_path)
    return image_path, mask_path


def test_the_synthetic_mask_really_round_trips_its_void(tmp_path):
    """테스트 도형 자체를 먼저 검증한다.

    저장이 void를 잃으면 아래 테스트들은 void를 전혀 시험하지 않으면서 통과할
    수 있다 — 검증되는 입력과 실제 입력이 갈리는 자리다.
    """
    _, mask_path = _synthetic_pair(tmp_path, size=(400, 300))

    raw = np.array(Image.open(mask_path))

    assert set(np.unique(raw).tolist()) == {BACKGROUND_LABEL, 1, VOID_LABEL}


def test_mask_lands_on_the_object_after_the_transform(tmp_path):
    """정직성 장치 2. 변환 후에도 마스크가 사각형 위에 정확히 겹쳐야 한다.

    이미지 쪽 사각형은 빨강 채널만 크다. 마스크가 참인 픽셀에서 빨강이 초록보다
    크고, 마스크 밖이면서 void도 아닌 픽셀에서는 그렇지 않아야 한다.
    """
    image_path, mask_path = _synthetic_pair(tmp_path, size=(400, 300))

    x, mask = load_image_and_mask(image_path, mask_path)

    obj = instance_mask(mask, 1)
    void = void_mask(mask)
    background = ~obj & ~void
    assert obj.any() and background.any()

    red, green = x[0].numpy(), x[1].numpy()
    assert (red[obj] > green[obj]).all(), "마스크 안인데 빨강이 아니다 — 정렬이 어긋났다"
    assert (red[background] < red[obj].min()).all(), "마스크 밖인데 빨강이다"


def test_alignment_holds_for_a_non_square_image(tmp_path):
    """Resize(224)는 짧은 변만 224로 맞춘다. 이미지와 마스크가 같은 순서로
    같은 변환을 받지 않으면 세로로 긴 이미지에서 먼저 어긋난다."""
    image_path, mask_path = _synthetic_pair(tmp_path, size=(400, 600))

    x, mask = load_image_and_mask(image_path, mask_path)

    obj = instance_mask(mask, 1)
    assert obj.any(), "크롭이 사각형을 통째로 잘라냈다면 테스트 도형을 다시 볼 것"
    red, green = x[0].numpy(), x[1].numpy()
    assert (red[obj] > green[obj]).all()


def test_mask_resize_uses_nearest_and_invents_no_ids(tmp_path):
    """마스크 값은 밝기가 아니라 id다. 보간하면 없던 id가 생긴다."""
    _, mask_path = _synthetic_pair(tmp_path, size=(400, 300))

    mask = load_mask(mask_path)

    assert set(np.unique(mask).tolist()) <= {BACKGROUND_LABEL, 1, VOID_LABEL}


def test_void_is_neither_object_nor_background(tmp_path):
    _, mask_path = _synthetic_pair(tmp_path, size=(400, 300))

    mask = load_mask(mask_path)

    assert void_mask(mask).any()
    assert not (instance_mask(mask, 1) & void_mask(mask)).any()
    assert instance_ids(mask) == [1]


def test_instance_ids_skips_background_and_void():
    mask = np.array([[0, 1, 2], [255, 1, 0]], dtype=np.uint8)
    assert instance_ids(mask) == [1, 2]


def test_instance_mask_refuses_background_and_void():
    mask = np.array([[0, 1], [255, 1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        instance_mask(mask, 0)
    with pytest.raises(ValueError):
        instance_mask(mask, 255)


def test_image_path_for_swaps_only_the_extension(tmp_path):
    assert image_path_for(Path("x/2007_000032.png"), tmp_path) == (
        tmp_path / "2007_000032.jpg"
    )


def test_load_mask_rejects_a_colour_converted_file(tmp_path):
    """RGB로 변환해 읽으면 id가 팔레트 색으로 바뀐다. 그 경로를 막는다."""
    _, mask_path = _synthetic_pair(tmp_path, size=(400, 300))
    rgb_path = tmp_path / "rgb.png"
    Image.open(mask_path).convert("RGB").save(rgb_path)

    with pytest.raises(ValueError, match="팔레트"):
        load_mask(rgb_path)
