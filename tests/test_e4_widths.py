"""폭 정렬의 관문. 이 대역을 벗어난 칸이 있으면 요인 대비에 용량 교란이 남는다."""
import pytest

from experiments.e4_widths import (
    PARAM_TARGET,
    PARAM_TOLERANCE,
    count_params,
    load_cell_config,
    load_common_config,
    within_budget,
)
from models.registry import E4_CELLS, build_e4_model


def test_budget_band_is_the_documented_one():
    assert PARAM_TARGET == 6_860_000
    assert PARAM_TOLERANCE == 0.05


def test_within_budget_rejects_the_stock_spread():
    """스톡 A 5.43M과 C 10.55M은 둘 다 대역 밖이다 — 그게 이 태스크의 존재 이유다."""
    assert not within_budget(5_430_000)
    assert not within_budget(10_550_000)
    assert within_budget(PARAM_TARGET)


@pytest.mark.parametrize("cell", E4_CELLS)
def test_committed_config_lands_in_the_budget(cell):
    """configs에 고정된 폭이 실제로 대역 안인지 매 실행 확인한다."""
    cfg = load_cell_config(cell)
    n = count_params(build_e4_model(cell, cfg, num_classes=200, img_size=64))
    assert within_budget(n), f"{cell}: {n / 1e6:.2f}M이 대역 밖이다"


def test_hierarchical_cells_use_the_rebalanced_depths():
    """[2,10,2,2] — 본체를 8x8 = 64토큰에 놓는 재배치. 스톡 [2,2,10,2]가 아니다."""
    for cell in ("c_cmt_ti", "d_hvim"):
        assert load_cell_config(cell)["depths"] == [2, 10, 2, 2]


def test_flat_cells_use_patch_8():
    for cell in ("a_deit_ti", "b_vim_ti"):
        assert load_cell_config(cell)["patch_size"] == 8


def test_common_config_pins_the_recipe():
    common = load_common_config()
    assert common["epochs"] == 300
    assert common["batch_size"] == 256
    assert common["lr"] == pytest.approx(2.5e-4)
    assert common["warmup_epochs"] == 5
    assert common["weight_decay"] == pytest.approx(0.05)
    assert common["seeds"] == [1, 2, 3]


def test_recipe_keys_reach_the_code_that_uses_them():
    """yaml의 값과 실제 동작이 갈라지지 않는지 확인한다.

    이 파일의 다른 테스트가 "yaml이 레시피를 고정한다"고 단언하지만, 그 키를 코드가
    읽지 않으면 그 단언은 yaml 파일 안에서만 참이다. mixup·cutmix·label_smoothing·
    crop_scale은 실제 학습 경로를 만드는 함수의 기본값과 일치해야 한다 — 이 테스트가
    깨지면 둘 중 한 곳만 고쳐진 것이다.
    """
    from data.tiny_imagenet import TRAIN_CROP_SCALE, build_mixup

    common = load_common_config()
    mixup = build_mixup(num_classes=10)
    assert mixup.mixup_alpha == pytest.approx(common["mixup"])
    assert mixup.cutmix_alpha == pytest.approx(common["cutmix"])
    assert mixup.label_smoothing == pytest.approx(common["label_smoothing"])
    assert tuple(common["crop_scale"]) == TRAIN_CROP_SCALE


def test_record_only_keys_are_marked_as_such():
    """동작을 바꾸지 못하는 키는 yaml에서 [record-only]로 표시되어야 한다.

    amp와 ema는 코드가 읽지 않는다(fp16은 bench/train.py에 고정, EMA 경로는 구현이
    없다). 표시가 없으면 값을 고치는 것이 조용한 no-op이 되고, 읽는 사람은 어느 키가
    run을 통제하는지 한눈에 알 수 없다.
    """
    from pathlib import Path

    text = Path("configs/e4_common.yaml").read_text(encoding="utf-8")
    for key in ("amp", "ema"):
        line = next(ln for ln in text.splitlines() if ln.startswith(f"{key}:"))
        assert "[record-only]" in line, line
