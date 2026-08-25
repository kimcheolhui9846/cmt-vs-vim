"""폭 정렬의 관문. 이 대역을 벗어난 칸이 있으면 요인 대비에 용량 교란이 남는다."""
import tempfile
from pathlib import Path

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


def test_recipe_keys_reach_the_code_that_uses_them(monkeypatch):
    """yaml의 값이 오케스트레이터의 호출을 타고 실제 객체까지 가는지 확인한다.

    이 파일의 다른 테스트가 "yaml이 레시피를 고정한다"고 단언하지만, 그 키를 코드가
    읽지 않으면 그 단언은 yaml 파일 안에서만 참이다.

    이전 판은 yaml 값을 `build_mixup`·`build_train_transform`의 **기본 인자**와
    비교했다. 그건 배선을 재는 것이 아니라 두 곳의 값이 우연히 같은지를 재는 것이라,
    yaml에서 레시피를 정당하게 재조정하는 순간 — 배선을 만든 목적이 바로 그것이다 —
    배선이 멀쩡한데도 빨개진다. 그래서 여기서는 yaml에 없는 값을 일부러 넣고,
    `experiments.e4_ablation.main`이 그 값을 그대로 실어 보내는지를 본다.
    """
    from experiments import e4_ablation

    tweaked = {**load_common_config(), "mixup": 0.42, "cutmix": 0.37,
               "label_smoothing": 0.03, "crop_scale": [0.31, 0.97], "seeds": [1],
               "epochs": 1}
    seen = {}

    def spy_mixup(num_classes, **kwargs):
        seen["mixup"] = (num_classes, kwargs)
        return object()

    def spy_loaders(root, batch_size, workers, size, crop_scale=None):
        seen["crop_scale"] = crop_scale
        return object(), object()

    monkeypatch.setattr(e4_ablation, "load_common_config", lambda *a, **k: tweaked)
    monkeypatch.setattr(e4_ablation, "load_cell_config", lambda *a, **k: {})
    monkeypatch.setattr(e4_ablation, "snapshot", lambda: {})
    monkeypatch.setattr(e4_ablation, "ensure_tiny_imagenet", lambda *a, **k: Path("."))
    monkeypatch.setattr(e4_ablation, "build_mixup", spy_mixup)
    monkeypatch.setattr(e4_ablation, "build_loaders", spy_loaders)
    monkeypatch.setattr(e4_ablation, "build_e4_model", lambda *a, **k: object())
    monkeypatch.setattr(e4_ablation, "count_params", lambda model: 0)
    monkeypatch.setattr(e4_ablation, "train", lambda *a, **k: {
        "epochs_done": 1, "top1": 0.1, "top5": 0.2, "hours": 0.0,
    })

    with tempfile.TemporaryDirectory() as tmp:
        e4_ablation.main(out_dir=tmp)

    num_classes, kwargs = seen["mixup"]
    assert num_classes == tweaked["num_classes"]
    assert kwargs["mixup_alpha"] == pytest.approx(0.42)
    assert kwargs["cutmix_alpha"] == pytest.approx(0.37)
    assert kwargs["label_smoothing"] == pytest.approx(0.03)
    assert seen["crop_scale"] == (0.31, 0.97)


def test_crop_scale_reaches_the_transform():
    """넘긴 crop_scale이 실제 변환의 scale로 도착하는지 본다.

    yaml 값과 이 함수의 기본 인자가 같은지는 배선의 증거가 아니다 — 값을 바꿔
    넣고 그 값이 도착하는지를 봐야 한다.
    """
    from data.tiny_imagenet import build_train_transform

    transform = build_train_transform(64, crop_scale=(0.31, 0.97))
    scales = [tuple(t.scale) for t in transform.transforms if hasattr(t, "scale")]
    assert (0.31, 0.97) in scales


def test_search_reads_configs_from_the_root_it_is_given():
    """`search`가 현재 작업 디렉터리가 아니라 인자로 받은 root를 읽어야 한다.

    형제 로더 둘은 이미 `root`를 받는데 이 함수만 받지 않아, 저장소 루트 밖에서
    부르면 조용히 다른 곳을 보거나 죽었다.
    """
    from experiments.e4_widths import search

    with tempfile.TemporaryDirectory() as tmp:
        # num_classes를 10으로 줄인 configs를 따로 만든다. head 크기가
        # num_classes에 비례하므로 파라미터 수가 실제로 달라진다.
        (Path(tmp) / "e4_common.yaml").write_text(
            "num_classes: 10\nimg_size: 64\n", encoding="utf-8")
        candidates = [{"embed_dim": 216, "depth": 12, "patch_size": 8}]
        _, small = search("a_deit_ti", candidates, root=tmp)

    _, full = search("a_deit_ti", candidates, root="configs")
    assert small < full  # root를 무시했다면 두 값이 같다


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
