import pytest
import torch

from models.registry import E4_CELLS, MODEL_NAMES, build_e4_model, build_model


def test_registry_lists_the_three_models_under_comparison():
    assert MODEL_NAMES == ("deit_s", "cmt_s", "vim_s")


def test_unknown_name_raises_with_a_useful_message():
    with pytest.raises(ValueError, match="vim_xl"):
        build_model("vim_xl")


def test_deit_s_has_the_published_parameter_count():
    """DeiT-S는 22M. 크게 어긋나면 잘못된 변형을 불러온 것이다."""
    model = build_model("deit_s", pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    assert 21e6 < params < 23e6, f"{params / 1e6:.1f}M"


def test_deit_s_accepts_a_non_default_resolution():
    """해상도 sweep의 전제. 384²가 안 되면 E1 자체가 성립하지 않는다."""
    model = build_model("deit_s", pretrained=False, img_size=384).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 384, 384))
    assert out.shape == (1, 1000)


FLAT = {"embed_dim": 192, "depth": 12, "patch_size": 8}
FLAT_VIM = {"embed_dim": 192, "depth": 24, "patch_size": 8}
HIER = {"embed_dims": [52, 104, 208, 416], "depths": [2, 10, 2, 2], "stem_channel": 16}


def test_e4_cells_are_the_four_documented_names():
    assert E4_CELLS == ("a_deit_ti", "b_vim_ti", "c_cmt_ti", "d_hvim")


def test_unknown_cell_fails_loudly():
    with pytest.raises(ValueError, match="알 수 없는"):
        build_e4_model("e_something", FLAT)


@pytest.mark.parametrize("cell,cfg", [
    ("a_deit_ti", FLAT), ("b_vim_ti", FLAT_VIM),
    ("c_cmt_ti", HIER), ("d_hvim", HIER),
])
def test_every_cell_builds_a_200_class_head(cell, cfg):
    model = build_e4_model(cell, cfg, num_classes=200, img_size=64)
    assert sum(p.numel() for p in model.parameters()) > 0
    assert model.head.out_features == 200


def test_flat_cells_produce_the_documented_token_count():
    """patch 8, 64px -> 8x8 = 64 토큰. 이 수가 바뀌면 예산 추정이 전부 어긋난다."""
    model = build_e4_model("a_deit_ti", FLAT, img_size=64)
    assert model.patch_embed.num_patches == 64


@pytest.mark.parametrize("cell,cfg", [
    ("a_deit_ti", FLAT), ("b_vim_ti", FLAT_VIM),
    ("c_cmt_ti", HIER), ("d_hvim", HIER),
])
def test_every_cell_actually_receives_drop_path(cell, cfg):
    """레시피가 한 칸만 달라지면 그 차이가 요인 효과에 섞이고 흔적이 남지 않는다.

    인자 이름이 프레임워크마다 달라서 조용히 빠지기 쉬운 지점이다 — 특히 CMT의
    dp는 stochastic depth가 아니다.
    """
    with_dp = build_e4_model(cell, cfg, img_size=64, drop_path=0.1)
    without = build_e4_model(cell, cfg, img_size=64, drop_path=0.0)

    def probs(model):
        return sorted(
            module.drop_prob
            for module in model.modules()
            if type(module).__name__ == "DropPath"
        )

    assert max(probs(with_dp) or [0.0]) > 0.0, f"{cell}에 drop path가 걸리지 않았다"
    assert max(probs(without) or [0.0]) == 0.0
