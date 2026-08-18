import pytest
import torch

from models.registry import MODEL_NAMES, build_model


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
