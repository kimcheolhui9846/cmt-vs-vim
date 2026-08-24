"""D칸이 C칸에서 연산자만 바뀐 것인지 확인한다.

상호작용 (D-B) - (C-A)가 해석되려면 B->D 조작이 A->C와 같아야 한다. 그 조건은
"격자와 depths가 같고 attention만 Mamba로 바뀌었다"로 환원된다.
"""
import pytest
import torch

from models.cmt_official import CMT
from models.hvim import HierarchicalVim, build_hvim, stage_grids

DEPTHS = (2, 10, 2, 2)
DIMS = (52, 104, 208, 416)


def _cmt(img_size=64):
    """C칸과 같은 인자로 CMT를 만든다.

    drop_path_rate와 dp를 둘 다 넘기는 것이 핵심이다 — CMT에서 dp는 stochastic
    depth가 아니라 head 앞 classifier dropout이고, drop_path_rate의 기본값은 0이다.
    """
    return CMT(img_size=img_size, num_classes=200, embed_dims=list(DIMS),
               stem_channel=16, num_heads=[1, 2, 4, 8], depths=list(DEPTHS),
               mlp_ratios=[3.6] * 4, qkv_bias=True, qk_ratio=1,
               sr_ratios=[8, 4, 2, 1], drop_path_rate=0.1, dp=0.1)


def test_hvim_has_no_unused_relative_pos_parameters():
    """CMT의 relative_pos는 nn.Parameter다. attention을 들어내면 쓰이지 않으면서
    파라미터 예산만 먹는다 — 예산 판정이 거짓이 된다."""
    model = build_hvim()
    leftovers = [name for name, _ in model.named_parameters() if "relative_pos" in name]
    assert leftovers == []


def test_hvim_stage_grids_match_cmt():
    assert stage_grids(build_hvim(img_size=64)) == stage_grids(_cmt(img_size=64))


def test_hvim_stage_grids_are_the_documented_ones():
    """64px에서 16-8-4-2. 본체(depth 10)가 8x8 = 64토큰에 놓인다."""
    assert stage_grids(build_hvim(img_size=64)) == [16, 8, 4, 2]


def test_hvim_block_counts_match_cmt():
    model = build_hvim()
    counts = [len(model.blocks_a), len(model.blocks_b),
              len(model.blocks_c), len(model.blocks_d)]
    assert counts == list(DEPTHS)


def test_hvim_keeps_the_conv_locality_of_cmt():
    """LPU(depth-wise conv)와 IRFFN이 남아 있어야 B->D가 A->C와 같은 조작이 된다."""
    block = build_hvim().blocks_b[0]
    assert block.proj.groups == block.proj.in_channels  # LPU는 depth-wise
    assert hasattr(block.mlp, "proj")                   # IRFFN의 3x3 conv


def test_hvim_uses_bidirectional_mamba_not_attention():
    block = build_hvim().blocks_b[0]
    assert not hasattr(block, "attn")
    assert block.mixer.bimamba_type == "v2"


def test_hvim_and_cmt_share_the_same_drop_path_schedule():
    """CMT의 dp는 stochastic depth가 아니라 classifier dropout이다.

    두 인자를 섞으면 C는 drop path 0, D는 0.1로 갈라지고 그 차이가 D-C에 그대로
    섞인다 — 요인 설계가 통제하려던 바로 그 종류의 교란이다.
    """
    hvim = build_hvim(drop_path_rate=0.1, dp=0.1)
    cmt = _cmt()

    def rates(model):
        out = []
        for name in ("blocks_a", "blocks_b", "blocks_c", "blocks_d"):
            for block in getattr(model, name):
                layer = block.drop_path
                out.append(getattr(layer, "drop_prob", 0.0))
        return out

    assert rates(hvim) == pytest.approx(rates(cmt))


def test_hvim_classifier_dropout_matches_cmt():
    assert build_hvim(dp=0.1)._drop.p == _cmt()._drop.p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba 커널은 CUDA 전용")
def test_hvim_forward_returns_class_logits():
    model = build_hvim().cuda().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64, device="cuda"))
    assert out.shape == (2, 200)
