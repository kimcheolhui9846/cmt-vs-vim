"""D칸이 C칸에서 연산자만 바뀐 것인지 확인한다.

상호작용 (D-B) - (C-A)가 해석되려면 B->D 조작이 A->C와 같아야 한다. 그 조건은
"격자와 depths가 같고 attention만 Mamba로 바뀌었다"로 환원된다.
"""
import pytest
import torch
import torch.nn as nn

from models.cmt_official import CMT
from models.hvim import HierarchicalVim, build_hvim, stage_grids

DEPTHS = (2, 10, 2, 2)
DIMS = (52, 104, 208, 416)


def _cmt(img_size=64, drop_path_rate=0.1, dp=0.1):
    """C칸과 같은 인자로 CMT를 만든다.

    drop_path_rate와 dp를 둘 다 넘기는 것이 핵심이다 — CMT에서 dp는 stochastic
    depth가 아니라 head 앞 classifier dropout이고, drop_path_rate의 기본값은 0이다.
    둘을 다른 값으로 넘길 수 있게 열어 둔 것은, 두 인자가 뒤바뀌어도 기본값(둘 다
    0.1)에서는 결과가 똑같아 보여 테스트가 눈감아 주는 사고를 막기 위해서다.
    """
    return CMT(img_size=img_size, num_classes=200, embed_dims=list(DIMS),
               stem_channel=16, num_heads=[1, 2, 4, 8], depths=list(DEPTHS),
               mlp_ratios=[3.6] * 4, qkv_bias=True, qk_ratio=1,
               sr_ratios=[8, 4, 2, 1], drop_path_rate=drop_path_rate, dp=dp)


def _drop_path_rates(model):
    """네 stage 전체에 걸친 블록별 drop_path 확률."""
    out = []
    for name in ("blocks_a", "blocks_b", "blocks_c", "blocks_d"):
        for block in getattr(model, name):
            layer = block.drop_path
            out.append(getattr(layer, "drop_prob", 0.0))
    return out


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

    assert _drop_path_rates(hvim) == pytest.approx(_drop_path_rates(cmt))


def test_hvim_classifier_dropout_matches_cmt():
    assert build_hvim(dp=0.1)._drop.p == _cmt()._drop.p


def test_hvim_drop_path_schedule_ignores_dp():
    """drop_path_rate와 dp가 서로 다를 때도 블록별 drop_path는 drop_path_rate만
    따라야 한다. 기본값(둘 다 0.1)만 테스트하면 hvim.py의 linspace 줄에서
    drop_path_rate 자리에 dp를 잘못 넣어도 (예: 계승 이슈에서 지적한 뒤바뀜) 두
    값이 같으니 통과해 버린다 — dp!=drop_path_rate로 그 뒤바뀜을 실제로 잡는다.
    """
    hvim = build_hvim(drop_path_rate=0.1, dp=0.9)
    cmt = _cmt(drop_path_rate=0.1, dp=0.9)

    assert _drop_path_rates(hvim) == pytest.approx(_drop_path_rates(cmt))


def test_hvim_classifier_dropout_follows_dp_not_drop_path_rate():
    """분류기 dropout은 dp를 따라야지 drop_path_rate를 따르면 안 된다."""
    hvim = build_hvim(drop_path_rate=0.1, dp=0.9)
    assert hvim._drop.p == pytest.approx(0.9)
    assert hvim._drop.p != pytest.approx(0.1)


def test_hvim_lpu_and_irffn_convs_use_cmt_init_not_pytorch_default():
    """MambaBlock의 proj(LPU)와 mlp(IRFFN) 안의 Conv2d는 CMT._init_weights가 쓰는
    kaiming_normal_(mode='fan_out')를 따라야 한다.

    CMT.__init__의 self.apply(self._init_weights)는 super().__init__() 안에서,
    즉 HierarchicalVim이 blocks_a~d를 MambaBlock으로 갈아 끼우기 전에 끝난다.
    재초기화를 하지 않으면 새로 만든 Conv2d는 PyTorch 기본값(kaiming_uniform_,
    mode='fan_in', a=sqrt(5))으로 남아 CMT와 가중치 스케일이 달라진다 — 이 차이는
    forward shape에는 드러나지 않고 어떤 출력 파일에도 남지 않는다.
    """
    model = build_hvim()
    ratios = []
    for stage in ("blocks_a", "blocks_b", "blocks_c", "blocks_d"):
        for block in getattr(model, stage):
            for subtree in (block.proj, block.mlp):
                for m in subtree.modules():
                    if isinstance(m, nn.Conv2d):
                        _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        expected_std = (2.0 / fan_out) ** 0.5  # kaiming_normal_ 기본 gain(relu)=sqrt(2)
                        actual_std = m.weight.detach().std().item()
                        ratios.append(actual_std / expected_std)

    mean_ratio = sum(ratios) / len(ratios)
    # PyTorch 기본 kaiming_uniform_(mode='fan_in', a=sqrt(5))으로 남아 있으면 이
    # 비율이 1에서 크게 벗어난다(대략 2배 이상) — 0.15 허용오차면 충분히 구분된다.
    assert mean_ratio == pytest.approx(1.0, abs=0.15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba 커널은 CUDA 전용")
def test_hvim_forward_returns_class_logits():
    model = build_hvim().cuda().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64, device="cuda"))
    assert out.shape == (2, 200)
