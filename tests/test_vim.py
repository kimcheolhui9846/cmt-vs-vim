"""Vim-S 통합과 fused op FLOPs 회계.

계획 전체에서 가장 조용히 틀리기 쉬운 지점이다. Vim-S 는 bimamba v2 + fast path 라
conv1d · x_proj · dt_proj · selective scan 이 커스텀 autograd Function 하나로 묶여
나간다. fvcore 는 핸들러가 없는 연산을 0 으로 세므로, 핸들러가 scan 만 세면 나머지
셋이 블록마다 양방향으로 사라진다.

항 하나를 빠뜨리는 회귀를 실제로 잡으려면 합계만 봐서는 안 된다. 그래서 항별 계산을
`fused_op_flop_terms` 순수 함수로 빼고, 테스트가 항을 하나씩 직접 검증한다.
"""
import types

import pytest
import torch

from bench.flops import count_flops
from models.registry import build_model
from models.registry import traceable
from models.vim import (
    VIM_OP_HANDLERS,
    fused_op_flop_handler,
    fused_op_flop_terms,
)

# Vim-S: embed_dim 384, d_state 16, d_conv 4, expand 2 → d_inner 768,
# dt_rank = ceil(384/16) = 24. 224² 에 cls token 하나가 붙어 seqlen 197.
VIM_S = dict(batch=1, seqlen=197, d_inner=768, d_conv=4, dt_rank=24, d_state=16)


def test_vim_s_has_the_published_parameter_count():
    """논문 표 2 기준 26M."""
    model = build_model("vim_s", pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    assert 25e6 < params < 28e6, f"{params / 1e6:.1f}M"


def test_scan_term_matches_the_formula_in_the_vim_paper():
    """한 방향 SSM = 8LDN. Vim 논문 식 (8)."""
    terms = fused_op_flop_terms(**VIM_S)
    assert terms["selective_scan"] == 8 * 197 * 768 * 16


def test_every_fused_matmul_is_accounted_for():
    """fused op 이 삼키는 네 항이 전부 있어야 한다. 하나라도 빠지면 과소 계상이다."""
    terms = fused_op_flop_terms(**VIM_S)

    assert terms["causal_conv1d"] == 197 * 768 * 4
    assert terms["x_proj"] == 197 * 768 * (24 + 2 * 16)
    assert terms["dt_proj"] == 197 * 24 * 768
    assert terms["selective_scan"] == 8 * 197 * 768 * 16
    assert set(terms) == {"causal_conv1d", "x_proj", "dt_proj", "selective_scan"}


def test_handler_returns_the_sum_of_every_term():
    terms = fused_op_flop_terms(**VIM_S)
    flops = fused_op_flop_handler(_fake_fused_inputs(**VIM_S), outputs=None)
    assert flops == sum(terms.values())


def test_handler_rejects_an_unexpected_argument_layout():
    """상수 이름만 맞고 인자 배치가 다르면 조용히 틀린 수를 내는 대신 죽어야 한다."""
    with pytest.raises((IndexError, ValueError)):
        fused_op_flop_handler([_node([1, 2])], outputs=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Vim 커널은 CUDA 전용")
def test_vim_flops_are_not_silently_zero():
    """핸들러 없이 세면 fused op 이 통째로 사라진다. 그 차이를 확인한다."""
    model = build_model("vim_s", pretrained=False)
    with traceable("vim_s", model):
        without = count_flops(model, input_shape=(3, 224, 224), device="cuda")
        with_handler = count_flops(
            model, input_shape=(3, 224, 224), op_handlers=VIM_OP_HANDLERS, device="cuda"
        )
    assert with_handler.total > without.total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Vim 커널은 CUDA 전용")
def test_vim_reports_no_uncounted_ops_once_handlers_are_registered():
    model = build_model("vim_s", pretrained=False)
    with traceable("vim_s", model):
        result = count_flops(
            model, input_shape=(3, 224, 224), op_handlers=VIM_OP_HANDLERS, device="cuda"
        )
    assert result.unexpected_uncounted_ops == (), (
        f"미등록: {result.unexpected_uncounted_ops} (전체: {result.uncounted_ops})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Vim 커널은 CUDA 전용")
def test_analytic_and_traced_flops_are_both_present_and_separate():
    """Vim 의 수치는 절반이 공식이다. 합계 하나로 뭉뚱그리면 출처가 사라진다."""
    model = build_model("vim_s", pretrained=False)
    with traceable("vim_s", model):
        result = count_flops(
            model,
            input_shape=(3, 224, 224),
            op_handlers=VIM_OP_HANDLERS,
            device="cuda",
        )
    assert result.analytic > 0, "fused op 이 공식으로 채워지지 않았다"
    assert result.traced > 0, "패치 임베딩·헤드 같은 일반 연산이 사라졌다"
    assert result.total == result.traced + result.analytic


def _node(shape):
    """fvcore 가 핸들러에 넘기는 형태를 흉내낸다: shape 를 가진 노드."""
    return types.SimpleNamespace(
        type=lambda: types.SimpleNamespace(sizes=lambda: shape)
    )


def _fake_fused_inputs(batch, seqlen, d_inner, d_conv, dt_rank, d_state):
    """MambaInnerFnNoOutProj.forward 의 인자 배치를 그대로 흉내낸다."""
    return [
        _node([batch, 2 * d_inner, seqlen]),        # xz
        _node([d_inner, 1, d_conv]),                # conv1d_weight
        _node([d_inner]),                           # conv1d_bias
        _node([dt_rank + 2 * d_state, d_inner]),    # x_proj_weight
        _node([d_inner, dt_rank]),                  # delta_proj_weight
        _node([d_inner, d_state]),                  # A
    ]
