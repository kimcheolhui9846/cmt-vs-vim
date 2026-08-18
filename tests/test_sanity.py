"""공개된 값과 대조해 측정 하네스 자체를 검증한다.

이 테스트가 실패하면 계측이 틀린 것이고, 그 상태로 잰 1024² 수치는 논문에 쓸 수 없다.

test_smoke.py 와 마찬가지로 CUDA가 없으면 skip 하지 않고 실패한다. 이건 실측 직전의
관문이지 이식성 있는 단위 테스트가 아니다 — 고정 환경 밖에서 조용히 통과하면 관문이
아니게 된다.
"""
import pytest
import torch

from bench.flops import count_flops
from models.registry import build_model, traceable

DEIT_S_PUBLISHED_FLOPS = 4.6e9  # DeiT 논문 보고값, 224²


def _handlers_and_device(name: str):
    """Vim만 CUDA에서 센다 — 커널이 CUDA 전용이라 CPU 트레이스가 불가능하다."""
    if name == "vim_s":
        from models.vim import VIM_OP_HANDLERS

        return VIM_OP_HANDLERS, "cuda"
    return {}, "cpu"


def test_deit_s_flops_match_the_published_value_within_5_percent():
    model = build_model("deit_s", pretrained=False)
    with traceable("deit_s", model):
        result = count_flops(model, (3, 224, 224))
    ratio = result.total / DEIT_S_PUBLISHED_FLOPS
    assert 0.95 < ratio < 1.05, (
        f"측정 {result.total / 1e9:.2f}G vs 공개값 4.6G (비율 {ratio:.3f}). "
        f"미등록 연산: {result.uncounted_ops}"
    )


def test_deit_s_flops_are_all_traced_not_analytic():
    """DeiT에는 핸들러가 붙지 않는다. analytic이 0이 아니면 회계가 새고 있다.

    DeiT 값이 전부 측정이라는 사실이 이 하네스의 기준점이다 — 공개값과 대조할 수
    있는 유일한 모델이고, 그래서 4.6G 대조가 하네스 전체를 검증한다.
    """
    model = build_model("deit_s", pretrained=False)
    with traceable("deit_s", model):
        result = count_flops(model, (3, 224, 224))
    assert result.analytic == 0


def test_fused_attention_hides_flops_unless_the_trace_context_unfuses_it():
    """이 컨텍스트가 막는다고 주장하는 회귀를 실제로 재현한다.

    timm의 fused SDPA는 attention matmul을 fvcore에게 감춘다. 컨텍스트 없이 세면
    공개값보다 확실히 작아야 한다 — 작아지지 않는다면 컨텍스트는 아무것도 하고 있지
    않은 것이고, 그러면 이 테스트는 이름값을 못 한다.
    """
    model = build_model("deit_s", pretrained=False)

    fused = count_flops(model, (3, 224, 224))
    with traceable("deit_s", model):
        unfused = count_flops(model, (3, 224, 224))

    assert fused.total < unfused.total
    assert "aten::scaled_dot_product_attention" in fused.uncounted_ops
    assert fused.unexpected_uncounted_ops != ()
    assert unfused.unexpected_uncounted_ops == ()


def test_trace_context_restores_the_fused_path():
    """latency는 반드시 원래의 fused 경로로 재야 한다. 되돌리지 않으면 조용히 느려진다."""
    model = build_model("deit_s", pretrained=False)
    before = [block.attn.fused_attn for block in model.blocks]
    with traceable("deit_s", model):
        pass
    assert [block.attn.fused_attn for block in model.blocks] == before


@pytest.mark.parametrize("name", ["deit_s", "cmt_s", "vim_s"])
def test_no_model_has_uncounted_ops_at_224(name):
    """미등록 연산이 남아 있으면 그 모델의 FLOPs는 과소 계상된다."""
    handlers, device = _handlers_and_device(name)
    if device == "cuda":
        assert torch.cuda.is_available(), "고정 환경에서 실행할 것 — Vim은 CUDA 전용"

    model = build_model(name, pretrained=False)
    with traceable(name, model):
        result = count_flops(
            model, (3, 224, 224), op_handlers=handlers, device=device
        )

    # 규약상 0인 elementwise·정규화 연산은 세 모델에 똑같이 빠지므로 허용한다.
    # 연산량을 실제로 지닌 연산이 남으면 그 모델의 FLOPs는 과소 계상된 값이다.
    assert result.unexpected_uncounted_ops == (), (
        f"{name} 미등록: {result.unexpected_uncounted_ops} "
        f"(전체 미등록: {result.uncounted_ops})"
    )


def test_vim_fused_op_is_too_large_a_share_to_lose_silently():
    """Vim FLOPs의 상당 부분이 fused op에서 온다는 사실 자체를 고정한다.

    핸들러가 사라지거나 연산자 이름이 어긋나면 이 비중이 통째로 0이 된다. 그때
    남는 수치는 '조금 틀린 값'이 아니라 '다른 모델의 값'이다. 공개된 Vim-S FLOPs
    보고값을 오프라인에서 확인할 수 없어 절대값 대조 대신 이 구조적 성질을 건다.
    """
    assert torch.cuda.is_available(), "고정 환경에서 실행할 것 — Vim은 CUDA 전용"
    from models.vim import VIM_OP_HANDLERS

    model = build_model("vim_s", pretrained=False)
    with traceable("vim_s", model):
        without = count_flops(model, (3, 224, 224), device="cuda")
        with_handlers = count_flops(
            model, (3, 224, 224), op_handlers=VIM_OP_HANDLERS, device="cuda"
        )

    assert without.total < with_handlers.total
    share = with_handlers.analytic / with_handlers.total
    assert share > 0.25, (
        f"fused op이 전체의 {share:.1%}뿐이다. 핸들러가 붙는 연산자가 바뀌었는지 "
        "확인할 것 — 이 값이 0에 가까우면 핸들러가 등록되지 않은 것이다."
    )
