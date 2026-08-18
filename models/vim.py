"""Vim-S 로더와 fused op FLOPs 회계.

fvcore는 핸들러가 없는 연산을 0으로 세고 아무 말도 하지 않는다. Vim-S가 정확히
이 경우인데, 계획이 예상한 것보다 범위가 넓다.

Vim-S는 bimamba_type="v2" + use_fast_path=True 라 블록마다
`mamba_inner_fn_no_out_proj` 를 방향당 한 번씩 부른다. 이 커스텀 autograd Function
하나가 causal conv1d · x_proj · dt_proj · selective scan 을 전부 삼킨다. 따라서
scan(8LDN)만 세는 핸들러는 나머지 세 항을 블록마다 양방향으로 잃는다.

fast path를 끄면 연산이 풀려 fvcore가 직접 셀 수 있을 것 같지만 그러면 안 된다 —
mamba_simple.py의 느린 경로는 bimamba v2를 구현하지 않아서 역방향 스캔
(conv1d_b / x_proj_b / dt_proj_b / A_b)을 통째로 건너뛴다. 단방향의 다른 모델을
재게 되고, SSM 연산량이 절반으로 줄어든다.

그래서 fast path를 그대로 두고, fused op이 삼킨 네 항을 핸들러가 전부 회계한다.
필요한 shape는 전부 fused op의 인자로 들어오므로 유도가 가능하다.

**단위 규약** — conv1d·x_proj·dt_proj는 fvcore와 같은 MAC 기준이다. selective scan
항만 Vim 논문 식 (8)의 8LDN을 그대로 쓴다(FLOP 기준). 두 규약이 섞여 있으므로
논문에 수치를 실을 때 이 사실을 명시할 것. 이 값들은 측정이 아니라 공식이라
`count_flops`가 traced와 분리해 반환한다.
"""
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn

from models import vim_official

_VIM_S_FACTORY = (
    vim_official.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
)

# fvcore가 보고하는 연산자 이름. 계획 Step 5의 절차로 실측 확인한 값이다 —
# 문자열이 틀리면 핸들러가 등록되지 않고 FLOPs가 조용히 0으로 남는다.
FUSED_OP = "prim::PythonOp.MambaInnerFnNoOutProj"
FUSED_ADD_NORM_OP = "prim::PythonOp.LayerNormFn"


def load_vim_small(pretrained: bool = False, img_size: int = 224) -> nn.Module:
    """E1은 연산 비용만 재므로 가중치가 필요 없다. models/cmt.py와 같은 원칙."""
    if pretrained:
        raise NotImplementedError(
            "Vim-S 가중치 로딩은 아직 없다. E1은 구조 비용만 재므로 필요하지 않고, "
            "체크포인트 로딩은 E2/E3 계획에서 구현한다."
        )
    return _VIM_S_FACTORY(img_size=img_size)


class _TracingRMSNorm(nn.Module):
    """mamba RMSNorm 의 순수 PyTorch 등가물. 트레이스 동안만 대신 들어간다.

    가중치 텐서는 원본과 같은 객체를 공유하므로 파라미터 수·값이 달라지지 않는다.
    """

    def __init__(self, weight: nn.Parameter, eps: float):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None or prenorm:
            raise ValueError(
                "트레이스용 RMSNorm 이 fused 경로처럼 호출됐다. fused_add_norm 이 "
                "정말 꺼졌는지 확인할 것 — 조용히 다른 경로를 재는 것이 최악이다."
            )
        dtype = x.dtype
        x = x.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype)


@contextmanager
def traceable(model: nn.Module):
    """FLOPs 트레이스 동안만 triton 정규화 커널을 우회한다.

    mamba 의 triton layer-norm 커널은 `BLOCK_N = min(..., next_power_of_2(N))` 을
    constexpr 인자로 넘기는데, torch.jit.trace 안에서는 `x.shape[-1]` 이 파이썬 int 가
    아니라 추적되는 값이 되어 BLOCK_N 이 constexpr 이 아니게 된다. 그래서 fvcore 의
    트레이스만 CompilationError 로 죽는다 — eager 실행은 멀쩡하다.

    두 가지를 함께 되돌려야 한다. fused_add_norm 을 끄는 것만으로는 부족한데,
    Vim-S 는 rms_norm=True 라 self.norm 자체가 mamba 의 RMSNorm 이고 그 forward 가
    같은 triton 커널을 부르기 때문이다.

    이렇게 재는 것이 정당한 이유는 두 가지다. (1) 두 경로의 수식이 같다 — fused
    add-norm 은 add + RMSNorm 을 커널 하나로 합친 것뿐이고, 실측 출력 차이가
    6.7e-07(fp32 잡음 수준)이다. (2) fvcore 는 정규화 연산을 규약상 세지 않으므로
    어느 경로든 FLOPs 기여가 0이다. DeiT·CMT 의 LayerNorm 도 같은 규약으로 빠지므로
    세 모델의 회계 기준이 여전히 같다.

    latency·메모리·throughput 은 반드시 원래의 fused 경로로 재야 한다. 그쪽은 커널
    융합이 곧 성능이고, 이 저장소가 검증하려는 주장이 바로 그 성능이다. 그래서
    영구히 바꾸지 않고 컨텍스트 안에서만 바꾼 뒤 되돌린다.
    """
    from mamba_ssm.ops.triton.layernorm import RMSNorm

    flag_targets = [
        module
        for module in [model, *getattr(model, "layers", [])]
        if hasattr(module, "fused_add_norm")
    ]
    if not flag_targets:
        raise ValueError(
            "fused_add_norm 속성을 가진 모듈이 없다. Vim 구현이 바뀌었는지 확인할 것 — "
            "조용히 넘어가면 트레이스가 실패하거나 다른 경로를 재게 된다."
        )

    swaps = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, RMSNorm)
    ]
    if not swaps:
        raise ValueError(
            "RMSNorm 모듈을 찾지 못했다. Vim-S 는 rms_norm=True 라 반드시 있어야 한다."
        )

    saved_flags = [module.fused_add_norm for module in flag_targets]
    for module in flag_targets:
        module.fused_add_norm = False
    for parent, name, child in swaps:
        setattr(parent, name, _TracingRMSNorm(child.weight, child.eps))
    try:
        yield model
    finally:
        for module, previous in zip(flag_targets, saved_flags):
            module.fused_add_norm = previous
        for parent, name, child in swaps:
            setattr(parent, name, child)


def fused_op_flop_terms(
    batch: int, seqlen: int, d_inner: int, d_conv: int, dt_rank: int, d_state: int
) -> dict[str, int]:
    """fused op이 삼킨 항을 하나씩 계산한다.

    합계만 내면 항 하나를 빠뜨리는 회귀가 테스트를 통과해버린다. 판단을 항별로
    드러내 테스트가 각각을 직접 확인할 수 있게 한다.
    """
    tokens = batch * seqlen
    return {
        # depthwise causal conv — 출력 원소당 d_conv MAC
        "causal_conv1d": tokens * d_inner * d_conv,
        # x_proj: (tokens, d_inner) @ (d_inner, dt_rank + 2 * d_state)
        "x_proj": tokens * d_inner * (dt_rank + 2 * d_state),
        # dt_proj: (d_inner, dt_rank) @ (dt_rank, tokens)
        "dt_proj": tokens * dt_rank * d_inner,
        # selective scan — Vim 논문 식 (8)
        "selective_scan": 8 * tokens * d_inner * d_state,
    }


def fused_op_flop_handler(inputs, outputs) -> int:
    """MambaInnerFnNoOutProj 한 번의 연산량.

    인자 배치는 mamba-1p1p1 의 MambaInnerFnNoOutProj.forward 를 따른다:
        (xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, A, ...)

    양방향이라 블록마다 두 번 불리고 fvcore가 각 호출을 세므로 여기서 2를 곱하지
    않는다. 배치가 어긋나면 조용히 틀린 수를 내는 대신 죽는다 — 이 핸들러가 조용히
    틀리는 것이 이 프로젝트에서 가장 비싼 실패다.
    """
    if len(inputs) < 6:
        raise ValueError(
            f"fused op 인자가 6개 미만이다({len(inputs)}개). mamba-1p1p1의 "
            "MambaInnerFnNoOutProj 배치가 바뀌었는지 확인할 것."
        )

    xz = inputs[0].type().sizes()
    conv1d_weight = inputs[1].type().sizes()
    x_proj_weight = inputs[3].type().sizes()
    delta_proj_weight = inputs[4].type().sizes()
    a = inputs[5].type().sizes()

    ranks = (len(xz), len(conv1d_weight), len(x_proj_weight), len(delta_proj_weight), len(a))
    if ranks != (3, 3, 2, 2, 2):
        raise ValueError(
            f"fused op 인자의 rank가 예상과 다르다: {ranks} (기대 (3, 3, 2, 2, 2)). "
            "인자 배치가 바뀌면 shape 해석이 조용히 틀어진다."
        )

    batch, two_d_inner, seqlen = xz
    return sum(
        fused_op_flop_terms(
            batch=batch,
            seqlen=seqlen,
            d_inner=two_d_inner // 2,
            d_conv=conv1d_weight[2],
            dt_rank=delta_proj_weight[1],
            d_state=a[1],
        ).values()
    )


def fused_add_norm_flop_handler(inputs, outputs) -> int:
    """fused add-norm(LayerNormFn)은 0으로 센다 — 의도된 0이다.

    fvcore는 aten::layer_norm 같은 정규화·elementwise 연산을 기본적으로 세지 않는다.
    DeiT-S와 CMT-S의 LayerNorm도 같은 규약으로 빠져 있으므로, Vim만 세면 오히려
    비교가 불공정해진다. 핸들러를 등록하는 이유는 값을 채우기 위해서가 아니라,
    미등록 연산 목록에 남아 sanity check를 막는 것을 피하면서 '왜 0인지'를 코드에
    남기기 위해서다.
    """
    return 0


VIM_OP_HANDLERS: dict[str, Callable] = {
    FUSED_OP: fused_op_flop_handler,
    FUSED_ADD_NORM_OP: fused_add_norm_flop_handler,
}
