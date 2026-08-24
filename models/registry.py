"""비교 대상 세 모델의 단일 진입점.

파라미터가 22~26M로 이미 정렬되어 있어 별도 조정 없이 통제가 성립한다.
"""
from contextlib import contextmanager

import torch.nn as nn

MODEL_NAMES = ("deit_s", "cmt_s", "vim_s")


def build_model(name: str, pretrained: bool = True, img_size: int = 224) -> nn.Module:
    if name not in MODEL_NAMES:
        raise ValueError(
            f"알 수 없는 모델 '{name}'. 사용 가능: {', '.join(MODEL_NAMES)}"
        )
    if name == "deit_s":
        return _build_deit_s(pretrained=pretrained, img_size=img_size)
    if name == "cmt_s":
        from models.cmt import load_cmt_small

        return load_cmt_small(pretrained=pretrained, img_size=img_size)
    if name == "vim_s":
        from models.vim import load_vim_small

        return load_vim_small(pretrained=pretrained, img_size=img_size)
    raise NotImplementedError(f"'{name}'은 이후 태스크에서 붙인다")


def _build_deit_s(pretrained: bool, img_size: int) -> nn.Module:
    import timm

    return timm.create_model(
        "deit_small_patch16_224",
        pretrained=pretrained,
        img_size=img_size,
    )


@contextmanager
def traceable(name: str, model: nn.Module):
    """FLOPs를 셀 동안만 융합 커널을 풀어 놓는다.

    융합 커널은 fvcore의 트레이서에게 불투명하다. 안에 든 matmul이 그래프에 나타나지
    않으므로 핸들러가 없으면 통째로 0이 된다. 이 저장소에서 실제로 두 번 일어났다 —
    DeiT는 timm의 fused SDPA가 attention matmul을 삼켜 4.25G로 측정됐고(공개값 4.6G),
    Vim은 fused op이 conv1d·x_proj·dt_proj·scan을 전부 삼켰다.

    푸는 편이 정직한 이유는 융합 여부가 **연산량을 바꾸지 않기** 때문이다. 같은 수식을
    같은 횟수로 계산하고 커널 경계만 다르다. 반대로 latency·메모리·throughput은 절대
    이 컨텍스트 안에서 재면 안 된다 — 거기서는 커널 융합이 곧 성능이고, 그 성능이
    이 저장소가 검증하려는 주장이다.
    """
    if name == "vim_s":
        from models.vim import traceable as vim_traceable

        with vim_traceable(model):
            yield model
        return

    if name == "deit_s":
        attentions = [block.attn for block in model.blocks]
        saved = [attention.fused_attn for attention in attentions]
        for attention in attentions:
            attention.fused_attn = False
        try:
            yield model
        finally:
            for attention, previous in zip(attentions, saved):
                attention.fused_attn = previous
        return

    yield model
