"""모델을 만드는 단일 진입점. 두 실험군이 여기를 지난다.

E1~E3의 세 모델(`MODEL_NAMES`)은 공표 설정 그대로이며 파라미터가 22~26M로 이미
정렬되어 있어 별도 조정 없이 통제가 성립한다.

E4의 네 칸(`E4_CELLS`)은 다르다. 스톡 설정이 A 5.43M / B 6.86M / C 10.55M로 두 배
가까이 벌어져 있어, `experiments/e4_widths.py`의 명시적 탐색으로 폭과 depth를 골라
6.86M +-5%에 맞춘 조정 모델이다. 그래서 이 칸들의 폭은 반드시 configs에서 들어온다.
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


E4_CELLS = ("a_deit_ti", "b_vim_ti", "c_cmt_ti", "d_hvim")


def build_e4_model(cell: str, cfg: dict, num_classes: int = 200,
                   img_size: int = 64, drop_path: float = 0.1) -> nn.Module:
    """E4의 네 칸을 만드는 단일 진입점.

    폭은 cfg로만 들어온다 — 코드에 기본값을 박아 두면 configs와 조용히 어긋난다.

    drop_path는 네 칸에 반드시 같은 값으로 걸려야 한다. 레시피가 한 칸만 달라지면
    그 차이가 요인 효과에 그대로 섞이고, 어느 결과에도 흔적이 남지 않는다. 인자
    이름이 프레임워크마다 다르므로(timm·VisionMamba는 drop_path_rate, CMT도
    drop_path_rate — CMT의 dp는 stochastic depth가 아니라 classifier dropout이다)
    여기서 한 번에 번역한다.

    같은 이유로 classifier dropout(CMT의 `dp`)도 네 칸에 0.0으로 맞춘다. C·D에만
    dp=0.1을 주면 상호작용 항 (D-B)-(C-A)에서는 상쇄되지만 구조 주효과
    (C+D)/2 - (A+B)/2 — 사전 등록 예측 1번 — 에는 그대로 남는다. A는 timm ViT의
    drop_rate 기본값 0, B는 VisionMamba의 drop_rate 기본값 0이라 head 앞에 dropout이
    없으므로, 네 칸을 맞추는 방향은 0.0이다. dropout에는 파라미터가 없으므로 6.86M
    예산 정렬은 이 값에 영향을 받지 않는다(tests/test_e4_widths.py가 확인한다).
    """
    if cell not in E4_CELLS:
        raise ValueError(
            f"알 수 없는 칸 '{cell}'. 사용 가능: {', '.join(E4_CELLS)}"
        )

    if cell == "a_deit_ti":
        import timm

        return timm.create_model(
            "deit_tiny_patch16_224", pretrained=False, img_size=img_size,
            patch_size=cfg["patch_size"], embed_dim=cfg["embed_dim"],
            depth=cfg["depth"], num_classes=num_classes,
            drop_path_rate=drop_path,
        )

    if cell == "b_vim_ti":
        from models import vim_official

        return vim_official.VisionMamba(
            img_size=img_size, patch_size=cfg["patch_size"],
            stride=cfg["patch_size"], embed_dim=cfg["embed_dim"],
            depth=cfg["depth"], rms_norm=True, residual_in_fp32=True,
            fused_add_norm=True, final_pool_type="mean", if_abs_pos_embed=True,
            if_rope=False, if_rope_residual=False, bimamba_type="v2",
            if_cls_token=True, if_divide_out=True, use_middle_cls_token=True,
            num_classes=num_classes, drop_path_rate=drop_path,
        )

    if cell == "c_cmt_ti":
        from models import cmt_official

        return cmt_official.CMT(
            img_size=img_size, num_classes=num_classes,
            embed_dims=list(cfg["embed_dims"]), stem_channel=cfg["stem_channel"],
            num_heads=[1, 2, 4, 8], depths=list(cfg["depths"]),
            mlp_ratios=[3.6] * 4, qkv_bias=True, qk_ratio=1,
            sr_ratios=[8, 4, 2, 1], drop_path_rate=drop_path, dp=0.0,
        )

    from models.hvim import build_hvim

    return build_hvim(
        img_size=img_size, num_classes=num_classes,
        embed_dims=tuple(cfg["embed_dims"]), depths=tuple(cfg["depths"]),
        stem_channel=cfg["stem_channel"], drop_path_rate=drop_path, dp=0.0,
    )
