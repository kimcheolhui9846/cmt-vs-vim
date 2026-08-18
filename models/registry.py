"""비교 대상 세 모델의 단일 진입점.

파라미터가 22~26M로 이미 정렬되어 있어 별도 조정 없이 통제가 성립한다.
"""
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
    raise NotImplementedError(f"'{name}'은 이후 태스크에서 붙인다")


def _build_deit_s(pretrained: bool, img_size: int) -> nn.Module:
    import timm

    return timm.create_model(
        "deit_small_patch16_224",
        pretrained=pretrained,
        img_size=img_size,
    )
