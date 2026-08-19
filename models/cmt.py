"""CMT-S 로더.

224²에서는 공개 가중치를 그대로 로드한다(E2). 다른 해상도에서 필요한 상대 위치
bias 보간은 아직 없다 — 그건 실제로 그 해상도를 재는 E3(dilution) 계획에서
구현한다.
"""
import torch.nn as nn

from models.cmt_official import cmt_s as _cmt_s_official


def load_cmt_small(pretrained: bool = False, img_size: int = 224) -> nn.Module:
    model = _cmt_s_official(img_size=img_size)
    if not pretrained:
        return model

    if img_size != 224:
        raise ValueError(
            f"CMT-S 공개 가중치는 224²로 학습됐다. img_size={img_size}는 상대 위치 "
            "bias 보간이 필요하고, 보간한 가중치로 잰 값은 공개 정확도와 대응하지 "
            "않는다. E2는 224²만 쓴다."
        )

    import torch

    from models.checkpoints import fetch, unwrap_state_dict

    state = unwrap_state_dict(torch.load(fetch("cmt_s"), map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model
