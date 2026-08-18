"""CMT-S 로더.

E1은 FLOPs·latency·메모리만 재고 셋 다 가중치와 무관하므로, 여기서는 구조만
세운다. 사전학습 가중치 로딩과 상대 위치 bias 보간은 그것이 실제로 필요한
E2(ERF)·E3(dilution) 계획에서 구현한다.
"""
import torch.nn as nn

from models.cmt_official import cmt_s as _cmt_s_official


def load_cmt_small(pretrained: bool = False, img_size: int = 224) -> nn.Module:
    if pretrained:
        raise NotImplementedError(
            "CMT-S 가중치 로딩은 아직 없다. E1은 구조 비용만 재므로 필요하지 않고, "
            "로딩과 상대 위치 bias 보간은 E2/E3 계획에서 구현한다."
        )
    return _cmt_s_official(img_size=img_size)
