"""최종 특징맵의 중심 토큰을 집는다.

이 파일이 틀리면 E2 전체가 조용히 틀린다. cls 토큰을 집으면 ERF가 전역으로 퍼져
'완벽한 등방'이라는 그럴듯한 오답이 나오고, 예외도 경고도 나지 않는다.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn

# 어느 모듈의 출력을 최종 특징맵으로 볼지. 추측이 아니라 세 모델의 모든 모듈에
# forward hook을 걸어 실측한 이름이다(계획 Step 1). 세 개 중 둘이 계획의 추측과
# 달랐다.
#
#   deit_s: 'blocks' → (1, 197, 384)   cls가 맨 앞. 마지막 LayerNorm('norm')이
#           아니라 그 **직전** residual stream이다. 이유는 random_init 조건이다.
#           LayerNorm 출력은 채널 평균이 0이라 채널 합이 Σ_c γ_c·x̂_c + Σ_c β_c 인데,
#           timm의 초기값이 γ=1·β=0이라 이 합이 해석적으로 정확히 0이 된다.
#           **사전학습 가중치에서는 성립하지 않는다** — 실측 γ는 std 0.119, 범위
#           0.118~1.810이고 β도 0이 아니라, natural·noise 조건이었다면 'norm'도
#           그냥 동작했을 것이다. 걸리는 것은 pretrained=False로 만드는 random_init
#           조건이고, 그건 E2의 세 조건 중 하나라 버릴 수 없다. 그 조건에서 'norm'을
#           집으면 VOC 16장 중 11장은 gradient가 **정확히 0**이고 나머지 5장은 peak가
#           1e-8 수준의 반올림 잡음인데, accumulate_erf가 이미지마다 peak로 나누므로
#           그 잡음이 1.0으로 증폭된다. 결과는 예외도 NaN도 아닌 그럴듯한 지도이고
#           비등방 지수 1.0696이 나온다 — 반올림에서 뽑아낸 숫자다. 그래서 세 조건에
#           걸쳐 하나로 쓸 수 있는 'blocks'를 집는다.
#
#           **둘은 등가물이 아니다.** 스칼라가 토큰 i에만 의존하므로 gradient가 닿는
#           support는 같지만, 잔차 스트림에 실리는 채널 가중이 다르다
#           (∂/∂h Σ_c LN(h)_c = J_LN^T·1 ≠ 1). 비등방 지수는 support가 아니라 크기
#           맵의 2차 모먼트이므로 support가 같다고 지수가 같아지지는 않는다. 사전학습
#           가중치로 실제로 재면 가깝기는 하다 — 비등방 지수 natural 1.0208 vs
#           1.0371(상대차 1.57%), noise 1.0361 vs 1.0375(0.14%), 맵 상관 0.91·0.98 —
#           이지만 이는 '같다'가 아니라 '이 선택이 결론을 뒤집지 않는다'는 뜻이다.
#   vim_s:  'norm_f' → (1, 197, 384)   cls가 98번에 끼어 있다. RMSNorm은 평균을
#           빼지 않으므로 DeiT 같은 상쇄가 없다.
#   cmt_s:  '_swish' → (1, 1280, 7, 7) CMT에는 'norm'이라는 모듈이 아예 없다.
#           _fc(1x1 conv)·_bn·_swish는 공간을 섞지 않으므로 이 7x7 격자는
#           마지막 stage 토큰(1, 49, 512)과 그대로 정렬된다.
FEATURE_MODULE = {
    "deit_s": "blocks",
    "vim_s": "norm_f",
    "cmt_s": "_swish",
}


def vim_center_sequence_index(num_patches: int) -> int:
    """Vim의 시퀀스에서 중심 패치가 있는 위치.

    Vim은 cls 토큰을 M//2에 '끼워 넣는다'(vim_official.py:427). 중심 패치는
    격자 (g//2, g//2)이고, 그 인덱스가 삽입 위치보다 뒤이므로 1칸 밀린다.

    M=196이면 cls가 98번이고 중심 패치는 105번 → 106번이다. 시퀀스 길이 197의
    '가운데'인 98을 집으면 정확히 cls 토큰이다 — 실제로 forward_features가
    돌려주는 값이 hidden_states[:, 98]임을 확인했다.
    """
    grid = int(num_patches ** 0.5)
    patch_index = (grid // 2) * grid + (grid // 2)
    cls_position = num_patches // 2
    return patch_index + 1 if patch_index >= cls_position else patch_index


@contextmanager
def _module_final_norm(model: nn.Module):
    """캡처하는 동안만 Vim의 마지막 norm을 모듈 호출로 되돌린다.

    Vim은 fused_add_norm=True라 마지막 정규화를 `self.norm_f(...)`가 아니라
    `rms_norm_fn(hidden, self.norm_f.weight, ...)`로 부른다. 그래서 norm_f에 건
    forward hook이 **한 번도 불리지 않는다** — 훅만 걸어 두고 안심하면 캡처가
    조용히 비는 자리다.

    끄면 같은 수식이 모듈 경로로 흐른다. 두 경로의 로짓 차이가 실측 0.0이었으므로
    측정 대상이 달라지지 않는다. 그래도 영구히 바꾸지는 않는다 — latency·메모리는
    반드시 융합 경로로 재야 하기 때문이다(models/vim.py의 `traceable`과 같은 이유).

    fused_add_norm 속성이 없는 모델(DeiT·CMT)에서는 아무것도 하지 않는다.
    """
    if not getattr(model, "fused_add_norm", False):
        yield
        return
    model.fused_add_norm = False
    try:
        yield
    finally:
        model.fused_add_norm = True


def _capture(model: nn.Module, module_name: str, x: torch.Tensor) -> torch.Tensor:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"'{module_name}' 모듈이 없다. 모델 구현이 바뀌었는지 확인할 것")

    captured = {}

    def record(_module, _inputs, output):
        captured.setdefault("out", output)

    handle = modules[module_name].register_forward_hook(record)
    try:
        with _module_final_norm(model):
            model(x)
    finally:
        handle.remove()
    if "out" not in captured:
        raise RuntimeError(f"'{module_name}'이 forward에서 불리지 않았다")
    return captured["out"]


def center_token_scalar(
    model_name: str, model: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """최종 특징맵 중심 토큰의 채널 합. 배치별 스칼라 (B,)."""
    features = _capture(model, FEATURE_MODULE[model_name], x)

    if features.dim() == 4:  # (B, C, H, W) — CMT
        _, _, height, width = features.shape
        return features[:, :, height // 2, width // 2].sum(dim=1)

    _, tokens, _ = features.shape
    if model_name == "vim_s":
        index = vim_center_sequence_index(num_patches=tokens - 1)
    elif model_name == "deit_s":
        grid = int((tokens - 1) ** 0.5)
        index = 1 + (grid // 2) * grid + (grid // 2)  # cls가 맨 앞
    else:
        raise ValueError(
            f"'{model_name}'의 토큰 배치를 모른다. cls 토큰이 어디 있는지 확인하지 "
            "않고 시퀀스의 가운데를 집으면 정확히 이 파일이 막으려는 오답이 나온다."
        )
    return features[:, index, :].sum(dim=1)
