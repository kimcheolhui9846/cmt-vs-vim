"""프로브가 cls 토큰이 아니라 중심 패치를 집는지 확인한다.

cls 토큰을 집으면 ERF가 이미지 전체에 퍼져 비등방 지수가 1.0에 수렴한다 —
'Vim은 등방'이라는, 논문을 반증하는 방향의 그럴듯한 오답이다. 아무것도 깨지지
않으므로 결과만 봐서는 알 수 없다.
"""
import torch

from models.probes import center_token_scalar
from models.registry import MODEL_NAMES, build_model

CENTER_PATCH = (112, 112)
CORNER = (8, 8)

# 각 모델의 중심 토큰이 실제로 덮는 픽셀 범위 (포함). 224²에서
#   deit_s·vim_s: 14x14 격자, 16px 패치, 중심 (7, 7) → 112~127
#   cmt_s:        7x7 격자, 32px 셀,  중심 (3, 3) → 96~127
# 112 기준 대칭 창(96~128)이 아니다 — 그 창은 DeiT·Vim에서 오답인 패치 (6, 6)을
# 통과시키면서 정답 패치의 끝을 1픽셀만 넘겨 준다. 창이 아니라 '자기 중심 패치
# 소속'을 단언한다.
CENTER_TOKEN_PIXELS = {
    "deit_s": (112, 127),
    "vim_s": (112, 127),
    "cmt_s": (96, 127),
}


def _erf_row(model_name: str) -> torch.Tensor:
    """중심 토큰 스칼라를 입력으로 미분한 gradient 크기 지도 (224, 224).

    Vim-S의 fused_add_norm 경로는 CUDA 전용 Triton 커널이라 CPU에서는 실행 자체가
    안 된다(tests/test_pretrained.py의 같은 사실). 스킵하지 않고 CUDA를 요구한다 —
    없으면 여기서 하드 실패한다.

    입력은 상수 영상이 아니라 고정 시드의 난수다. 전부 0인 영상은 값이 공간적으로
    똑같아서 퇴화한 입력이고, CMT의 depthwise conv backward가 실제로 NaN을 돌려준다
    (`ConvolutionBackward0 returned nan values`). E2가 재는 것은 VOC 자연 영상이므로
    상수 영상의 퇴화까지 맞출 이유가 없다.
    """
    assert torch.cuda.is_available(), "고정 환경에서 실행할 것 — Vim은 CUDA 전용"
    torch.manual_seed(0)
    model = build_model(model_name, pretrained=False, img_size=224).eval().cuda()
    x = torch.randn(1, 3, 224, 224, device="cuda").requires_grad_(True)
    center_token_scalar(model_name, model, x).sum().backward()
    return x.grad.abs().sum(dim=1)[0]


def test_the_probe_looks_at_the_center_not_everywhere():
    """랜덤 초기화 모델의 ERF는 좁다(Luo et al.). 중심의 gradient가 구석보다
    뚜렷하게 커야 한다. cls 토큰을 집었다면 둘이 비슷해진다."""
    for name in MODEL_NAMES:
        grad = _erf_row(name)
        center = float(grad[CENTER_PATCH[0], CENTER_PATCH[1]])
        corner = float(grad[CORNER[0], CORNER[1]])
        assert center > 10 * corner, f"{name}: 중심 {center:.3e} vs 구석 {corner:.3e}"


def test_the_peak_sits_inside_the_models_own_center_token():
    """peak가 그 모델의 중심 토큰이 덮는 픽셀 안에 있어야 한다.

    모델마다 격자가 달라서 덮는 범위도 다르다. 112 기준 대칭 창으로 뭉뚱그리면
    DeiT·Vim에서 패치 하나만큼 어긋난 창이 되어, 옆 패치를 집는 오답을 통과시킨다.
    """
    for name in MODEL_NAMES:
        grad = _erf_row(name)
        row, col = divmod(int(grad.argmax()), grad.shape[1])
        low, high = CENTER_TOKEN_PIXELS[name]
        assert low <= row <= high and low <= col <= high, (
            f"{name}: peak ({row}, {col})가 중심 토큰 범위 {low}~{high} 밖이다"
        )


def test_vim_center_index_skips_the_inserted_cls_token():
    """Vim은 cls를 M//2=98에 끼워 넣는다. 시퀀스의 '가운데'가 곧 cls다."""
    from models.probes import vim_center_sequence_index

    assert vim_center_sequence_index(num_patches=196) == 106
