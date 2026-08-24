import pytest
import torch

from models.registry import build_model


def test_cmt_s_has_the_published_parameter_count():
    """논문 표 2 기준 25M."""
    model = build_model("cmt_s", pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    assert 24e6 < params < 27e6, f"{params / 1e6:.1f}M"


def test_cmt_s_runs_at_224():
    model = build_model("cmt_s", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 1000)


def test_cmt_s_runs_at_384():
    """해상도 sweep의 전제. 384²가 안 되면 E1이 성립하지 않는다."""
    model = build_model("cmt_s", pretrained=False, img_size=384).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 384, 384))
    assert out.shape == (1, 1000)


def test_cmt_s_at_384_actually_rebuilds_the_stage_grids():
    """출력 shape만 보면 img_size를 통째로 무시해도 통과한다 — 클래스 수는
    해상도와 무관하기 때문이다. stage별 격자가 실제로 커졌는지 직접 확인한다."""
    small = build_model("cmt_s", pretrained=False, img_size=224)
    large = build_model("cmt_s", pretrained=False, img_size=384)

    def first_stage_patches(model):
        for module in model.modules():
            if hasattr(module, "num_patches"):
                return module.num_patches
        raise AssertionError("num_patches를 가진 모듈이 없다 — 구조 가정이 틀렸다")

    assert first_stage_patches(large) > first_stage_patches(small)


def test_pretrained_weights_are_refused_at_a_resolution_other_than_224():
    """"224²만" 제약의 유일한 집행 장치다. CMT-S 공개 가중치는 224²에서 학습됐고,
    다른 해상도는 상대 위치 bias 보간이 필요하다 — 보간한 가중치로 잰 값은 공개
    정확도와 대응하지 않는다. 가드가 사라지면 예외 없이 잘못된 가중치로 측정이
    진행되고 결과는 그럴듯해 보인다. 체크포인트를 받기 전에 터지므로 이 테스트는
    네트워크를 건드리지 않는다."""
    with pytest.raises(ValueError, match="224"):
        build_model("cmt_s", pretrained=True, img_size=384)
