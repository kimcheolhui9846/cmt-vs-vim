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
