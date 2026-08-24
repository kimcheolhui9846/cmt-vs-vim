"""가중치가 실제로 들어갔는지 확인한다.

부분 로드된 모델은 예외 없이 조용히 랜덤 모델처럼 행동한다. 그 상태로 ERF를 재면
'학습이 이방성을 만든다'는 결론이 통째로 무의미해진다.
"""
import torch

from data.voc import ensure_voc, load_images, sample_image_paths
from models.registry import MODEL_NAMES, build_model


def _natural_batch(n: int = 4) -> torch.Tensor:
    images = ensure_voc()
    return load_images(sample_image_paths(list(images.glob("*.jpg")), n, seed=0))


def _mean_entropy(model: torch.nn.Module, x: torch.Tensor) -> float:
    # Vim-S의 fused_add_norm 경로는 CUDA 전용 Triton 커널이라 CPU에서는 아예
    # 실행되지 않는다(tests/test_vim.py의 "Vim 커널은 CUDA 전용" 스킵 사유와 동일
    # 사실). 여기서는 스킵하지 않고 가능하면 CUDA로 옮겨 실제로 돌린다 — CUDA가
    # 없으면 이 아래에서 그대로 하드 실패한다.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)
    with torch.no_grad():
        probs = model(x).softmax(dim=-1)
    return float(-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean())


def test_every_model_loads_pretrained_weights():
    for name in MODEL_NAMES:
        build_model(name, pretrained=True, img_size=224)


def test_pretrained_is_more_confident_than_random_init():
    """엔트로피가 랜덤 초기화와 같으면 가중치가 안 들어간 것이다.

    1000-클래스 균등 분포의 엔트로피는 ln(1000) ≈ 6.91이다. 학습된 모델은 자연
    이미지에서 그보다 훨씬 낮아야 한다.
    """
    x = _natural_batch()
    for name in MODEL_NAMES:
        trained = _mean_entropy(build_model(name, pretrained=True).eval(), x)
        random_init = _mean_entropy(build_model(name, pretrained=False).eval(), x)
        assert trained < random_init - 1.0, f"{name}: {trained:.2f} vs {random_init:.2f}"
