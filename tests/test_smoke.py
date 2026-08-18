"""고정 실행 환경이 실제로 서 있는지 확인하는 관문.

여기가 깨진 상태로 진행하면 이후 측정이 전부 무의미하다. 그래서 skip 하지 않고
실패하게 뒀다 — 고정 환경 밖에서 이 파일이 통과하면 그게 더 나쁜 신호다.
"""
import pytest
import torch


def test_cuda_is_available():
    assert torch.cuda.is_available(), "CUDA를 못 찾음 — WSL2 GPU 패스스루 확인"


def test_selective_scan_kernel_imports():
    """Vim의 CUDA 커널이 빌드되었는지. 순수 PyTorch 대체는 허용하지 않는다."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    assert selective_scan_fn is not None


def test_selective_scan_runs_on_gpu():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, dim, seqlen, state = 1, 16, 64, 8
    u = torch.randn(batch, dim, seqlen, device="cuda")
    delta = torch.rand(batch, dim, seqlen, device="cuda")
    A = -torch.rand(dim, state, device="cuda")
    B = torch.randn(batch, state, seqlen, device="cuda")
    C = torch.randn(batch, state, seqlen, device="cuda")

    out = selective_scan_fn(u, delta, A, B, C)
    assert out.shape == (batch, dim, seqlen)
    assert torch.isfinite(out).all()
