"""GPU 메모리 예산.

WSL2/WDDM 은 VRAM 이 모자라도 OOM 을 내지 않고 시스템 RAM 으로 넘긴다(sysmem
fallback). 그러면 OOM 이 영영 발생하지 않아 "8GB 에 무엇이 들어가는가"라는 E1 의
질문 자체가 측정 불가능해진다. 실제로 첫 실행에서 전용 VRAM 7.9GB 에 더해 공유
시스템 메모리 10GB 를 쓰면서 셀 하나가 50분 넘게 끝나지 않았다.

그래서 PyTorch 할당자에 상한을 걸어 드라이버가 넘기기 **전에** OOM 이 나게 만든다.
"""
import pytest
import torch

from bench.budget import DEFAULT_MEMORY_FRACTION, apply_memory_budget


def test_budget_is_a_fraction_of_total_device_memory(monkeypatch):
    applied = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("P", (), {"total_memory": 8 * 1024**3})(),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device=0: applied.update(fraction=fraction),
    )

    cap = apply_memory_budget(fraction=0.5)

    assert cap == 4 * 1024**3
    assert applied["fraction"] == 0.5


def test_budget_fails_when_another_process_already_took_the_memory(monkeypatch):
    """상한만 걸고 실제 여유를 확인하지 않으면, 다른 프로세스가 VRAM 을 쥐고 있을 때
    또 조용히 spill 로 샌다. 그 경우 재보다 먼저 멈춰야 한다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("P", (), {"total_memory": 8 * 1024**3})(),
    )
    # 여유가 2GiB 뿐인데 6.4GiB 를 쓰겠다고 하는 상황
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction, device=0: None
    )

    with pytest.raises(RuntimeError, match="여유"):
        apply_memory_budget(fraction=0.8)


def test_no_cuda_reports_no_budget(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert apply_memory_budget() is None


def test_default_fraction_leaves_headroom_for_context_and_desktop():
    """CUDA 컨텍스트(수백 MiB)와 데스크톱이 쓰는 VRAM 은 이 상한 밖에 있다.
    1.0 에 가깝게 잡으면 상한을 지켜도 드라이버가 넘긴다."""
    assert 0.5 <= DEFAULT_MEMORY_FRACTION <= 0.9
