"""GPU 메모리 예산을 고정한다.

WSL2/WDDM 은 VRAM 이 모자라도 OOM 을 내지 않고 시스템 RAM 으로 넘긴다(sysmem
fallback). 넘어간 뒤에도 실행은 계속되지만 PCIe 를 타므로 10~50배 느려지고, 무엇보다
**OOM 이 영영 발생하지 않는다.** 그러면 `find_max_batch` 는 실제 한계를 만나지 못한 채
상한까지 올라가고, "8GB 에서 어느 모델이 어느 해상도에 OOM 나는가"라는 E1 의 질문이
측정 불가능해진다.

첫 실행에서 실제로 겪었다 — 전용 VRAM 7.9GB 에 더해 공유 시스템 메모리 10GB 를 쓰면서
`deit_s@512²` 한 셀이 50분 넘게 끝나지 않았고, 그 앞의 384² 셀은 처리량이 25배
떨어져 있었다(그때 이미 새고 있었다).

PyTorch 할당자에 상한을 걸면 드라이버가 넘기기 전에 할당자가 먼저 OOM 을 낸다.
그러면 `bench.memory.is_oom` 이 이미 처리하는 정상 경로로 들어오고, OOM 은 즉시
발생하므로 실행 시간도 예측 가능해진다.
"""
import torch

# 전체 VRAM 대비 비율. 1.0 에 가깝게 잡으면 안 된다 — CUDA 컨텍스트(수백 MiB)와
# 데스크톱이 쓰는 VRAM 은 이 상한 밖에 있어서, 상한을 지켜도 드라이버가 넘긴다.
DEFAULT_MEMORY_FRACTION = 0.80


def apply_memory_budget(
    fraction: float = DEFAULT_MEMORY_FRACTION, device: int = 0
) -> int | None:
    """할당자 상한을 걸고 그 바이트 수를 돌려준다. CUDA 가 없으면 None.

    상한만 걸고 끝내면 안 된다. 다른 프로세스가 이미 VRAM 을 쥐고 있으면 상한을
    지켜도 물리 메모리가 모자라 또 spill 로 샌다. 그건 조용히 일어나므로, 재기
    전에 실제 여유를 확인하고 모자라면 멈춘다.
    """
    if not torch.cuda.is_available():
        return None

    total = torch.cuda.get_device_properties(device).total_memory
    cap = int(total * fraction)

    free, _ = torch.cuda.mem_get_info(device)
    if free < cap:
        raise RuntimeError(
            f"GPU 여유 메모리가 예산보다 적다: 여유 {free / 1024**2:.0f} MiB < "
            f"예산 {cap / 1024**2:.0f} MiB. 다른 프로세스가 VRAM 을 쓰고 있으면 "
            "그만큼 시스템 메모리로 새면서 측정이 무효가 된다. GPU 를 비우고 다시 실행할 것."
        )

    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return cap
