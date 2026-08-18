# E1 해상도 sweep 실측 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DeiT-S / CMT-S / Vim-S를 224²~1024²에서 실측해, 논문 표 1의 추정 FLOPs를 재현 가능한 실측값으로 대체한다.

**Architecture:** 측정 로직(`bench/`)과 모델 로딩(`models/`)을 분리하고, 실험 스크립트(`experiments/`)는 둘을 조합해 CSV만 쓴다. 그림(`figures/`)은 CSV만 읽는다. 측정 모듈은 GPU 없이도 토이 모델로 검증 가능하게 설계해, 하네스 자체의 정확성을 실제 실험 전에 확인한다.

**Tech Stack:** Python 3.10.13, PyTorch 2.1.1+cu118, fvcore, timm, mamba-1p1p1, causal_conv1d, pytest, matplotlib

## Global Constraints

- **실측 실행 환경(고정)**: Python 3.10.13, torch 2.1.1+cu118, `causal_conv1d>=1.1.0`, `mamba-1p1p1`, WSL2. `results/`에 커밋되는 모든 수치는 이 환경에서 나와야 한다. Vim 커널이 버전에 민감하고, selective scan이 Linux를 요구한다.
- **개발·단위테스트 환경(무관)**: `bench/`와 `figures/`의 단위 테스트는 토이 모델과 CPU만 쓰므로 버전에 의존하지 않는다. WSL2 준비 전까지 Windows(Python 3.12, torch 2.6)에서 개발·검증해도 되지만, **그 환경에서 나온 측정값을 `results/`에 커밋하지 않는다.** Task 1의 스모크 테스트가 고정 환경 진입 관문이며, 통과 후 전체 테스트를 고정 환경에서 한 번 더 돌린다.
- 순수 PyTorch selective scan 대체 금지. 5~10배 느려져 latency 측정이 무의미해진다.
- 정밀도는 **fp32 고정**. AMP는 부록 전용이며 본 실험에서 섞지 않는다.
- latency는 **batch=1, 워밍업 50회 후 100회 측정의 중앙값**.
- 측정 대상 해상도: **224², 384², 512², 768², 1024²**.
- 측정값은 `results/`에 원시 csv/json으로 커밋한다. 그림과 표는 반드시 그 파일을 읽어 생성한다. 손으로 옮겨 적은 숫자를 산출물에 넣지 않는다.
- OOM은 실패가 아니라 기록할 결과다. 어느 모델이 어느 해상도에서 넘어갔는지 남긴다.
- 모든 결과 파일에 환경 스냅샷(GPU·드라이버·torch·CUDA·git commit)을 포함한다.

---

### Task 1: 툴체인 고정과 스모크 테스트

WSL2와 CUDA 커널이 실제로 동작하는지부터 확인한다. 이게 깨진 상태로 진행하면 이후 모든 측정이 무의미하다.

**Files:**
- Create: `requirements.txt`
- Create: `tests/test_smoke.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: 없음
- Produces: 검증된 실행 환경. 이후 모든 태스크가 전제한다.

- [ ] **Step 1: requirements.txt 작성**

```
torch==2.1.1+cu118
torchvision==0.16.1+cu118
--extra-index-url https://download.pytorch.org/whl/cu118
timm==0.9.12
fvcore==0.1.5.post20221221
causal_conv1d>=1.1.0
mamba-ssm==1.1.1
matplotlib==3.8.2
pandas==2.1.4
pytest==7.4.4
```

- [ ] **Step 2: 실패하는 스모크 테스트 작성**

```python
# tests/test_smoke.py
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
```

- [ ] **Step 3: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mamba_ssm'`

- [ ] **Step 4: 의존성 설치**

WSL2 Ubuntu 셸에서:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`causal_conv1d`와 `mamba-ssm`은 소스 빌드라 수 분 걸린다. 빌드가 실패하면 `nvcc --version`이 11.8인지 먼저 확인한다.

- [ ] **Step 5: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS 3건

빌드가 끝내 실패하면 공식 Docker 이미지로 전환한다(스펙의 리스크 대응). 순수 PyTorch 대체는 선택지가 아니다.

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt tests/
git commit -m "chore: pin the toolchain and prove the Vim CUDA kernel runs"
```

---

### Task 2: 환경 스냅샷

모든 결과 파일에 붙일 재현 정보를 만든다.

**Files:**
- Create: `bench/__init__.py`
- Create: `bench/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: 없음
- Produces: `bench.env.snapshot() -> dict[str, str | None]` — 키는 `python`, `torch`, `cuda`, `gpu`, `driver`, `git_commit`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_env.py
from bench.env import snapshot

REQUIRED_KEYS = {"python", "torch", "cuda", "gpu", "driver", "git_commit"}


def test_snapshot_has_all_required_keys():
    assert set(snapshot().keys()) == REQUIRED_KEYS


def test_snapshot_values_are_str_or_none():
    for key, value in snapshot().items():
        assert value is None or isinstance(value, str), f"{key}가 {type(value)}"


def test_git_commit_is_a_sha():
    commit = snapshot()["git_commit"]
    assert commit is not None
    assert len(commit) == 40
    int(commit, 16)  # 16진수가 아니면 여기서 터진다
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench'`

- [ ] **Step 3: 구현**

```python
# bench/env.py
"""결과 파일에 붙일 환경 정보. 재현이 안 되는 수치는 논문에 쓸 수 없다."""
import platform
import subprocess

import torch


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def snapshot() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
    }
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_env.py -v`
Expected: PASS 3건

- [ ] **Step 5: 커밋**

```bash
git add bench/ tests/test_env.py
git commit -m "feat: capture the environment every result file has to carry"
```

---

### Task 3: FLOPs 계측

fvcore는 **등록되지 않은 연산을 조용히 0으로 센다.** Vim의 selective scan이 여기 해당하므로, 커스텀 핸들러를 주입할 수 있는 형태로 만들고 미등록 연산 목록을 함께 반환해 침묵을 막는다.

**Files:**
- Create: `bench/flops.py`
- Test: `tests/test_flops.py`

**Interfaces:**
- Consumes: 없음
- Produces: `bench.flops.count_flops(model, input_shape, op_handlers=None) -> FlopResult`
  - `FlopResult`는 `traced: int`, `uncounted_ops: tuple[str, ...]` 필드를 가진 dataclass
  - `input_shape`는 배치를 뺀 `(C, H, W)`
  - `op_handlers`는 `{연산자_이름: 핸들러_함수}` 매핑

- [ ] **Step 1: 실패하는 테스트 작성**

`nn.Linear(10, 20)`의 MAC 수는 10×20 = 200으로 해석적으로 알 수 있다. 하네스가 이 값을 맞히지 못하면 실제 모델 수치도 믿을 수 없다.

```python
# tests/test_flops.py
import torch
import torch.nn as nn

from bench.flops import count_flops


class TinyLinear(nn.Module):
    """Linear(10, 20) 한 겹. MAC = 10 * 20 = 200."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc(x)


def test_counts_a_known_linear_exactly():
    result = count_flops(TinyLinear(), input_shape=(10,))
    assert result.traced == 200


def test_reports_nothing_uncounted_for_supported_ops():
    result = count_flops(TinyLinear(), input_shape=(10,))
    assert result.uncounted_ops == ()


class CustomOpModule(nn.Module):
    """fvcore가 모르는 연산을 부르는 모듈."""

    def forward(self, x):
        return torch._C._nn.gelu(x) * x.sum()


def test_uncounted_ops_are_surfaced_not_silently_zero():
    """미등록 연산이 조용히 0으로 세어지면 Vim FLOPs가 통째로 사라진다."""
    result = count_flops(CustomOpModule(), input_shape=(10,))
    assert result.uncounted_ops != (), "미등록 연산이 보고되지 않았다"
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_flops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.flops'`

- [ ] **Step 3: 구현**

```python
# bench/flops.py
"""FLOPs 계측.

fvcore는 핸들러가 없는 연산을 0으로 센다. 그 침묵이 가장 위험한 실패 모드라
(Vim의 selective scan이 통째로 사라진다) 미등록 연산을 항상 함께 반환한다.
"""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


@dataclass(frozen=True)
class FlopResult:
    traced: int
    uncounted_ops: tuple[str, ...]


def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
    op_handlers: dict[str, Callable] | None = None,
    device: str = "cpu",
) -> FlopResult:
    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)

    analysis = FlopCountAnalysis(model, x)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)

    for name, handler in (op_handlers or {}).items():
        analysis.set_op_handle(name, handler)

    total = analysis.total()
    uncounted = tuple(sorted(analysis.unsupported_ops().keys()))
    return FlopResult(traced=total, uncounted_ops=uncounted)
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_flops.py -v`
Expected: PASS 3건

`set_op_handle`이나 `unsupported_ops()`에서 `AttributeError`가 나면 설치된 fvcore 버전의 API가 다른 것이다. `python -c "from fvcore.nn import FlopCountAnalysis; print([m for m in dir(FlopCountAnalysis) if not m.startswith('_')])"`로 실제 메서드명을 확인하고 맞춘다.

- [ ] **Step 5: 커밋**

```bash
git add bench/flops.py tests/test_flops.py
git commit -m "feat: count FLOPs and refuse to hide ops fvcore cannot trace"
```

---

### Task 4: latency 계측

**Files:**
- Create: `bench/latency.py`
- Test: `tests/test_latency.py`

**Interfaces:**
- Consumes: 없음
- Produces: `bench.latency.measure_latency(model, input_shape, warmup=50, iters=100, device="cuda") -> float` — 중앙값 밀리초

- [ ] **Step 1: 실패하는 테스트 작성**

실제 시간값은 결정적이지 않으므로, 호출 횟수와 반환 타입으로 계약을 검증한다.

```python
# tests/test_latency.py
import torch
import torch.nn as nn

from bench.latency import measure_latency


class CountingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        self.calls += 1
        return self.fc(x)


def test_runs_warmup_plus_measured_iterations():
    model = CountingModule()
    measure_latency(model, input_shape=(4,), warmup=3, iters=7, device="cpu")
    assert model.calls == 10


def test_returns_a_positive_float():
    latency = measure_latency(
        CountingModule(), input_shape=(4,), warmup=2, iters=5, device="cpu"
    )
    assert isinstance(latency, float)
    assert latency > 0


def test_does_not_track_gradients():
    """grad를 켜고 재면 학습 경로 비용이 섞여 추론 latency가 아니게 된다."""

    class GradProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.saw_grad_enabled = None
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            self.saw_grad_enabled = torch.is_grad_enabled()
            return self.fc(x)

    model = GradProbe()
    measure_latency(model, input_shape=(4,), warmup=1, iters=1, device="cpu")
    assert model.saw_grad_enabled is False
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_latency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.latency'`

- [ ] **Step 3: 구현**

```python
# bench/latency.py
"""추론 latency. CUDA는 비동기라 벽시계로 재면 커널 큐잉 시간만 재게 된다."""
import statistics
import time

import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    warmup: int = 50,
    iters: int = 100,
    device: str = "cuda",
) -> float:
    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)
    use_cuda = device.startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if use_cuda:
            torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(x)
                samples.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(samples)
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_latency.py -v`
Expected: PASS 3건

- [ ] **Step 5: 커밋**

```bash
git add bench/latency.py tests/test_latency.py
git commit -m "feat: time inference with CUDA events instead of the wall clock"
```

---

### Task 5: peak VRAM과 OOM 기록

8GB에서 고해상도 attention은 OOM이 예상된다. 이건 크래시가 아니라 논문의 메모리 주장을 뒷받침하는 데이터다.

**Files:**
- Create: `bench/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: 없음
- Produces: `bench.memory.measure_peak_memory(fn) -> MemoryResult`
  - `MemoryResult`는 `peak_allocated_bytes: int | None`, `peak_reserved_bytes: int | None`, `status: str` 필드를 가진 dataclass
  - `status`는 `"ok"`, `"oom"`, `"no_cuda"` 중 하나
  - `bench.memory._is_oom(exc) -> bool`

두 통계를 모두 기록한다. `max_memory_allocated`는 텐서 수요량이라 재현성이 높고 논문에서 흔히 쓰는 값이며, `max_memory_reserved`는 캐싱 얼로케이터가 실제 잡은 VRAM이라 8GB 한계선과 직결된다. 비용이 사실상 0이므로 둘 다 남기고 논문에서 어느 쪽을 인용할지는 나중에 정한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_memory.py
import pytest
import torch

from bench.memory import measure_peak_memory


def test_oom_is_recorded_as_data_not_raised():
    def blows_up():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    result = measure_peak_memory(blows_up)
    assert result.status == "oom"
    assert result.peak_allocated_bytes is None
    assert result.peak_reserved_bytes is None


def test_runtime_error_worded_as_oom_is_also_recorded_as_oom():
    """CUDA OOM은 OutOfMemoryError로만 오지 않는다. cuDNN 실패 등에서는 메시지에
    'out of memory'가 든 평범한 RuntimeError로 샌다. 이걸 놓치면 sweep이
    고해상도 셀에서 통째로 죽는데, 하필 거기가 논문 주장이 걸린 자리다."""

    def blows_up():
        raise RuntimeError("CUDA error: out of memory when allocating workspace")

    result = measure_peak_memory(blows_up)
    assert result.status == "oom"


def test_unrelated_runtime_error_still_propagates():
    """RuntimeError를 전부 삼키면 진짜 버그가 가짜 메모리 한계로 논문에 실린다."""

    def has_a_bug():
        raise RuntimeError("shape '[2, 3]' is invalid for input of size 7")

    with pytest.raises(RuntimeError, match="invalid for input"):
        measure_peak_memory(has_a_bug)


def test_other_exceptions_still_propagate():
    """OOM만 삼킨다. 진짜 버그를 'oom'으로 기록하면 결과가 오염된다."""

    def has_a_bug():
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        measure_peak_memory(has_a_bug)


def test_no_cuda_is_reported_distinctly_from_oom(monkeypatch):
    """세 상태 중 하나가 커버리지 0으로 나가면 안 된다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = measure_peak_memory(lambda: None)
    assert result.status == "no_cuda"
    assert result.peak_allocated_bytes is None
    assert result.peak_reserved_bytes is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_successful_run_reports_both_statistics():
    def allocates():
        torch.zeros(1024, 1024, device="cuda")

    result = measure_peak_memory(allocates)
    assert result.status == "ok"
    assert result.peak_allocated_bytes > 0
    assert result.peak_reserved_bytes >= result.peak_allocated_bytes
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.memory'`

- [ ] **Step 3: 구현**

```python
# bench/memory.py
"""peak VRAM. OOM은 실패가 아니라 기록할 결과다."""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MemoryResult:
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    status: str


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM이 오는 두 가지 형태를 모두 인정한다.

    `torch.cuda.OutOfMemoryError`가 표준 경로지만, cuDNN workspace 할당 실패
    등에서는 메시지에 'out of memory'가 든 평범한 RuntimeError로 샌다. 후자를
    놓치면 sweep이 고해상도 셀에서 죽고, 하필 그 셀이 논문의 메모리 주장이
    걸린 자리다. 반대로 RuntimeError를 전부 삼키면 진짜 버그가 가짜 메모리
    한계로 결과표에 실리므로, 메시지를 확인해 구분한다.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def measure_peak_memory(fn: Callable[[], object]) -> MemoryResult:
    if not torch.cuda.is_available():
        try:
            fn()
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            return MemoryResult(None, None, "oom")
        return MemoryResult(None, None, "no_cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        fn()
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return MemoryResult(None, None, "oom")

    torch.cuda.synchronize()
    return MemoryResult(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        status="ok",
    )
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_memory.py -v`
Expected: PASS 6건 (CUDA 없으면 마지막 1건 skip)

- [ ] **Step 5: 커밋**

```bash
git add bench/memory.py tests/test_memory.py
git commit -m "feat: record peak VRAM and treat OOM as a measurement"
```

---

### Task 6: 모델 레지스트리와 DeiT-S

세 모델을 이름 하나로 부를 수 있는 단일 진입점을 만든다. DeiT부터 붙인다 — timm에 있어 가장 간단하고, 두 원논문의 공통 기준선이다.

**Files:**
- Create: `models/__init__.py`
- Create: `models/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `models.registry.build_model(name: str, pretrained: bool = True, img_size: int = 224) -> nn.Module`
  - `models.registry.MODEL_NAMES: tuple[str, ...]` — `("deit_s", "cmt_s", "vim_s")`
  - 알 수 없는 이름이면 `ValueError`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_registry.py
import pytest
import torch

from models.registry import MODEL_NAMES, build_model


def test_registry_lists_the_three_models_under_comparison():
    assert MODEL_NAMES == ("deit_s", "cmt_s", "vim_s")


def test_unknown_name_raises_with_a_useful_message():
    with pytest.raises(ValueError, match="vim_xl"):
        build_model("vim_xl")


def test_deit_s_has_the_published_parameter_count():
    """DeiT-S는 22M. 크게 어긋나면 잘못된 변형을 불러온 것이다."""
    model = build_model("deit_s", pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    assert 21e6 < params < 23e6, f"{params / 1e6:.1f}M"


def test_deit_s_accepts_a_non_default_resolution():
    """해상도 sweep의 전제. 384²가 안 되면 E1 자체가 성립하지 않는다."""
    model = build_model("deit_s", pretrained=False, img_size=384).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 384, 384))
    assert out.shape == (1, 1000)
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.registry'`

- [ ] **Step 3: 구현**

```python
# models/registry.py
"""비교 대상 세 모델의 단일 진입점.

파라미터가 22~26M로 이미 정렬되어 있어 별도 조정 없이 통제가 성립한다.
"""
import torch.nn as nn

MODEL_NAMES = ("deit_s", "cmt_s", "vim_s")


def build_model(name: str, pretrained: bool = True, img_size: int = 224) -> nn.Module:
    if name not in MODEL_NAMES:
        raise ValueError(
            f"알 수 없는 모델 '{name}'. 사용 가능: {', '.join(MODEL_NAMES)}"
        )
    if name == "deit_s":
        return _build_deit_s(pretrained=pretrained, img_size=img_size)
    raise NotImplementedError(f"'{name}'은 이후 태스크에서 붙인다")


def _build_deit_s(pretrained: bool, img_size: int) -> nn.Module:
    import timm

    return timm.create_model(
        "deit_small_patch16_224",
        pretrained=pretrained,
        img_size=img_size,
    )
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_registry.py -v`
Expected: PASS 4건

- [ ] **Step 5: 커밋**

```bash
git add models/ tests/test_registry.py
git commit -m "feat: put the three compared models behind one builder"
```

---

### Task 7: CMT-S 통합

**Files:**
- Create: `models/cmt.py`
- Modify: `models/registry.py` — `_build_cmt_s` 추가하고 `build_model`에서 분기
- Test: `tests/test_cmt.py`

**Interfaces:**
- Consumes: `models.registry.build_model`
- Produces: `build_model("cmt_s", ...)`가 동작. `models.cmt.load_cmt_small(pretrained, img_size) -> nn.Module`

**가중치 로딩은 이 계획의 범위가 아니다.** E1은 FLOPs·latency·메모리만 재고 셋 다 가중치와 무관하다. 사전학습 체크포인트가 실제로 필요한 것은 ERF를 재는 E2와 dilution을 재는 E3이므로, 로딩과 상대 위치 bias 보간은 그 계획에서 구현한다. 여기서 미리 쓰면 어떤 테스트도 거치지 않는 추측성 코드가 된다. `pretrained=True`는 무엇을 해야 하는지 알려주는 `NotImplementedError`를 던진다.

- [ ] **Step 1: 공식 구현 벤더링**

`huawei-noah/Efficient-AI-Backbones`의 `cmt_pytorch/cmt.py`를 `models/cmt_official.py`로 복사한다. 라이선스 헤더(파일 상단 Huawei 저작권 주석)를 지우지 않는다. 출처 URL과 받은 날짜를 파일 상단에 덧붙인다.

체크포인트는 받지 않는다 — E1에 필요 없다.

- [ ] **Step 2: 실패하는 테스트 작성**

```python
# tests/test_cmt.py
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


def test_pretrained_weights_are_not_silently_skipped():
    """가중치 로딩은 E2/E3 계획으로 미뤘다. 조용히 무시하면 나중에 무작위 초기화
    모델로 ERF를 재고도 아무도 모른다."""
    import pytest

    with pytest.raises(NotImplementedError, match="E2"):
        build_model("cmt_s", pretrained=True)
```

- [ ] **Step 3: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_cmt.py -v`
Expected: FAIL — `NotImplementedError: 'cmt_s'은 이후 태스크에서 붙인다`

- [ ] **Step 4: 구현**

```python
# models/cmt.py
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
```

`models/registry.py`의 `build_model`에서 분기를 추가한다:

```python
    if name == "cmt_s":
        from models.cmt import load_cmt_small

        return load_cmt_small(pretrained=pretrained, img_size=img_size)
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_cmt.py -v`
Expected: PASS 5건

공식 소스는 확인해두었다. `cmt_s(pretrained=False, **kwargs)`가 존재하고 `CMT.__init__`이 `img_size`를 받으므로 위 호출이 그대로 동작한다. `embed_dims=[64,128,256,512]`, `depths=[3,3,16,3]`으로 논문 표 2의 CMT-S와 일치한다.

import는 `timm.models.helpers` 등 timm 0.x 경로를 쓰지만 timm 1.0.28에서 deprecation shim으로 모두 동작한다(FutureWarning만 발생). **벤더링한 파일의 import를 손대지 않는다** — 고정 환경의 timm 0.9.12에서 원본 그대로 돌아가야 하고, 여기서 1.x 경로로 고쳐두면 그쪽이 깨진다.

- [ ] **Step 6: 커밋**

```bash
git add models/cmt.py models/cmt_official.py models/registry.py tests/test_cmt.py
git commit -m "feat: build CMT-S at any sweep resolution"
```

---

### Task 8: Vim-S 통합과 selective scan FLOPs 핸들러

**이 태스크가 계획 전체에서 가장 조용히 틀리기 쉬운 지점이다.** fvcore는 selective scan을 0으로 세고, 그러면 Vim의 FLOPs가 실제보다 훨씬 작게 나와 "Vim이 압도적"이라는 잘못된 결론이 나온다.

**Files:**
- Create: `models/vim.py`
- Modify: `models/registry.py` — `vim_s` 분기 추가
- Test: `tests/test_vim.py`

**Interfaces:**
- Consumes: `models.registry.build_model`, `bench.flops.count_flops`
- Produces:
  - `build_model("vim_s", ...)`가 동작
  - `models.vim.selective_scan_flop_handler(inputs, outputs) -> int`
  - `models.vim.VIM_OP_HANDLERS: dict[str, Callable]` — `count_flops`의 `op_handlers`에 그대로 넘긴다

- [ ] **Step 1: 공식 구현 벤더링**

`hustvl/Vim`의 `vim/models_mamba.py`를 `models/vim_official.py`로 복사한다. 출처와 커밋 해시를 상단 주석에 남긴다. 체크포인트는 HuggingFace에서 Vim-S를 받아 `checkpoints/vim_s.pth`에 둔다.

- [ ] **Step 2: 실패하는 테스트 작성**

```python
# tests/test_vim.py
import torch

from bench.flops import count_flops
from models.registry import build_model
from models.vim import VIM_OP_HANDLERS, selective_scan_flop_handler


def test_vim_s_has_the_published_parameter_count():
    """논문 표 2 기준 26M."""
    model = build_model("vim_s", pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    assert 25e6 < params < 28e6, f"{params / 1e6:.1f}M"


def test_selective_scan_flops_match_the_analytic_formula():
    """한 방향 8MDN. Vim 논문 식 (8)."""
    seqlen, dim, state = 196, 384, 16
    flops = selective_scan_flop_handler(
        inputs=_fake_scan_inputs(seqlen=seqlen, dim=dim, state=state),
        outputs=None,
    )
    assert flops == 8 * seqlen * dim * state


def test_vim_flops_are_not_silently_zero():
    """핸들러 없이 세면 selective scan이 통째로 사라진다. 그 차이를 확인한다."""
    model = build_model("vim_s", pretrained=False)
    without = count_flops(model, input_shape=(3, 224, 224))
    with_handler = count_flops(
        model, input_shape=(3, 224, 224), op_handlers=VIM_OP_HANDLERS
    )
    assert with_handler.traced > without.traced


def test_vim_reports_no_uncounted_ops_once_handlers_are_registered():
    model = build_model("vim_s", pretrained=False)
    result = count_flops(
        model, input_shape=(3, 224, 224), op_handlers=VIM_OP_HANDLERS
    )
    assert result.uncounted_ops == (), f"미등록: {result.uncounted_ops}"


def _fake_scan_inputs(seqlen: int, dim: int, state: int):
    """fvcore가 핸들러에 넘기는 형태를 흉내낸다: shape를 가진 인자 목록."""
    import types

    def node(shape):
        return types.SimpleNamespace(type=lambda: types.SimpleNamespace(sizes=lambda: shape))

    return [
        node([1, dim, seqlen]),   # u
        node([1, dim, seqlen]),   # delta
        node([dim, state]),       # A
    ]
```

- [ ] **Step 3: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_vim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.vim'`

- [ ] **Step 4: 구현**

```python
# models/vim.py
"""Vim-S 로더와 selective scan FLOPs 핸들러.

fvcore는 핸들러가 없는 연산을 0으로 센다. selective scan은 커스텀 CUDA
커널이라 여기 해당하고, 그대로 두면 Vim의 연산량이 통째로 사라져 "Vim이
압도적으로 효율적"이라는 잘못된 결론이 나온다. 아래 핸들러는 Vim 논문 식
(8)의 8MDN을 쓴다 — 측정이 아니라 해석적 값이므로 결과 CSV에서 별도 열로
분리해 출처를 드러낸다.
"""
from typing import Callable

import torch
import torch.nn as nn

from models.vim_official import vim_small as _vim_small_official

CHECKPOINT_PATH = "checkpoints/vim_s.pth"

# fvcore가 보고하는 연산자 이름. 실제 이름은 Step 5에서 확인해 맞춘다.
SELECTIVE_SCAN_OP = "prim::PythonOp.SelectiveScanFn"


def load_vim_small(pretrained: bool = True, img_size: int = 224) -> nn.Module:
    model = _vim_small_official(img_size=img_size)
    if not pretrained:
        return model
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    return model


def selective_scan_flop_handler(inputs, outputs) -> int:
    """한 방향 SSM = 8MDN (Vim 논문 식 8).

    inputs[0]은 u, shape (batch, dim, seqlen). inputs[2]는 A, shape (dim, state).
    """
    u_shape = inputs[0].type().sizes()
    a_shape = inputs[2].type().sizes()
    _, dim, seqlen = u_shape
    state = a_shape[1]
    return 8 * seqlen * dim * state


VIM_OP_HANDLERS: dict[str, Callable] = {
    SELECTIVE_SCAN_OP: selective_scan_flop_handler,
}
```

`models/registry.py`에 분기를 추가한다:

```python
    if name == "vim_s":
        from models.vim import load_vim_small

        return load_vim_small(pretrained=pretrained, img_size=img_size)
```

- [ ] **Step 5: 실제 연산자 이름 확인**

`SELECTIVE_SCAN_OP` 문자열이 틀리면 핸들러가 등록되지 않고 조용히 0이 유지된다. `test_vim_reports_no_uncounted_ops_once_handlers_are_registered`가 실패하면 실제 이름을 찍어 상수를 고친다:

```bash
python -c "
from bench.flops import count_flops
from models.registry import build_model
r = count_flops(build_model('vim_s', pretrained=False), (3, 224, 224))
print(r.uncounted_ops)
"
```

출력에 나온 이름을 `SELECTIVE_SCAN_OP`에 넣는다. 양방향이라 연산자가 두 번 불리면 fvcore가 각 호출을 세므로 핸들러에서 2배를 곱하지 않는다.

- [ ] **Step 6: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_vim.py -v`
Expected: PASS 4건

- [ ] **Step 7: 커밋**

```bash
git add models/vim.py models/vim_official.py models/registry.py tests/test_vim.py
git commit -m "feat: load Vim-S and stop fvcore from counting selective scan as zero"
```

---

### Task 9: throughput (8GB에 들어가는 최대 배치)

스펙의 측정 항목 중 마지막. batch=1 latency와 달리, 이건 "8GB로 실제로 얼마나 처리하는가"를 잰다. 최대 배치를 찾는 과정 자체가 OOM을 유발하므로 탐색 로직이 필요하다.

**Files:**
- Create: `bench/throughput.py`
- Test: `tests/test_throughput.py`

**Interfaces:**
- Consumes: `bench.memory.is_oom`
- Produces:
  - `bench.throughput.find_max_batch(model, input_shape, device="cuda", limit=512) -> int` — 들어가는 최대 배치. 1도 안 들어가면 `0`
  - `bench.throughput.measure_throughput(model, input_shape, device="cuda", limit=512) -> ThroughputResult`
  - `ThroughputResult`는 `batch: int`, `images_per_sec: float | None` 필드를 가진 dataclass

- [ ] **Step 0: `_is_oom`을 공개 이름으로 승격**

`bench/memory.py`의 `_is_oom`을 `is_oom`으로 이름만 바꾼다(호출부도 함께). 배치 탐색은 일부러 OOM을 유발하는 코드라 같은 판정이 필요한데, 두 모듈이 각자 다른 기준을 쓰면 메모리 측정에서는 OOM으로 기록되는 상황이 배치 탐색에서는 예외로 터진다. `tests/test_memory.py`는 `measure_peak_memory`를 통해서만 검증하므로 이 이름 변경에 영향받지 않는다. 변경 후 `pytest tests/test_memory.py -v`가 여전히 6건 통과하는지 확인한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_throughput.py
import torch
import torch.nn as nn

from bench.throughput import find_max_batch, measure_throughput


class FitsUpTo(nn.Module):
    """batch가 threshold를 넘으면 OOM을 던지는 가짜 모델."""

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        if x.shape[0] > self.threshold:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return self.fc(x)


def test_finds_the_largest_batch_that_fits():
    assert find_max_batch(FitsUpTo(6), input_shape=(4,), device="cpu") == 6


def test_finds_exact_power_of_two_boundary():
    assert find_max_batch(FitsUpTo(8), input_shape=(4,), device="cpu") == 8


def test_returns_zero_when_even_batch_one_fails():
    assert find_max_batch(FitsUpTo(0), input_shape=(4,), device="cpu") == 0


def test_respects_the_upper_limit():
    """상한이 없으면 큰 모델에서 탐색이 끝없이 늘어난다."""
    assert find_max_batch(FitsUpTo(9999), input_shape=(4,), device="cpu", limit=32) == 32


def test_throughput_reports_batch_and_rate():
    result = measure_throughput(FitsUpTo(4), input_shape=(4,), device="cpu")
    assert result.batch == 4
    assert result.images_per_sec > 0


def test_throughput_is_none_when_nothing_fits():
    result = measure_throughput(FitsUpTo(0), input_shape=(4,), device="cpu")
    assert result.batch == 0
    assert result.images_per_sec is None


class RaisesOomWordedRuntimeError(nn.Module):
    """OOM이 평범한 RuntimeError로 새는 형태. bench.memory와 판정이 일치해야 한다."""

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        if x.shape[0] > self.threshold:
            raise RuntimeError("CUDA error: out of memory allocating workspace")
        return self.fc(x)


def test_batch_search_treats_oom_worded_runtime_error_as_not_fitting():
    """배치 탐색은 일부러 OOM을 유발한다. 여기서 판정이 좁으면 탐색이 죽는다."""
    assert find_max_batch(
        RaisesOomWordedRuntimeError(6), input_shape=(4,), device="cpu"
    ) == 6


class HasABug(nn.Module):
    def forward(self, x):
        raise RuntimeError("shape '[2, 3]' is invalid for input of size 7")


def test_batch_search_does_not_swallow_unrelated_runtime_errors():
    """진짜 버그를 '안 들어감'으로 처리하면 max_batch가 조용히 0이 된다."""
    import pytest

    with pytest.raises(RuntimeError, match="invalid for input"):
        find_max_batch(HasABug(), input_shape=(4,), device="cpu")
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_throughput.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.throughput'`

- [ ] **Step 3: 구현**

```python
# bench/throughput.py
"""8GB에 들어가는 최대 배치와 그때의 처리량.

배치를 2배씩 올려 실패 지점을 잡고, 그 구간을 이분 탐색한다. 선형 증가는
큰 모델에서 너무 느리고, 2배만 쓰면 6과 8을 구분하지 못한다.
"""
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from bench.memory import is_oom


@dataclass(frozen=True)
class ThroughputResult:
    batch: int
    images_per_sec: float | None


def _fits(model: nn.Module, input_shape: tuple[int, ...], batch: int, device: str) -> bool:
    """`bench.memory`와 같은 OOM 판정을 쓴다. 두 모듈이 기준을 달리하면,
    메모리 측정에서는 oom으로 기록되는 상황이 배치 탐색에서는 예외로 터진다."""
    try:
        with torch.no_grad():
            model(torch.zeros(batch, *input_shape, device=device))
    except RuntimeError as exc:
        if not is_oom(exc):
            raise
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return False
    return True


def find_max_batch(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
    limit: int = 512,
) -> int:
    model = model.to(device).eval()

    if not _fits(model, input_shape, 1, device):
        return 0

    low = 1
    high = 2
    while high <= limit and _fits(model, input_shape, high, device):
        low = high
        high *= 2

    high = min(high, limit + 1)
    while low + 1 < high:
        mid = (low + high) // 2
        if _fits(model, input_shape, mid, device):
            low = mid
        else:
            high = mid
    return min(low, limit)


def measure_throughput(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
    limit: int = 512,
    warmup: int = 5,
    iters: int = 20,
) -> ThroughputResult:
    batch = find_max_batch(model, input_shape, device=device, limit=limit)
    if batch == 0:
        return ThroughputResult(batch=0, images_per_sec=None)

    model = model.to(device).eval()
    x = torch.zeros(batch, *input_shape, device=device)
    use_cuda = device.startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if use_cuda:
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)

    return ThroughputResult(
        batch=batch, images_per_sec=batch / statistics.median(samples)
    )
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_throughput.py tests/test_memory.py -v`
Expected: PASS 14건 (throughput 8건 + memory 6건). memory 쪽은 Step 0의 이름 변경이 기존 동작을 깨지 않았음을 확인하는 것이다.

- [ ] **Step 5: 커밋**

```bash
git add bench/throughput.py tests/test_throughput.py
git commit -m "feat: find the largest batch 8GB holds and time it"
```

---

### Task 10: E1 오케스트레이션

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/e1_resolution_sweep.py`
- Test: `tests/test_e1.py`

**Interfaces:**
- Consumes: `bench.env.snapshot`, `bench.flops.count_flops`, `bench.latency.measure_latency`, `bench.memory.measure_peak_memory`, `bench.throughput.measure_throughput`, `models.registry.build_model`
- Produces:
  - `experiments.e1_resolution_sweep.RESOLUTIONS: tuple[int, ...]` — `(224, 384, 512, 768, 1024)`
  - `experiments.e1_resolution_sweep.COLUMNS: list[str]`
  - `experiments.e1_resolution_sweep.run_sweep(model_names, resolutions, out_dir) -> pandas.DataFrame`
  - CSV 열: `model, resolution, params, flops_traced, flops_uncounted_ops, latency_ms, peak_allocated_bytes, peak_reserved_bytes, max_batch, images_per_sec, status, error`

실행이 15셀 한 시간 규모라는 점이 이 태스크의 설계를 좌우한다. 두 가지가 따라온다.

**셀마다 CSV를 다시 쓴다.** 마지막에 한 번만 쓰면 14번째 셀에서 예외가 났을 때 앞의 13셀이 통째로 사라진다. 실패한 셀은 `status="error"`와 `error` 메시지를 담은 행으로 남겨, 무엇이 빠졌는지 나중에 알 수 있게 한다. 실패를 예외로 터뜨려 실행을 끝내지 않는다 — 그 대가가 GPU 한 시간이다.

**FLOPs를 가장 먼저 잰다.** `count_flops`는 CPU 트레이스라 OOM이 날 수 없다. 메모리에 안 들어가는 해상도에서도 연산량은 논문 표에 필요한 값이므로, OOM 때문에 FLOPs까지 잃으면 안 된다. 순서를 FLOPs(CPU) → latency+메모리(GPU) → throughput(GPU)로 두면 장치 왕복도 한 번으로 줄어든다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_e1.py
import pandas as pd

import experiments.e1_resolution_sweep as e1
from bench.memory import MemoryResult
from experiments.e1_resolution_sweep import COLUMNS, RESOLUTIONS, run_sweep

EXPECTED_COLUMNS = [
    "model",
    "resolution",
    "params",
    "flops_traced",
    "flops_uncounted_ops",
    "latency_ms",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "max_batch",
    "images_per_sec",
    "status",
    "error",
]


def _stub_row(model_name: str, resolution: int) -> dict:
    row = {column: None for column in COLUMNS}
    row.update(
        model=model_name,
        resolution=resolution,
        flops_uncounted_ops="",
        status="ok",
    )
    return row


def test_sweeps_the_five_resolutions_from_the_spec():
    assert RESOLUTIONS == (224, 384, 512, 768, 1024)


def test_writes_one_row_per_model_resolution_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "_measure_one", _stub_row)

    df = run_sweep(
        model_names=("deit_s", "cmt_s"), resolutions=(224, 384), out_dir=tmp_path
    )

    assert len(df) == 4
    assert list(df.columns) == EXPECTED_COLUMNS


def test_persists_a_csv_and_an_env_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "_measure_one", _stub_row)

    run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert (tmp_path / "sweep.csv").exists()
    assert (tmp_path / "env.json").exists()


def test_a_failing_cell_does_not_lose_the_cells_already_measured(tmp_path, monkeypatch):
    """15셀 한 시간짜리 실행이 중간에 죽어도 앞의 결과는 남아야 한다.
    이 방어가 없으면 GPU 한 시간이 예외 하나로 사라진다."""
    calls = {"n": 0}

    def flaky(model_name, resolution):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("이 셀만 터진다")
        return _stub_row(model_name, resolution)

    monkeypatch.setattr(e1, "_measure_one", flaky)

    df = run_sweep(
        model_names=("deit_s",), resolutions=(224, 384, 512), out_dir=tmp_path
    )

    assert len(df) == 3
    assert list(df["status"]) == ["ok", "error", "ok"]
    assert "이 셀만 터진다" in df.iloc[1]["error"]

    on_disk = pd.read_csv(tmp_path / "sweep.csv")
    assert len(on_disk) == 3


def test_csv_exists_before_the_sweep_finishes(tmp_path, monkeypatch):
    """마지막에 한 번만 쓰면 중간 실패 시 파일 자체가 없다."""
    seen = []

    def record_then_stub(model_name, resolution):
        seen.append((tmp_path / "sweep.csv").exists())
        return _stub_row(model_name, resolution)

    monkeypatch.setattr(e1, "_measure_one", record_then_stub)
    run_sweep(model_names=("deit_s",), resolutions=(224, 384), out_dir=tmp_path)

    assert seen[1] is True, "두 번째 셀을 재기 전에 첫 셀이 이미 디스크에 있어야 한다"


def test_oom_rows_are_kept_with_status_not_dropped(tmp_path, monkeypatch):
    """OOM은 결과다. 행이 사라지면 메모리 주장의 증거가 사라진다."""
    monkeypatch.setattr(
        e1, "measure_peak_memory", lambda fn: MemoryResult(None, None, "oom")
    )

    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["status"] == "oom"


def test_flops_survive_an_oom_row(tmp_path, monkeypatch):
    """FLOPs는 CPU 트레이스라 OOM과 무관하다. 메모리에 안 들어가는 해상도에서도
    연산량은 논문 표에 들어갈 값이므로 함께 잃으면 안 된다."""
    monkeypatch.setattr(
        e1, "measure_peak_memory", lambda fn: MemoryResult(None, None, "oom")
    )

    df = run_sweep(model_names=("deit_s",), resolutions=(224,), out_dir=tmp_path)

    assert df.iloc[0]["status"] == "oom"
    assert df.iloc[0]["flops_traced"] > 0
    assert df.iloc[0]["params"] > 0
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_e1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.e1_resolution_sweep'`

- [ ] **Step 3: 구현**

```python
# experiments/e1_resolution_sweep.py
"""E1 — 해상도 sweep 실측.

논문 표 1은 손으로 계산한 추정값이었다. 이 스크립트가 그 자리를 대체할 실측
CSV를 만든다. 그림과 표는 이 CSV만 읽는다.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from bench.env import snapshot
from bench.flops import count_flops
from bench.latency import measure_latency
from bench.memory import measure_peak_memory
from bench.throughput import measure_throughput
from models.registry import MODEL_NAMES, build_model

RESOLUTIONS = (224, 384, 512, 768, 1024)

COLUMNS = [
    "model",
    "resolution",
    "params",
    "flops_traced",
    "flops_uncounted_ops",
    "latency_ms",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "max_batch",
    "images_per_sec",
    "status",
    "error",
]


def _op_handlers(model_name: str) -> dict:
    if model_name == "vim_s":
        from models.vim import VIM_OP_HANDLERS

        return VIM_OP_HANDLERS
    return {}


def _blank_row(model_name: str, resolution: int) -> dict:
    row = {column: None for column in COLUMNS}
    row.update(model=model_name, resolution=resolution, flops_uncounted_ops="")
    return row


def _measure_one(model_name: str, resolution: int) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (3, resolution, resolution)

    row = _blank_row(model_name, resolution)
    row["status"] = "ok"

    model = build_model(model_name, pretrained=False, img_size=resolution)
    row["params"] = sum(p.numel() for p in model.parameters())

    # FLOPs를 먼저 잰다. CPU 트레이스라 OOM이 날 수 없으므로, 메모리에 들어가지
    # 않는 셀에서도 연산량은 남는다.
    flops = count_flops(model, shape, op_handlers=_op_handlers(model_name))
    row["flops_traced"] = flops.traced
    row["flops_uncounted_ops"] = ";".join(flops.uncounted_ops)

    # latency 측정을 peak memory 측정 안에서 한 번만 돌린다. 밖에서 또 부르면
    # 1024²에서 150 iteration을 두 번 돌게 된다.
    captured: dict[str, float] = {}

    def _timed_run() -> None:
        captured["latency_ms"] = measure_latency(model, shape, device=device)

    memory = measure_peak_memory(_timed_run)
    row["peak_allocated_bytes"] = memory.peak_allocated_bytes
    row["peak_reserved_bytes"] = memory.peak_reserved_bytes
    row["status"] = memory.status
    row["latency_ms"] = captured.get("latency_ms")

    if memory.status == "oom":
        return row

    throughput = measure_throughput(model, shape, device=device)
    row["max_batch"] = throughput.batch
    row["images_per_sec"] = throughput.images_per_sec
    return row


def _error_row(model_name: str, resolution: int, exc: BaseException) -> dict:
    row = _blank_row(model_name, resolution)
    row["status"] = "error"
    row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_sweep(
    model_names: tuple[str, ...] = MODEL_NAMES,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    out_dir: Path | str = "results/e1",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    (out_dir / "env.json").write_text(json.dumps(snapshot(), indent=2))

    rows: list[dict] = []
    for name in model_names:
        for resolution in resolutions:
            try:
                rows.append(_measure_one(name, resolution))
            except Exception as exc:  # 한 셀의 실패로 전체를 잃지 않는다
                rows.append(_error_row(name, resolution, exc))
            # 셀마다 다시 쓴다. 한 시간짜리 실행이 도중에 죽어도 앞의 결과는 남는다.
            pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows, columns=COLUMNS)

    failed = df[df["status"] == "error"]
    if not failed.empty:
        cells = ", ".join(f"{r.model}@{r.resolution}" for r in failed.itertuples())
        print(f"경고: {len(failed)}개 셀 실패 — {cells}. 숫자를 쓰기 전에 확인할 것.")

    return df


if __name__ == "__main__":
    print(run_sweep().to_string(index=False))
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_e1.py -v`
Expected: PASS 7건

- [ ] **Step 5: 커밋**

```bash
git add experiments/ tests/test_e1.py
git commit -m "feat: sweep the models into a CSV that survives a failed cell"
```

---

### Task 11: 측정 파이프라인 sanity check

**스펙이 요구하는 검증 관문이다.** 측정 도구가 틀렸으면 나머지가 전부 무의미하므로, 알려진 공개값과 대조해 하네스를 검증한다.

**Files:**
- Create: `tests/test_sanity.py`

**Interfaces:**
- Consumes: `bench.flops.count_flops`, `models.registry.build_model`
- Produces: 없음 (검증 전용)

- [ ] **Step 1: 실패할 수 있는 검증 테스트 작성**

```python
# tests/test_sanity.py
"""공개된 값과 대조해 측정 하네스 자체를 검증한다.

이 테스트가 실패하면 계측이 틀린 것이고, 그 상태로 잰 1024² 수치는 논문에
쓸 수 없다.
"""
import pytest

from bench.flops import count_flops
from models.registry import build_model

DEIT_S_PUBLISHED_FLOPS = 4.6e9  # DeiT 논문 보고값, 224²


def test_deit_s_flops_match_the_published_value_within_5_percent():
    result = count_flops(build_model("deit_s", pretrained=False), (3, 224, 224))
    ratio = result.traced / DEIT_S_PUBLISHED_FLOPS
    assert 0.95 < ratio < 1.05, (
        f"측정 {result.traced / 1e9:.2f}G vs 공개값 4.6G (비율 {ratio:.3f}). "
        f"미등록 연산: {result.uncounted_ops}"
    )


@pytest.mark.parametrize("name", ["deit_s", "cmt_s", "vim_s"])
def test_no_model_has_uncounted_ops_at_224(name):
    """미등록 연산이 남아 있으면 그 모델의 FLOPs는 과소 계상된다."""
    handlers = {}
    if name == "vim_s":
        from models.vim import VIM_OP_HANDLERS

        handlers = VIM_OP_HANDLERS
    result = count_flops(
        build_model(name, pretrained=False), (3, 224, 224), op_handlers=handlers
    )
    assert result.uncounted_ops == (), f"{name} 미등록: {result.uncounted_ops}"
```

- [ ] **Step 2: 테스트 실행**

Run: `pytest tests/test_sanity.py -v`
Expected: PASS 4건

실패하면 **다음 태스크로 넘어가지 않는다.** 비율이 2에 가까우면 fvcore가 MAC이 아니라 FLOP을 세는 설정인 것이고, 1보다 한참 작으면 미등록 연산이 남아 있는 것이다. 후자는 실패 메시지의 `uncounted_ops` 목록이 직접 알려준다.

- [ ] **Step 3: 커밋**

```bash
git add tests/test_sanity.py
git commit -m "test: check the harness against published FLOPs before trusting it"
```

---

### Task 12: E1 그림 생성

**Files:**
- Create: `figures/__init__.py`
- Create: `figures/e1_plot.py`
- Test: `tests/test_e1_plot.py`

**Interfaces:**
- Consumes: `results/e1/sweep.csv`
- Produces: `figures.e1_plot.plot_sweep(csv_path, out_path) -> Path` — 3단 그림(FLOPs / latency / peak VRAM 대 해상도) PNG

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_e1_plot.py
import pandas as pd
import pytest

from figures.e1_plot import plot_sweep


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_writes_a_png_from_the_csv(tmp_path):
    csv = _write_csv(
        tmp_path / "sweep.csv",
        [
            {
                "model": "deit_s",
                "resolution": 224,
                "params": 22_000_000,
                "flops_traced": 4.6e9,
                "flops_uncounted_ops": "",
                "latency_ms": 5.0,
                "peak_allocated_bytes": 1_000_000,
                "peak_reserved_bytes": 2_000_000,
                "status": "ok",
            }
        ],
    )
    out = plot_sweep(csv, tmp_path / "e1.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_refuses_an_empty_csv(tmp_path):
    """빈 입력에서 조용히 빈 그림을 내면 논문에 빈 그림이 실린다."""
    csv = _write_csv(tmp_path / "sweep.csv", [])
    with pytest.raises(ValueError, match="비어"):
        plot_sweep(csv, tmp_path / "e1.png")


def test_oom_rows_do_not_become_zero_points(tmp_path):
    """OOM을 0으로 그리면 '메모리를 안 쓴다'로 읽힌다."""
    csv = _write_csv(
        tmp_path / "sweep.csv",
        [
            {
                "model": "deit_s",
                "resolution": 1024,
                "params": 22_000_000,
                "flops_traced": None,
                "flops_uncounted_ops": "",
                "latency_ms": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "status": "oom",
            }
        ],
    )
    out = plot_sweep(csv, tmp_path / "e1.png")
    assert out.exists()
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `pytest tests/test_e1_plot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'figures.e1_plot'`

- [ ] **Step 3: 구현**

```python
# figures/e1_plot.py
"""E1 그림. CSV만 읽는다 — 손으로 넣은 숫자가 그림에 들어가지 않게."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("flops_traced", "FLOPs", 1e9, "GFLOPs"),
    ("latency_ms", "Latency", 1.0, "ms"),
    ("peak_allocated_bytes", "Peak VRAM (allocated)", 1024**3, "GiB"),
    ("peak_reserved_bytes", "Peak VRAM (reserved)", 1024**3, "GiB"),
]


def plot_sweep(csv_path: Path | str, out_path: Path | str) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path}가 비어 있다 — 먼저 sweep을 실행할 것")

    out_path = Path(out_path)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4))

    for ax, (column, title, scale, unit) in zip(axes, PANELS):
        for model, group in df.groupby("model"):
            ok = group[group["status"] == "ok"].sort_values("resolution")
            if not ok.empty:
                ax.plot(
                    ok["resolution"], ok[column] / scale, marker="o", label=model
                )
            # OOM은 0이 아니라 표식으로 남긴다.
            for _, row in group[group["status"] == "oom"].iterrows():
                ax.axvline(row["resolution"], linestyle=":", alpha=0.4)
                ax.annotate(
                    f"{row['model']} OOM",
                    xy=(row["resolution"], ax.get_ylim()[1] * 0.5),
                    rotation=90,
                    fontsize=7,
                    ha="right",
                )
        ax.set_xlabel("input resolution (px)")
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(plot_sweep("results/e1/sweep.csv", "results/e1/e1_sweep.png"))
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

Run: `pytest tests/test_e1_plot.py -v`
Expected: PASS 3건

- [ ] **Step 5: 커밋**

```bash
git add figures/__init__.py figures/e1_plot.py tests/test_e1_plot.py
git commit -m "feat: plot the sweep from the CSV"
```

**실제 sweep은 여기서 돌리지 않는다.** 이 태스크가 내놓는 것은 그림을 그리는 코드이지 측정값이 아니다. 실측은 Task 13에서, 고정 환경에서 수행한다.

---

### Task 13: 고정 환경에서 실측 실행

앞선 태스크들은 하네스를 만들었다. 이 태스크가 논문에 들어갈 숫자를 만든다.

**선행 조건** (모두 충족되어야 시작한다):
- Task 1의 스모크 테스트 통과 — WSL2에서 selective scan CUDA 커널이 실제로 돈다
- Task 8 완료 — `build_model("vim_s")`가 동작하고 FLOPs 핸들러가 등록되어 있다
- Task 11의 sanity check 통과 — DeiT-S 224² FLOPs가 공개값 4.6G의 ±5% 이내

**Files:**
- Create: `results/e1/sweep.csv`, `results/e1/env.json`, `results/e1/e1_sweep.png`

**Interfaces:**
- Consumes: `experiments.e1_resolution_sweep.run_sweep`, `figures.e1_plot.plot_sweep`
- Produces: 논문 표 1을 대체할 실측 데이터

- [ ] **Step 1: 고정 환경에서 전체 테스트 재확인**

Run: `pytest tests/ -v`

Windows(Python 3.12 / torch 2.6)에서 개발한 하네스가 고정 환경(Python 3.10.13 / torch 2.1.1+cu118)에서도 같게 동작하는지 확인한다. 결과가 다르면 실측 전에 원인을 밝힌다 — 하네스가 환경에 따라 다르게 굴면 그 하네스로 잰 숫자는 못 쓴다.

- [ ] **Step 2: sweep 실행**

```bash
python -m experiments.e1_resolution_sweep
```

3 모델 × 5 해상도 = 15셀. 배치 탐색이 매 셀에서 OOM까지 올려보므로 한 시간 규모를 예상한다. 실행 중 GPU를 다른 용도로 쓰지 않는다 — latency와 peak memory가 오염된다.

- [ ] **Step 3: 결과의 정직성 확인**

`results/e1/sweep.csv`를 열어 다음을 확인한다. 하나라도 어긋나면 숫자를 쓰지 않는다.

- `flops_uncounted_ops` 열이 **모든 행에서 비어 있는가.** 비어 있지 않다면 그 행의 FLOPs는 과소 계상된 값이다. 특히 `vim_s` 행을 확인한다 — selective scan이 빠지면 Vim이 실제보다 훨씬 효율적으로 보인다.
- `env.json`의 torch·CUDA·GPU가 고정 환경과 일치하는가.
- OOM 행이 `status == "oom"`으로 남아 있는가. 사라졌다면 기록이 아니라 소실이다.

- [ ] **Step 4: 그림 생성**

```bash
python -m figures.e1_plot
```

- [ ] **Step 5: 결과 커밋**

```bash
git add results/e1/
git commit -m "measure: record the resolution sweep the paper's Table 1 estimated"
```

이 시점에서 `results/e1/sweep.csv`가 논문 표 1을 대체할 실측 데이터다.

---

## 이 계획이 끝나면

- 논문 표 1의 추정 FLOPs를 실측값으로 교체할 수 있다.
- 이론 교차점(M = 2d = 768)과 실측 latency 교차점을 대비할 수 있다.
- 8GB에서 어느 모델이 어느 해상도에 OOM 나는지가 기록된다.
- `bench/` 하네스가 E2·E3·E4에서 그대로 재사용된다.

## 후속 계획

- 계획 2: E2 — ERF 정량 측정 (비등방 지수, 주축 각도, 수직/수평 감쇠율 비)
- 계획 3: E3 — effective attention 환원과 dilution 커버리지
- 계획 4: E4 — 2×2 요인 학습 ablation
- 계획 5: 논문 v2 · 포트폴리오 프로젝트 · 벨로그 후속 글
