# 인계 문서 — 2026-08-18

다음 세션이 이 파일부터 읽으면 이어서 작업할 수 있다.

## 이 저장소가 하려는 일

「CMT와 Vision Mamba의 심층 비교 분석」(김철희·이강국·심규성, 2026-05-15 대한전자공학회
제출)은 이론 분석은 갖췄지만 **직접 측정한 수치가 하나도 없다.** 표 1의 FLOPs는 손으로
계산한 추정값이고, 표 2의 83.5% / 80.5%와 3.3절 ablation 분해는 전부 원논문 인용이다.
Abstract는 "Experimental and theoretical analysis shows"라고 쓰지만 실제로는 theoretical만
있다.

이 저장소는 그 구멍을 실측으로 메운다. 최종 산출물은 셋이다 — 논문 v2, 포트폴리오
프로젝트 항목, 벨로그 후속 글(1편은 그대로 두고 2편 신규).

전체 설계: `docs/superpowers/specs/2026-08-18-cmt-vs-vim-experiments-design.md`

## 실험 구성 (승인된 설계)

| ID | 실험 | 검증 대상 | 학습 |
|----|------|-----------|------|
| E1 | 해상도 sweep 실측 | 표 1의 추정 FLOPs, cross-over point | 불필요 |
| E2 | ERF 정량 측정 | 3.1절 등방성 vs scan 정합 이방성 | 불필요 |
| E3 | Softmax dilution 정량화 | 3.2절 큰 객체 희석 주장 | 불필요 |
| E4 | 2×2 요인 학습 ablation | 3.3절 inductive bias 주장 | 필요 |

E4는 2×2 요인 설계다. 행이 연산자(Attention/SSM), 열이 구조(평면/계층 4-stage).
D칸이 Hierarchical Vim이며 논문 결론부가 예고한 향후 연구가 여기 들어온다.

**요인 정의의 한계** — C칸 CMT-Ti는 DeiT-Ti와 계층 구조만 다른 게 아니라 conv locality
(LPU, IRFFN, LMHSA)도 함께 들어온다. 따라서 "구조 주효과"는 순수 계층 효과가 아니라
"CMT식 구조적 prior 묶음"의 효과다. 분리하려면 2×2×2 = 24 run이 필요해 예산을 넘는다.
논문에 한계로 명시할 것. 이 정의가 D칸 구성도 강제한다 — B→D 조작이 A→C와 같아야
상호작용을 해석할 수 있으므로, Hierarchical Vim도 계층화와 conv locality를 함께 넣는다.

## 현재 상태

- 브랜치: `feat/e1-resolution-sweep` (main에 머지 안 됨, 원격에 푸시됨)
- 테스트: **48건 전부 통과** (timm deprecation 경고 3건만 남음 — 무해)
- 계획: `docs/superpowers/plans/2026-08-18-e1-resolution-sweep.md` (13 태스크)
- SDD 원장: `.superpowers/sdd/2026-08-18-e1-resolution-sweep/progress.md` — git-ignored,
  태스크별 완료·수정 라운드·이월된 minor가 전부 기록돼 있다

### 완료된 태스크

| Task | 산출물 | 비고 |
|------|--------|------|
| 2 | `bench/env.py` | 환경 스냅샷 (GPU·드라이버·torch·CUDA·git commit) |
| 3 | `bench/flops.py` | FLOPs + **미등록 연산 보고** |
| 4 | `bench/latency.py` | CUDA event 계측 |
| 5 | `bench/memory.py` | peak VRAM(allocated/reserved 둘 다) + OOM을 데이터로 |
| 6 | `models/registry.py` | `build_model` 단일 진입점, DeiT-S |
| 7 | `models/cmt.py`, `cmt_official.py` | CMT-S (upstream 바이트 동일 벤더링) |
| 9 | `bench/throughput.py` | 최대 배치 이진 탐색 + 처리량 |
| 10 | `experiments/e1_resolution_sweep.py` | sweep 오케스트레이션 |
| 12 | `figures/e1_plot.py` | CSV → 4패널 그림 |

Task 8, 11이 없으므로 `build_model("vim_s")`는 아직 `NotImplementedError`를 던진다.
`experiments/e1_resolution_sweep.py`를 그냥 실행하면 vim_s 5셀이 `status="error"`
행으로 기록된다 — 죽지는 않지만 그 CSV는 불완전하다.

### 남은 태스크 — 전부 WSL2 필요

| Task | 내용 | 막힌 이유 |
|------|------|-----------|
| 1 | 스모크 테스트 | `mamba_ssm` import + CUDA 커널 실행 확인 |
| 8 | Vim-S 통합 + selective scan FLOPs 핸들러 | 같음 |
| 11 | sanity check (DeiT-S 4.6G 대조) | vim_s 포함 파라미터화라 Task 8 의존 |
| 13 | 고정 환경에서 실측 실행 | Task 1·8·11 전부 선행 |

**순서: 1 → 8 → 11 → 13.**

## 선행 조건 — WSL2 설치

컨트롤러 세션이 비관리자라 대신 설치할 수 없다. 사용자가 직접:

```
시작 메뉴 → PowerShell 우클릭 → "관리자 권한으로 실행"
wsl --install
(재부팅)
```

설치 후 WSL2 안에서 고정 툴체인을 만든다. Vision Mamba의 selective scan CUDA 커널이
Linux를 요구하며, 순수 PyTorch로 대체하면 5~10배 느려져 latency 측정이 무의미해진다.

```
Python  3.10.13
torch   2.1.1+cu118
causal_conv1d >= 1.1.0
mamba-1p1p1
```

## 다음 세션에서 반드시 알아야 할 것

### 1. Windows에서 잰 수치는 논문에 못 쓴다

지금까지의 개발·테스트는 Windows(Python 3.12.10 / torch 2.6.0+cu124)에서 했다.
`bench/` 모듈은 토이 모델과 CPU만 쓰므로 버전 무관하지만, **`results/`에 커밋되는 측정값은
반드시 고정 환경에서 나와야 한다.** 현재 `results/`에는 `.gitkeep`뿐이다 — 그 상태가 맞다.

Task 13 Step 1이 고정 환경에서 `pytest tests/`를 재확인하는 관문이다. 결과가 Windows와
다르면 실측 전에 원인을 밝힐 것.

### 2. 이 프로젝트에서 가장 비싼 실패 모드

**fvcore는 핸들러가 없는 연산을 조용히 0으로 센다.** Vim의 selective scan이 정확히 이
경우이며, 놓치면 Vim의 FLOPs가 통째로 사라져 "Vim이 압도적으로 효율적"이라는 그럴듯한
오답이 나온다. 아무것도 깨진 것처럼 보이지 않는다는 점이 위험하다.

`count_flops`는 항상 `uncounted_ops`를 함께 반환하고, Task 11의 sanity check는 미등록
연산이 하나라도 남으면 실패한다. Task 8에서 `SELECTIVE_SCAN_OP` 문자열이 실제 연산자
이름과 다르면 핸들러가 등록되지 않고 조용히 0이 유지되므로, 브리프의 확인 절차를 반드시
밟을 것.

### 3. 진행 중 발견해 계획을 고친 것들

리뷰가 잡아낸 것 중 계획 자체의 결함이었던 항목:

- **OOM 판정이 좁았다** — `torch.cuda.OutOfMemoryError`만 잡으면 cuDNN workspace 실패
  경로의 OOM을 놓쳐 고해상도 셀에서 sweep이 죽는다. `is_oom`이 메시지 기반 판정을 함께
  하고, `bench/memory.py`와 `bench/throughput.py`가 이를 공유한다.
- **sweep이 중간 실패 시 전부 잃었다** — 마지막에 한 번만 CSV를 썼다. 기본 `MODEL_NAMES`에
  `vim_s`가 있어 실제 실행 시 10셀(GPU 1시간)을 재고 죽을 게 확정이었다. 이제 셀마다
  다시 쓰고 실패는 `status="error"` 행으로 남는다.
- **OOM 행에서 FLOPs를 버렸다** — `count_flops`는 CPU 트레이스라 OOM이 불가능한데 OOM이면
  재기 전에 return했다. 이제 FLOPs를 가장 먼저 잰다.
- **가중치 로딩을 E2/E3로 이관** — E1은 연산 비용만 재므로 가중치가 불필요하다.
  `build_model(..., pretrained=True)`는 `NotImplementedError`를 던진다.
- **그림이 측정 실패를 감췄다** — `status`가 `error`나 `no_cuda`인 행이 흔적 없이
  빠져, 측정에 실패한 셀과 아직 재지 않은 셀이 구분되지 않았다. 출처 없는 수치가
  문제였던 논문에 "왜 비었는지 알 수 없는 빈칸"을 넣는 셈이었다. 이제 `MISSING_STATUSES`가
  셋 다 색과 라벨로 구분한다.

### 테스트가 이름값을 하는지 확인할 것

Task 12에서 리뷰어가 OOM을 0으로 그리는 회귀를 코드에 직접 주입했더니 테스트 3건이
**그대로 통과**했다. `test_oom_rows_do_not_become_zero_points`가 `out.exists()`만
단언하고 있었기 때문이다 — PNG는 어느 쪽이든 만들어진다.

고친 방식은 단언 강화가 아니라 구조 변경이었다. "무엇을 그릴지" 판단을 `plotted_series`와
`missing_cells` 두 순수 함수로 빼서, matplotlib 내부를 뒤지지 않고 직접 검증한다.
남은 태스크에서도 같은 질문을 할 것 — **이 테스트는 자기가 막는다고 주장하는 회귀를
넣었을 때 실제로 실패하는가.**

그림에 그려지는 문자열은 전부 영어로 둔다. matplotlib 기본 폰트에 한글 글리프가 없어
PNG에 네모 상자로 찍힌다. 코드 주석과 docstring은 한글 그대로다.

### 4. 논문 개정 시 반영할 사실

CMT-S 공식 구현의 실제 파라미터는 **26.26M**이다. 논문 3.3절은 "동일한 25M 이하 파라미터
규모"라고 쓰는데 원논문 보고치(25.1M)와 공식 구현이 어긋난다. E1이 실측 파라미터를 CSV에
남기므로 개정 시 이 문장을 실측에 맞출 것.

## 이 계획 이후

- 계획 2: E2 ERF 정량 측정
- 계획 3: E3 effective attention 환원과 dilution 커버리지
- 계획 4: E4 2×2 요인 학습 ablation (Tiny-ImageNet, seed 3, epochs 300)
- 계획 5: 논문 v2 · 포트폴리오 프로젝트 · 벨로그 후속 글

E2·E3부터는 사전학습 체크포인트가 필요하다.

| 모델 | 파라미터 | ImageNet Top-1 | 출처 |
|------|----------|----------------|------|
| CMT-Small | 26.26M(실측) | 83.5% | huawei-noah/Efficient-AI-Backbones Releases |
| Vim-S | 26M | 80.5% | hustvl/Vim (HuggingFace) |
| DeiT-S | 22.05M(실측) | 79.8% | timm |
