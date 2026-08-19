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
- 테스트: **97건 전부 통과** (고정 환경 기준. Windows에서는 `mamba_ssm`이 없어
  `tests/test_vim.py`가 수집 단계에서 실패한다 — 정상이다)
- 계획: `docs/superpowers/plans/2026-08-18-e1-resolution-sweep.md` (13 태스크 **전부 완료**)
- SDD 원장: `.superpowers/sdd/2026-08-18-e1-resolution-sweep/progress.md` — git-ignored,
  태스크별 완료·수정 라운드·이월된 minor가 전부 기록돼 있다

### 완료된 태스크

| Task | 산출물 | 비고 |
|------|--------|------|
| 1 | `tests/test_smoke.py`, `requirements.txt` | 고정 환경 구축. skip 없이 하드 실패하는 3건 |
| 2 | `bench/env.py` | 환경 스냅샷 (GPU·드라이버·torch·CUDA·git commit) |
| 3 | `bench/flops.py` | FLOPs + **미등록 연산 보고**, traced/analytic 분리 |
| 4 | `bench/latency.py` | CUDA event 계측 + **반복 측정** |
| 5 | `bench/memory.py` | peak VRAM(allocated/reserved 둘 다) + OOM을 데이터로 |
| 6 | `models/registry.py` | `build_model` 단일 진입점, DeiT-S |
| 7 | `models/cmt.py`, `cmt_official.py` | CMT-S (upstream 바이트 동일 벤더링) |
| 8 | `models/vim.py`, `vim_official.py`, `rope.py` | Vim-S + fused op FLOPs 핸들러 |
| 9 | `bench/throughput.py` | 최대 배치 이진 탐색 + 처리량 |
| 10 | `experiments/e1_resolution_sweep.py` | sweep 오케스트레이션 |
| 11 | `tests/test_sanity.py` | DeiT-S 공개값 4.6G 대조 (실측 4.6083G, 비율 1.002) |
| 12 | `figures/e1_plot.py` | CSV → 5패널 그림 |
| 13 | `results/e1/` | 고정 환경 실측 (`sweep.csv`, `env.json`, `e1_sweep.png`) |

## 고정 측정 환경 (구축 완료)

WSL2 Ubuntu 위에 conda 환경 `/opt/conda/envs/e1`로 만들어 뒀다. 전체 핀과 빌드
절차는 `requirements.txt`에 있다 — causal_conv1d는 PyPI sdist에 `csrc/`가 없어
GitHub 태그에서, mamba는 stock mamba-ssm이 아니라 Vim 포크(`mamba-1p1p1`)에서
빌드해야 한다. 그 파일을 읽지 않고 재구축하려 들면 하루를 잃는다.

Linux가 필요한 이유는 Vision Mamba의 selective scan CUDA 커널이다. 순수 PyTorch로
대체하면 5~10배 느려져 latency 측정이 무의미해진다.

```
Python  3.10.13    torch 2.1.1+cu118    CUDA 11.8
timm    0.9.12     mamba-1p1p1 (Vim 포크)    causal_conv1d 1.1.0
RTX 3070 Ti (sm_86), 드라이버 591.86
```

저장소 명령을 이 환경에서 돌리는 래퍼는 스크래치패드의 `wsl/run.sh`에 있다.
없으면 다시 만들면 된다 — `CUDA_HOME`·`PATH`·`LD_LIBRARY_PATH`를 위 env로 잡고
`CC=gcc-11`을 걸어 저장소 루트에서 실행하는 8줄짜리다.

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

### 5. latency 반복 측정 — 실행 안의 편차는 작고, 실행 사이의 편차가 크다

`bench/latency.py`가 측정 블록(워밍업 50 + 계측 100)을 3번 반복하고, sweep이
`latency_min_ms`·`latency_max_ms`·`latency_repeats_ms`를 CSV에 남긴다. 반복마다
워밍업을 다시 도는 게 핵심이다 — 밖으로 빼면 첫 블록이 만든 클럭·할당자 상태를
물려받아 편차가 측정에서 지워진다.

**그런데 이 반복은 잡으려던 것을 잡지 못했다.** 한 프로세스 안의 반복은 전부
잘 맞는다(최대 1.10배, 15셀 중 12셀이 1.03배 이내). 반면 같은 셀을 서로 다른
실행에서 잰 값은 vim_s@224에서 30.005 ms와 18.134 ms로 **1.66배** 갈렸다.

| | 최대 편차 | 어디서 |
|---|---|---|
| 실행 안 (반복 3회) | 1.10배 | deit_s@224 |
| 실행 사이 (독립 sweep 2회) | 1.66배 | vim_s@224 |

즉 배치 1 latency의 불확실성은 프로세스 경계에 있다. `LATENCY_SPREAD_WARN = 1.2`
경고는 이번 실행에서 한 번도 뜨지 않았는데, 그게 "재현된다"는 뜻이 아니다.

**결정 (2026-08-19)**: 독립 프로세스 반복 sweep은 하지 않는다. 그 GPU 시간이
사는 것은 논문 주장에 쓰이지 않는 구간의 정밀도다 — 표 1의 주장은 고해상도
cross-over에 걸려 있고, 그 구간(768² 이상)은 실행 안·사이 모두 1.04배 이내다.
대신 한계로 명시한다.

**논문에 쓸 때**:

- 768²·1024² latency는 단일 값으로 인용해도 된다.
- 224²·384²는 단일 값으로 인용하지 말 것. "실행 간 최대 1.66배 변동"을 함께
  적고, 그 위에 결론을 세우지 말 것.
- 근거 데이터는 손으로 옮겨 적지 말 것. 독립 두 실행의 원본이 git에 그대로 있다 —
  실행 A는 `fa35e51:results/e1/sweep.csv`, 실행 B는 `e90ba8e:results/e1/sweep.csv`.
  둘 다 같은 고정 환경에서 나왔고, `git show <해시>:<경로>`로 꺼내 대조하면 위
  숫자가 재현된다.

## E2 — ERF 정량 측정 완료 (2026-08-19)

브랜치 `feat/e2-erf`, 계획 `docs/superpowers/plans/2026-08-19-e2-erf.md`(9 태스크
전부 완료), SDD 원장 `.superpowers/sdd/2026-08-19-e2-erf/progress.md`.

측정은 고정 WSL2 환경에서 세 모델(`deit_s`·`cmt_s`·`vim_s`) × 세 조건(`natural`·
`noise`·`random_init`) × 5개 표본 크기(16~256)로 돌렸다. 결과는
`results/e2/{erf_metrics.csv, erf_maps.npz, env.json, images.txt, e2_erf.png}`.

### 실측 도중 발견해 고친 결함 두 가지

1. **`bench/erf.py`의 `decay_ratio`가 배열 경계를 확인하지 않았다.** ERF 피크가
   경계에서 64px보다 가까운 셀에서 `IndexError`를 던졌는데, `experiments/e2_erf.py`의
   셀별 `try/except`가 `accumulate_erf` 호출만 감싸고 지표 계산은 감싸지 않아 이
   예외가 측정 전체를 죽였다. 첫 실측 시도가 정확히 이 지점(`cmt_s`/`noise`/N=16)에서
   두 번 연속 크래시했다 — 프로세스 킬이 아니라 재현 가능한 파이썬 예외였다.
   `decay_window()`가 피크에서 실제로 쓸 수 있는 반경(상하좌우 여유와
   `max_distance` 중 최솟값)을 계산하고, 그 반경이 `MIN_DECAY_WINDOW=8`(패치 크기
   16의 절반, 기울기 적합에 필요한 최소 표본) 미만이면 조용히 좁은 창으로
   계산하지 않고 명시적으로 `ValueError`를 던진다. `e2_erf.py`는 이제 지표 계산까지
   같은 `try` 블록에 넣어, 한 셀이 어떤 이유로 실패해도 그 셀만 `status="error"`
   행이 되고 나머지 44셀은 살아남는다.
2. **정직성 게이트 기준 2(모든 맵의 피크가 중심 ±16 이내)가 `noise` 조건까지
   포함하고 있었는데, 이건 게이트의 설계 의도 밖이었다.** `cmt_s`는 `natural`·
   `random_init`에서는 피크가 중심 ±16 이내로 멀쩡한데 `noise`(사전학습 가중치 +
   `torch.randn` 입력)에서만 5개 표본 크기 전부 피크가 이미지 모서리(중심에서
   ~110px)로 튄다. 같은 모델의 다른 두 조건이 정상이므로 프로브 인덱스 매핑은
   무죄이고 — 학습 분포 밖 입력에서 conv+BatchNorm 스택이 보이는 실제 반응으로
   본다. 게이트를 `natural`·`random_init`에만 적용하도록
   `docs/superpowers/specs/2026-08-19-e2-erf-design.md`·
   `docs/superpowers/plans/2026-08-19-e2-erf.md`를 고쳤다. cls 토큰 오인을 잡는
   핵심 게이트(기준 3, `random_init` vs `natural` 질량 반경 비교)는 건드리지
   않았다.

### 정직성 검사 4기준 — 전부 통과

1. `status`: 45행 중 40행 `ok`, 5행(`cmt_s`/`noise` 전 표본 크기) `error` — 전부
   위 1번 결함과 같은, 설명된 단일 원인.
2. `natural`·`random_init` 9개 model×condition의 맵 전부 피크가 중심 ±16 이내.
3. **`random_init` 질량 반경 < `natural` 질량 반경, 세 모델 전부**(N=256):
   `deit_s` 66.6 < 80.0, `cmt_s` 47.6 < 84.2, `vim_s` 14.9 < 84.1. `vim_s`가 가장
   여유가 크다 — 프로브가 cls 토큰이 아니라 실제 중심 토큰을 읽는다는 가장 강한
   증거.
4. `status="ok"`인 9개 조합 전부 N=256에서 수렴(마지막 두 점 상대 변화 ≤5%).

### 실측값 (natural, N=256) — 사전 등록 예측과 대조

| model | 비등방 지수(예측) | 비등방 지수(실측) | 감쇠비(예측) | 감쇠비(실측) |
|---|---|---|---|---|
| CMT-S | 1.0–1.2 | **1.036** (범위 내) | ≈1.0 | 1.075 |
| Vim-S | **>1.3** | **1.048** (크게 미달) | >1.0 (수직이 가파름) | **1.401** (부합, 셋 중 최댓값) |
| DeiT-S | ≈1.0 | 1.003 (부합) | ≈1.0 | 1.058 |

**예측을 고치지 않고 어긋난 그대로 적는다: Vim-S의 2차 모먼트 비등방 지수(1.048)는
예측(>1.3)에 크게 못 미치고, CMT(1.036)·DeiT(1.003)와 사실상 같은 수준(거의
등방)이다.** 반면 감쇠비(1.401)는 예측 방향과 일치하고 세 모델 중 가장 크다.

**두 지표가 갈리는 게 이 실험의 핵심 결과다.** `anisotropy_index`는 ERF 전체의
2차 모먼트(공분산 타원)로 "덩어리 모양"을 재는 지표라 분포의 벌크(대부분의 질량)에
지배된다. `decay_ratio`는 중심에서 각 축 방향으로 gradient가 얼마나 빨리
줄어드는지(꼬리 형태)를 직접 잰다. `figures/e2_plot.py`가 그린
`results/e2/e2_erf.png`에서 `vim_s`의 `natural`·`noise` 히트맵을 보면 수평
방향(scan 방향)으로 격자 무늬가 눈에 띄게 이어지는데, 이건 covariance 타원으로는
거의 안 잡히지만 꼬리 감쇠 프로파일에는 남아 있다. 즉 **Vim의 ERF는 "덩어리
모양"으로는 CMT·DeiT와 구분되지 않을 만큼 등방에 가깝지만, 꼬리의 감쇠 방향성은
남아 있다** — 논문 3.1절의 "1D scan 방향에 정합된 anisotropic"이라는 서술은 이
실측으로는 지지되지 않고, 대신 "2차 모먼트 기준으로는 등방에 가까우나 감쇠
프로파일에 방향성이 남는다"로 다시 써야 한다.

각도(`principal_angle_deg`)는 세 모델 전부 비등방 지수가 1에 가까운 영역(1.0~1.05)
이라 Task 4가 확립한 캐비아트대로 수치적으로 ill-conditioned하다 — 개별 각도값
(deit_s 77.8°, cmt_s 84.3°, vim_s 17.8°)을 논문에서 방향 근거로 쓰지 않는다.

`noise` 조건은 게이트 대상은 아니지만 그 자체로 결과다: `cmt_s`만 이 조건에서
피크가 이미지 모서리로 튄다(`deit_s`·`vim_s`는 `noise`에서도 멀쩡히 중심에
남는다). 분포 밖 입력에 대한 CMT 고유의 반응으로 보이며, 원인(BatchNorm 통계·
zero-padding 경계 아티팩트 추정)은 미확인 — 필요하면 별도로 조사할 것.

### 논문 3.1절에 반영할 문장 (초안)

> CMT-S와 DeiT-S는 자연 이미지 입력에서 ERF 공분산 기반 비등방 지수가 각각 1.04,
> 1.00으로 거의 등방이다. Vision Mamba-S의 비등방 지수는 1.05로 이들과 사실상
> 구분되지 않아, 서술로 주장했던 "scan 방향에 정합된 뚜렷한 비등방"은 2차 모먼트
> 기준으로는 성립하지 않는다. 다만 축별 감쇠 기울기 비(수직/수평)는 Vim이 1.40으로
> CMT(1.08)·DeiT(1.06)보다 뚜렷이 크며, 이는 예측한 방향과 일치한다 — Vim의 ERF는
> 전체 형태로는 등방에 가깝지만 꼬리의 감쇠 속도에는 scan 방향 구조가 남아 있다.

### 다음 세션 제안

- `cmt_s`/`noise`의 피크 이탈 원인(BatchNorm 통계 vs conv 경계 아티팩트)을
  조사할 가치가 있다 — 이번 태스크 범위 밖으로 남겨뒀다.
- E3 착수 전, `models/probes.py`·`bench/erf.py`의 관례(중심 토큰 인덱싱, 경계
  가드)를 참고할 것 — E3도 비슷한 인덱스 오독 위험을 안고 있다.

## 이 계획 이후

- ~~계획 2: E2 ERF 정량 측정~~ **완료 (2026-08-19)**
- 계획 3: E3 effective attention 환원과 dilution 커버리지
- 계획 4: E4 2×2 요인 학습 ablation (Tiny-ImageNet, seed 3, epochs 300)
- 계획 5: 논문 v2 · 포트폴리오 프로젝트 · 벨로그 후속 글

E2·E3부터는 사전학습 체크포인트가 필요하다.

| 모델 | 파라미터 | ImageNet Top-1 | 출처 |
|------|----------|----------------|------|
| CMT-Small | 26.26M(실측) | 83.5% | ggjy/CMT.pytorch Releases |
| Vim-S | 26M | 80.5% | hustvl/Vim (HuggingFace) |
| DeiT-S | 22.05M(실측) | 79.8% | timm |
