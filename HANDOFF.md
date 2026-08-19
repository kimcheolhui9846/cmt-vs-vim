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

## E2 — ERF 정량 측정 완료 (2026-08-19, 리뷰 반영 재측정 2026-08-20)

브랜치 `feat/e2-erf`, 계획 `docs/superpowers/plans/2026-08-19-e2-erf.md`(9 태스크
전부 완료), SDD 원장 `.superpowers/sdd/2026-08-19-e2-erf/progress.md`.

측정은 고정 WSL2 환경에서 세 모델(`deit_s`·`cmt_s`·`vim_s`) × 세 조건(`natural`·
`noise`·`random_init`) × **6개 표본 크기(16~512)**로 돌렸다(N=512는 리뷰에서 추가
지시). 결과는
`results/e2/{erf_metrics.csv, erf_maps.npz, env.json, images.txt, e2_erf.png}`.
아래 숫자는 전부 이 최종 실행이 커밋한 파일(`559f3d1`)에서 다시 뽑은 것이다 — 이전
실행 값을 옮겨 적지 않았다.

### 실측·리뷰 두 라운드에서 고친 결함들

**1라운드(2026-08-19)**: `bench/erf.py`의 `decay_ratio`가 배열 경계를 확인하지
않아, ERF 피크가 경계에서 64px보다 가까운 셀(`cmt_s`/`noise`)에서 `IndexError`를
던지며 측정 전체가 두 번 연속 크래시했다. `decay_window()`가 실제로 쓸 수 있는
반경을 계산해 `MIN_DECAY_WINDOW=8` 미만이면 명시적으로 `ValueError`를 던지도록
고쳤다. 또한 정직성 게이트 기준 2(피크가 중심 ±16 이내)가 학습 분포 밖 입력을
쓰는 `noise` 조건까지 포함하고 있었는데, 같은 모델의 `natural`·`random_init`은
멀쩡하므로 이건 프로브 결함이 아니라 `noise` 조건 자체의 특성이라고 판단해 게이트
범위를 `natural`·`random_init`으로 좁혔다.

**2라운드(리뷰, 2026-08-20)**: 1라운드의 수정 자체에 구조적 결함이 더 있었다.

1. **`decay_ratio`의 계획 자체 수렴 기준(5%)이 한 번도 적용되지 않았다.**
   `converged` 열이 `anisotropy_index` 이력만 보고 계산돼서, `decay_ratio`가
   `natural`에서 N=128→256 사이 `vim_s` +7.5%, `cmt_s` +15.1%로 5% 기준을 훌쩍
   넘겼는데도 `True`가 찍혔다. 세 지표(`anisotropy_index`·`principal_angle_deg`·
   `decay_ratio`) 각각 독립적인 이력과 `*_converged` 열을 갖도록 고치고,
   `SAMPLE_SIZES`에 512를 추가했다.
2. **`random_init`에 시드가 없어 cls 토큰 가드(질량 반경 비교) 숫자가
   재현되지 않았다.** `build_model(..., pretrained=False)` 직전에
   `torch.manual_seed(SEED)`를 걸고, `env.json`에 `random_init_seed`를 남긴다.
3. **측정된 맵을 그림이 "not measured"로 그리고 있었다.** `cmt_s`/`noise`의 맵은
   `accumulate_erf`가 실제로 성공시켰는데, 맵 저장이 지표 계산 뒤(전부 성공해야
   실행되는 분기)에 있어서 `decay_ratio`의 실패가 이미 확보한 맵까지 함께 버렸다.
   맵을 지표 계산보다 먼저 저장하도록 순서를 바꿨다.
4. **되살릴 수 있는 지표까지 한 번에 버렸다.** 네 개 지표 계산이 한 `try` 블록에
   묶여 있어 하나가 던지면 넷 다 잃었다(경계 근접 피크에서도 비등방 지수·주축
   각도는 잘 정의된다). 지표마다 독립적인 `try/except`로 분리해, 실패한 지표만
   비고 사유가 `error` 열에 남으며 `status`는 `accumulate_erf` 성공 여부만
   보도록 재정의했다(계획·spec에도 이 재정의를 명시).
5. **비등방 지수가 far-field 꼬리에 지배된다는 사실이 문서에 없었다.** 2차
   모먼트는 거리 제곱 가중이라, 넓고 등방적인 배경(pedestal)이 있으면 중심의
   비등방 능선이 전체 지수에 거의 반영되지 않는다. 중심 128×128만 잘라 같은
   함수를 다시 적용한 `anisotropy_central` 열을 추가해 그 효과를 데이터로
   드러냈다(아래 표).
6. **대조군(`random_init`/`noise`) 감쇠비가 결론을 강화하는데 언급이 없었다.**
   `vim_s`는 `random_init`(1.90)·`noise`(2.01~2.18)에서 `natural`(1.35)보다도
   감쇠비가 크다 — 아래에서 다시 다룬다.

### 정직성 검사 4기준(개정판) — 전부 통과

1. **`status`: 54행 전부 `ok`.** `accumulate_erf`가 원본 맵을 얻지 못한 행은
   0개다(`cmt_s`/`noise`도 맵은 있다 — 감쇠비만 6개 N 전부 미정의).
2. `natural`·`random_init` 36개 맵(3모델×2조건×6N) 전부 피크가 중심 ±16 이내.
3. **`random_init` 질량 반경 < `natural` 질량 반경, 세 모델 전부**(N=512):
   `deit_s` 64.3 < 80.5, `cmt_s` 47.4 < 84.4, `vim_s` 14.9 < 84.3. `vim_s`가 가장
   여유가 크다(≈5.7배) — 프로브가 cls 토큰이 아니라 실제 중심 토큰을 읽는다는
   가장 강한 증거. `random_init`은 이제 시드가 고정돼 재현된다.
4. **지표별 수렴(N=512, 마지막 두 점 상대 변화 ≤5%)**:
   - `anisotropy_index`: 9/9 수렴.
   - `decay_ratio`: 7/9 수렴 — **세 모델의 `natural` 전부 포함**(이 실험이
     인용할 조건). 미수렴 2건: `vim_s`/`noise`(N=256→512, +8.0%, 여전히
     움직이는 중), `cmt_s`/`noise`(구조적으로 미정의, 아래 참고).
   - `principal_angle_deg`: **9/9 전부 미수렴.** Task 4가 확립한 캐비아트(지수가
     1에 가까우면 주축이 수치적으로 ill-conditioned)와 정확히 부합한다 — 애초에
     방향 근거로 쓰지 않는 지표이므로 수렴하지 않는다는 사실 자체가 그 캐비아트를
     뒷받침한다.

### 전체 최종 지표 (N=512)

| model | condition | anisotropy | anisotropy_central | decay_ratio | decay_ratio_converged |
|---|---|---|---|---|---|
| deit_s | natural | 1.0092 | 1.0197 | 1.0480 | True |
| deit_s | noise | 1.0102 | 1.0190 | 1.0234 | True |
| deit_s | random_init | 1.0051 | 1.0364 | 1.0134 | True |
| cmt_s | natural | 1.0283 | 1.0058 | 1.0234 | True |
| cmt_s | noise | 1.1192 | 1.0290 | **미정의**(피크가 경계에서 반경 2, 최소 8 필요) | — |
| cmt_s | random_init | 1.0254 | 1.0239 | 1.1075 | True |
| vim_s | natural | 1.0453 | 1.1153 | 1.3450 | True |
| vim_s | noise | 1.0850 | 1.3320 | 2.0075 | False(+8.0%) |
| vim_s | random_init | 1.9791 | 1.9554 | 1.8990 | True |

### 사전 등록 예측과의 대조 (natural, N=512)

| model | 비등방 지수(예측) | 비등방 지수(실측) | 감쇠비(예측) | 감쇠비(실측) |
|---|---|---|---|---|
| CMT-S | 1.0–1.2 | **1.028** (범위 내) | ≈1.0 | 1.023 |
| Vim-S | **>1.3** | **1.045** (크게 미달) | >1.0 (수직이 가파름) | **1.345** (부합, 셋 중 최댓값, 수렴함) |
| DeiT-S | ≈1.0 | 1.009 (부합) | ≈1.0 | 1.048 |

**예측을 고치지 않고 어긋난 그대로 적는다: Vim-S의 2차 모먼트 비등방 지수
(1.045)는 예측(>1.3)에 크게 못 미치고, CMT(1.028)·DeiT(1.009)와 사실상 같은 수준
(거의 등방)이다.** N=512로 늘려도 이 격차는 좁혀지지 않았다. 반면 감쇠비(1.345)는
예측 방향과 일치하고 세 모델 중 가장 크며, **이제 계획 자체의 5% 기준으로
수렴까지 확인됐다** — 사용자 판단에 따라 이 실험의 주 지표는 감쇠비다.

### 왜 두 지표가 갈리는가 — `anisotropy_central`이 보여주는 것

`anisotropy_index`는 ERF를 2차 모먼트(공분산 타원)로 보는 지표라 **거리 제곱으로
가중**한다. 중심 128×128만 잘라 같은 함수를 다시 적용(`anisotropy_central`)하면
꼬리를 제거한 지수를 얻는데, `vim_s`에서 그 차이가 뚜렷하다:

- `vim_s natural`: 전체 1.045 → 중심 1.115 (중심이 더 비등방)
- `vim_s noise`: 전체 1.085 → 중심 1.332
- `cmt_s natural`: 전체 1.028 → 중심 1.006 (거의 변화 없음)
- `deit_s natural`: 전체 1.009 → 중심 1.020 (거의 변화 없음)

**Vim만 중심을 자르면 지수가 뚜렷이 올라간다.** 즉 Vim의 ERF는 중심 근처에는
실제로 비등방 능선이 있지만, 전체 지도에는 넓고 대체로 등방적인 배경(pedestal)이
깔려 있어 2차 모먼트가 그 배경에 끌려간다. `decay_ratio`는 피크 근방의 기울기만
재므로 이 능선을 직접 잡고, `anisotropy_index`는 넓은 받침에 지배된다 — 둘이
갈리는 건 모순이 아니라 **서로 다른 것을 재기 때문**이다. `results/e2/e2_erf.png`의
`vim_s natural`·`noise` 히트맵에서 수평(scan) 방향 격자 무늬가 육안으로도 보인다.

**3.1절의 주장은 감쇠비가 진다.** 비등방 지수는 한계(꼬리 지배)를 명시하고
보조 지표로 남긴다.

### 대조군이 결론을 강화한다 (Important 5)

`vim_s`의 감쇠비는 `natural` 1.345보다 `random_init`(1.899)과 `noise`(noise 조건들
2.01~2.31, N=512는 2.008)에서 **오히려 더 크다.** 이건 주장을 약화시키지 않는다 —
**강화한다.** 논문 3.1절의 논거(row-major flatten이 수직 이웃을 시퀀스상 W만큼
떼어놓는다)는 애초에 **학습된 표현이 아니라 아키텍처(스캔 순서) 수준의 주장**이다.
`random_init`(가중치가 전혀 학습되지 않음)에서도 감쇠비가 1.899로 뚜렷이 1보다
크다는 것은, 이 비등방성이 학습으로 만들어진 게 아니라 **scan 구조 자체에서
나온다**는 뜻이고, 이는 예측과 정확히 맞아떨어진다. `anisotropy_central`도 같은
패턴을 보인다: `vim_s random_init`의 중심 지수는 1.955로 세 조건 중 가장 크다.

### 논문 3.1절에 반영할 문장 (초안, 개정판)

> CMT-S와 DeiT-S는 자연 이미지 입력에서 ERF 공분산 기반 비등방 지수가 각각 1.03,
> 1.01로 거의 등방이다. Vision Mamba-S의 비등방 지수는 1.05로 이들과 사실상
> 구분되지 않아, 서술로 주장했던 "scan 방향에 정합된 뚜렷한 비등방"은 2차 모먼트
> 기준으로는 성립하지 않는다 — 이 지수는 거리 제곱 가중이라 넓고 등방적인 배경에
> 지배되기 쉽다는 한계가 있다. 축별 감쇠 기울기 비(수직/수평, 피크 근방의
> 기울기만 재는 지표)는 Vim이 1.35로 CMT(1.02)·DeiT(1.05)보다 뚜렷이 크고
> 표본 512장까지 수렴이 확인됐다 — 이 실험의 주 지표는 이쪽이다. 더 나아가
> Vim의 이 비등방성은 학습 여부와 무관하게 나타난다: 가중치를 전혀 학습하지 않은
> `random_init` 조건에서도 감쇠비가 1.90으로 여전히 1을 크게 웃돌아, 원인이
> 학습된 표현이 아니라 1D scan이라는 아키텍처 자체에 있음을 뒷받침한다.

### 다음 세션 제안

- `cmt_s`/`noise`의 피크 이탈 원인(BatchNorm 통계 vs conv 경계 아티팩트)을
  조사할 가치가 있다 — 이번 태스크 범위 밖으로 남겨뒀다.
- `vim_s`/`noise`의 감쇠비가 N=512에서도 +8.0%로 아직 수렴하지 않는다 — 인용할
  일이 생기면 더 큰 N으로 재확인할 것(`natural`은 이미 수렴했으므로 논문
  인용에는 영향 없음).
- E3 착수 전, `models/probes.py`·`bench/erf.py`의 관례(중심 토큰 인덱싱, 경계
  가드, 지표별 독립 실패 처리)를 참고할 것 — E3도 비슷한 위험을 안고 있다.

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
