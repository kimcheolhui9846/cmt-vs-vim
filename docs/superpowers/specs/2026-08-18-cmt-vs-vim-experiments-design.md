# CMT vs Vision Mamba — 실측 실험 설계

작성일: 2026-08-18

## 배경

「CMT와 Vision Mamba의 심층 비교 분석」(김철희, 이강국, 심규성)은 CNN-Transformer
하이브리드와 상태공간모델을 아키텍처·복잡도·수용 영역·이산화·수학적 패러다임의
다섯 축에서 대비한 3페이지 논문이다. 분석의 뼈대는 갖췄지만 **직접 측정한 수치가
하나도 없다.**

- 표 1의 FLOPs는 손으로 계산한 추정값이다.
- 표 2의 ImageNet 83.5% / 80.5%는 원논문 인용이다.
- 3.3절의 기여도 분해(계층구조 1.5%p, CNN stem 0.5%p, LPU 0.8%p, IRFFN 0.6%p)는
  CMT 논문의 ablation을 그대로 옮긴 것이다.
- Abstract는 "Experimental and theoretical analysis shows"라고 쓰지만 실제로는
  theoretical만 있다.

이 문서는 그 구멍을 실측으로 메우는 실험을 설계한다.

## 목표

1. 논문의 추정 FLOPs 표를 실측 표로 교체한다.
2. 인용에 의존하는 3.3절 ablation을 직접 수행한 통제 실험으로 대체한다.
3. 정성적으로만 서술된 ERF 차이와 softmax dilution을 정량 지표로 만든다.
4. 결과를 논문 v2, 포트폴리오 프로젝트, 벨로그 후속 글 세 갈래로 산출한다.

## 비목표

- ImageNet 전체 재학습. 8GB VRAM으로 불가능하다.
- 새 아키텍처 제안. E4의 Hierarchical Vim은 요인 분리를 위한 통제된 재구현이며,
  VMamba가 이미 같은 방향을 제시했다. 신규성을 주장하지 않는다.
- 원논문 보고 수치의 반박. 스케일이 달라 직접 비교 대상이 아니다.

## 제약

| 항목 | 값 |
|------|-----|
| GPU | RTX 3070 Ti, VRAM 8GB, 1장 |
| 기간 | 1~2개월 |
| OS | Windows 11 + WSL2 (Vim CUDA 커널이 Linux 요구) |

Vision Mamba의 selective scan을 순수 PyTorch로 대체하면 5~10배 느려져 latency
측정이 무의미해진다. 공식 CUDA 커널 사용은 선택이 아니라 선행 조건이다.

### 고정 툴체인

```
Python  3.10.13
torch   2.1.1+cu118
causal_conv1d >= 1.1.0
mamba-1p1p1
```

### 사전학습 체크포인트 (가용성 확인 완료)

| 모델 | 파라미터 | ImageNet Top-1 | 출처 |
|------|----------|----------------|------|
| CMT-Small | 25M | 83.5% | huawei-noah/Efficient-AI-Backbones, GitHub Releases |
| Vim-S | 26M | 80.5% | hustvl/Vim, HuggingFace |
| DeiT-S | 22M | 79.8% | timm |

파라미터가 22~26M로 이미 정렬되어 있어 별도 조정 없이 통제가 성립한다. DeiT는 두
원논문이 공통 기준선으로 삼았으므로 정렬 축으로 함께 측정한다.

## E1 — 해상도 sweep 실측

**검증 대상**: 논문 표 1(추정 FLOPs), 3.4절 cross-over point, "고해상도에서 Vim이
2.8배 FPS, 86.8% 메모리 절감"

DeiT-S / CMT-S / Vim-S를 224², 384², 512², 768², 1024²에서 측정한다.

| 측정 항목 | 방법 |
|-----------|------|
| FLOPs | fvcore 그래프 트레이스 |
| latency | CUDA event, batch=1, 워밍업 50회 후 100회 중앙값 |
| peak VRAM | `torch.cuda.max_memory_allocated` |
| throughput | 8GB에 들어가는 최대 배치에서 img/s |

정밀도는 fp32로 고정한다. AMP 결과는 부록으로만 둔다 — 두 축을 동시에 움직이면
해상도 효과와 정밀도 효과가 섞인다.

**예상되는 기여**: 논문 3.4절의 M = 2d = 768은 self-attention *내부에서* quadratic
항이 linear 항을 추월하는 지점이지, 모델 대 모델의 교차점이 아니다. 실측 latency의
교차점은 이와 다를 것으로 예상한다. CMT는 stage 4에서 k=1이라 full attention이
남아 있고, Vim은 커널 호출 오버헤드를 갖기 때문이다. "이론 교차점 vs 실측 교차점"의
대비 자체가 논문에 없던 결과다.

8GB에서 1024²는 attention 계열이 OOM 날 것으로 예상한다. **OOM은 실패가 아니라
기록할 결과다** — 메모리 주장의 직접 증거이므로 어느 해상도에서 어느 모델이 넘어갔는지
`results/`에 남긴다.

## E2 — Effective Receptive Field 정량 측정

**검증 대상**: 논문 3.1절 "CMT의 ERF는 2D Gaussian-like, Vim의 ERF는 1D scan
방향에 정합된 anisotropic"

Luo et al.(2016)의 방법으로 출력 중심 픽셀의 입력에 대한 gradient를 다수 이미지에
평균해 ERF map을 만든다. 핵심은 "이방성"을 서술이 아니라 숫자로 정의하는 것이다.

| 지표 | 정의 | 등방일 때 |
|------|------|-----------|
| 비등방 지수 | ERF 2차 모먼트 공분산의 √(λ_max/λ_min) | 1.0 |
| 주축 각도 | 주축과 수평 scan 방향 사이 각도 | 정의되지 않음 |
| 수직/수평 감쇠율 비 | 중심에서 각 축으로의 gradient 감쇠 기울기 비 | 1.0 |

세 번째 지표는 논문 3.1절의 "row-major flatten 시 수직으로 인접한 픽셀이 시퀀스
상 W만큼 떨어진다"를 직접 겨냥한다. 이 주장이 맞다면 Vim의 수직 감쇠가 수평보다
가팔라야 한다.

## E3 — Softmax dilution 정량화

**검증 대상**: 논문 3.2절 "객체가 K개 토큰에 걸치면 attention 평균 가중치가 1/K로
희석되지만 SSM은 상태에 누적되므로 희석되지 않는다"

세 실험 중 난이도가 가장 높다. attention 가중치와 SSM의 상태 노름은 척도가 달라
직접 비교하면 안 된다. Ali et al.의 effective attention으로 SSM을 attention 형태로
환원해 같은 축에 올린다.

```
w(t, s) = C_t · Ā_{t:s+1} · B̄_s
```

이 환원이 E3 작업량의 대부분을 차지한다.

**측정**: 객체가 차지하는 토큰 수 K에 대해, 기여도 상위 토큰 중 객체 내부에 있는
비율(커버리지)을 K 구간별로 집계한다. 논문 주장대로면 attention 계열은 K가 커질수록
커버리지가 떨어지고 SSM은 유지된다.

객체 마스크가 필요하므로 ImageNet-S 또는 PASCAL VOC를 사용한다.

**우선순위**: E1, E2보다 낮다. effective attention 환원이 막히면 E3를 제외해도
논문 v2가 성립하도록 의존 관계를 두지 않는다.

## E4 — 통제 학습 ablation (2×2 요인 설계)

**검증 대상**: 논문 3.3절 "차이의 대부분은 CMT가 주입한 2D 지역성과 multi-scale이라는
inductive bias에서 발생한다", 그리고 "Vim에 동등한 구조적 prior를 주입한 후속
모델에서는 격차가 1%p 미만으로 좁혀진다"

|  | 평면 구조 | 계층 4-stage |
|---|---|---|
| **Attention** | A. DeiT-Ti | C. CMT-Ti |
| **SSM** | B. Vim-Ti | D. Hierarchical Vim |

논문이 인용한 ablation은 CMT 내부에서만 뜯어본 값이다. 같은 구조적 prior가 SSM
쪽에서도 같은 효과를 내는지는 확인된 바 없다. 2×2 설계는 이를 분리한다.

- 구조 주효과 = 계층 평균 − 평면 평균
- 연산자 주효과 = Attention 평균 − SSM 평균
- **상호작용 = (D − B) − (C − A)**

상호작용이 핵심이다. 양수면 "계층 구조가 SSM에 더 큰 이득을 준다"는 뜻이고, 이는
논문이 인용으로만 처리한 VMamba·LocalMamba의 격차 축소를 본인 실험으로 뒷받침한다.

### 요인 정의의 한계 — 반드시 논문에 명시할 것

C칸의 CMT-Ti는 DeiT-Ti와 계층 구조만 다른 것이 아니다. LPU의 depth-wise conv,
IRFFN의 conv locality, LMHSA의 K/V 축소가 함께 들어온다. 따라서 위에서 "구조
주효과"라 부른 값은 **순수한 계층 구조의 효과가 아니라 "CMT식 구조적 prior 묶음
(계층 + conv locality)"의 효과**다.

이 둘을 분리하려면 conv locality를 세 번째 요인으로 넣어 2×2×2 = 8칸, seed 3개
기준 24 run이 필요하다. 1~2개월 예산을 넘어선다.

따라서 요인 이름을 "구조적 prior"로 규정하고, 계층 구조와 conv locality를 분리하지
않았다는 점을 논문의 한계로 명시한다. 이를 감추고 "계층 구조의 효과"라고 쓰면
논문 3.3절이 인용 수치를 자기 실험처럼 서술했던 것과 같은 종류의 문제를 반복하게
된다.

이 정의는 D칸의 구성을 강제한다. 상호작용 (D−B) − (C−A)가 의미를 가지려면 **B→D의
조작과 A→C의 조작이 같아야** 한다. A→C가 "계층 + conv locality"를 함께 넣으므로,
D의 Hierarchical Vim도 계층화와 conv locality를 **함께** 넣은 구성이어야 한다.
계층화만 적용하면 두 팔의 조작이 달라져 상호작용을 해석할 수 없다. 결과적으로 D는
VMamba·LocalMamba 계열과 가까워지는데, 이는 요인 설계의 대칭성이 요구하는 바이며
해당 연구들을 인용해 관계를 명시한다.

### 데이터셋

Tiny-ImageNet (200 클래스, 64×64, 학습 100k / 검증 10k).

CIFAR-100을 쓰지 않는 이유: 32×32를 patch 8로 자르면 토큰이 16개뿐이라 시퀀스
길이에 관한 주장이 아예 드러나지 않는다. 64×64가 최소선이다.

### 통제변인

`configs/*.yaml`에 고정한다.

- 파라미터 예산 ±5% 내 정렬
- epochs 300, AdamW, 동일 lr schedule
- DeiT augmentation recipe 동일 적용
- seed 3개, 결과는 평균과 표준편차 병기

### 비용

4칸 × 3 seed = 12 run. run당 5~10시간 추정으로 총 60~120시간(3~5일 연속).
GPU를 다른 용도로도 써야 해서 일정이 밀리면 seed 3→2, epochs 300→200 순으로
줄인다. 칸 수는 줄이지 않는다 — 하나라도 빠지면 요인 설계가 무너진다.

### 구현 부담

D칸이 가장 무겁다. 1D 시퀀스에 stage 간 다운샘플링을 넣으려면 2D reshape → patch
merge → 재flatten 경로를 직접 구현해야 한다. VMamba가 같은 문제를 4방향 scan으로
풀었으므로 반드시 인용하고, 본 실험의 D는 신규 제안이 아니라 요인 분리를 위한
통제된 재구현임을 논문에 명시한다.

## 측정 파이프라인 검증

측정 도구 자체가 틀렸으면 나머지가 전부 무의미하다. 각 실험에 알려진 값과 대조하는
sanity check를 선행한다.

| 실험 | 검증 | 기준값 |
|------|------|--------|
| E1 | DeiT-S 224² FLOPs | 공개값 4.6G |
| E2 | 랜덤 초기화 모델의 ERF 폭 | Luo et al. 보고대로 좁게 |
| E4 | Vim-Ti / DeiT-Ti 체크포인트 재평가 | 76.1% / 72.2% |

각 스크립트는 실행 시 GPU·드라이버·torch·CUDA 버전을 `results/`에 자동 기록한다.

## 재현성 원칙

측정값은 `results/`에 원시 형태(csv/json)로 커밋하고, 그림과 표는 반드시 그 파일을
읽어 생성한다. 손으로 옮겨 적은 숫자는 논문에 넣지 않는다. 개정 전 논문의 가장 큰
약점이 "이 수치가 어디서 나왔는가"에 답할 수 없다는 점이었고, 같은 실수를 반복하지
않기 위한 구조적 장치다.

## 산출물

### 1. 논문 v2

| 현재 | 개정 후 |
|------|---------|
| Abstract "Experimental and theoretical analysis" | 실험이 붙어 참이 됨 |
| 표 1 (추정 FLOPs) | E1 실측 + 이론/실측 교차점 대비 |
| 3.1 ERF 정성 서술 | E2 비등방 지수 수치 |
| 3.2 dilution 서술 | E3 커버리지 곡선 |
| 3.3 인용 ablation | E4 2×2 표 (인용 병기, 근거는 본인 실험) |
| 결론 "향후 연구: Hierarchical Vim" | 수행 완료로 변경 |

실험 설정을 기술하는 절을 신설한다. 분량이 3페이지에서 6~8페이지로 늘어나므로
**투고처 양식과 페이지 상한을 먼저 확인해야 한다.** 상한이 낮으면 E3를 부록으로
빼거나 투고처를 재선정한다.

### 2. 포트폴리오 프로젝트

`content/projects/cmt-vs-vision-mamba.ts` (Portfolio Implementation 저장소).

기존 `Project` 스키마에 그대로 맞는다. `experiments[]` 배열에 E4의 2×2 결과가
들어간다. `deployment`는 이 프로젝트에 해당 사항이 없으므로 템플릿 규칙대로 필드째
생략한다. `links`는 GitHub(public 전환 후), 논문 PDF, 벨로그 글.

### 3. 벨로그 후속 글

기존 글 「ViT의 한계를 극복하는 두 시선: CMT vs Vision Mamba (SSM) 심층 분석」은
이론 분석으로 완결되어 있으므로 수정하지 않는다. 같은 "AI Study" 시리즈에 실험편을
2편으로 추가하고, 1편 말미에 후속 링크를 단다. "이론으로 예측한 것을 직접 재봤다"는
서사가 자연스럽게 만들어진다.

대표 이미지는 E2의 ERF heatmap을 쓴다. 세 실험 중 시각적 대비가 가장 명확하다.

## 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| WSL2 미설치 | 전체 차단 | 관리자 권한 `wsl --install` + 재부팅. 사용자 선행 작업 |
| mamba-ssm 빌드 실패 | E1 latency, E4 B·D칸 차단 | 공식 Docker 이미지로 대체 |
| E3 effective attention 환원 실패 | E3 소실 | E3를 논문에서 제외. 다른 실험에 의존 관계 없음 |
| D칸 구현이 예산 초과 | 2×2 붕괴 | 구현을 E4 착수 전 별도 검증 단계로 분리 |
| 투고처 페이지 상한 | 실험 일부 누락 | 착수 전 양식 확인 |
| 8GB에서 학습 배치가 지나치게 작음 | 학습 시간 폭증 | gradient accumulation, 필요 시 epochs 축소 |

## 열린 항목

- 투고처와 마감일이 미정이다. 논문 v2 작성 시작 전에 확정해야 페이지 배분이 가능하다.
