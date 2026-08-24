# CMT vs Vision Mamba — 실측 비교 실험

CNN-Transformer 하이브리드(CMT)와 상태공간모델(Vision Mamba)의 비교 분석 논문에
**직접 측정한 실험 근거**를 붙이기 위한 저장소.

기존 논문은 아키텍처·복잡도·수용 영역을 이론적으로 대비했지만, 표에 실린 수치가
전부 원논문 인용이거나 추정값이었다. 이 저장소는 그 두 구멍 — 추정 FLOPs 표와
인용 ablation — 을 실측으로 대체하는 것을 목표로 한다.

## 실험 구성

| ID | 실험 | 검증 대상 | 학습 필요 |
|----|------|-----------|-----------|
| E1 | 해상도 sweep 실측 | 복잡도 cross-over point, 고해상도 효율 주장 | 아니오 |
| E2 | Effective Receptive Field 정량 측정 | CMT의 등방성 vs Vim의 scan 정합 이방성 | 아니오 |
| E3 | Softmax dilution 정량화 | 큰 객체에서 attention이 희석된다는 주장 | 아니오 |
| E4 | 통제 학습 ablation | 정확도 차이가 inductive bias에서 온다는 주장 | 예 |

E1~E3는 공개된 ImageNet 사전학습 체크포인트로 측정한다. E4는 Tiny-ImageNet 규모의
**축소 대리 실험**이며, ImageNet 원 수치와 직접 비교할 수 없다는 점을 결과 해석에서
명시한다.

## 실행 환경

Vision Mamba의 selective scan CUDA 커널이 Linux를 요구하므로 WSL2에서 실행한다.
순수 PyTorch 구현으로 대체하면 커널 최적화가 빠져 latency 측정이 무의미해진다.

```
Python  3.10.13
torch   2.1.1+cu118
causal_conv1d >= 1.1.0
mamba-1p1p1
```

## 사전학습 체크포인트

| 모델 | 파라미터 | ImageNet Top-1 | 출처 |
|------|----------|----------------|------|
| CMT-Small | 25M | 83.5% | ggjy/CMT.pytorch (GitHub Releases) |
| Vim-S | 26M | 80.5% | hustvl/Vim (HuggingFace) |
| DeiT-S | 22M | 79.8% | timm |

DeiT는 두 원논문이 공통 기준선으로 삼았으므로 정렬 기준으로 함께 측정한다.

## 디렉터리

```
models/       CMT, Vim, DeiT 정의와 E4용 변형
experiments/  E1~E4 실행 스크립트
configs/      변형별 통제 설정 — 통제변인을 파일로 고정한다
results/      원시 측정값 (csv/json). 커밋 대상
figures/      results/ 를 읽어 그림을 생성하는 스크립트
docs/         논문 개정본과 실험 노트
```

## 재현성 원칙

측정값은 `results/`에 원시 형태로 커밋하고, 그림과 표는 반드시 그 파일을 읽어
생성한다. 손으로 옮겨 적은 숫자를 논문에 넣지 않는다 — 개정 전 논문의 가장 큰
약점이 "이 수치가 어디서 나왔는가"에 답할 수 없다는 점이었다.
