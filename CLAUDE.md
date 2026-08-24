# cmt-vs-vim

김철희·이강국·심규성, "CMT와 Vision Mamba의 심층 비교 분석"(2026-05-15 대한전자공학회
제출) 논문의 **근거 없는 주장을 실측으로 대체하는** private 측정 저장소다. 포트폴리오
사이트와는 무관한 별개 저장소다.

**작업을 시작하기 전에 `HANDOFF.md`를 먼저 읽는다.** 현재 상태, 완료된 실험, 남은
태스크, 그리고 "이미 판단이 끝난 것"이 전부 거기 있다. 실험별 산출물은
`results/e1`, `results/e2`, `results/e3`의 각 `README.md`에 출처가 적혀 있다.

## 측정은 반드시 고정 환경에서 돌린다

Windows에서 잰 수치는 논문에 쓸 수 없다. 모든 python·pytest는 저장소 루트에서
이 래퍼를 통과해야 한다:

```
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run.sh python -m pytest -q
```

`tools/run.sh`는 고정 환경(`/opt/conda/envs/e1`, Python 3.10.13, torch 2.1.1+cu118,
CUDA 11.8, RTX 3070 Ti)이 없으면 exit 1로 죽는다. 죽으면 고쳐서 돌리지 말고 왜 없는지
확인한다 — 조용히 다른 python으로 돌아간 측정값이 가장 비싼 실패다.

## 이 저장소의 규칙

- **커밋 저자는 사용자 단독이다.** Claude/Anthropic co-author 트레일러를 넣지 않는다.
- **수치를 손으로 다시 계산하지 않는다.** 문서에 쓰는 모든 숫자는 `results/*/`의 CSV나
  그 실험의 사실 파일에서 나와야 한다. 반올림도 출처를 따른다.
- **사전 등록한 예측표는 측정 후에 고치지 않는다.** 빗나간 예측은 빗나간 그대로 적는다
  — 이 저장소에서 가장 값나가는 발견 몇 개가 빗나간 예측에서 나왔다.
- **ASCII 제약의 범위**: CSV의 범주 값과 matplotlib 캔버스에 그려지는 문자열만 해당한다.
  Markdown 산문과 CSV의 `error` 열은 한글이어도 된다.
- 문서는 한국어로 쓴다.
