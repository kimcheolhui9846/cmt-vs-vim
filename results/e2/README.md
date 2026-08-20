# results/e2 — E2 ERF 측정 결과의 출처

이 디렉터리의 숫자가 어디서 왔는지, 어느 것이 측정이고 어느 것이 사후 계산인지
적어 둔다. 지금까지 이 정보는 커밋 메시지에만 있었고, `results/e2/`만 여는 사람은
볼 수 없었다.

## 파일

| 파일 | 내용 |
|------|------|
| `erf_maps.npz` | 원본 ERF 맵 54개. 키 형식 `{model}__{condition}__n{N}` (예: `vim_s__natural__n512`). 각 맵은 224×224 float64. |
| `erf_metrics.csv` | 맵에서 뽑은 지표 54행. 아래 "사후 계산된 열" 참고. |
| `env.json` | 측정 환경 스냅샷. 아래 "env.json의 git_commit" 참고. |
| `images.txt` | `natural` 조건에 실제로 쓴 VOC 이미지 512장의 파일명. N이 작은 행은 이 목록의 **앞에서부터** 잘라 쓴다. |
| `e2_erf.png` | `figures/e2_plot.py`가 위 두 파일에서 그린다. 언제든 다시 그릴 수 있다. |

**맵이 원본이다.** CSV의 모든 지표는 `erf_maps.npz`에서 다시 계산할 수 있고,
아래 backfill들이 실제로 그렇게 만들어졌다. 지표가 의심스러우면 맵에서 다시
계산해서 대조할 것 — CSV를 손으로 고치지 말 것.

## 맵을 만든 커밋

`erf_maps.npz`와 `env.json`은 **`559f3d1`**("measure: re-run with N=512 and the
decoupled per-metric pipeline")이 커밋한 실행에서 나왔다. 그 실행을 만든 코드는
`559f3d1` 시점의 트리다 — 특히 지표별 독립 try/except(`64e0214`)와 맵 우선
저장이 들어간 뒤의 코드다.

## env.json의 `git_commit`이 `d9b45a2`인 이유

`env.json`의 `git_commit`은 `d9b45a2`인데, 이 커밋은 `64e0214`(지표 분리)보다
**앞선다.** 즉 그 해시를 체크아웃하면 커밋된 CSV 스키마를 만들 수 없는 코드가
나온다. 오기가 아니라 기록 방식의 한계다:

`bench/env.py`의 `snapshot()`은 실행 시점의 **HEAD**를 읽는다. 이 실행을 시작할 때
파이프라인 수정은 아직 작업 트리에만 있었고 커밋되지 않은 상태였으므로, HEAD는
여전히 직전 측정 커밋 `d9b45a2`를 가리켰다. 기록된 값은 "그때 HEAD가 무엇이었나"
로는 정확하고, "이 결과를 낸 코드가 무엇인가"로는 부정확하다.

**후자를 알고 싶으면 `559f3d1`을 보라.** `git_commit` 필드는 고치지 않는다 —
측정 시점에 실제로 기록된 값이고, 사후에 바꾸면 기록이 아니라 사후 서술이 된다.

## 사후 계산된(backfill) 열

아래 열들은 측정 실행이 내보낸 것이 아니라, **커밋된 맵과 CSV에서 나중에
파생시킨** 것이다. 재측정은 하지 않았다.

| 열 | 만든 커밋 | 방법 |
|----|-----------|------|
| `anisotropy_central_converged` | `fe2dc77` | 이미 기록된 `anisotropy_central` 계열에 `bench.erf.has_converged`를 N 오름차순으로 누적 적용. |
| `principal_angle_converged` (재계산) | `c178fdd` | `bench.erf.has_converged_deg`(절대 1.0° 기준)로 재계산. 상대 5% 기준이 0에 가까운 각도에서만 비정상적으로 빡빡해져 54행 중 9행이 잘못 미수렴으로 찍혀 있었다. |
| `peak_row`, `peak_col` | `c178fdd` | `bench.erf.peak_location`을 맵에 적용. |
| `mass_radius`, `rms_radius` | `c178fdd` | `bench.erf.mass_radius`(질량 50% 반경, 설계 문서의 정의) / `rms_radius`(Task 9 게이트 스크립트가 쓴 정의)를 맵에 적용. 둘 다 배열의 실제 중심 (111.5, 111.5) 기준이다. |
| `anisotropy_central_crop` | `c178fdd` | 상수 128 — 실행에 쓰인 `experiments.e2_erf.CENTRAL_CROP_SIZE` 값. |
| `error` (cmt_s/noise 6행, 문구만) | `c178fdd` | 같은 맵에 새 `decay_ratio` 가드를 다시 적용해 사유 문장을 다시 생성. 어느 셀이 왜 실패하는지는 그대로다. |

`c178fdd`의 backfill은 쓰기 전에 **이미 커밋돼 있던 네 지표를 맵에서 HEAD
코드로 전부 다시 계산해** 커밋된 값과 대조했다. 최대 절대차는
`principal_angle_deg` 7.1e-15, 나머지 셋은 1e-15 미만이었고 `decay_window`는
54행 전부 64로 동일했다. 즉 맵과 거기서 나온 숫자는 바뀌지 않았다.

## 재현에는 고정 툴체인이 필요하다

`erf_maps.npz`를 다시 만들려면 `requirements.txt`의 고정 WSL2 환경이 있어야
한다(Python 3.10.13 / torch 2.1.1+cu118 / CUDA 11.8 / RTX 3070 Ti). 다른
툴체인에서는 **같은 시드로도 같은 맵이 나오지 않는다** — torch 2.6/CPU로
다시 만들었을 때 `random_init` 가중치부터 달라져 최대 절대차 0.49가 났다.
`env.json`의 torch·CUDA 핀은 장식이 아니라 재현성 주장의 일부다.

같은 고정 환경 안에서도 모든 모델이 비트 단위로 결정적이지는 않다. 완전히 같은
입력(`noise` 조건)으로 두 번 잰 두 실행을 대조하면 **`deit_s`는 5셀 전부 비트
단위로 동일**하고, **`vim_s`는 5셀 전부 다르다**(최대 절대차 `anisotropy`
1.1e-06, `principal_angle_deg` 1.2e-04). `cmt_s`는 실행 A의 해당 셀이 전부
NaN이라 판정할 수 없다.

즉 관측된 비결정성은 **`vim_s`에 한정**되며, 파이프라인 공통 경로가 아니다 —
`deit_s`도 같은 `patch_embed` conv를 쓰는데 비트 단위로 재현된다. 원인은
`vim_s` 고유의 융합 커널 쪽이다. 이 크기는 결론에 영향을 주지 않지만
**"비트 단위 재현"을 모델 전반에 대해 주장하지는 말 것.** (두 실행 대조 전체는
`HANDOFF.md`의 "두 실행 대조가 말해 주는 것" 절 참고.)

## 이 디렉터리에서 인용하면 안 되는 것

- `noise` 조건은 정직성 게이트와 논문 인용 대상이 **아니다.** 사전학습 가중치에
  학습 분포 밖 입력(`torch.randn`)을 먹이는 대조 조건이다.
- `cmt_s`/`noise`의 `decay_ratio` 6칸은 비어 있다. 측정 실패가 아니라 그 조건에서
  ERF 피크가 이미지 모서리로 튀어 감쇠 창을 확보할 수 없기 때문이다 — 사유가
  `error` 열에 그대로 있다.
- 세 번째 유효숫자는 샘플링 잡음 안이다. `HANDOFF.md`의 두 실행 대조를 볼 것.
