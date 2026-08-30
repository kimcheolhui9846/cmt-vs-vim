# 벨로그 2편 실험편 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 이 저장소는 사용자가 요청하지 않는 한 subagent를 띄우지 않는다.

**Goal:** 벨로그 "AI Study" 시리즈 2편(실험편)의 붙여넣기 가능한 마크다운 초안과,
그 초안의 수치가 커밋된 데이터에서 왔는지 검사하는 테스트를 만든다.

**Architecture:** 초안은 `docs/velog/`에 마크다운 한 파일로 둔다. 수치 규율은
논문과 반대 방향의 검사기로 강제한다 — 논문은 본문에 손으로 쓴 숫자가 "없어야"
통과하지만(매크로를 쓰므로), 마크다운에는 매크로가 없으니 나오는 수치가 전부
`paper/generated/macros.tex`의 값 중 하나인지를 본다. 그림은 새로 뽑지 않고
`results/e1~e4`의 기존 PNG를 상대 경로로 참조한다.

**Tech Stack:** Markdown, pytest, 기존 고정 환경 래퍼(`tools/run.sh`).

## Global Constraints

- **모든 python·pytest는 고정 환경 래퍼를 통과한다.**
  `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run.sh python -m pytest -q`
  래퍼가 exit 1로 죽으면 고쳐서 돌리지 말고 왜 없는지 확인한다.
- **수치를 손으로 다시 계산하지 않는다.** 글에 쓰는 모든 숫자는
  `paper/generated/macros.tex` 또는 `results/*/`의 사실 파일에서 나온다.
  반올림도 출처를 따른다.
- **커밋 저자는 사용자 단독이다.** Claude/Anthropic co-author 트레일러를 넣지 않는다.
- **문서는 한국어로 쓴다.** 초안 본문도 한국어다.
- **1편을 수정하지 않는다.** 후속 링크 한 줄은 사용자가 벨로그에서 직접 단다.
- **새 측정을 하지 않는다.** 글을 쓰다 나오는 "이것도 재면 좋겠다"는 전부
  "남은 일"로 적고 넘어간다.
- **v1 논문(`docs/paper-v1.pdf`)의 문장을 재사용하지 않는다.** 벨로그 1편은
  사용자 본인 글이므로 인용해도 된다.
- **저장소·논문 링크를 넣지 않는다.** "곧 공개 예정" 같은 문장도 넣지 않는다.

### 인용 금지 목록 (모든 태스크에 적용)

- 구조 주효과(`\StructureEffect` = 6.94)만 단독 인용해도 된다.
- **상호작용은 부호만 쓴다.** 크기(1.86)를 성과로 쓰지 않는다.
- **연산자 주효과도 크기를 쓰지 않는다.**
- **raw precision을 기준선 없이 쓰지 않는다.**
- **224²·384²의 latency를 단독 인용하지 않는다.**
- **CMT-S와 DeiT-S의 순서를 주장하지 않는다.**

---

## File Structure

| 파일 | 책임 |
|---|---|
| `docs/velog/2026-08-30-experiments-post.md` | 초안 본문. 이 계획의 산출물 |
| `docs/velog/numbers_allowlist.txt` | 매크로에 없지만 정당한 수치 + 그 출처 |
| `tests/test_velog_prose.py` | 초안의 수치가 매크로/allowlist에서 왔는지 검사 |
| `HANDOFF.md` | 벨로그 절 추가 (Task 9) |

---

### Task 1: 검사기와 초안 뼈대

**Files:**
- Create: `tests/test_velog_prose.py`
- Create: `docs/velog/2026-08-30-experiments-post.md`
- Create: `docs/velog/numbers_allowlist.txt`

**Interfaces:**
- Produces: `macro_values() -> set[str]`, `allowed() -> set[str]`,
  `prose_lines(text) -> list[tuple[int, str]]`, `offenders(lines, known) -> list[str]`.
  Task 2~8은 이 검사기를 통과시키는 것으로 검증한다.

- [ ] **Step 1: 초안 뼈대와 allowlist를 만든다**

`docs/velog/2026-08-30-experiments-post.md`:

```markdown
# (제목 미정 — Task 9에서 정한다)

> 이 글은 [1편](https://velog.io/@kimcheolhui1217/논문-리뷰-ViT의-한계를-극복하는-두-시선-CMT-vs-Vision-Mamba-SSM-심층-분석)의 후속입니다.
```

`docs/velog/numbers_allowlist.txt`:

```
# 매크로에 없지만 본문에 써도 되는 수. 각 항목에 출처를 적는다.
# paper/generated/macros.tex에 있는 값은 여기 적을 필요가 없다.
#
# 형식:  값  # 출처
86.8%   # Vim 원논문이 보고한 메모리 절감(1248², 다른 하드웨어). 우리가 재현하지 못한 인용값이다
48%     # 15.245h -> 7.957h 감소율. results/e4/README.md가 쓰는 값과 같다
7%      # 4.306h -> 4.596h 증가율. results/e4/README.md가 쓰는 값과 같다
66%     # 연산자 주효과의 표준편차/평균(0.45/0.68). results/e4/README.md의 인용 규칙 절
```

- [ ] **Step 2: 검사기의 자기 테스트를 먼저 쓴다**

`tests/test_velog_prose.py`에 아래를 그대로 넣는다.

```python
"""벨로그 초안의 측정 수치가 커밋된 데이터에서 온 값인지 검사한다.

논문 검사기(tests/test_paper_prose.py)와 방향이 반대다. 논문은 본문에 손으로
쓴 숫자가 "없어야" 통과한다 - 수치를 매크로로 부르기 때문이다. 마크다운에는
매크로가 없으므로, 여기서는 본문에 나오는 수치가 전부 매크로의 값 중
하나인지를 본다.

CSV가 바뀌어 매크로가 갱신됐을 때 글이 옛 숫자를 들고 있으면 아무 표시 없이
조용히 틀린 글이 된다. 이 저장소는 그 실패를 이미 두 번 겪었다 - E1 README의
1.66배, HANDOFF의 "1.35와 1.40".
"""
import re
from pathlib import Path

DRAFT = Path("docs/velog/2026-08-30-experiments-post.md")
MACROS = Path("paper/generated/macros.tex")
ALLOWLIST = Path("docs/velog/numbers_allowlist.txt")

# 측정값처럼 생긴 것: 소수 둘 이상, 또는 숫자에 붙은 퍼센트.
# 앞뒤의 (?<![\d.]) / (?![\d.])가 "3.10.13" 같은 버전 문자열을 통째로 비껴간다.
SUSPECT = re.compile(
    r"(?<![\d.])\d+\.\d{2,}(?![\d.])|(?<![\d.])\d+(?:\.\d+)?\s*%")
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")


def normalise(token):
    """부호와 퍼센트를 떼어 비교 가능한 형태로 만든다.

    본문은 "+6.94%p"나 "−0.0875"로 쓰고 매크로는 "6.94"나 "-0.0875"로 갖고
    있다. 유니코드 빼기(U+2212)도 함께 떼야 한글 조판에서 쓴 부호가
    통과한다.
    """
    return token.strip().rstrip("%").strip().lstrip("+-\u2212")


def macro_values():
    return {normalise(value)
            for _, value in MACRO.findall(MACROS.read_text(encoding="utf-8"))}


def allowed():
    if not ALLOWLIST.exists():
        return set()
    return {normalise(line.split("#")[0])
            for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
            if line.split("#")[0].strip()}


def prose_lines(text):
    """코드 펜스 밖의 줄만, 원래 줄 번호를 달아 돌려준다.

    펜스 안은 명령어나 출력이라 검사 대상이 아니다. 줄 번호를 유지해야
    실패 메시지로 바로 찾아갈 수 있다.
    """
    lines, inside = [], False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            inside = not inside
            continue
        if not inside:
            lines.append((lineno, line))
    return lines


def offenders(lines, known):
    found = []
    for lineno, line in lines:
        for match in SUSPECT.finditer(line):
            if normalise(match.group(0)) not in known:
                found.append(f"{lineno}: {match.group(0).strip()!r}")
    return found


def test_draft_exists():
    assert DRAFT.is_file(), f"{DRAFT} 가 없다"


def test_macros_are_readable():
    values = macro_values()
    assert len(values) > 20, "macros.tex를 읽지 못했다. 논문 표를 먼저 생성할 것"
    assert "6.94" in values


def test_no_unsourced_measurements():
    known = macro_values() | allowed()
    found = offenders(prose_lines(DRAFT.read_text(encoding="utf-8")), known)
    assert not found, (
        "초안에 출처 없는 수치가 있다. paper/generated/macros.tex의 값을 쓰거나, "
        "매크로에 없는 정당한 수치라면 docs/velog/numbers_allowlist.txt에 출처와 "
        "함께 추가할 것:\n" + "\n".join(found))


def test_allowlist_entries_carry_a_source():
    for lineno, line in enumerate(
            ALLOWLIST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        assert "#" in line, (
            f"{ALLOWLIST}:{lineno}: allowlist 항목에는 출처를 적는다 "
            f"(`값  # 출처`). 출처 없는 예외는 규칙을 조용히 비운다.")


def test_the_checker_catches_a_number_absent_from_macros():
    """검사기가 이름값을 하는지 본다.

    이것이 없으면 정규식이 아무것도 잡지 못하게 바뀌어도
    test_no_unsourced_measurements가 조용히 통과한다.
    """
    lines = prose_lines("구조 효과는 9.99%p였습니다.\n")
    assert offenders(lines, macro_values()) == ["1: '9.99'"]


def test_the_checker_accepts_a_macro_value():
    lines = prose_lines("구조 효과는 6.94%p였습니다.\n")
    assert offenders(lines, macro_values()) == []


def test_the_checker_ignores_fenced_code_blocks():
    text = "```\npytest --cov=99.99\n```\n본문입니다.\n"
    assert offenders(prose_lines(text), macro_values()) == []


def test_sign_and_percent_are_normalised():
    """본문의 표기와 매크로의 표기가 달라도 같은 값으로 봐야 한다."""
    values = macro_values()
    assert normalise("+6.94%p".rstrip("p")) in values
    assert normalise("\u22120.0875") in values


def test_version_strings_are_not_flagged():
    """`3.10.13` 같은 버전은 측정값이 아니다."""
    assert offenders(prose_lines("Python 3.10.13에서 쟀습니다.\n"), set()) == []
```

- [ ] **Step 3: 테스트를 돌려 통과하는지 본다**

Run:
```
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run.sh python -m pytest -q tests/test_velog_prose.py
```
Expected: 9건 전부 PASS. 실패하면 정규식이나 `normalise`를 고치되,
`test_the_checker_catches_a_number_absent_from_macros`를 약화시키는 방향으로
고치지 않는다.

- [ ] **Step 4: 커밋**

```bash
git add tests/test_velog_prose.py docs/velog/
git commit -m "test: 벨로그 초안의 수치가 매크로에서 왔는지 검사한다"
```

---

### Task 2: 도입과 "어떻게 쟀는가"

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

**Interfaces:**
- Consumes: Task 1의 검사기

- [ ] **Step 1: 도입을 쓴다**

담을 것:
- 1편은 이론으로 끝났다. 표도 그림도 유도였고 **직접 잰 수치가 하나도 없었다**
- 넉 달 뒤 전부 재봤다. 여섯 중 **둘이 적중, 하나는 지표에 따라 갈렸고, 셋이
  어긋났다**
- 어긋난 셋 중 하나는 **1편이 스스로 쓴 문장**이다. 그게 이 글에서 제일 아픈
  대목이라고 미리 말해 둔다
- 이 글의 결론을 한 줄로: **주장이 참인 축과 내가 재는 축이 달랐다**

**하지 말 것:** 도입에서 여섯 판정을 전부 나열하지 않는다. 채점표 서사가
죽는다.

- [ ] **Step 2: "어떻게 쟀는가"를 쓴다 (짧게, 400자 안팎)**

셋만 적는다.

1. **고정 환경 래퍼** — 모든 python이 `tools/run.sh`를 통과하고 환경이 다르면
   exit 1로 죽는다. Python `\PyVersion`, torch `\TorchVersion`, CUDA
   `\CudaVersion`, `\GpuName`. 조용히 다른 python으로 돌아간 측정값이 가장 비싼
   실패이기 때문이다. 특히 SSM은 컴파일된 CUDA 커널 없이 돌면 한 자릿수 느려지고,
   그 latency는 실수가 아니라 결과처럼 보인다
2. **사전 등록** — 실험마다 예측을 측정 전에 확정하고 측정 후 고치지 않았다.
   두 번 빗나갔고 그대로 싣는다(뒤에서 다룬다)
3. **수치는 CSV에서** — 문서에 쓰는 숫자를 손으로 옮기지 않는다. 이 글도
   테스트가 검사한다

- [ ] **Step 3: 검사기를 돌린다**

Run:
```
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run.sh python -m pytest -q tests/test_velog_prose.py
```
Expected: PASS. 실패하면 매크로 값을 쓰거나 allowlist에 출처와 함께 추가한다.

- [ ] **Step 4: 커밋**

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편 도입과 측정 규율 절을 쓴다"
```

---

### Task 3: 채점 1·2 — 복잡도

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

- [ ] **Step 1: 채점 1(cross-over 위치)을 쓴다**

**1편이 뭐라 했나 —** 그대로 인용한다:

> 이차항($2M^2d$)이 선형항($4Md^2$)을 추월하는 지점을 유도하면 $M \approx 2d$가 나옵니다.

**어떻게 쟀나 —** 세 모델을 224²·384²·512²·768²·1024²에서 재고 전체 모델의
FLOPs를 비교했다. 블록 하나의 식으로 예측한 것을 통째 모델에서 확인하는 것이라
자명하게 성립하지 않는다 — 모델은 블록 하나가 아니다.

**결과 —** 384²에서 Vim이 DeiT보다 비싸고 512²에서 싸진다. 예측 구간 안이다.

**판정 — 적중.** 이 글에서 1편이 온전히 이긴 유일한 항목이다.

그림: `![해상도별 FLOPs와 throughput](results/e1/e1_sweep.png)`

- [ ] **Step 2: 채점 2(이득의 크기)를 쓴다**

**여기서 출처를 갈아탄다는 것을 명시한다.** 1편은 크기를 정량화하지 않았다.
54배는 **논문 v1의 표 1**에 있다. "1편이 하지 않은 말"이라고 분명히 적고
넘어간다.

**결과 —** 1024²에서 Vim `\VimFlopsHigh` GFLOPs 대 DeiT `\DeitFlopsHigh` GFLOPs.
2.0배다.

**왜 어긋났나 —** 표 1이 self-attention에는 투영 항까지 세고 SSM에는 스캔 커널만
셌다. Mamba 블록에도 입력·출력 투영과 파라미터를 만드는 투영이 있고 전부 채널
폭의 제곱에 비례한다. 상태 폭은 수십이고 채널 폭은 수백이라 **투영 항이 스캔
항보다 한 자릿수 크다.** 한쪽만 세면 54배가 나온다.

- [ ] **Step 3: fvcore 함정을 이 절 안에 녹인다**

별도 절로 빼지 않는다. 담을 것:
- 자동 FLOP 카운터는 **추적하지 못하는 융합 커널을 통째로 0으로 센다**
- 이 저장소에서 두 번 겪었다 — DeiT는 fused SDPA가 attention matmul을 삼켰고,
  Vim은 fused op이 conv1d·투영·스캔을 전부 삼켰다
- **두 모델 다 과소 계수되지만 SSM 쪽이 훨씬 심하다.** 연산의 더 많은 부분이 한
  번의 융합 호출 안에 있기 때문이다. 즉 **이 비교를 가장 흔한 방법으로 재면
  SSM에 유리한 쪽으로 편향된다**
- 재는 동안만 융합을 푸는 장치를 만들었다. 융합 여부가 연산량을 바꾸지 않으므로
  푸는 편이 정직하다. **반대로 latency·메모리·throughput은 그 안에서 재면 안
  된다** — 거기서는 커널 융합이 곧 성능이다

- [ ] **Step 4: 검사기를 돌리고 커밋한다**

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편 복잡도 채점을 쓴다 - 위치는 맞고 크기는 틀렸다"
```

---

### Task 4: 채점 3 — "Vim만이 유일한 실용적 대안"

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

**이 태스크가 이 글에서 가장 뾰족하다.** 두 이야기를 한 문단에 섞지 않는다.

- [ ] **Step 1: 1편의 문장을 인용한다**

> 하지만 1024x1024 이상의 Gigapixel, 의료/위성 영상 등에서는 Vim만이 유일한 실용적 대안입니다.

이것은 인용값이 아니라 **1편이 스스로 쓴 서술**이라고 밝힌다.

- [ ] **Step 2: 측정과 대조한다**

1024², 배치 1, `\GpuName`:

- peak allocated: Vim `\VimPeakHigh` MiB 대 DeiT `\DeitPeakHigh` MiB — **Vim이 더
  쓴다**
- 최대 배치: Vim `\VimMaxBatchHigh` 대 DeiT `\DeitMaxBatchHigh` — **Vim이 먼저
  터진다**

**판정 — 반대다.** 1편이 "유일한 실용적 대안"이라 부른 그 해상도에서, 메모리가
먼저 모자라는 쪽이 Vim이다. 스캔의 중간 활성을 들고 있어야 하기 때문이다.
선형 시간 연산자가 메모리에도 검소하리라는 직관이 여기서는 성립하지 않는다.

- [ ] **Step 3: 원논문 인용값은 따로 적는다**

**문단을 나눈다.** Vim 원논문의 FPS 2.8배와 메모리 86.8% 절감은 1248²에서 다른
하드웨어로 잰 값이다. 우리는 1024², 소비자 GPU 하나다. **이건 반증이 아니라
재현 범위의 한계**로 적는다. 해상도도 장치도 메모리 예산도 다르고, 어느 것이든
차이를 설명할 수 있다.

**섞지 말아야 하는 이유를 한 줄로 적는다:** 인용값의 재현 실패는 조건 차이로
설명되지만, 자기가 쓴 서술이 측정과 어긋나는 것은 조건과 무관하다.

- [ ] **Step 4: 검사기를 돌리고 커밋한다**

`86.8%`는 allowlist에 이미 있다. 없으면 출처와 함께 추가한다.

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편에서 1편이 스스로 쓴 문장을 채점한다"
```

---

### Task 5: 채점 4 — 수용 영역

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

- [ ] **Step 1: 1편의 두 문장을 인용한다**

> CMT (2D 등방적 - Isotropic)... 중심에서 퍼져나가는 종 모양(Gaussian-like)을 가집니다.

> Vim (1D 스캔 편향 - Anisotropic)... 스캔 방향으로 편향된 비대칭적 수용 영역을 가집니다.

- [ ] **Step 2: 지표에 따라 갈린다는 것을 쓴다**

- **공분산 기반 비등방 지수**로는 세 모델이 구분되지 않는다. 자연 이미지에서
  전부 1 근처다. 이 지표는 거리 제곱 가중이라 **넓고 등방적인 배경이 값을
  지배한다.** scan이 만드는 모양은 피크 근방에 있는데 그걸 배경이 덮는다
- **축별 감쇠비**로 재면 Vim이 `\VimDecayNatural`, 나머지 둘은 1 근처다
- 그 격차는 측정 불확실성보다 한 자릿수 크다 — 다른 이미지 집합으로 두 번 돌려
  같은 표본 크기에서 대조하면 `\VimDecayRunDrift`% 움직인다

**CMT-S와 DeiT-S의 순서는 주장하지 않는다.** 그 차이는 불확실성 안이고,
세 모델의 프로브 지점이 구조상 같을 수도 없다.

- [ ] **Step 3: 가장 센 증거가 미학습이라는 것을 쓴다**

가중치를 전혀 학습하지 않은 Vim의 ERF는 비등방 지수가 `\VimAnisoRandom`,
**그 타원의 주축이 수평에서 `\VimAngleRandom`도 이내**다. 같은 조건에서 감쇠비는
`\VimDecayRandom`으로 학습된 모델보다 오히려 크다.

**여기가 이 절의 요점이다.** 비등방성은 학습이 만든 것이 아니라 row-major
flatten이라는 구조가 애초에 갖고 있는 것이다. 학습은 그것을 부분적으로
**상쇄한다.** 그래서 학습 후 감쇠비가 더 작고, 세 지표 중 가장 둔한 공분산
지수는 남은 것을 찾지 못한다.

미학습 모델은 보통 대조군인데 여기서는 **현상이 가장 큰 조건**이다. 아무것도
보정할 기회가 없었기 때문이다.

그림: `![세 모델의 ERF](results/e2/e2_erf.png)` — 대표 이미지로 지정한다.

**판정 — 조건부 적중.** 1편의 서술은 맞았지만, 맞는 축이 1편이 생각한 축과
달랐다.

- [ ] **Step 4: 검사기를 돌리고 커밋한다**

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편 수용 영역 채점을 쓴다 - 학습이 아니라 구조다"
```

---

### Task 6: 채점 5 — dilution

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

- [ ] **Step 1: 1편의 문장을 인용하고, 그 논증이 맞다는 것부터 인정한다**

> Softmax는 가중치 합이 1이어야 합니다. 객체가 50개의 토큰을 차지하면 각 가중치는 $1/50$로 희석됩니다.

**정규화에 대한 이 논증 자체는 맞다.** 틀린 것은 거기서 끌어낸 결론이다.

- [ ] **Step 2: 읽는 법을 먼저 적는다 (이 순서를 지킨다)**

raw precision@K는 세 모델 모두 **객체가 커질수록 올라간다.** 이건 dilution의
반증이 아니라 산술이다 — 객체가 커지면 무작위 순위도 더 많이 맞힌다.
**기준선이 모델보다 빨리 오르므로 정보를 담는 값은 기준선 초과분이다.**

**기준선 없이 raw precision을 인용하면 이 실험이 정반대로 서술된다.** 그래서
초과분만 적는다.

- [ ] **Step 3: 결과와 기전을 쓴다**

**결과 —** Vim의 기준선 초과분이 **여섯 면적 구간 전부에서** 세 모델 중 최하다.
dilution 논증이 예측하는 큰 객체에서만이 아니라 전부다.

**왜 논증이 결론에 닿지 못하나 —** 둘이다.
1. **상태 용량.** 정규화 제약이 없다는 것이 용량을 만들어 주지는 않는다. 상태가
   객체 정보를 담지 못하면 그것을 $K$로 나누는 것이 없다는 사실은 아무 도움이
   안 된다. 재미있는 건 **1편도 SSM의 상태가 좁은 파이프라는 걸 다른 절에서
   이미 적었다는 점**이다 — 두 절이 서로를 반박하고 있었고 측정이 한쪽 손을 들어
   줬다
2. **스캔 순서.** row-major로 펴면 2차원 객체의 토큰이 시퀀스에서 붙어 있지
   않다. 행이 바뀔 때마다 배경이 끼어든다

**교차 확인 —** 스캔 순서가 원인이라면 세로로 긴 객체가 가로로 긴 객체보다
불리해야 한다. 실제로 tall−wide precision이 DeiT `\DeitTallWide`,
CMT `\CmtTallWide`로 0과 구분되지 않는데 Vim만 `\VimTallWide`
(z = `\VimTallWideZ`)이고, 미학습에서는 `\VimTallWideRandom`
(z = `\VimTallWideRandomZ`)으로 더 벌어진다. 다른 목적으로 만든 두 실험이 같은
구조적 원인을 가리킨다.

- [ ] **Step 4: 프로브 지점 한계를 이 절 안에 녹인다**

세 모델에 구조적으로 같은 층이 없어서 각각 역할이 가장 가까운 지점에서 읽는다.
DeiT의 지점을 바꿔 다시 재면 CMT와의 거리가 절반으로 준다. 그래서 **두 attention
모델의 순서는 어디서도 주장하지 않는다.**

"Vim이 가장 낮다"는 대안 지점에서도 살아남지만, **그 확인은 전체 평균에서만
했고 구간별로는 다시 보지 않았다.** "여섯 구간 전부"라는 형태는 원래 지점에만
기댄다 — 이 단서를 빼지 않는다.

그림: `![면적 구간별 기준선 초과분](results/e3/e3_coverage.png)`

**판정 — 반박.**

- [ ] **Step 5: 검사기를 돌리고 커밋한다**

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편 dilution 채점을 쓴다 - 방향이 반대다"
```

---

### Task 7: 채점 6 — 구조 대 연산자, 그리고 비용의 상호작용

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

- [ ] **Step 1: 출처를 밝히고 문제를 세운다**

1편은 성능 차이를 분해하지 않았다. 분해는 **논문 v1이 CMT 원논문의 ablation
표를 인용해** 한 것이다. 그 표는 **한 아키텍처에서 측정된 것**이고, 그걸 두
아키텍처 사이의 격차에 갖다 쓰려면 "그 아래 토큰 혼합 연산자가 달라져도 각
성분이 같은 몫을 낸다"를 가정해야 한다. 요인 설계는 정확히 그 가정을 시험하려고
있는 도구다.

- [ ] **Step 2: 2×2 설계와 D칸 제작을 쓴다**

- 축 둘: 구조(평면 / 계층+conv), 연산자(attention / 선택적 스캔)
- 네 칸: A 평면·attention, B 평면·SSM, C 계층·attention, **D 계층·SSM**
- **D는 직접 만들었다.** CMT Block에서 attention 자리만 양방향 Mamba로 바꾸고
  LPU와 IRFFN은 그대로 뒀다. 그래야 B→D와 A→C가 **같은 조작**이 되어 상호작용
  항이 해석 가능해진다
- **VMamba의 4방향 스캔은 일부러 쓰지 않았다.** 쓰면 B→D가 구조와 스캔 방향 수를
  동시에 바꾸게 되고, 상호작용 항에 스캔 방향 효과가 섞여 분리할 수 없다
- 파라미터를 `\ParamBudget`M ±`\ParamTolerance`%로 통제했고 네 칸 전부 대역
  안이다
- 네 칸 모두 같은 레시피, 300 epoch, seed 3개

- [ ] **Step 3: 결과를 인용 규칙대로 쓴다**

칸 평균: A `\CellA`%, B `\CellB`%, C `\CellC`%, D `\CellD`%.

- **구조 주효과 `\StructureEffect`%p (표준편차 `\StructureStd`).** 자기 편차의
  열한 배이고 세 seed 모두에서 같은 크기다. **이 실험에서 단독으로 인용할 수
  있는 유일한 값이다.** 방향은 1편·논문의 예상과 같다 — 대부분이 구조에서 온다
- **상호작용은 부호만.** 세 seed 모두 양수라 방향은 살아남지만 크기가 seed 사이에
  다섯 배 가까이 흔들린다. 그래서 "계층 구조가 SSM에 최소한 attention만큼은 도움이
  된다, 우리가 돌린 모든 seed에서"까지만 말하고 **크기는 이 실험이 정하지 못한다**
  고 적는다
- **연산자 주효과도 크기를 쓰지 않는다.** 표준편차가 평균의 66%다. 대신 더 약하고
  여전히 말할 값이 있는 문장을 쓴다 — **이 시퀀스 길이에서는 연산자 선택이 구조
  선택보다 훨씬 덜 중요하다**

그림: `![2×2 요인 결과](results/e4/e4_factorial.png)`

**판정 — 인용을 자체 측정으로 교체. 구조가 압도한다는 방향은 적중.**

- [ ] **Step 4: 정확도에 없던 상호작용이 비용에 있었다는 것을 쓴다**

계층을 얹으면 attention은 `\HoursA`h → `\HoursC`h로 7% 느려지고, SSM은
`\HoursB`h → `\HoursD`h로 48% 빨라진다.

**기전은 시퀀스 길이다.** 계층 본체에서 토큰 수가 단계마다 반감하므로 스캔이
짧아진다. attention은 64토큰에서 이미 싸서 얻을 것이 없고 conv stem과 LPU
오버헤드만 붙는다.

**요점:** 정확도 축에서는 상호작용이 미확정인데 비용 축에서는 뚜렷하고 크다.
아키텍처 비교를 정확도 하나로 요약하면 **이 실험에서 가장 큰 상호작용을 통째로
놓친다.** 이건 사전 등록 예측표에 없던 관찰이라 사후 발견으로 적는다.

- [ ] **Step 5: 검사기를 돌리고 커밋한다**

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편 2x2 요인 채점과 비용 상호작용을 쓴다"
```

---

### Task 8: 빗나간 예측 둘, 내가 틀린 추론 하나, 마무리

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`

- [ ] **Step 1: 빗나간 예측 둘을 쓴다**

**dilution 예측 — 척도가 판정을 만들었다.** 큰 객체로 갈수록 얼마나 가파르게
떨어지는지를 예측했는데, **비율로 보면 세 모델이 비슷하게 떨어지고 절대 하락으로
보면 Vim이 가장 적게 잃는다.** 다만 그건 Vim이 가장 낮은 값에서 시작해 잃을 것이
적기 때문이다. 어느 척도를 고르느냐가 판정을 만든다면 그 판정은 발견이 아니다.
그래서 척도 선택이 끼어들 자리가 없는 **수준(level)**으로 결론을 옮겼다.

**요인 실험의 검증 대역 — 위로 벗어났다.** 네 칸이 모두 45~60%에 들 것으로
등록했는데 계층 칸 둘이 위로 넘었다. **위로 벗어나는 것도 자명하게 안전하지
않다.** 파이프라인 결함을 먼저 의심했고 둘로 배제했다 — 네 칸이 같은 loader를
쓰는데 평면 칸은 대역 안이었고(라벨 누수라면 넷 다 올라간다), 검증 집합을 직접
열어 200클래스·10,000장·클래스당 정확히 50장·증강 없는 평가 변환을 확인했다.
대역 상한이 낮게 잡혀 있었던 것이다. **예측표는 고치지 않았다.**

- [ ] **Step 2: 내가 틀린 추론을 쓴다**

이 절이 이 글에서 가장 값나가는 대목이다. 숨기지 않는다.

A 칸의 seed 간 차이가 0.88%p인 것을 보고 **"연산자 효과는 잡음에 묻힌다"고
판단했는데, 그 추론 자체가 틀렸다.** 요인 효과는 네 칸의 평균끼리 빼기 때문에
네 칸에 공통으로 걸리는 seed 변동이 상쇄된다. **칸의 분산과 효과의 분산은 다른
양이다.**

그런데 그렇다고 효과가 확정된 것도 아니었다. seed 2개까지는 연산자 효과의
표준편차가 0.02로 아주 작았는데 seed 3을 채우자 0.45로 커졌다. **표본 두 개로
잰 표준편차는 신뢰 구간이라기보다 "두 값이 얼마나 벌어졌나"에 가깝다.** 그래서
결국 연산자 주효과의 크기는 인용하지 않기로 했다.

**틀린 방향으로 한 번, 맞는 방향으로 한 번 틀린 셈이다.** 사전 등록이 없었다면
둘 다 결과에 맞춰 조용히 정당화됐을 것이다.

- [ ] **Step 3: 마무리를 쓴다**

어긋난 넷을 나란히 놓으면 모양이 같다.

- 복잡도: 논증은 **스캔 커널**에 대해 참인데 수치는 **블록** 전체인 것처럼 읽혔다
- 수용 영역: 주장은 **피크 근방**에서 참인데 지표는 **넓은 배경**을 쟀다
- dilution: 논증은 **정규화**에 대해 참인데 결론은 **표현 용량**에 대한 것이었다

셋 다 "주장이 틀렸다"도 "맞다"도 정확하지 않다. 정확한 문장은 **축을 밝히는
것**이다 — 주장은 그 논증이 다루는 축에서 참이고, 측정은 다른 축에서 이뤄졌다.

네 번째("Vim만이 유일한 실용적 대안")는 성격이 다르다. 그건 축의 문제가 아니라
**측정 없이 쓴 문장**이었고, 그래서 그냥 어긋났다.

**마지막 문단:** 1편을 쓸 때 나는 인용에 출처가 있으면 근거가 있다고 생각했다.
출처가 있다는 것과 **내 조건에서 성립한다는 것**은 다른 말이다. 이 글의 여섯
줄 중 그 차이가 만든 것이 셋이다.

- [ ] **Step 4: 남은 일을 짧게 적는다**

conv와 계층이 분리되지 않았다(2×2×2 = 24 run 필요), 64토큰은 SSM에 불리한
영역이다(평면 칸을 256토큰으로 올리면 약 312시간), Tiny-ImageNet 하나다.

- [ ] **Step 5: 검사기를 돌리고 커밋한다**

```bash
git add docs/velog/
git commit -m "docs: 벨로그 2편 빗나간 예측과 마무리를 쓴다"
```

---

### Task 9: 제목, 전체 검사, HANDOFF, 게시 안내

**Files:**
- Modify: `docs/velog/2026-08-30-experiments-post.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: 제목 후보 셋을 내고 사용자가 고른다**

1편이 「ViT의 한계를 극복하는 두 시선: CMT vs Vision Mamba (SSM) 심층 분석」이므로
2편은 **측정을 드러내는 쪽**이 좋다. 후보:

1. 「내 논문의 주장 여섯 개를 직접 재봤다: CMT vs Vision Mamba 실측편」
2. 「이론으로 예측한 것을 4일 동안 재봤습니다: CMT vs Vision Mamba 실험편」
3. 「CMT vs Vision Mamba 2편: 여섯 주장 중 셋이 어긋났다」

`AskUserQuestion`으로 묻는다. 고른 것을 초안 첫 줄에 넣는다.

- [ ] **Step 2: 1편 본문과 인용문을 다시 대조한다**

Run: RSS(`https://api.velog.io/rss/@kimcheolhui1217`)를 다시 읽어
Task 3·4·5·6에서 인용한 문장 다섯이 **글자 그대로 맞는지**, 앞뒤 문맥이 뜻을
바꾸지 않는지 확인한다.

**이 단계를 건너뛰지 않는다.** 스펙에 인용문을 박아 둔 것은 자동 추출 결과이고,
이 글의 주제가 정확히 "확인하지 않은 인용"이다. 자기 글에서 같은 짓을 하면
글 전체가 무너진다.

- [ ] **Step 3: 인용 금지 목록을 사람이 직접 대조한다**

테스트가 잡지 못하는 종류다. 초안을 열고 하나씩 확인한다.

- 상호작용의 **크기**를 성과처럼 쓰지 않았는가
- 연산자 주효과의 **크기**를 쓰지 않았는가
- raw precision을 기준선 없이 쓰지 않았는가
- 224²·384²의 latency를 단독으로 쓰지 않았는가
- CMT-S와 DeiT-S의 순서를 어디서도 주장하지 않았는가
- 프로브 지점 한계를 dilution 절 안에 적었는가
- 저장소·논문 링크나 "곧 공개" 문구가 들어가지 않았는가

- [ ] **Step 4: 전체 테스트를 돌린다**

Run:
```
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' wsl bash tools/run.sh python -m pytest -q
```
Expected: 기존 413건 + Task 1이 더한 9건이 전부 통과.

- [ ] **Step 5: 초안 말미에 게시 절차를 주석으로 남긴다**

본문이 아니라 마크다운 주석(`<!-- -->`)으로 둔다. 벨로그에 붙여넣을 때 함께
복사돼도 렌더링되지 않는다.

```markdown
<!--
게시 절차
1. 벨로그 에디터에 본문을 붙여넣는다
2. 그림 넷을 업로드하고 상대 경로를 업로드 URL로 바꾼다
   results/e1/e1_sweep.png / e2/e2_erf.png / e3/e3_coverage.png / e4/e4_factorial.png
3. 시리즈를 "AI Study"로 지정한다
4. 대표 이미지를 e2_erf.png로 설정한다
5. 1편 말미에 이 글 링크를 단다
저장소가 공개되면 마지막에 링크 문단을 하나 덧붙인다.
-->
```

- [ ] **Step 6: HANDOFF에 벨로그 절을 추가한다**

담을 것: 초안 위치, 검사기가 논문과 반대 방향이라는 것과 그 이유, 1편이 실제로
한 말과 하지 않은 말(54배·ablation 분해는 논문 v1에만 있다), 게시는 사용자가
한다는 것, 저장소 공개는 여전히 학회 답을 기다린다는 것.
"이 계획 이후" 목록에서 계획 5를 완료로 표시한다.

- [ ] **Step 7: 커밋**

```bash
git add docs/velog/ HANDOFF.md
git commit -m "docs: 벨로그 2편 제목을 정하고 전체 검사를 돌린다"
```

---

## Self-Review

**1. 스펙 커버리지**

| 스펙 항목 | 태스크 |
|---|---|
| 초안 파일 `docs/velog/2026-08-30-experiments-post.md` | 1 |
| 검사기 `tests/test_velog_prose.py` (논문과 반대 방향) | 1 |
| allowlist + 출처 강제 | 1 |
| 채점표 서사, 네 토막 구조 | 3~7 |
| 1편이 실제로 한 말 인용 | 3, 4, 5, 6 |
| 1편에 없는 것 둘(54배·분해) 구분 | 3 Step 2, 7 Step 1 |
| 3번이 가장 뾰족하다 / 두 이야기 분리 | 4 |
| fvcore 함정을 복잡도 절에 녹임 | 3 Step 3 |
| 프로브 지점 한계를 dilution 절에 녹임 | 6 Step 4 |
| 비용의 상호작용 | 7 Step 4 |
| 빗나간 예측 둘 + 틀린 추론 | 8 |
| 마무리 = 축의 차이 | 8 Step 3 |
| 그림 넷, 대표 이미지 E2 | 3, 5, 6, 7 |
| 인용 금지 목록 | Global Constraints + 9 Step 3 |
| 하지 않을 것(링크·새 측정·1편 수정·v1 문장) | Global Constraints |
| 게시 절차 | 9 Step 5 |
| 열린 항목: 제목 | 9 Step 1 |
| 1편 본문 재대조 | 9 Step 2 |

빠진 것 없음.

**2. 자리표시자 점검**

"제목 미정"은 Task 9 Step 1이 해결하며 후보 셋을 이미 적었다. 그 외 TBD·TODO
없음. 모든 코드 단계에 실제 코드가 있다.

**3. 이름 일관성**

`macro_values()` / `allowed()` / `prose_lines()` / `offenders()` / `normalise()`는
Task 1에서 정의하고 Task 2~9는 `pytest`로만 부른다. 파일 경로
`docs/velog/2026-08-30-experiments-post.md`와 `docs/velog/numbers_allowlist.txt`는
전 태스크에서 같은 이름을 쓴다.
