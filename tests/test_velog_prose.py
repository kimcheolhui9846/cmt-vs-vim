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
