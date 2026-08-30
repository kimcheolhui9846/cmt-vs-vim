"""본문에 손으로 쓴 측정 수치가 없는지 검사한다.

스펙의 핵심 규칙 — 논문의 모든 수치는 CSV에서 나온다 — 을 기계로 강제한다.
사람이 지키기로 한 규칙은 언젠가 깨지고, 깨져도 아무 표시가 남지 않는다.

측정처럼 보이는 수치만 잡는다: 소수점 두 자리 이상, 또는 퍼센트가 붙은 수.
해상도(1024)나 패치 크기(16)처럼 정당한 정수는 애초에 잡지 않고, 그래도 걸리는
것은 allowlist에 이유와 함께 둔다.
"""
import re
from pathlib import Path

import pytest

SECTIONS = Path("paper/sections")
ALLOWLIST = Path("paper/numbers_allowlist.txt")

# 소수점 두 자리 이상, 또는 숫자 뒤 퍼센트.
SUSPECT = re.compile(r"\d+\.\d{2,}|\d+(?:\.\d+)?\s*\\?%")
# LaTeX 주석은 조판되지 않으므로 검사 대상이 아니다. \% 는 주석이 아니다.
COMMENT = re.compile(r"(?<!\\)%.*$")


def _allowed():
    if not ALLOWLIST.exists():
        return set()
    return {line.split("#")[0].strip()
            for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
            if line.split("#")[0].strip()}


def _section_files():
    return sorted(SECTIONS.glob("*.tex"))


def _offenders(path, allowed):
    found = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = COMMENT.sub("", raw)
        for match in SUSPECT.finditer(line):
            token = match.group(0).strip()
            if token not in allowed:
                found.append(f"{path}:{lineno}: {token!r}")
    return found


def test_sections_directory_exists():
    assert SECTIONS.is_dir(), "paper/sections/ 가 없다"


def test_no_hand_written_measurements():
    allowed = _allowed()
    offenders = []
    for path in _section_files():
        offenders += _offenders(path, allowed)
    assert not offenders, (
        "본문에 손으로 쓴 측정 수치가 있다. paper/generated/macros.tex의 "
        "매크로를 쓰거나, 측정값이 아니라면 paper/numbers_allowlist.txt에 "
        "이유와 함께 추가할 것:\n" + "\n".join(offenders))


def test_allowlist_entries_carry_a_reason():
    if not ALLOWLIST.exists():
        return
    for lineno, line in enumerate(
            ALLOWLIST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        assert "#" in line, (
            f"{ALLOWLIST}:{lineno}: allowlist 항목에는 이유를 적는다 "
            f"(`값  # 이유`). 이유 없는 예외는 규칙을 조용히 비운다.")


def test_the_checker_catches_a_planted_measurement(tmp_path):
    """검사기가 이름값을 하는지 본다.

    이 테스트가 없으면 정규식이 아무것도 잡지 못하게 바뀌어도
    test_no_hand_written_measurements가 조용히 통과한다.
    """
    planted = tmp_path / "planted.tex"
    planted.write_text(
        "Structure dominates by 6.94 points and CMT reaches 60.19\\%.\n",
        encoding="utf-8")
    found = _offenders(planted, set())
    assert len(found) == 2, f"심어 둔 수치 둘을 잡지 못했다: {found}"


def test_the_checker_ignores_latex_comments(tmp_path):
    commented = tmp_path / "commented.tex"
    commented.write_text("% structure is 6.94 points\nPlain text.\n",
                         encoding="utf-8")
    assert _offenders(commented, set()) == []


def test_the_checker_does_not_flag_escaped_percent_in_prose(tmp_path):
    """`\\%` 자체는 수치가 아니다. 앞에 숫자가 붙을 때만 잡아야 한다."""
    plain = tmp_path / "plain.tex"
    plain.write_text("We report top-1 accuracy in \\% throughout.\n",
                     encoding="utf-8")
    assert _offenders(plain, set()) == []


def test_macros_are_not_flagged(tmp_path):
    """매크로로 쓴 수치는 잡히면 안 된다 - 그것이 권장하는 방식이다."""
    macro = tmp_path / "macro.tex"
    macro.write_text(
        "Structure dominates by \\StructureEffect{} points.\n",
        encoding="utf-8")
    assert _offenders(macro, set()) == []
