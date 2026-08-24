"""관문 자체가 통과 판정을 제대로 내리는지 확인한다.

관문이 헐거우면 관문이 아니다 — 여기서는 임계값과 판정 로직만 검증하고,
실제 GPU 실행은 Task 7 Step 4에서 사람이 돌린다.
"""
import pytest

from experiments.e4_gate import GATE_THRESHOLD, gate_verdict


def test_threshold_is_the_documented_one():
    assert GATE_THRESHOLD == 0.95


def test_verdict_passes_only_above_the_threshold():
    assert gate_verdict({"a_deit_ti": 0.99, "b_vim_ti": 0.97}) == []
    assert gate_verdict({"a_deit_ti": 0.99, "b_vim_ti": 0.51}) == ["b_vim_ti"]


def test_verdict_reports_every_failing_cell_not_just_the_first():
    """첫 실패에서 멈추면 두 번째 칸의 고장을 다음 라운드에 또 발견하게 된다."""
    failing = gate_verdict({"a_deit_ti": 0.10, "b_vim_ti": 0.99, "c_cmt_ti": 0.20})
    assert failing == ["a_deit_ti", "c_cmt_ti"]
