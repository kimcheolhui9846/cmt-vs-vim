"""관문 자체가 통과 판정을 제대로 내리는지 확인한다.

관문이 헐거우면 관문이 아니다 — 여기서는 임계값과 판정 로직만 검증하고,
실제 GPU 실행은 Task 7 Step 4에서 사람이 돌린다.
"""
import pytest

from experiments.e4_gate import GATE_THRESHOLD, gate_verdict, select_subset_indices

IMAGES_PER_CLASS = 500  # Tiny-ImageNet train 분할의 클래스당 이미지 수


def test_threshold_is_the_documented_one():
    assert GATE_THRESHOLD == 0.95


def test_verdict_passes_only_above_the_threshold():
    assert gate_verdict({"a_deit_ti": 0.99, "b_vim_ti": 0.97}) == []
    assert gate_verdict({"a_deit_ti": 0.99, "b_vim_ti": 0.51}) == ["b_vim_ti"]


def test_verdict_reports_every_failing_cell_not_just_the_first():
    """첫 실패에서 멈추면 두 번째 칸의 고장을 다음 라운드에 또 발견하게 된다."""
    failing = gate_verdict({"a_deit_ti": 0.10, "b_vim_ti": 0.99, "c_cmt_ti": 0.20})
    assert failing == ["a_deit_ti", "c_cmt_ti"]


def test_select_subset_indices_spans_every_class_when_stride_matches_class_size():
    """클래스당 500장짜리 가짜 데이터셋에서 n=200을 고르면 200개 클래스 전부를
    건드려야 한다. 이게 무너지면 관문이 몇 안 되는 클래스만 외우고 통과해,
    "200장을 외운다"는 관문의 전제 자체가 거짓이 된다."""
    dataset_len = 200 * IMAGES_PER_CLASS
    indices = select_subset_indices(dataset_len, n=200)
    assert len(indices) == 200
    classes = {i // IMAGES_PER_CLASS for i in indices}
    assert len(classes) == 200


def test_select_subset_indices_returns_exactly_n_when_n_does_not_divide_evenly():
    """n이 데이터셋 크기를 나누어떨어지지 않아도 정확히 n개의 서로 다른
    인덱스를 돌려줘야 한다."""
    indices = select_subset_indices(1000, n=7)
    assert len(indices) == 7
    assert len(set(indices)) == 7
