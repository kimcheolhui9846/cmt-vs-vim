"""요인 효과 계산. 손으로 검산 가능한 값으로 고정한다."""
import pytest

from bench.factorial import (
    cell_means,
    complete_seed_count,
    incomplete_seeds,
    per_seed_effects,
    summarize,
)


def _rows(values: dict[tuple[str, int], float], status: str = "ok") -> list[dict]:
    return [
        {"cell": cell, "seed": str(seed), "top1": str(top1), "status": status}
        for (cell, seed), top1 in values.items()
    ]


SEED1 = {
    ("a_deit_ti", 1): 0.50, ("b_vim_ti", 1): 0.40,
    ("c_cmt_ti", 1): 0.60, ("d_hvim", 1): 0.58,
}


def test_effects_follow_the_documented_formulas():
    effects = per_seed_effects(_rows(SEED1))[1]
    # 구조 = (0.60+0.58)/2 - (0.50+0.40)/2 = 0.59 - 0.45
    assert effects["structure"] == pytest.approx(0.14)
    # 연산자 = (0.50+0.60)/2 - (0.40+0.58)/2 = 0.55 - 0.49
    assert effects["operator"] == pytest.approx(0.06)
    # 상호작용 = (0.58-0.40) - (0.60-0.50) = 0.18 - 0.10
    assert effects["interaction"] == pytest.approx(0.08)


def test_summarize_reports_mean_and_std_across_seeds():
    rows = _rows({**SEED1, **{
        ("a_deit_ti", 2): 0.50, ("b_vim_ti", 2): 0.40,
        ("c_cmt_ti", 2): 0.60, ("d_hvim", 2): 0.62,
    }})
    mean, std = summarize(rows)["interaction"]
    assert mean == pytest.approx(0.10)   # (0.08 + 0.12) / 2
    # 이 std > 0을 지우지 말 것. 효과를 seed별로 먼저 계산하고 그다음 평균내는지를
    # 강제하는 것이 저장소 전체에서 이 한 줄뿐이다. 효과 공식이 선형이라 칸별로 먼저
    # 평균낸 뒤 효과를 계산해도 mean은 같은 값이 나오고, 위의 mean 단언은 두 순서를
    # 구분하지 못한다. 칸 우선으로 평균내면 seed 간 분산이 사라져 std가 0이 되므로,
    # "효과가 seed 분산에 묻히는가"라는 이 실험의 판정 자체를 할 수 없게 된다.
    assert std > 0


def test_error_rows_are_not_counted_as_zero():
    """status를 안 보면 실패한 run이 top1 0으로 읽혀 효과가 통째로 왜곡된다."""
    rows = _rows(SEED1) + [
        {"cell": "d_hvim", "seed": "2", "top1": "", "status": "error"}
    ]
    assert list(per_seed_effects(rows)) == [1]


def test_incomplete_seed_is_reported_not_silently_dropped():
    rows = _rows(SEED1) + [
        {"cell": "a_deit_ti", "seed": "2", "top1": "0.5", "status": "ok"}
    ]
    assert incomplete_seeds(rows) == [2]


def test_cell_means_average_over_seeds():
    rows = _rows({**SEED1, ("a_deit_ti", 2): 0.60, ("b_vim_ti", 2): 0.40,
                  ("c_cmt_ti", 2): 0.60, ("d_hvim", 2): 0.58})
    mean, _ = cell_means(rows)["a_deit_ti"]
    assert mean == pytest.approx(0.55)


def test_summarize_signals_no_complete_seed_instead_of_reporting_zero():
    """완성된 seed가 없을 때 (0.0, 0.0)을 돌려주면 그림이 헤드라인 수치를 0으로
    주장한다 — 아무것도 재지 않은 상태를 측정 결과로 내놓는 것이다.

    진짜 0(네 칸이 정말 같은 점수를 낸 경우)과 구분되어야 하므로, 그 진짜 0도 함께
    확인한다.
    """
    assert summarize([])["interaction"] == (None, None)

    partial = _rows({("a_deit_ti", 1): 0.5, ("b_vim_ti", 1): 0.4})
    assert summarize(partial)["structure"] == (None, None)

    flat = _rows({("a_deit_ti", 1): 0.5, ("b_vim_ti", 1): 0.5,
                  ("c_cmt_ti", 1): 0.5, ("d_hvim", 1): 0.5})
    mean, std = summarize(flat)["interaction"]
    assert mean == pytest.approx(0.0)  # 이쪽은 진짜 측정된 0이다
    assert std == pytest.approx(0.0)


def test_complete_seed_count_counts_only_full_tables():
    assert complete_seed_count([]) == 0
    assert complete_seed_count(_rows(SEED1)) == 1
    assert complete_seed_count(
        _rows(SEED1) + _rows({("a_deit_ti", 2): 0.5})
    ) == 1
