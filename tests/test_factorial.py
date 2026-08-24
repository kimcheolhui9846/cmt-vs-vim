"""요인 효과 계산. 손으로 검산 가능한 값으로 고정한다."""
import pytest

from bench.factorial import (
    cell_means,
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
