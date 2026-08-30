"""그림이 측정 실패를 감추지 않는지 확인한다.

E1에서 리뷰어가 OOM을 0으로 그리는 회귀를 주입했더니 테스트 3건이 그대로 통과했다.
out.exists()만 단언하고 있었기 때문이다. 그래서 여기서는 "무엇을 그릴지"의 판단을
순수 함수로 빼서 직접 검증한다.
"""
import pytest

from bench.factorial import cell_means, complete_seed_count
from figures.e4_plot import (
    MISSING_STATUSES,
    bar_title,
    effect_caption,
    incomplete_seed_positions,
    missing_cells,
)


def test_error_rows_are_listed_as_missing():
    rows = [
        {"cell": "a_deit_ti", "seed": "1", "top1": "0.5", "status": "ok"},
        {"cell": "b_vim_ti", "seed": "1", "top1": "", "status": "error"},
    ]
    assert missing_cells(rows) == [("b_vim_ti", 1, "error")]


def test_successful_rows_are_never_listed_as_missing():
    rows = [{"cell": "a_deit_ti", "seed": "1", "top1": "0.5", "status": "ok"}]
    assert missing_cells(rows) == []


def test_every_missing_status_has_a_colour_and_label():
    """라벨이 없는 상태가 생기면 KeyError로 죽는다 — 조용히 빠지는 것보다 낫다."""
    for status, (colour, label) in MISSING_STATUSES.items():
        assert colour and label
        assert label.isascii()  # matplotlib 기본 폰트에 한글 글리프가 없다


def test_incomplete_seed_annotations_get_distinct_positions():
    """여러 seed가 동시에 미완성이면 라벨이 겹치지 않아야 읽을 수 있다.

    Fix round 1, finding 1: missing_cells 루프는 enumerate로 겹침을 이미 피하는데
    incomplete_seeds 루프는 모든 항목을 같은 좌표에 찍고 있었다.
    """
    rows = [
        {"cell": "a_deit_ti", "seed": "1", "top1": "0.5", "status": "ok"},
        {"cell": "b_vim_ti", "seed": "1", "top1": "0.5", "status": "ok"},
        {"cell": "c_cmt_ti", "seed": "1", "top1": "0.5", "status": "ok"},
        # d_hvim seed 1 없음 -> seed 1 미완성
        {"cell": "a_deit_ti", "seed": "2", "top1": "0.5", "status": "ok"},
        {"cell": "b_vim_ti", "seed": "2", "top1": "0.5", "status": "ok"},
        # c_cmt_ti, d_hvim seed 2 없음 -> seed 2 미완성
    ]
    positions = incomplete_seed_positions(rows)
    assert [seed for seed, _ in positions] == [1, 2]
    ys = [y for _, y in positions]
    assert len(set(ys)) == len(ys)  # 좌표가 서로 달라야 겹쳐 그려지지 않는다


def test_caption_says_n_a_when_no_seed_is_complete():
    """summarize가 (None, None)을 주는 상태를 "+0.00 +- 0.00"으로 찍으면, 캠페인
    중간에 그린 그림이 이 실험의 헤드라인 수치를 0으로 주장한다.
    """
    caption = effect_caption({
        "structure": (None, None),
        "operator": (None, None),
        "interaction": (None, None),
    })
    assert "interaction: n/a" in caption
    assert "+0.00" not in caption
    assert caption.isascii()


def test_caption_still_prints_a_genuine_zero_as_a_number():
    """진짜 0은 숫자로 적어야 한다 — n/a와 구분되지 않으면 반대 방향의 거짓말이다."""
    caption = effect_caption({"interaction": (0.0, 0.0)})
    assert caption == "interaction: +0.00 +- 0.00"


def test_caption_formats_measured_effects_as_percent():
    caption = effect_caption({"interaction": (0.0123, 0.0045)})
    assert caption == "interaction: +1.23 +- 0.45"


def test_bar_title_names_the_number_of_complete_seeds():
    """행이 아예 없는 seed(사전 등록한 seed 3 -> 2 축소)는 제목에 n이 없으면
    그림에서 보이지 않는다."""
    assert "n=0" in bar_title(0)
    assert "n=3" in bar_title(3)
    assert bar_title(3).isascii()


def test_bar_title_does_not_attach_n_to_the_bars():
    """n은 캡션의 효과에 붙어야 한다 — 막대에 붙이면 제목이 틀린 말을 한다.

    막대는 cell_means에서 나오고 그 함수는 칸별로 status == "ok"인 행을 전부
    평균낸다(미완성 seed의 행도 포함). 캡션의 세 효과는 네 칸이 모두 찬 seed에
    대해서만 계산된다. A가 성공 seed 3개, D가 2개인 중간 상태에서 이전 제목
    "mean +- std over n=2 complete seeds"는 3-seed 평균인 A 막대를 n=2라고
    주장했다.
    """
    bars_line, effects_line = bar_title(2).splitlines()
    assert "bars" in bars_line
    assert "n=" not in bars_line, bars_line
    assert "effects" in effects_line
    assert "n=2" in effects_line


def test_bar_title_describes_the_sample_cell_means_actually_uses():
    """제목의 막대 설명이 cell_means의 실제 표본과 맞는지 대조한다.

    A만 세 seed가 성공하고 D는 두 seed뿐인 상태를 만들어, complete_seed_count가
    2인데 A 막대는 세 run의 평균이라는 것을 직접 확인한다. 제목이 그 둘을 하나의
    n으로 뭉뚱그리면 안 되는 이유가 이 어긋남이다.
    """
    rows = [
        {"cell": cell, "seed": str(seed), "top1": "0.5", "status": "ok"}
        for seed in (1, 2)
        for cell in ("a_deit_ti", "b_vim_ti", "c_cmt_ti", "d_hvim")
    ] + [{"cell": "a_deit_ti", "seed": "3", "top1": "0.8", "status": "ok"}]

    assert complete_seed_count(rows) == 2
    assert cell_means(rows)["a_deit_ti"][0] == pytest.approx(0.6)  # (0.5+0.5+0.8)/3

    bars_line, effects_line = bar_title(complete_seed_count(rows)).splitlines()
    assert "all ok runs" in bars_line
    assert "n=2" in effects_line
