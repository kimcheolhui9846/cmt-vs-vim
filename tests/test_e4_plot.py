"""그림이 측정 실패를 감추지 않는지 확인한다.

E1에서 리뷰어가 OOM을 0으로 그리는 회귀를 주입했더니 테스트 3건이 그대로 통과했다.
out.exists()만 단언하고 있었기 때문이다. 그래서 여기서는 "무엇을 그릴지"의 판단을
순수 함수로 빼서 직접 검증한다.
"""
from figures.e4_plot import MISSING_STATUSES, missing_cells


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
