"""그림이 측정 실패를 감추지 않는지 확인한다.

E1에서 리뷰어가 OOM을 0으로 그리는 회귀를 주입했더니 테스트 3건이 그대로 통과했다.
out.exists()만 단언하고 있었기 때문이다. 그래서 여기서는 "무엇을 그릴지"의 판단을
순수 함수로 빼서 직접 검증한다.
"""
from figures.e4_plot import MISSING_STATUSES, incomplete_seed_positions, missing_cells


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
