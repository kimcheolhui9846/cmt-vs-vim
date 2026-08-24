"""오케스트레이션의 순수 부분. 99시간짜리 실행을 다시 돌리지 않고 검증한다."""
import csv

from experiments.e4_ablation import (
    RUN_COLUMNS,
    completed_runs,
    run_order,
    write_rows,
)


def test_seed_one_runs_all_four_cells_before_seed_two():
    """2x2 표가 한 번 완성되어야 조기에 신호를 본다.

    칸 우선으로 돌면 43시간짜리 b_vim_ti 세 개를 끝낼 때까지 표가 비어 있다.
    """
    order = run_order([1, 2, 3])
    assert [seed for _, seed in order[:4]] == [1, 1, 1, 1]
    assert {cell for cell, _ in order[:4]} == {
        "a_deit_ti", "b_vim_ti", "c_cmt_ti", "d_hvim"
    }
    assert len(order) == 12


def test_completed_runs_skips_only_successful_rows(tmp_path):
    """실패한 run은 다시 돌아야 한다. status를 안 보면 error 행이 완료로 읽힌다."""
    path = tmp_path / "runs.csv"
    write_rows(path, [
        {**{c: "" for c in RUN_COLUMNS},
         "cell": "a_deit_ti", "seed": 1, "status": "ok"},
        {**{c: "" for c in RUN_COLUMNS},
         "cell": "b_vim_ti", "seed": 1, "status": "error"},
    ])
    assert completed_runs(path) == {("a_deit_ti", 1)}


def test_completed_runs_on_missing_file_is_empty(tmp_path):
    assert completed_runs(tmp_path / "absent.csv") == set()


def test_write_rows_is_rewritten_every_call(tmp_path):
    """run마다 다시 쓴다. 마지막에 한 번만 쓰면 중간 실패가 앞선 결과를 전부 지운다."""
    path = tmp_path / "runs.csv"
    write_rows(path, [{**{c: "" for c in RUN_COLUMNS}, "cell": "a_deit_ti", "seed": 1}])
    write_rows(path, [
        {**{c: "" for c in RUN_COLUMNS}, "cell": "a_deit_ti", "seed": 1},
        {**{c: "" for c in RUN_COLUMNS}, "cell": "b_vim_ti", "seed": 1},
    ])
    with path.open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
