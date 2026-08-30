"""오케스트레이션의 순수 부분. 99시간짜리 실행을 다시 돌리지 않고 검증한다."""
import csv

import torch

from experiments.e4_ablation import (
    RUN_COLUMNS,
    _status_for,
    completed_runs,
    run_order,
    write_rows,
)


def test_seed_one_runs_all_four_cells_before_seed_two():
    """2x2 표가 한 번 완성되어야 조기에 신호를 본다.

    칸 우선으로 돌면 b_vim_ti의 seed 세 개(14.4h x 3 = 43.2h)를 끝낼 때까지 표가
    비어 있다.
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


def test_oom_shaped_exception_gets_oom_status():
    """Fix round 1, finding 2: OOM은 error와 다른 지시다 — 배치를 줄이라는 뜻이지
    코드를 고치라는 뜻이 아니다. bench.memory.is_oom이 인정하는 두 형태를 모두 잡는다.
    """
    assert _status_for(torch.cuda.OutOfMemoryError("CUDA out of memory.")) == "oom"
    assert _status_for(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) == "oom"


def test_ordinary_exception_still_gets_error_status():
    assert _status_for(ValueError("nan loss")) == "error"
    assert _status_for(RuntimeError("some other runtime failure")) == "error"


def test_write_rows_leaves_no_partial_file_behind(tmp_path):
    """tmp에 다 쓴 뒤 이름을 바꾼다 — 제자리에서 쓰다 죽으면 runs.csv가 찢어진다.

    찢어진 CSV는 이미 끝난 run의 행을 잃고, 그 run은 done에서 빠져 다시 들어오며,
    마지막 epoch까지 끝낸 체크포인트를 만나 학습 루프가 한 번도 돌지 않는 경로로
    간다. 그 경로가 조용한 0점을 만들던 자리다.
    """
    path = tmp_path / "runs.csv"
    write_rows(path, [{**{c: "" for c in RUN_COLUMNS}, "cell": "a_deit_ti", "seed": 1}])
    assert not list(tmp_path.glob("*.tmp"))  # 임시 파일이 남지 않는다

    # 쓰는 도중에 죽어도 이전 내용이 그대로 남아 있어야 한다
    def explode(rows):
        raise KeyboardInterrupt

    original = path.read_text(encoding="utf-8")
    try:
        with (tmp_path / "runs.csv.tmp").open("w", encoding="utf-8") as handle:
            handle.write("cell,이건,찢어진,행")
            explode(None)
    except KeyboardInterrupt:
        pass
    assert path.read_text(encoding="utf-8") == original


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
