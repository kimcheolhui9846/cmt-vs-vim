import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import experiments.e2_erf as e2
from experiments.e2_erf import COLUMNS, CONDITIONS, SAMPLE_SIZES, run_erf


def _stub_erf(model_name, model, images, device="cuda"):
    axis = np.arange(224) - 112
    x, y = np.meshgrid(axis, axis)
    return np.exp(-(x**2) / (2 * 30**2) - (y**2) / (2 * 10**2))


def _offline(monkeypatch):
    """단위 테스트가 네트워크를 건드리지 않게 막는다.

    이걸 빼면 체크포인트 500MB와 VOC 2GB를 테스트마다 받으러 간다. 느린 게
    문제가 아니라, 네트워크가 없는 곳에서 테스트가 실패해 '코드가 깨졌다'로
    읽힌다.
    """
    monkeypatch.setattr(e2, "accumulate_erf", _stub_erf)
    monkeypatch.setattr(e2, "_images_for", lambda *a, **k: torch.zeros(4, 3, 224, 224))
    monkeypatch.setattr(e2, "_image_names", lambda n: [f"img_{i}.jpg" for i in range(n)])
    monkeypatch.setattr(e2, "build_model", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(e2, "_checkpoint_hashes", lambda: {})


def test_the_three_conditions_are_declared():
    assert set(CONDITIONS) == {"natural", "noise", "random_init"}


def test_one_row_per_model_condition_and_sample_size(tmp_path, monkeypatch):
    _offline(monkeypatch)

    df = run_erf(model_names=("deit_s",), sample_sizes=(2, 4), out_dir=tmp_path)

    assert len(df) == 1 * 3 * 2
    assert list(df.columns) == COLUMNS


def test_the_erf_maps_are_saved_next_to_the_metrics(tmp_path, monkeypatch):
    """그림은 커밋된 원시 데이터에서 나와야 한다. 지표만 남기면 heatmap을 다시
    그릴 수 없다."""
    _offline(monkeypatch)

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    maps = np.load(tmp_path / "erf_maps.npz")
    assert "deit_s__natural__n4" in maps
    assert maps["deit_s__natural__n4"].shape == (224, 224)


def test_every_sample_size_keeps_its_own_map(tmp_path, monkeypatch):
    """지적 2: 안쪽 for n in sample_sizes 루프가 (model, condition) 키만 쓰면 N마다
    덮어써서 CSV엔 N별 행이 남는데 맵은 마지막 N 것만 남는다. 키에 n을 넣어
    N마다 따로 남아야 한다 — sample_sizes 두 개 이상으로 돌려야 이 결함이
    드러난다."""
    _offline(monkeypatch)

    run_erf(model_names=("deit_s",), sample_sizes=(2, 4), out_dir=tmp_path)

    maps = np.load(tmp_path / "erf_maps.npz")
    for n in (2, 4):
        key = f"deit_s__natural__n{n}"
        assert key in maps
        assert maps[key].shape == (224, 224)


def test_env_json_records_the_checkpoint_hashes(tmp_path, monkeypatch):
    """어느 가중치로 잰 값인지 없으면 나중에 받은 다른 체크포인트의 결과와
    구분할 수 없다."""
    _offline(monkeypatch)
    monkeypatch.setattr(e2, "_checkpoint_hashes", lambda: {"cmt_s": "abc123"})

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    env = json.loads((tmp_path / "env.json").read_text())
    assert env["checkpoints"] == {"cmt_s": "abc123"}


def test_the_image_list_is_recorded(tmp_path, monkeypatch):
    """어떤 이미지로 잰 값인지 없으면 재현이 불가능하다."""
    _offline(monkeypatch)

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    assert (tmp_path / "images.txt").read_text().splitlines() == [
        "img_0.jpg", "img_1.jpg", "img_2.jpg", "img_3.jpg"
    ]


def test_a_failing_cell_does_not_lose_the_rest(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("이 조건만 터진다")
        return _stub_erf(*args, **kwargs)

    _offline(monkeypatch)
    monkeypatch.setattr(e2, "accumulate_erf", flaky)

    df = run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    assert len(df) == 3
    assert list(df["status"]).count("error") == 1
    assert pd.read_csv(tmp_path / "erf_metrics.csv").shape[0] == 3


def test_a_failing_metric_does_not_lose_the_rest(tmp_path, monkeypatch):
    """지표 계산(anisotropy_index/principal_angle_deg/decay_ratio)에서 나는
    예외도 accumulate_erf의 예외와 똑같이 그 셀만 error로 남기고 나머지는
    살아남아야 한다. 예전엔 이 계산이 try 블록 밖(else 분기)에 있어서 여기서
    예외가 나면 실행 전체가 죽었다 — cmt_s/noise 실측에서 decay_ratio가 실제로
    이 경로로 죽어 이후 모델·조건이 통째로 사라진 바 있다."""
    calls = {"n": 0}

    def flaky_decay_ratio(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("피크가 경계에 너무 가깝다")
        return 1.0

    _offline(monkeypatch)
    monkeypatch.setattr(e2, "decay_ratio", flaky_decay_ratio)

    df = run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    assert len(df) == 3
    assert list(df["status"]).count("error") == 1
    assert pd.read_csv(tmp_path / "erf_metrics.csv").shape[0] == 3


class _FakeVocDir:
    """glob("*.jpg")만 흉내내는 가짜 디렉터리. 실제 파일시스템을 건드리지 않는다."""

    def __init__(self, paths):
        self._paths = paths

    def glob(self, pattern):
        return iter(self._paths)


def test_images_for_nests_across_sample_sizes(monkeypatch):
    """지적 1: random.Random(seed).sample(pool, n)은 n마다 겹치지 않는 전혀 다른
    집합을 준다 — Random(0).sample(pool, 16)은 Random(0).sample(pool, 256)의
    부분집합이 아니다. data.voc.sample_image_paths는 모킹하지 않고 그대로 써서
    이 성질을 실제로 재현한다. _images_for는 항상 max_n을 한 번 뽑고 앞에서
    자르는 방식으로 이 문제를 피해야 한다: 작은 n에 쓴 이미지가 큰 n에도
    그대로, 앞부분으로 남아 있어야 한다."""
    pool = [Path(f"img_{i:04d}.jpg") for i in range(64)]
    monkeypatch.setattr(e2, "ensure_voc", lambda *a, **k: _FakeVocDir(pool))

    captured = []

    def fake_load(paths):
        captured.append(list(paths))
        return torch.zeros(len(paths), 3, 224, 224)

    monkeypatch.setattr(e2, "load_images", fake_load)

    e2._images_for("natural", 8, max_n=32)
    e2._images_for("natural", 32, max_n=32)

    small_paths, large_paths = captured
    assert small_paths == large_paths[:8]


def test_noise_images_nest_across_sample_sizes():
    """noise 조건도 같은 원리다 — 같은 seed로 max_n을 한 번 뽑고 앞에서 자른다.
    네트워크나 데이터셋을 건드리지 않는 순수 함수라 모킹 없이 그대로 돌린다."""
    small = e2._images_for("noise", 2, max_n=4)
    large = e2._images_for("noise", 4, max_n=4)

    assert torch.equal(small, large[:2])


def test_run_erf_asks_for_the_same_max_n_at_every_sample_size(tmp_path, monkeypatch):
    """run_erf가 _images_for에 넘기는 max_n이 조건·N과 무관하게 sample_sizes의
    최댓값으로 고정되는지 확인한다. 안 그러면 nested 슬라이싱의 기준 pool이
    호출마다 달라져 부분집합 성질이 깨진다."""
    _offline(monkeypatch)
    calls = []

    def spy(condition, n, max_n):
        calls.append((condition, n, max_n))
        return torch.zeros(n, 3, 224, 224)

    monkeypatch.setattr(e2, "_images_for", spy)

    run_erf(model_names=("deit_s",), sample_sizes=(2, 4, 8), out_dir=tmp_path)

    assert calls
    assert all(max_n == 8 for _, _, max_n in calls)
