import json

import numpy as np
import pandas as pd

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
    import torch

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
    assert "deit_s__natural" in maps
    assert maps["deit_s__natural"].shape == (224, 224)


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
