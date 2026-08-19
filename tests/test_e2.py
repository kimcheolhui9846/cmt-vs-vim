import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import experiments.e2_erf as e2
from experiments.e2_erf import CONDITIONS, SAMPLE_SIZES, run_erf

EXPECTED_COLUMNS = [
    "model",
    "condition",
    "n_images",
    "anisotropy",
    "anisotropy_converged",
    "anisotropy_central",
    "anisotropy_central_converged",
    "principal_angle_deg",
    "principal_angle_converged",
    "decay_ratio",
    "decay_window",
    "decay_ratio_converged",
    "status",
    "error",
]


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
    assert list(df.columns) == EXPECTED_COLUMNS


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


def test_env_json_records_the_random_init_seed(tmp_path, monkeypatch):
    """random_init은 매번 새로 무작위 초기화된 모델을 잰다. 어떤 시드로
    고정했는지 없으면 cls 토큰 가드(질량 반경 비교) 숫자가 재현되지 않는다는
    사실 자체를 나중에 알아볼 수 없다."""
    _offline(monkeypatch)

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    env = json.loads((tmp_path / "env.json").read_text())
    assert env["random_init_seed"] == e2.SEED


def test_random_init_is_seeded_for_reproducibility(tmp_path, monkeypatch):
    """random_init 조건은 build_model(..., pretrained=False)가 실행마다 새로
    무작위 초기화한 모델을 준다. 시드를 걸지 않으면 실행마다 다른 모델을
    재는 셈이라, 이 실험의 핵심 정직성 장치(랜덤 초기화 vs 학습된 모델의
    질량 반경 비교)의 숫자가 재현되지 않는다."""
    captured_weights = []

    def fake_build_model(name, pretrained=True, **kwargs):
        layer = torch.nn.Linear(3, 3)
        captured_weights.append(layer.weight.detach().clone())
        return layer

    _offline(monkeypatch)
    monkeypatch.setattr(e2, "build_model", fake_build_model)

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path / "a")
    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path / "b")

    # CONDITIONS 순서는 (natural, noise, random_init)이므로 각 실행에서
    # build_model이 세 번 불린다 — 세 번째(인덱스 2)가 random_init.
    first_random_init = captured_weights[2]
    second_random_init = captured_weights[5]
    assert torch.equal(first_random_init, second_random_init)


def test_the_image_list_is_recorded(tmp_path, monkeypatch):
    """어떤 이미지로 잰 값인지 없으면 재현이 불가능하다."""
    _offline(monkeypatch)

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    assert (tmp_path / "images.txt").read_text().splitlines() == [
        "img_0.jpg", "img_1.jpg", "img_2.jpg", "img_3.jpg"
    ]


def test_a_failing_cell_does_not_lose_the_rest(tmp_path, monkeypatch):
    """accumulate_erf 자체가 실패하면(원본조차 못 얻음) 그 셀만 status="error"로
    남고 나머지는 살아남는다 — 가장 심각한 실패 모드다."""
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


def test_a_failing_metric_leaves_only_that_metric_blank(tmp_path, monkeypatch):
    """decay_ratio 하나가 던져도 accumulate_erf는 성공했으므로 그 셀은
    status="ok"로 남아야 한다 — "측정 실패"가 아니라 "이 지표만 정의되지
    않음"이기 때문이다. anisotropy/principal_angle_deg는 decay_ratio와
    무관하게 독립적으로 계산되므로 그대로 채워진다. 실패 사유는 error 열에
    남는다. (예전엔 지표 네 개가 한 try 블록에 묶여 있어 하나가 던지면
    넷 다, 그리고 그 셀 전체가 status="error"로 사라졌다 — cmt_s/noise
    실측에서 실제로 이 경로로 맵까지 함께 사라진 바 있다.)"""
    calls = {"n": 0}

    def flaky_decay_ratio(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("피크가 경계에 너무 가깝다")
        return 1.0, 64

    _offline(monkeypatch)
    monkeypatch.setattr(e2, "decay_ratio", flaky_decay_ratio)

    df = run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    assert len(df) == 3
    assert list(df["status"]).count("error") == 0
    failed = df[df["error"].notna()]
    assert len(failed) == 1
    assert "decay_ratio" in failed.iloc[0]["error"]
    assert pd.isna(failed.iloc[0]["decay_ratio"])
    assert not pd.isna(failed.iloc[0]["anisotropy"])
    assert not pd.isna(failed.iloc[0]["principal_angle_deg"])


def test_the_map_survives_a_metric_failure(tmp_path, monkeypatch):
    """decay_ratio가 던져도 accumulate_erf가 이미 만든 맵은 npz에 남아야
    한다. 예전엔 맵 저장이 지표 계산 뒤(성공한 경우에만 실행되는 else
    분기)에 있어서, decay_ratio 하나의 실패로 원본 맵까지 함께 사라져
    그림이 실제로 측정된 셀을 "not measured"로 그렸다."""
    calls = {"n": 0}

    def flaky_decay_ratio(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # CONDITIONS 순서상 noise가 두 번째
            raise ValueError("피크가 경계에 너무 가깝다")
        return 1.0, 64

    _offline(monkeypatch)
    monkeypatch.setattr(e2, "decay_ratio", flaky_decay_ratio)

    run_erf(model_names=("deit_s",), sample_sizes=(4,), out_dir=tmp_path)

    maps = np.load(tmp_path / "erf_maps.npz")
    assert "deit_s__noise__n4" in maps


def test_convergence_is_tracked_per_metric(tmp_path, monkeypatch):
    """네 지표(anisotropy/anisotropy_central/principal_angle/decay_ratio) 각각
    독립적으로 수렴 이력을 본다. 예전엔 anisotropy_index 이력 하나만 보고
    통짜 converged 열을 채워서, decay_ratio가 계획의 5% 기준으로 전혀
    수렴하지 않았는데도 anisotropy만 보고 True가 찍힌 적이 있었다(Critical 1).

    이 결함은 "네 지표가 전부 같이 움직이는" 스텁으로는 재현되지 않는다 —
    그 경우 넷 다 우연히 같은 방향으로 수렴/미수렴해서 통짜 열이든 지표별
    열이든 결과가 똑같아 보인다. 여기서는 한쪽(anisotropy)은 매 N마다 5배씩
    계속 벌어지게, 다른 쪽(principal_angle/decay_ratio)은 매번 완전히 같은
    값을 주게 만들어 실제로 갈라놓는다."""
    _offline(monkeypatch)

    # anisotropy_index는 한 셀에서 두 번 불린다(전체 맵, 중심 크롭). 둘 다 같은
    # 발산 시퀀스를 쓰게 해서 anisotropy_converged와 anisotropy_central_converged
    # 둘 다 미수렴 상태를 유지하는지 함께 확인한다.
    diverging = iter([1.0, 1.0, 5.0, 5.0, 25.0, 25.0])
    monkeypatch.setattr(e2, "anisotropy_index", lambda erf: next(diverging))
    monkeypatch.setattr(e2, "principal_angle_deg", lambda erf: 30.0)
    monkeypatch.setattr(e2, "decay_ratio", lambda erf: (1.0, 64))

    df = run_erf(model_names=("deit_s",), sample_sizes=(4, 8, 16), out_dir=tmp_path)

    natural = df[df["condition"] == "natural"].sort_values("n_images")
    last = natural.iloc[-1]
    # 5.0 -> 25.0은 +400% 변화라 anisotropy 계열은 계속 미수렴이어야 한다.
    assert bool(last["anisotropy_converged"]) is False
    assert bool(last["anisotropy_central_converged"]) is False
    # 매번 같은 값을 주는 두 지표는 상대 변화 0%로 즉시 수렴해야 한다.
    assert bool(last["principal_angle_converged"]) is True
    assert bool(last["decay_ratio_converged"]) is True


def test_anisotropy_central_is_recorded(tmp_path, monkeypatch):
    """중심 128²만 잘라 다시 잰 비등방 지수와 그 수렴 플래그. 2차 모먼트
    지수가 far-field 꼬리에 얼마나 좌우되는지를 전체값과 나란히 놓고 봐야
    하고, 다른 세 지표와 마찬가지로 수렴 여부도 독립적으로 남아야 한다."""
    _offline(monkeypatch)

    df = run_erf(model_names=("deit_s",), sample_sizes=(4, 8), out_dir=tmp_path)

    assert df["anisotropy_central"].notna().all()
    assert df["anisotropy_central_converged"].notna().all()


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
