import hashlib

import pytest
import torch

from models.checkpoints import CHECKPOINTS, fetch, sha256_of, unwrap_state_dict


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"cmt vs vim")
    assert sha256_of(path) == hashlib.sha256(b"cmt vs vim").hexdigest()


def test_an_existing_file_is_not_downloaded_again(tmp_path, monkeypatch):
    """394MB를 매 실행 다시 받으면 아무도 이 실험을 돌리지 않는다."""
    (tmp_path / "cmt_small.pth").write_bytes(b"already here")
    calls = []
    monkeypatch.setattr(
        "models.checkpoints._download", lambda url, dest: calls.append(url)
    )

    path = fetch("cmt_s", root=tmp_path)

    assert path.read_bytes() == b"already here"
    assert calls == []


def test_unwrap_finds_the_weights_under_either_wrapper():
    """체크포인트마다 감싸는 키가 다르다. 감싼 채로 load_state_dict에 넘기면
    '키가 하나도 안 맞는다'는 엉뚱한 오류가 난다."""
    weights = {"head.weight": torch.zeros(2, 2)}
    assert unwrap_state_dict({"model": weights}) is weights
    assert unwrap_state_dict({"state_dict": weights}) is weights
    assert unwrap_state_dict(weights) is weights


def test_unwrap_refuses_something_that_is_not_a_state_dict():
    """조용히 빈 dict를 돌려주면 strict=False였을 때 랜덤 가중치로 측정하게 된다."""
    with pytest.raises(ValueError, match="state_dict"):
        unwrap_state_dict({"epoch": 300, "args": {}})


def test_every_checkpoint_entry_has_a_url_and_a_filename():
    for name, entry in CHECKPOINTS.items():
        assert entry.url.startswith("https://"), name
        assert entry.filename.endswith(".pth"), name
