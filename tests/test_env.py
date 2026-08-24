from bench.env import snapshot

REQUIRED_KEYS = {"python", "torch", "cuda", "gpu", "driver", "git_commit"}


def test_snapshot_has_all_required_keys():
    assert set(snapshot().keys()) == REQUIRED_KEYS


def test_snapshot_values_are_str_or_none():
    for key, value in snapshot().items():
        assert value is None or isinstance(value, str), f"{key}가 {type(value)}"


def test_git_commit_is_a_sha():
    commit = snapshot()["git_commit"]
    assert commit is not None
    assert len(commit) == 40
    int(commit, 16)  # 16진수가 아니면 여기서 터진다
