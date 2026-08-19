from pathlib import Path

import pytest
import torch

from data.voc import load_images, sample_image_paths


def _paths(n: int) -> list[Path]:
    return [Path(f"img_{i:04d}.jpg") for i in range(n)]


def test_same_seed_gives_the_same_sample():
    """이미지가 달라지면 ERF도 달라진다. 재현되지 않는 샘플은 측정을 무효로 만든다."""
    first = sample_image_paths(_paths(100), n=10, seed=0)
    second = sample_image_paths(_paths(100), n=10, seed=0)
    assert first == second


def test_different_seeds_give_different_samples():
    assert sample_image_paths(_paths(100), 10, seed=0) != sample_image_paths(
        _paths(100), 10, seed=1
    )


def test_sample_order_does_not_depend_on_input_order():
    """파일시스템 순회 순서는 OS마다 다르다. 정렬하지 않으면 같은 seed로도
    다른 이미지가 뽑힌다."""
    forward = sample_image_paths(_paths(100), 10, seed=0)
    backward = sample_image_paths(list(reversed(_paths(100))), 10, seed=0)
    assert forward == backward


def test_asking_for_more_than_exists_fails_loudly():
    """조용히 적게 반환하면 N=256으로 잰 줄 알았던 값이 실은 N=40이 된다."""
    with pytest.raises(ValueError, match="256"):
        sample_image_paths(_paths(40), n=256, seed=0)
