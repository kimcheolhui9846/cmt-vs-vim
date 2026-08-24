"""사전학습 체크포인트 확보와 무결성 기록.

SHA256을 코드에 박지 않는 이유: 아직 받아 보지 않은 값을 게이트로 걸면 첫 실행이
반드시 실패하고, 그때 하는 일은 실패한 값을 복사해 넣는 것뿐이라 아무것도 검증되지
않는다. 대신 실행마다 계산해 results/e2/env.json에 기록한다 — 나중에 다른 가중치로
잰 결과와 섞이는 것을 그 기록이 막는다.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Checkpoint:
    url: str
    filename: str
    reported_top1: float


# 2026-08-19 확인. CMT는 huawei-noah/Efficient-AI-Backbones 릴리스에 없다 —
# 그 저장소의 cmt_pytorch/README.md가 원저자 개인 저장소로 링크한다.
CHECKPOINTS = {
    "cmt_s": Checkpoint(
        url="https://github.com/ggjy/CMT.pytorch/releases/download/release-v1/cmt_small.pth",
        filename="cmt_small.pth",
        reported_top1=83.5,
    ),
    "vim_s": Checkpoint(
        url="https://huggingface.co/hustvl/Vim-small-midclstok/resolve/main/vim_s_midclstok_80p5acc.pth",
        filename="vim_s_midclstok_80p5acc.pth",
        reported_top1=80.5,
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    from torch.hub import download_url_to_file

    download_url_to_file(url, str(dest), progress=True)


def fetch(name: str, root: str | Path = "checkpoints") -> Path:
    if name not in CHECKPOINTS:
        raise ValueError(f"알 수 없는 체크포인트 '{name}'")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / CHECKPOINTS[name].filename
    if not dest.exists():
        _download(CHECKPOINTS[name].url, dest)
    return dest


def unwrap_state_dict(obj: dict) -> dict[str, torch.Tensor]:
    """체크포인트가 감싼 실제 가중치를 꺼낸다."""
    for key in ("model", "state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    if any(isinstance(value, torch.Tensor) for value in obj.values()):
        return obj
    raise ValueError(
        f"state_dict를 찾을 수 없다. 최상위 키: {sorted(obj)[:10]}"
    )
