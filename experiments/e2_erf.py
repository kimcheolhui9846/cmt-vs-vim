"""E2 — ERF 정량 측정.

논문 3.1절은 "CMT는 등방, Vim은 scan 방향 이방"을 그림 없이 서술로만 편다.
이 스크립트가 그 서술을 대체할 숫자를 만든다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bench.env import snapshot
from bench.erf import (
    accumulate_erf,
    anisotropy_index,
    decay_ratio,
    has_converged,
    principal_angle_deg,
)
from data.voc import ensure_voc, load_images, sample_image_paths
from models.registry import MODEL_NAMES, build_model

CONDITIONS = ("natural", "noise", "random_init")
SAMPLE_SIZES = (16, 32, 64, 128, 256)
SEED = 0

COLUMNS = [
    "model",
    "condition",
    "n_images",
    "anisotropy",
    "principal_angle_deg",
    "decay_ratio",
    "converged",
    "status",
    "error",
]


def _images_for(condition: str, n: int) -> torch.Tensor:
    if condition == "noise":
        return torch.randn(n, 3, 224, 224, generator=torch.Generator().manual_seed(SEED))
    paths = sample_image_paths(sorted(ensure_voc().glob("*.jpg")), n, seed=SEED)
    return load_images(paths)


def _checkpoint_hashes() -> dict[str, str]:
    from models.checkpoints import CHECKPOINTS, fetch, sha256_of

    return {name: sha256_of(fetch(name)) for name in CHECKPOINTS}


def _image_names(n: int) -> list[str]:
    """실제로 쓴 이미지 파일 이름. results/e2/images.txt에 들어간다."""
    paths = sample_image_paths(sorted(ensure_voc().glob("*.jpg")), n, seed=SEED)
    return [path.name for path in paths]


def run_erf(
    model_names: tuple[str, ...] = MODEL_NAMES,
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES,
    out_dir: Path | str = "results/e2",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = snapshot()
    env["checkpoints"] = _checkpoint_hashes()
    env["seed"] = SEED
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    (out_dir / "images.txt").write_text("\n".join(_image_names(max(sample_sizes))))

    rows: list[dict] = []
    maps: dict[str, np.ndarray] = {}

    for name in model_names:
        for condition in CONDITIONS:
            history: list[float] = []
            model = build_model(name, pretrained=condition != "random_init")
            for n in sample_sizes:
                row = {column: None for column in COLUMNS}
                row.update(model=name, condition=condition, n_images=n, status="ok")
                try:
                    erf = accumulate_erf(
                        name, model, _images_for(condition, n), device=device
                    )
                except Exception as exc:  # 한 조건의 실패로 전체를 잃지 않는다
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    continue

                history.append(anisotropy_index(erf))
                row.update(
                    anisotropy=history[-1],
                    principal_angle_deg=principal_angle_deg(erf),
                    decay_ratio=decay_ratio(erf),
                    converged=has_converged(history),
                )
                rows.append(row)
                maps[f"{name}__{condition}"] = erf

            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                out_dir / "erf_metrics.csv", index=False
            )
            np.savez_compressed(out_dir / "erf_maps.npz", **maps)

    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    print(run_erf().to_string(index=False))
