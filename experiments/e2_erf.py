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


def _images_for(condition: str, n: int, max_n: int) -> torch.Tensor:
    """condition의 이미지 n장.

    항상 max_n장을 한 번 뽑고 앞에서 n장만 자른다. random.Random(seed).sample은
    n마다 겹치지 않는 전혀 다른 집합을 주기 때문에 — Random(0).sample(pool, 16)은
    Random(0).sample(pool, 256)의 부분집합이 아니다 — N마다 새로 뽑으면 이미지
    집합이 통째로 바뀌어 수렴 곡선이 수렴과 샘플링 변동을 뒤섞는다. 여기서는
    N을 키울 때 이미지가 추가되기만 해야 순수한 수렴만 보인다.
    """
    if condition == "noise":
        pool = torch.randn(
            max_n, 3, 224, 224, generator=torch.Generator().manual_seed(SEED)
        )
        return pool[:n]
    paths = sample_image_paths(sorted(ensure_voc().glob("*.jpg")), max_n, seed=SEED)
    return load_images(paths[:n])


def _checkpoint_hashes() -> dict[str, str]:
    from models.checkpoints import CHECKPOINTS, fetch, sha256_of

    return {name: sha256_of(fetch(name)) for name in CHECKPOINTS}


def _image_names(n: int) -> list[str]:
    """실제로 쓴 이미지 파일 이름. results/e2/images.txt에 들어간다.

    run_erf가 항상 max(sample_sizes)로 호출하므로, 이 목록의 앞에서 n장을 자른
    것이 _images_for(condition, n, max_n)이 natural/random_init에 실제로 쓰는
    이미지와 같다 — 둘 다 같은 seed로 같은 개수(max_n)를 sample_image_paths에
    요청한다.
    """
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
    max_n = max(sample_sizes)

    env = snapshot()
    env["checkpoints"] = _checkpoint_hashes()
    env["seed"] = SEED
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    (out_dir / "images.txt").write_text("\n".join(_image_names(max_n)))

    csv_path = out_dir / "erf_metrics.csv"
    maps_path = out_dir / "erf_maps.npz"

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
                        name, model, _images_for(condition, n, max_n), device=device
                    )
                except Exception as exc:  # 한 셀의 실패로 전체를 잃지 않는다
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                else:
                    history.append(anisotropy_index(erf))
                    row.update(
                        anisotropy=history[-1],
                        principal_angle_deg=principal_angle_deg(erf),
                        decay_ratio=decay_ratio(erf),
                        converged=has_converged(history),
                    )
                    maps[f"{name}__{condition}__n{n}"] = erf
                    np.savez_compressed(maps_path, **maps)

                rows.append(row)
                # 셀마다 다시 쓴다. 긴 실행이 도중에 죽어도 앞의 결과는 남는다.
                # try/except는 파이썬 예외만 잡는다 — OOM 킬러나 드라이버 크래시는
                # 못 잡으므로, 그 순간까지의 결과가 디스크에 있어야 한다.
                pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)

    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    print(run_erf().to_string(index=False))
