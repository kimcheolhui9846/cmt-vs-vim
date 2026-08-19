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
    central_crop,
    decay_ratio,
    has_converged,
    principal_angle_deg,
)
from data.voc import ensure_voc, load_images, sample_image_paths
from models.registry import MODEL_NAMES, build_model

CONDITIONS = ("natural", "noise", "random_init")
SAMPLE_SIZES = (16, 32, 64, 128, 256, 512)
SEED = 0

COLUMNS = [
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
    # random_init 조건은 build_model(..., pretrained=False)가 매번 새로
    # 무작위 초기화한 모델을 준다. 시드 없이는 실행마다 다른 모델을 재는
    # 셈이라 cls 토큰 가드(질량 반경 비교)의 숫자가 재현되지 않는다 — 아래
    # 루프에서 random_init을 만들기 직전에 이 시드로 고정한다.
    env["random_init_seed"] = SEED
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    (out_dir / "images.txt").write_text("\n".join(_image_names(max_n)))

    csv_path = out_dir / "erf_metrics.csv"
    maps_path = out_dir / "erf_maps.npz"

    rows: list[dict] = []
    maps: dict[str, np.ndarray] = {}

    for name in model_names:
        for condition in CONDITIONS:
            ani_history: list[float] = []
            central_history: list[float] = []
            angle_history: list[float] = []
            decay_history: list[float] = []
            if condition == "random_init":
                torch.manual_seed(SEED)
            model = build_model(name, pretrained=condition != "random_init")
            for n in sample_sizes:
                row = {column: None for column in COLUMNS}
                row.update(model=name, condition=condition, n_images=n, status="ok")

                try:
                    erf = accumulate_erf(
                        name, model, _images_for(condition, n, max_n), device=device
                    )
                except Exception as exc:  # 원본조차 못 얻은, 가장 심각한 실패
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)
                    continue

                # 맵을 지표 계산보다 먼저 저장한다. 예전엔 지표 계산이 전부
                # 끝난 뒤(else 분기)에만 저장해서, decay_ratio 하나가 던지면
                # accumulate_erf가 이미 성공시킨 맵까지 함께 버려졌다 — 그림이
                # 실제로 측정된 셀을 "not measured"로 그리는 결과를 낳았다.
                maps[f"{name}__{condition}__n{n}"] = erf
                np.savez_compressed(maps_path, **maps)

                errors: list[str] = []

                try:
                    ai = anisotropy_index(erf)
                    ai_central = anisotropy_index(central_crop(erf))
                    ani_history.append(ai)
                    central_history.append(ai_central)
                    row.update(
                        anisotropy=ai,
                        anisotropy_converged=has_converged(ani_history),
                        anisotropy_central=ai_central,
                        anisotropy_central_converged=has_converged(central_history),
                    )
                except Exception as exc:
                    errors.append(f"anisotropy: {type(exc).__name__}: {exc}")

                try:
                    pa = principal_angle_deg(erf)
                    angle_history.append(pa)
                    row.update(
                        principal_angle_deg=pa,
                        principal_angle_converged=has_converged(angle_history),
                    )
                except Exception as exc:
                    errors.append(f"principal_angle: {type(exc).__name__}: {exc}")

                try:
                    ratio, window = decay_ratio(erf)
                    decay_history.append(ratio)
                    row.update(
                        decay_ratio=ratio,
                        decay_window=window,
                        decay_ratio_converged=has_converged(decay_history),
                    )
                except Exception as exc:
                    errors.append(f"decay_ratio: {type(exc).__name__}: {exc}")

                # status는 accumulate_erf 성공 여부만 본다 — 개별 지표가
                # 실패해도 그 셀은 "측정됐지만 이 지표만 정의되지 않음"이지
                # "측정 실패"가 아니다. 실패한 지표와 사유는 error에 남긴다.
                if errors:
                    row["error"] = "; ".join(errors)

                rows.append(row)
                # 셀마다 다시 쓴다. 긴 실행이 도중에 죽어도 앞의 결과는 남는다.
                # try/except는 파이썬 예외만 잡는다 — OOM 킬러나 드라이버 크래시는
                # 못 잡으므로, 그 순간까지의 결과가 디스크에 있어야 한다.
                pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)

    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    print(run_erf().to_string(index=False))
