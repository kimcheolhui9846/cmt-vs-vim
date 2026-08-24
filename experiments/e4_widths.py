"""네 칸을 같은 파라미터 예산에 맞추는 탐색.

학습 없이 파라미터 카운트만 본다. 스톡 설정은 A 5.43M / B 6.86M / C 10.55M로
C가 A의 1.94배이고, 그대로 두면 구조 주효과에 용량 차이가 섞인다.

기준점은 B(Vim-Ti 스톡)의 6.86M이다. B만 스톡이고 A·C·D는 조정된 모델이므로
문서와 논문에서 조정 사실을 이름에 단다.
"""
from pathlib import Path

import yaml

from models.registry import build_e4_model

PARAM_TARGET = 6_860_000
PARAM_TOLERANCE = 0.05


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def within_budget(n: int) -> bool:
    return abs(n - PARAM_TARGET) <= PARAM_TARGET * PARAM_TOLERANCE


def load_common_config(root: str | Path = "configs") -> dict:
    return yaml.safe_load((Path(root) / "e4_common.yaml").read_text(encoding="utf-8"))


def load_cell_config(cell: str, root: str | Path = "configs") -> dict:
    return yaml.safe_load(
        (Path(root) / f"e4_{cell}.yaml").read_text(encoding="utf-8")
    )


def search(cell: str, candidates: list[dict]) -> tuple[dict, int]:
    """예산에 가장 가까운 후보를 고른다. 대역 안이 하나도 없으면 죽는다."""
    scored = []
    for cfg in candidates:
        n = count_params(build_e4_model(cell, cfg, num_classes=200, img_size=64))
        scored.append((abs(n - PARAM_TARGET), n, cfg))
    scored.sort(key=lambda item: item[0])
    _, n, cfg = scored[0]
    if not within_budget(n):
        raise RuntimeError(
            f"{cell}: 후보 {len(candidates)}개 중 대역 안이 없다. 최선 {n / 1e6:.2f}M"
        )
    return cfg, n


def main() -> None:
    """후보를 훑어 각 칸의 폭을 정하고 화면에 표로 낸다.

    이 함수는 configs를 자동으로 덮어쓰지 않는다 — 사람이 보고 고른 값을 손으로
    yaml에 적는다. 자동으로 쓰면 재현할 때 어떤 후보 목록이었는지가 사라진다.
    """
    grids = {
        "a_deit_ti": [
            {"embed_dim": d, "depth": 12, "patch_size": 8} for d in range(192, 241, 6)
        ],
        "b_vim_ti": [{"embed_dim": 192, "depth": 24, "patch_size": 8}],
        "c_cmt_ti": [
            {"embed_dims": [b, b * 2, b * 4, b * 8], "depths": [2, 10, 2, 2],
             "stem_channel": 16}
            for b in range(38, 55, 2)
        ],
        "d_hvim": [
            {"embed_dims": [b, b * 2, b * 4, b * 8], "depths": [2, 10, 2, 2],
             "stem_channel": 16}
            for b in range(38, 55, 2)
        ],
    }
    for cell, candidates in grids.items():
        cfg, n = search(cell, candidates)
        print(f"{cell:<12} {n / 1e6:6.2f}M  {cfg}")


if __name__ == "__main__":
    main()
