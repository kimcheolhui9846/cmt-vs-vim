"""2x2 요인 효과. 그림과 README가 같은 함수를 쓴다.

seed별로 효과를 먼저 계산하고 그다음 평균낸다. 칸별로 먼저 평균내면 seed 간
분산이 사라져 "효과가 seed 분산에 묻히는가"를 판정할 수 없다.
"""
import statistics

FLAT_ATTENTION = "a_deit_ti"
FLAT_SSM = "b_vim_ti"
HIER_ATTENTION = "c_cmt_ti"
HIER_SSM = "d_hvim"
CELLS = (FLAT_ATTENTION, FLAT_SSM, HIER_ATTENTION, HIER_SSM)


def _by_seed(rows: list[dict]) -> dict[int, dict[str, float]]:
    table: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get("top1"):
            continue
        table.setdefault(int(row["seed"]), {})[row["cell"]] = float(row["top1"])
    return table


def per_seed_effects(rows: list[dict]) -> dict[int, dict[str, float]]:
    """네 칸이 모두 찬 seed에 대해서만 효과를 계산한다."""
    effects = {}
    for seed, cells in sorted(_by_seed(rows).items()):
        if set(cells) != set(CELLS):
            continue
        a, b = cells[FLAT_ATTENTION], cells[FLAT_SSM]
        c, d = cells[HIER_ATTENTION], cells[HIER_SSM]
        effects[seed] = {
            "structure": (c + d) / 2 - (a + b) / 2,
            "operator": (a + c) / 2 - (b + d) / 2,
            "interaction": (d - b) - (c - a),
        }
    return effects


EFFECT_NAMES = ("structure", "operator", "interaction")


def summarize(rows: list[dict]) -> dict[str, tuple[float | None, float | None]]:
    """효과별 (평균, 표준편차). 완성된 seed가 하나도 없으면 (None, None)이다.

    (0.0, 0.0)을 돌려주면 안 된다. 캠페인 중간에 그린 그림이 "interaction:
    +0.00 +- 0.00"으로 읽혀, 아무것도 재지 않은 상태를 이 실험의 헤드라인 수치가
    0이라는 측정 결과로 주장하게 된다. 진짜 0(네 칸이 정말 같은 점수를 낸 경우)과
    "아직 잴 수 없다"는 서로 다른 사실이므로 값의 종류로 구분한다.
    """
    effects = per_seed_effects(rows)
    out = {}
    for name in EFFECT_NAMES:
        values = [effects[seed][name] for seed in sorted(effects)]
        if not values:
            out[name] = (None, None)
            continue
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        out[name] = (statistics.fmean(values), std)
    return out


def complete_seed_count(rows: list[dict]) -> int:
    """네 칸이 모두 찬 seed의 수. 그림 제목이 n을 밝히는 데 쓴다."""
    return len(per_seed_effects(rows))


def cell_means(rows: list[dict]) -> dict[str, tuple[float, float]]:
    per_cell: dict[str, list[float]] = {cell: [] for cell in CELLS}
    for cells in _by_seed(rows).values():
        for cell, top1 in cells.items():
            per_cell.setdefault(cell, []).append(top1)
    return {
        cell: (
            statistics.fmean(values) if values else 0.0,
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for cell, values in per_cell.items()
    }


def incomplete_seeds(rows: list[dict]) -> list[int]:
    """네 칸이 다 차지 않은 seed. 조용히 빠지면 표가 왜 비었는지 알 수 없다."""
    return [
        seed for seed, cells in sorted(_by_seed(rows).items())
        if set(cells) != set(CELLS)
    ]
