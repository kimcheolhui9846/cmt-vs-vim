"""E3의 판단이 들어가는 계산 전부. 모델도 GPU도 거치지 않는 순수 함수다.

여기서 틀리면 결과가 예외 없이 한 방향으로 편향된다. 그래서 이 파일의 모든
함수는 numpy 배열만 받고, 합성 입력으로 전부 검증된다.
"""
import numpy as np

MIN_PATCH_COVERAGE = 0.5
"""질의 패치가 마스크 안에 있다고 볼 최소 덮임 비율.

'초과'로 비교한다. 정확히 절반만 덮인 패치는 그 토큰이 보는 픽셀의 절반이
배경이라, 이 실험이 재려는 '객체 안에서 던진 질의'가 아니다.
"""


def patch_coverage(object_mask: np.ndarray, grid: int) -> np.ndarray:
    """각 패치 셀이 마스크로 덮인 비율 (grid, grid)."""
    height, width = object_mask.shape
    if height != width:
        raise ValueError(f"정사각 마스크만 받는다 — {object_mask.shape}")
    if height % grid:
        raise ValueError(f"{height}는 격자 {grid}로 나누어떨어지지 않는다")
    cell = height // grid
    return (
        object_mask.astype(np.float64)
        .reshape(grid, cell, grid, cell)
        .mean(axis=(1, 3))
    )


def query_patch(
    object_mask: np.ndarray, grid: int, min_coverage: float = MIN_PATCH_COVERAGE
) -> tuple[int, int] | None:
    """마스크 안에 있으면서 무게중심에 가장 가까운 패치의 (행, 열).

    후보가 없으면 None이다. 무게중심 자체를 쓰지 않는 이유가 이 함수의 존재
    이유다 — 오목한 객체는 무게중심이 마스크 바깥에 떨어지고, 그러면 질의가
    배경에 놓인 채 낮은 precision@K가 나오면서 "이 모델은 객체를 통합하지
    못한다"로 읽힌다. 예외도 경고도 없다.

    동점은 (행, 열) 오름차순으로 깬다. 부동소수 거리가 정확히 같은 경우가
    드물지만, 남겨 두면 numpy 버전에 따라 결과가 달라져 재현이 깨진다.
    """
    coverage = patch_coverage(object_mask, grid)
    candidates = np.argwhere(coverage > min_coverage)
    if len(candidates) == 0:
        return None

    rows, cols = np.nonzero(object_mask)
    centroid_row, centroid_col = rows.mean(), cols.mean()

    cell = object_mask.shape[0] // grid
    centers = (candidates + 0.5) * cell - 0.5
    distance = (centers[:, 0] - centroid_row) ** 2 + (centers[:, 1] - centroid_col) ** 2

    order = np.lexsort((candidates[:, 1], candidates[:, 0], distance))
    best = candidates[order[0]]
    return int(best[0]), int(best[1])


def population_size(void: np.ndarray) -> int:
    """순위 모집단 크기 N. void를 뺀 픽셀 수다."""
    return int((~void).sum())


def object_pixels(object_mask: np.ndarray, void: np.ndarray) -> int:
    """K. void를 뺀 마스크 픽셀 수다."""
    return int((object_mask & ~void).sum())


def random_baseline(k: int, n: int) -> float:
    """무작위 정렬의 precision@K 기댓값. 정확히 K/N이다.

    절대 임계값이 아니라 계산된 값이므로 게이트를 통과시키려고 조정할 여지가
    없다 — E1에서 달성 불가능한 절대 기준을 걸었다가 가드를 푸는 방향으로 간
    적이 있다. 모든 보고 수치를 이 바닥과 함께 싣는다.
    """
    if n <= 0:
        raise ValueError(f"모집단 크기가 {n}이다")
    return k / n


def _valid(
    attribution: np.ndarray, object_mask: np.ndarray, void: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """void를 뺀 (기여도, 객체 여부) 1차원 쌍."""
    if not (attribution.shape == object_mask.shape == void.shape):
        raise ValueError(
            f"모양이 다르다: {attribution.shape}, {object_mask.shape}, {void.shape}"
        )
    keep = ~void
    return attribution[keep], (object_mask & keep)[keep]


def precision_at_k(
    attribution: np.ndarray, object_mask: np.ndarray, void: np.ndarray
) -> float:
    """상위 K개 기여도 픽셀 중 마스크 안의 비율. K는 마스크 픽셀 수다.

    주 지표다. 순위만 쓰므로 기여도의 척도에 면역이고 — 세 모델의 gradient
    크기는 애초에 같은 단위가 아니다 — K가 달라도 값이 비교 가능하다.
    """
    scores, inside = _valid(attribution, object_mask, void)
    k = int(inside.sum())
    if k == 0:
        raise ValueError("K가 0이다 — void를 뺀 마스크가 비었다")
    order = np.argsort(-scores, kind="stable")
    return float(inside[order][:k].sum()) / k


def mass_fraction(
    attribution: np.ndarray, object_mask: np.ndarray, void: np.ndarray
) -> float:
    """마스크가 가져가는 기여도 질량의 비율.

    보조 지표다. precision@K와 달리 크기를 쓰므로, 두 지표가 갈리면 그 자체가
    정보다 — 순위는 맞는데 질량이 흩어져 있다는 뜻이다.
    """
    scores, inside = _valid(attribution, object_mask, void)
    total = scores.sum()
    if not total > 0:
        raise ValueError(f"기여도 질량이 0이다(sum={total}) — 비율이 정의되지 않는다")
    return float(scores[inside].sum() / total)
