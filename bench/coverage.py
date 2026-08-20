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
