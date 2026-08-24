"""E3의 판단이 들어가는 계산 전부. 모델도 GPU도 거치지 않는 순수 함수다.

여기서 틀리면 결과가 예외 없이 한 방향으로 편향된다. 그래서 이 파일의 모든
함수는 numpy 배열만 받고, 합성 입력으로 전부 검증된다.
"""
import numpy as np
import pandas as pd

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


AREA_BINS = (
    (0.00, 0.02, "<2%"),
    (0.02, 0.05, "2-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, 0.40, "20-40%"),
    (0.40, 1.01, ">=40%"),
)
"""면적 비율 K/224²의 구간. 라벨은 그림과 CSV에 그대로 들어가므로 ASCII다.

경계는 아래를 포함하고 위를 제외한다(0.02는 "2-5%"). 마지막 구간의 위 끝이
1.01인 것은 비율 1.0을 포함시키기 위한 것이다 — 화면 전체를 덮는 객체가
구간 없이 떨어지면 예외가 난다.
"""

ASPECT_THRESHOLD = 1.5
"""가로형/세로형을 가르는 bounding box 종횡비.

세로형 기준은 이 값의 역수다. 대칭이어야 E2와의 교차 검증이 기준의 비대칭이
아니라 모델의 성질을 잰다.
"""

LOW_SAMPLE_MIN = 30
"""이 미만인 구간은 값을 싣되 논문에 인용하지 않는다.

E2는 수렴 곡선으로 불확실성을 다뤘지만, E3는 객체마다 독립 측정값이 나오므로
표준오차가 더 정직한 불확실성 척도다. 표본이 적으면 그 표준오차 자체를
믿을 수 없다.
"""


def area_bin(fraction: float) -> str:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"면적 비율이 [0, 1] 밖이다: {fraction}")
    for low, high, label in AREA_BINS:
        if low <= fraction < high:
            return label
    raise ValueError(f"구간을 찾지 못했다: {fraction}")


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(top, left, bottom, right). 양 끝 포함이다."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("빈 마스크에는 bounding box가 없다")
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def aspect_ratio(mask: np.ndarray) -> float:
    """bounding box의 너비/높이."""
    top, left, bottom, right = bounding_box(mask)
    return (right - left + 1) / (bottom - top + 1)


def aspect_class(ratio: float) -> str:
    """E2와의 교차 검증에 쓰는 종횡비 범주.

    E2는 Vim의 ERF가 수평 능선이고 수직 감쇠가 더 가파름(감쇠비 1.345)을
    측정했다. 그렇다면 Vim은 세로로 긴 객체에서 더 나빠야 한다. 측정 비용은
    0이다 — 같은 데이터의 사후 그룹화다.
    """
    if ratio >= ASPECT_THRESHOLD:
        return "wide"
    if ratio <= 1 / ASPECT_THRESHOLD:
        return "tall"
    return "square"


def aggregate(df: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    """측정된 행만 모아 평균·표준오차·표본 수를 낸다.

    무작위 기준선을 같은 표에 넣는 것이 요점이다. 기준선은 인스턴스마다
    K/N으로 달라서 구간별 평균을 함께 내야 "이 값이 바닥을 얼마나 넘는가"를
    같은 행에서 읽을 수 있다.

    표본이 1이면 표준오차가 NaN이다(ddof=1). 그건 결함이 아니라 사실이므로
    0으로 채우지 않는다 — 채우면 오차 막대가 없는 점이 정밀한 값처럼 보인다.
    """
    measured = df[df["status"] == "ok"]
    grouped = measured.groupby(list(group_columns), dropna=False)
    out = grouped.agg(
        precision_mean=("precision_at_k", "mean"),
        precision_sem=("precision_at_k", "sem"),
        mass_mean=("mass_fraction", "mean"),
        mass_sem=("mass_fraction", "sem"),
        baseline_mean=("random_baseline", "mean"),
        n=("precision_at_k", "size"),
    ).reset_index()
    out["low_sample"] = out["n"] < LOW_SAMPLE_MIN
    return out


def expected_cells(
    model_names: tuple[str, ...], conditions: tuple[str, ...]
) -> set[tuple[str, str]]:
    """이 실행이 재기로 한 (모델, 조건) 전부."""
    return {(model, condition) for model in model_names for condition in conditions}


def common_subset(df: pd.DataFrame, cells: set[tuple[str, str]]) -> pd.DataFrame:
    """`cells`의 모든 (모델, 조건)이 측정에 성공한 인스턴스만 남긴다.

    CMT의 격자는 7×7이라 셀 하나가 32×32다. 작은 객체는 어떤 셀도 과반으로
    덮지 못해 질의 후보가 없고, 그러면 CMT의 표본만 큰 객체 쪽으로 쏠린다 —
    하필 이 실험이 재려는 축이 객체 크기다. 부분집합이 다른 채로 평균을 내면
    그 쏠림이 곧 모델 차이로 읽힌다.

    **기대 셀을 인자로 받고 df에서 유추하지 않는 것이 이 함수의 핵심이다.**
    유추하면 같은 함정이 한 층 위에서 재현된다 — 한 모델이 통째로 빠진 실행
    (OOM 킬러나 드라이버 크래시는 try/except가 잡지 못한다)에서는 기준 개수가
    조용히 줄어들어, 두 모델만 비교한 결과가 완전한 것처럼 보인다. status로
    표시된 실패와 달리 '행이 아예 없는' 실패는 df 안에 흔적을 남기지 않으므로
    경고조차 뜨지 않는다.

    제외된 인스턴스는 사라지지 않는다. `coverage.csv`에 status와 함께 남아
    있고, README가 격자별 제외 수를 적는다.
    """
    present = set(map(tuple, df[["model", "condition"]].drop_duplicates().to_numpy()))
    missing = cells - present
    if missing:
        raise ValueError(
            f"측정 행이 하나도 없는 셀이 있다: {sorted(missing)}. "
            "실행이 도중에 죽었는지 확인할 것 — 남은 셀만으로 평균을 내면 "
            "빠진 모델이 있었다는 사실 자체가 결과에서 사라진다."
        )

    measured = df[df["status"] == "ok"]
    if measured.empty:
        return measured.reset_index(drop=True)
    sizes = measured.groupby(["image", "instance_id"])["model"].transform("size")
    return measured[sizes == len(cells)].reset_index(drop=True)
