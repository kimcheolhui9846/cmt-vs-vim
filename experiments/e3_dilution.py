"""E3 — softmax dilution 정량 측정.

논문 3.2절은 "객체가 K개 토큰에 걸치면 attention 가중치가 1/K로 희석되지만 SSM은
희석되지 않는다"고 적었지만 근거 수치가 없다. 이 스크립트가 그 자리를 채운다.

**이 측정이 검증하는 것은 그 수식이 아니라 그 귀결이다.** gradient 귀속도로 세
모델을 같은 축에 올리기 때문에, "가중치가 1/K가 된다"를 직접 재지 않고 "객체가
커지면 기여도가 객체 밖으로 새는가"를 잰다. 논문에 쓸 때 이 구분을 명시할 것.
"""
import random
from pathlib import Path

import pandas as pd

from bench.coverage import (
    area_bin,
    aspect_class,
    aspect_ratio,
    bounding_box,
    object_pixels,
    population_size,
    query_patch,
    random_baseline,
)
from models.probes import PATCH_GRID_AT_224
from data.voc_masks import (
    image_path_for,
    instance_ids,
    instance_mask,
    load_mask,
    void_mask,
)

SEED = 0

MIN_MASK_PIXELS = 512
"""크롭 후 이보다 작은 인스턴스는 제외한다.

상위 K개가 너무 작으면 precision이 잡음이 된다. 512px는 DeiT·Vim의 16×16 패치
두 개쯤이다.
"""

INSTANCES_PER_BIN = 100
"""면적 구간마다 뽑는 인스턴스 수. 여섯 구간이므로 표본은 600개다.

균등 추출이 아니라 **구간별 층화 추출**을 쓴다. 이 실험이 재는 축이 곧 객체
면적이고 집계도 구간별로 하므로, 구간마다 표본 수가 같은 것이 이상적이다.

균등 추출로는 가장 중요한 칸이 빈다. 고정 환경 실측(자격 인스턴스 5042개):
균등 500개를 뽑으면 세 모델 모두 질의 가능한 것은 420개인데, `<2%` 구간에는
**10개**만 남아 저표본 기준 30에 한참 못 미친다. 원인은 데이터가 아니라 CMT의
격자다 — 7×7이라 셀 하나가 32×32=1024px이고, 1003px 미만인 객체가 한 셀을
과반으로 덮으려면 513px 이상이 그 셀에 몰려야 한다. 구간별 통과율은
`<2%` 0.20, `2-5%` 0.74, `5-10%` 0.97, 나머지는 1.00에 가깝다.

그 `<2%` 칸이 하필 논문 3.2절의 1/K 희석이 가장 세게 걸려야 하는 지점이라,
비면 결과의 핵심이 빈다.

100으로 잡은 근거는 실측이다 — 세 모델 모두 질의 가능한 인스턴스가 구간별로
`<2%` 114, `2-5%` 728, `5-10%` 849, `10-20%` 1083, `20-40%` 967, `>=40%` 559개
있다. 가장 빠듯한 `<2%`에 14개 여유가 남는다. 모자라면 `sample_instances`가
조용히 적게 돌려주지 않고 예외를 던진다.
"""

QUERY_GRIDS = tuple(sorted(set(PATCH_GRID_AT_224.values())))
"""세 모델이 쓰는 서로 다른 격자 크기 — 224²에서 (7, 14).

모델 이름이 아니라 격자 크기로 도는 이유는 DeiT와 Vim이 같은 14×14를 쓰기
때문이다. 질의 후보 존재 여부는 격자에만 달려 있으므로 같은 계산을 두 번 하지
않는다.
"""


def queryable_column(grid: int) -> str:
    """그 격자에서 질의 후보가 있었는지 기록하는 열 이름."""
    return f"queryable_grid{grid}"

CATALOG_COLUMNS = [
    "image",
    "instance_id",
    "area_px",
    "area_fraction",
    "area_bin",
    "k",
    "void_px",
    "population_n",
    "random_baseline",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "aspect_ratio",
    "aspect_class",
] + [queryable_column(grid) for grid in QUERY_GRIDS] + ["measurable_by_all"]
"""`objects.csv`의 열. 격자별 질의 가능 여부를 남기는 이유는 정직성 요구다 —
설계 문서가 "제외 사유와 개수를 기록한다"고 정했고, 격자별로 남겨야 README가
"CMT의 7×7 격자가 5042개 중 742개를 제외했다"처럼 적을 수 있다.
"""


def instance_rows(mask_path: Path, image_dir: Path, size: int = 224) -> list[dict]:
    """마스크 한 장에서 자격을 갖춘 인스턴스의 설명 행들."""
    mask = load_mask(mask_path, size=size)
    void = void_mask(mask)
    population = population_size(void)
    if not image_path_for(mask_path, image_dir).exists():
        raise FileNotFoundError(f"{mask_path.name}에 대응하는 JPEG이 없다")

    rows = []
    for instance_id in instance_ids(mask):
        obj = instance_mask(mask, instance_id)
        k = object_pixels(obj, void)
        if k < MIN_MASK_PIXELS:
            continue
        top, left, bottom, right = bounding_box(obj)
        ratio = aspect_ratio(obj)
        fraction = k / (size * size)
        # 격자마다 질의 후보가 있는지 여기서 한 번 판정해 목록에 남긴다.
        # Task 7이 모델마다 다시 계산하지만, 표본을 뽑는 시점에는 그 결과가
        # 필요하고 — 뽑을 대상이 세 모델 모두 잴 수 있는 것이어야 한다 —
        # 그때 모델을 세 번 만들 이유는 없다.
        queryable = {
            queryable_column(grid): query_patch(obj, grid) is not None
            for grid in QUERY_GRIDS
        }
        rows.append({
            "image": mask_path.stem,
            "instance_id": instance_id,
            "area_px": k,
            "area_fraction": fraction,
            "area_bin": area_bin(fraction),
            "k": k,
            "void_px": int(void.sum()),
            "population_n": population,
            "random_baseline": random_baseline(k, population),
            "bbox_top": top,
            "bbox_left": left,
            "bbox_bottom": bottom,
            "bbox_right": right,
            "aspect_ratio": ratio,
            "aspect_class": aspect_class(ratio),
            **queryable,
            "measurable_by_all": all(queryable.values()),
        })
    return rows


def build_catalog(mask_paths: list[Path], image_dir: Path) -> pd.DataFrame:
    """자격을 갖춘 모든 인스턴스의 목록. 파일 이름으로 정렬한다.

    정렬하는 이유는 `data.voc.sample_image_paths`와 같다 — 파일시스템 순회
    순서가 OS마다 달라서, 정렬하지 않으면 같은 시드로도 다른 객체가 뽑힌다.
    """
    rows: list[dict] = []
    for path in sorted(mask_paths):
        rows.extend(instance_rows(path, image_dir))
    return (
        pd.DataFrame(rows, columns=CATALOG_COLUMNS)
        .sort_values(["image", "instance_id"])
        .reset_index(drop=True)
    )


def sample_instances(
    catalog: pd.DataFrame, per_bin: int = INSTANCES_PER_BIN, seed: int = SEED
) -> pd.DataFrame:
    """면적 구간마다 per_bin개씩, 세 모델 모두 질의 가능한 것 중에서 뽑는다.

    모집단을 `measurable_by_all`로 먼저 좁히는 이유는 집계 단계의
    `common_subset`이 어차피 같은 일을 하기 때문이다. 뽑을 때 좁히면 같은
    실행량으로 구간마다 원하는 표본 수를 정확히 얻는다.

    **이 좁힘은 `<2%` 구간에 편향을 남긴다** — 그 구간에서 CMT가 질의할 수 있는
    것은 픽셀이 한 셀에 몰린 '뭉친' 객체 20%뿐이라, 그 칸의 비교는 작은 객체
    전체가 아니라 작고 뭉친 객체에 대한 비교다. 숨기지 말고 README에 적을 것.

    난수는 구간 순서를 고정한 하나의 Random 인스턴스로 소비한다. 구간마다 새
    시드를 만들지 않는 이유는 문자열이나 튜플을 시드로 쓰면 재현이 파이썬 해시
    설정에 걸리기 때문이다.
    """
    from bench.coverage import AREA_BINS

    pool = catalog[catalog["measurable_by_all"]]
    rng = random.Random(seed)
    picks = []
    for _, _, label in AREA_BINS:
        rows = pool[pool["area_bin"] == label].sort_values(["image", "instance_id"])
        if len(rows) < per_bin:
            raise ValueError(
                f"구간 {label}에 세 모델 모두 질의 가능한 인스턴스가 {len(rows)}개뿐이다 "
                f"(구간당 {per_bin}개 필요). 구간당 수를 줄이거나 그 구간을 한계로 적을 것."
            )
        chosen = sorted(rng.sample(range(len(rows)), per_bin))
        picks.append(rows.iloc[chosen])
    return (
        pd.concat(picks)
        .sort_values(["image", "instance_id"])
        .reset_index(drop=True)
    )
