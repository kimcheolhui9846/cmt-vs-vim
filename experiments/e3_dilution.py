"""E3 — softmax dilution 정량 측정.

논문 3.2절은 "객체가 K개 토큰에 걸치면 attention 가중치가 1/K로 희석되지만 SSM은
희석되지 않는다"고 적었지만 근거 수치가 없다. 이 스크립트가 그 자리를 채운다.

**이 측정이 검증하는 것은 그 수식이 아니라 그 귀결이다.** gradient 귀속도로 세
모델을 같은 축에 올리기 때문에, "가중치가 1/K가 된다"를 직접 재지 않고 "객체가
커지면 기여도가 객체 밖으로 새는가"를 잰다. 논문에 쓸 때 이 구분을 명시할 것.
"""
import json
import random
from pathlib import Path

import pandas as pd
import torch

from bench.attribution import gradient_map
from bench.coverage import (
    area_bin,
    aspect_class,
    aspect_ratio,
    bounding_box,
    mass_fraction,
    object_pixels,
    population_size,
    precision_at_k,
    query_patch,
    random_baseline,
)
from bench.env import snapshot
from data.voc import ensure_voc
from models.probes import PATCH_GRID_AT_224, query_token_scalar
from data.voc_masks import (
    image_path_for,
    instance_ids,
    instance_mask,
    load_image_and_mask,
    load_mask,
    mask_dir,
    void_mask,
)
from models.registry import MODEL_NAMES, build_model

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


CONDITIONS = ("pretrained", "random_init")
"""`random_init`은 논지에 직결된다.

논문 3.2절의 주장 자체가 기전(softmax 정규화 vs 상태 누적)에 관한 것이므로,
Vim의 우위가 학습 후에만 보인다면 그 기전 설명은 약해진다. E2에서 가장 값진
결과가 이 대조군에서 나왔다 — 이방성이 학습이 아니라 구조에서 왔다는 것.
"""

MEASUREMENT_COLUMNS = CATALOG_COLUMNS + [
    "model",
    "condition",
    "query_row",
    "query_col",
    "precision_at_k",
    "mass_fraction",
    "status",
    "error",
]


def measure_instance(
    model_name: str,
    model,
    row: dict,
    image_dir: Path,
    masks_dir: Path,
    device: str,
    size: int = 224,
) -> dict:
    """인스턴스 하나의 측정 행. 예외를 밖으로 내보내지 않는다.

    E2와 달리 이미지별로 평균내지 않는다 — 객체마다 하나의 측정값이 나와야
    구간별 표준오차가 의미를 갖는다.

    마스크 디렉터리를 이미지 디렉터리에서 유추하지 않고 인자로 받는다. 유추하면
    테스트가 실제 배치와 다른 경로를 타게 되어, 검증되는 코드와 실행되는 코드가
    갈린다.
    """
    mask_path = Path(masks_dir) / f"{row['image']}.png"

    out = {column: None for column in MEASUREMENT_COLUMNS}
    out.update(row)
    out.update(model=model_name, status="ok")

    try:
        image, mask = load_image_and_mask(
            image_path_for(mask_path, image_dir), mask_path, size=size
        )
        obj = instance_mask(mask, int(row["instance_id"]))
        void = void_mask(mask)

        k = object_pixels(obj, void)
        population = population_size(void)
        fraction = k / (size * size)
        out.update(
            area_px=k, k=k, void_px=int(void.sum()), population_n=population,
            area_fraction=fraction, area_bin=area_bin(fraction),
            random_baseline=random_baseline(k, population),
        )
        ratio = aspect_ratio(obj)
        out.update(aspect_ratio=ratio, aspect_class=aspect_class(ratio))

        patch = query_patch(obj, PATCH_GRID_AT_224[model_name])
        if patch is None:
            # 아무 패치나 고르지 않는다. 질의가 배경에 놓인 채 낮은 precision이
            # 나오면 그건 "이 모델은 객체를 통합하지 못한다"로 읽힌다.
            out["status"] = "no_query_patch"
            return out
        out.update(query_row=patch[0], query_col=patch[1])

        attribution = gradient_map(
            lambda x: query_token_scalar(model_name, model, x, patch[0], patch[1]),
            image,
            device=device,
        )
        out.update(
            precision_at_k=precision_at_k(attribution, obj, void),
            mass_fraction=mass_fraction(attribution, obj, void),
        )
    except Exception as exc:  # 한 인스턴스의 실패로 3000회 실행을 잃지 않는다
        out.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return out


def _checkpoint_hashes() -> dict[str, str]:
    from models.checkpoints import CHECKPOINTS, fetch, sha256_of

    return {name: sha256_of(fetch(name)) for name in CHECKPOINTS}


def run_dilution(
    model_names: tuple[str, ...] = MODEL_NAMES,
    conditions: tuple[str, ...] = CONDITIONS,
    per_bin: int = INSTANCES_PER_BIN,
    out_dir: Path | str = "results/e3",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = ensure_voc()
    masks = mask_dir()
    catalog = build_catalog(sorted(masks.glob("*.png")), image_dir)
    sample = sample_instances(catalog, per_bin=per_bin, seed=SEED)
    sample.to_csv(out_dir / "objects.csv", index=False)

    env = snapshot()
    env["checkpoints"] = _checkpoint_hashes()
    env["seed"] = SEED
    env["random_init_seed"] = SEED
    env["instances_per_bin"] = per_bin
    env["n_instances"] = len(sample)
    env["measurable_by_all"] = int(catalog["measurable_by_all"].sum())
    env["min_mask_pixels"] = MIN_MASK_PIXELS
    env["eligible_instances"] = len(catalog)
    env["patch_grid"] = PATCH_GRID_AT_224
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))

    csv_path = out_dir / "coverage.csv"
    rows: list[dict] = []
    for name in model_names:
        for condition in conditions:
            if condition == "random_init":
                # E2에서 시드 없는 랜덤 초기화 때문에 문서 숫자가 재현되지 않아
                # 리뷰에서 잡혔다. 실행마다 다른 모델을 재는 셈이었다.
                torch.manual_seed(SEED)
            model = build_model(name, pretrained=condition == "pretrained")
            model = model.to(device).eval()
            for record in sample.to_dict("records"):
                row = measure_instance(
                    name, model, record, image_dir, masks, device
                )
                row["condition"] = condition
                rows.append(row)
                # 인스턴스마다 다시 쓴다. try/except는 파이썬 예외만 잡는다 —
                # OOM 킬러나 드라이버 크래시는 못 잡으므로, 그 순간까지의 결과가
                # 이미 디스크에 있어야 한다.
                #
                # 셀 단위로 쓰면 그 주장이 성립하지 않는다. 한 셀이 600개이므로
                # 599번째에서 죽으면 599행이 통째로 메모리에만 있다가 사라진다.
                # 재개 로직이 없어 어차피 전체를 다시 돌려야 하지만, 어디까지
                # 갔고 무엇이 이상했는지 볼 수 있느냐가 달라진다.
                #
                # 비용은 쟀다: /mnt/c(9p)에서 3600회 누적 기록이 약 37초로,
                # 3600회 backward에 몇 분 걸리는 실행에서 감당할 수 있다.
                pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS).to_csv(
                    csv_path, index=False
                )

    df = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)

    for status in ("no_query_patch", "error"):
        affected = df[df["status"] == status]
        if not affected.empty:
            counts = affected.groupby(["model", "condition"]).size().to_dict()
            print(f"경고: status={status} {len(affected)}행 — {counts}")

    from bench.coverage import common_subset, expected_cells

    # 기대 셀을 명시해 넘긴다. df에서 유추하면 한 모델이 통째로 빠진 실행에서
    # 기준 개수가 함께 줄어, 두 모델만 비교한 결과가 완전한 것처럼 보인다.
    kept = common_subset(df, expected_cells(model_names, conditions))
    print(
        f"공통 부분집합: {len(kept)}행 "
        f"(인스턴스 {kept[['image', 'instance_id']].drop_duplicates().shape[0]}개). "
        "집계는 이 부분집합만 쓴다 — 모델마다 다른 표본으로 평균을 내면 "
        "그 차이가 곧 모델 차이로 읽힌다."
    )
    return df


if __name__ == "__main__":
    print(run_dilution().groupby(["model", "condition", "status"]).size())
