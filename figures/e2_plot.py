"""E2 그림. results/e2/ 만 읽는다.

무엇을 그릴지 정하는 판단은 `final_metrics`와 `erf_panel_key`에 있다. 둘 다
순수 함수라 matplotlib을 거치지 않고 직접 검증할 수 있다.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = ("natural", "noise", "random_init")

# erf_maps.npz 키 형식: "{model}__{condition}__n{n}" (Task 7 개정판, 예:
# "deit_s__natural__n16"). 안쪽 N 루프가 조건당 맵 하나를 덮어써서 가장 큰
# N만 남던 버그를 고치며 바뀌었다 — CSV는 N마다 행이 남는데 맵은 하나뿐이던
# 문제. 모델 이름에도 밑줄이 들어가므로(deit_s, cmt_s, vim_s) condition은
# 정해진 셋으로 고정해 파싱하고 모델 이름은 그 앞을 그리디하게 먹는다.
MAP_KEY_PATTERN = re.compile(
    r"^(?P<model>.+)__(?P<condition>" + "|".join(CONDITIONS) + r")__n(?P<n>\d+)$"
)


def final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """모델·조건마다 가장 큰 N의 행. 실패한 행은 표에 올리지 않는다."""
    usable = df[df["status"] == "ok"]
    if usable.empty:
        return usable
    latest = usable.sort_values("n_images").groupby(["model", "condition"]).tail(1)
    return latest.sort_values(["model", "condition"]).reset_index(drop=True)


def parse_map_keys(keys) -> list[dict]:
    """npz 키를 {key, model, condition, n}으로 분해한다.

    형식에 맞지 않는 키는 조용히 건너뛴다 — 키 형식이 통째로 어긋났는지는
    이 함수가 아니라 `check_map_key_format`이 판단한다. 여기서 하나라도
    걸러내 버리면 "일부만 이상한 키"와 "전부 이상한 키"를 구분할 수 없다.
    """
    parsed = []
    for key in keys:
        match = MAP_KEY_PATTERN.match(key)
        if match:
            parsed.append({
                "key": key,
                "model": match.group("model"),
                "condition": match.group("condition"),
                "n": int(match.group("n")),
            })
    return parsed


def check_map_key_format(keys) -> None:
    """npz에 키가 있는데 하나도 형식에 맞지 않으면 형식 자체가 바뀐 것이다.

    "measured"과 "not measured"는 그림에서 같은 자리(플레이스홀더)로 처리해도
    되지만, "키 생성 코드가 잘못됐다"는 그 자리로 흡수되면 안 된다 — 흡수되면
    모든 패널이 조용히 not measured로 나오고 에러는 하나도 없이 빈 그림 같은
    PNG가 나온다. 그래서 이 경우만 따로 크게 실패시킨다.

    npz에 키가 아예 없는 경우(진짜로 아무것도 측정되지 않음)는 형식 문제가
    아니므로 통과시킨다 — 그 경우는 모든 셀이 "not measured"로 그려지는 게
    맞는 동작이다.
    """
    keys = list(keys)
    if keys and not parse_map_keys(keys):
        raise ValueError(
            "erf_maps.npz의 키가 'model__condition__nN' 형식과 맞지 않는다"
            f" (예: {keys[0]!r}). Task 7의 npz 키 포맷이 바뀌었을 수 있다."
        )


def format_metric(value, fmt: str = "{:.2f}") -> str:
    """지표 하나를 패널 제목용 문자열로. 값이 없으면(NaN/None) "n/a".

    맵은 accumulate_erf가 성공하는 순간 저장되고, 그 뒤 지표 각각은 독립적으로
    실패할 수 있다(예: decay_ratio가 피크 경계 근접으로 정의되지 않음). 그런
    셀도 맵은 있으므로 히트맵은 그려지는데, 없는 지표를 그냥 포맷하면
    "decay nan"처럼 무엇이 없는지 불분명한 문자열이 나온다. "n/a"가 "이
    지표만 정의되지 않았다"는 뜻을 분명히 한다 — 셀 전체가 "not measured"인
    것과는 다르다.
    """
    if pd.isna(value):
        return "n/a"
    return fmt.format(value)


def erf_panel_key(parsed: list[dict], model: str, condition: str) -> str | None:
    """이 (model, condition) 패널에 그릴 npz 키 — 가장 큰 N.

    final_metrics가 CSV에서 가장 큰 N의 행을 고르는 것과 일관되게, 히트맵도
    가장 큰 N의 맵을 보여준다. 해당하는 키가 하나도 없으면 None — 진짜로
    측정되지 않은 셀이라는 뜻이다.
    """
    candidates = [p for p in parsed if p["model"] == model and p["condition"] == condition]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["n"])["key"]


def plot_erf(csv_path: Path | str, npz_path: Path | str, out_path: Path | str) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path}가 비어 있다 — 먼저 실험을 실행할 것")

    maps = np.load(npz_path)
    keys = list(maps.keys())
    check_map_key_format(keys)
    parsed = parse_map_keys(keys)

    metrics = final_metrics(df)
    models = list(dict.fromkeys(metrics["model"]))
    fig, axes = plt.subplots(len(models), 4, figsize=(18, 4 * len(models)),
                             squeeze=False)

    for row, model in enumerate(models):
        for column, condition in enumerate(CONDITIONS):
            ax = axes[row][column]
            key = erf_panel_key(parsed, model, condition)
            if key is not None:
                ax.imshow(maps[key], cmap="viridis")
                cell = metrics.query("model == @model and condition == @condition")
                if not cell.empty:
                    ax.set_title(
                        f"{model} / {condition}\n"
                        f"anisotropy {format_metric(cell.iloc[0]['anisotropy'])}, "
                        f"decay {format_metric(cell.iloc[0]['decay_ratio'])}"
                    )
            else:
                ax.text(0.5, 0.5, "not measured", transform=ax.transAxes,
                        ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        curve = df[(df["model"] == model) & (df["condition"] == "natural")
                   & (df["status"] == "ok")].sort_values("n_images")
        ax = axes[row][3]
        ax.plot(curve["n_images"], curve["anisotropy"], marker="o", color="tab:blue",
                 label="anisotropy index")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("images averaged")
        ax.set_ylabel("anisotropy index", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_title(f"{model} / convergence")

        # decay_ratio가 주 지표다(anisotropy_index는 far-field 꼬리에 지배돼
        # 한계가 있다) — 같은 패널에 겹쳐 그려야 두 지표의 수렴 여부를 나란히
        # 볼 수 있다. 일부 N에서 decay_ratio가 정의되지 않을 수 있으므로
        # (예: 피크가 경계에 너무 가까움) 값이 있는 점만 그린다.
        decay_curve = curve[curve["decay_ratio"].notna()]
        if not decay_curve.empty:
            ax2 = ax.twinx()
            ax2.plot(decay_curve["n_images"], decay_curve["decay_ratio"], marker="s",
                     color="tab:orange", label="decay ratio")
            ax2.set_ylabel("decay ratio", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


if __name__ == "__main__":
    print(plot_erf("results/e2/erf_metrics.csv", "results/e2/erf_maps.npz",
                   "results/e2/e2_erf.png"))
