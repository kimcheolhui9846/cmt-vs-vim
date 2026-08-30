"""results/*/에서 논문용 그림 네 개를 만든다.

저장소용 그림(results/*/*.png)과 같은 코드 경로를 쓴다 — style 인자만 다르다.
새 플로팅 모듈을 만들면 CSV에서 그림까지 경로가 둘이 되고, 언젠가 갈라져
논문 그림과 저장소 그림이 다른 데이터를 보여 준다.
"""
import os
from pathlib import Path

from figures import e1_plot, e2_plot, e3_plot, e4_plot


def build(out_dir="paper/figures"):
    # matplotlib이 PDF에 /CreationDate를 박는다. 그대로 두면 다시
    # 만들 때마다 그림 네 개가 바뀐 것으로 보여, "생성물을 다시
    # 만들었을 때 워킹트리가 깨끗한가"라는 검사가 아무것도 검사하지
    # 못한다. 재현 빌드의 관례대로 시각을 고정한다.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        e1_plot.plot_sweep("results/e1/sweep.csv",
                           out / "fig1_complexity.pdf", style="paper"),
        e2_plot.plot_erf("results/e2/erf_metrics.csv",
                         "results/e2/erf_maps.npz",
                         out / "fig2_erf.pdf", style="paper"),
        e3_plot.plot_dilution("results/e3/coverage.csv",
                              out / "fig3_dilution.pdf", style="paper"),
        e4_plot.plot_e4("results/e4/runs.csv", "results/e4/curves",
                        out / "fig4_factorial.pdf", style="paper"),
    ]


if __name__ == "__main__":
    for path in build():
        print(path)
