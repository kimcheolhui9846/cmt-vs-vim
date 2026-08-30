"""논문용 그림이 저장소 그림과 같은 데이터 경로에서 나오는지 확인한다.

새 플로팅 모듈을 만들면 CSV에서 그림까지 경로가 둘이 되고, 언젠가 갈라져
논문 그림과 저장소 그림이 다른 데이터를 보여 준다. 그래서 기존 모듈에 style
인자만 더했고, 이 테스트가 그 구조를 지킨다.
"""
import inspect

import pytest

from figures import e1_plot, e2_plot, e3_plot, e4_plot, style as figstyle
from paper import make_figures

MODULES = (e1_plot, e2_plot, e3_plot, e4_plot)
PLOTTERS = (e1_plot.plot_sweep, e2_plot.plot_erf,
            e3_plot.plot_dilution, e4_plot.plot_e4)


@pytest.mark.parametrize("fn", PLOTTERS, ids=lambda f: f.__name__)
def test_every_plotter_takes_a_style(fn):
    params = inspect.signature(fn).parameters
    assert "style" in params, f"{fn.__name__}에 style 인자가 없다"
    assert params["style"].default == "repo", (
        f"{fn.__name__}의 style 기본값은 'repo'여야 한다 — 기존 호출자가 "
        f"모르는 사이에 논문용 크기로 바뀌면 안 된다")


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_paper_style_is_narrower_than_repo_style(module):
    """논문은 단 너비가 정해져 있다. 저장소용 그림은 화면에서 보려고 넓다."""
    assert module.FIGSIZE["paper"][0] < module.FIGSIZE["repo"][0]


@pytest.mark.parametrize("fn", PLOTTERS, ids=lambda f: f.__name__)
def test_unknown_style_is_rejected(fn, tmp_path):
    """오타가 조용히 저장소용 크기로 떨어지면 안 된다."""
    args = {
        "plot_sweep": ("results/e1/sweep.csv", tmp_path / "x.pdf"),
        "plot_erf": ("results/e2/erf_metrics.csv", "results/e2/erf_maps.npz",
                     tmp_path / "x.pdf"),
        "plot_dilution": ("results/e3/coverage.csv", tmp_path / "x.pdf"),
        "plot_e4": ("results/e4/runs.csv", "results/e4/curves",
                    tmp_path / "x.pdf"),
    }[fn.__name__]
    with pytest.raises(ValueError, match="style"):
        fn(*args, style="poster")


def test_style_helper_rejects_unknown_and_accepts_both():
    assert figstyle.dpi("paper") > figstyle.dpi("repo")
    with pytest.raises(ValueError, match="style"):
        figstyle.check("poster")


def test_e1_paper_style_drops_to_two_panels():
    """논문 그림 1은 cross-over를 보이는 것이 목적이다.

    저장소용 다섯 패널은 화면에서 훑어보기 위한 것이고, 논문 단 너비에 다섯을
    넣으면 축 라벨이 읽히지 않는다. 대신 FLOPs와 throughput만 남긴다.
    """
    repo_panels = e1_plot.panels_for("repo")
    paper_panels = e1_plot.panels_for("paper")
    assert len(repo_panels) == 5
    assert len(paper_panels) == 2
    columns = [panel[0] for panel in paper_panels]
    assert columns == ["flops_total", "images_per_sec"]


def test_build_writes_four_pdfs(tmp_path):
    written = make_figures.build(tmp_path)
    assert {p.name for p in written} == {
        "fig1_complexity.pdf", "fig2_erf.pdf",
        "fig3_dilution.pdf", "fig4_factorial.pdf"}
    for path in written:
        assert path.stat().st_size > 0, f"{path.name}이 비어 있다"


def test_paper_figures_are_vector_pdfs(tmp_path):
    """PNG를 PDF 확장자로 감싸 두면 인쇄에서 뭉갠다."""
    written = make_figures.build(tmp_path)
    for path in written:
        assert path.read_bytes().startswith(b"%PDF"), (
            f"{path.name}이 PDF가 아니다")
