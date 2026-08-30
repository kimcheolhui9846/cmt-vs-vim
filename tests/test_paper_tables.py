"""생성된 표와 매크로가 CSV와 일치하는지 검사한다.

이 테스트가 없으면 CSV가 바뀌어도 논문의 숫자가 조용히 옛 값으로 남는다.
E4 README에서 손으로 한 대조를 자동화한 것이다.
"""
import csv

import pytest

from bench.factorial import cell_means, summarize
from paper import make_tables

EXPECTED_FILES = {
    "macros.tex",
    "tab_e1_sweep.tex",
    "tab_e2_erf.tex",
    "tab_e3_excess.tex",
    "tab_e4_cells.tex",
    "tab_e4_effects.tex",
}


def _rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("generated")
    make_tables.build(out)
    return out


def test_build_writes_every_expected_file(generated):
    assert {p.name for p in generated.iterdir()} == EXPECTED_FILES


def test_e4_effects_table_matches_factorial_module(generated):
    rows = _rows("results/e4/runs.csv")
    text = (generated / "tab_e4_effects.tex").read_text(encoding="utf-8")
    for mean, std in summarize(rows).values():
        assert f"{mean * 100:+.2f}" in text
        assert f"{std * 100:.2f}" in text


def test_e4_cells_table_matches_cell_means(generated):
    rows = _rows("results/e4/runs.csv")
    text = (generated / "tab_e4_cells.tex").read_text(encoding="utf-8")
    for mean, std in cell_means(rows).values():
        assert f"{mean * 100:.2f}" in text
        assert f"{std * 100:.2f}" in text


def test_e1_table_carries_every_measured_row(generated):
    rows = _rows("results/e1/sweep.csv")
    text = (generated / "tab_e1_sweep.tex").read_text(encoding="utf-8")
    assert len(rows) == 15
    for row in rows:
        assert f"{float(row['flops_total']) / 1e9:.2f}" in text


def test_e1_table_marks_the_throughput_oom_cells(generated):
    """throughput이 빈 셀은 빈칸이 아니라 표식으로 나와야 한다.

    E1의 세 셀은 status가 'ok'인데 throughput만 OOM으로 비어 있다. 표에서
    그냥 빈칸으로 두면 재지 않은 것인지 0인지 구분되지 않는다.
    """
    rows = _rows("results/e1/sweep.csv")
    missing = [r for r in rows if not r["images_per_sec"]]
    assert len(missing) == 3
    text = (generated / "tab_e1_sweep.tex").read_text(encoding="utf-8")
    assert text.count(r"\textemdash") >= len(missing)


def test_macros_carry_the_headline_numbers(generated):
    text = (generated / "macros.tex").read_text(encoding="utf-8")
    for name in ("StructureEffect", "InteractionEffect", "VimAngleRandom",
                 "VimTallWideZ", "CmtParamsHigh", "HoursTotal",
                 "LatencySpreadHigh"):
        assert f"\\newcommand{{\\{name}}}" in text


def test_macro_values_come_from_the_csv(generated):
    rows = _rows("results/e4/runs.csv")
    structure_mean, _ = summarize(rows)["structure"]
    text = (generated / "macros.tex").read_text(encoding="utf-8")
    assert f"\\newcommand{{\\StructureEffect}}{{{structure_mean * 100:.2f}}}" in text


def test_stale_generated_files_are_removed(generated):
    stray = generated / "tab_obsolete.tex"
    stray.write_text("stale", encoding="utf-8")
    make_tables.build(generated)
    assert not stray.exists()


def test_committed_generated_files_match_the_csvs(tmp_path):
    """커밋된 paper/generated/ 가 지금의 CSV와 일치하는지 본다.

    이것이 이 파일에서 가장 중요한 검사다. 나머지는 생성기가 CSV를 옳게
    읽는지 보지만, 이것은 **논문이 실제로 \\input 하는 파일**이 최신인지 본다.
    CSV를 다시 측정하고 make_tables를 돌리는 것을 잊으면 여기서 잡힌다.
    """
    from pathlib import Path

    committed = Path("paper/generated")
    if not committed.is_dir():
        pytest.skip("paper/generated/ 가 아직 생성되지 않았다")
    make_tables.build(tmp_path)
    for name in EXPECTED_FILES:
        fresh = (tmp_path / name).read_text(encoding="utf-8")
        stored = (committed / name).read_text(encoding="utf-8")
        assert stored == fresh, (
            f"paper/generated/{name} 가 CSV와 어긋난다. "
            f"`python -m paper.make_tables`를 다시 돌릴 것.")


def test_the_four_experiments_share_one_environment():
    """논문 3절이 "하나의 고정 환경"을 주장한다. 그것이 참인지 본다.

    네 실험이 서로 다른 toolchain에서 나왔다면 그 주장은 거짓이고, 손으로 쓴
    버전 문자열로는 그 사실이 드러나지 않는다.
    """
    env = make_tables.shared_environment()
    assert set(env) == set(make_tables.ENV_KEYS)
    assert all(env.values()), f"빈 값이 있다: {env}"


def test_environment_mismatch_is_refused(tmp_path, monkeypatch):
    """한 실험만 다른 환경이면 매크로를 만들지 않고 죽어야 한다."""
    import json
    import shutil

    root = tmp_path / "results"
    for name in make_tables.EXPERIMENTS:
        (root / name).mkdir(parents=True)
        shutil.copy(f"results/{name}/env.json", root / name / "env.json")
    tampered = json.loads((root / "e2" / "env.json").read_text(encoding="utf-8"))
    tampered["torch"] = "2.9.9+cu999"
    (root / "e2" / "env.json").write_text(json.dumps(tampered), encoding="utf-8")

    monkeypatch.setattr(make_tables, "RESULTS", root)
    with pytest.raises(ValueError, match="torch"):
        make_tables.shared_environment()


def test_macros_carry_the_environment(generated):
    text = (generated / "macros.tex").read_text(encoding="utf-8")
    for name in ("PyVersion", "TorchVersion", "CudaVersion", "GpuName",
                 "DriverVersion"):
        assert rf"\newcommand{{\{name}}}" in text
