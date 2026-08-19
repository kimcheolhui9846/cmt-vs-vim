import numpy as np
import pandas as pd
import pytest

from figures.e2_plot import (
    check_map_key_format,
    erf_panel_key,
    final_metrics,
    parse_map_keys,
    plot_erf,
)

BASE = {
    "model": "vim_s",
    "condition": "natural",
    "n_images": 256,
    "anisotropy": 1.42,
    "principal_angle_deg": 3.1,
    "decay_ratio": 1.8,
    "converged": True,
    "status": "ok",
    "error": None,
}


def _row(**overrides):
    row = dict(BASE)
    row.update(overrides)
    return row


# --- final_metrics -----------------------------------------------------------


def test_final_metrics_takes_the_largest_sample_size():
    df = pd.DataFrame([_row(n_images=16, anisotropy=2.9), _row(n_images=256)])

    final = final_metrics(df)

    assert len(final) == 1
    assert final.iloc[0]["anisotropy"] == pytest.approx(1.42)


def test_unconverged_rows_are_kept_and_flagged():
    """수렴하지 않은 값을 버리면 '왜 빈칸인지 알 수 없는 표'가 된다. 남기되
    수렴 여부를 함께 싣는다."""
    df = pd.DataFrame([_row(converged=False)])

    final = final_metrics(df)

    assert len(final) == 1, "수렴하지 않은 행이 표에서 사라졌다"
    assert bool(final.iloc[0]["converged"]) is False


def test_failed_rows_never_reach_the_table():
    df = pd.DataFrame([_row(status="error", anisotropy=None)])

    assert final_metrics(df).empty


# --- npz 키 파싱 — model__condition__nN (Task 7 개정판) ------------------------


def test_parse_map_keys_extracts_model_condition_and_n():
    parsed = parse_map_keys(["deit_s__natural__n16", "vim_s__random_init__n256"])

    assert {"key": "deit_s__natural__n16", "model": "deit_s",
            "condition": "natural", "n": 16} in parsed
    assert {"key": "vim_s__random_init__n256", "model": "vim_s",
            "condition": "random_init", "n": 256} in parsed


def test_parse_map_keys_skips_keys_that_dont_match_the_format():
    """옛 형식(model__condition, N 없음)은 조용히 건너뛴다 — 형식이 통째로
    어긋났는지 판단하는 건 check_map_key_format의 몫이다."""
    parsed = parse_map_keys(["deit_s__natural", "not_a_key_at_all"])

    assert parsed == []


def test_erf_panel_key_picks_the_largest_n():
    parsed = parse_map_keys([
        "deit_s__natural__n16",
        "deit_s__natural__n64",
        "deit_s__natural__n256",
    ])

    assert erf_panel_key(parsed, "deit_s", "natural") == "deit_s__natural__n256"


def test_erf_panel_key_returns_none_for_a_genuinely_unmeasured_cell():
    parsed = parse_map_keys(["deit_s__natural__n16"])

    assert erf_panel_key(parsed, "deit_s", "noise") is None


# --- 키 형식이 통째로 어긋난 경우는 크게 실패한다 -------------------------------


def test_check_map_key_format_raises_when_no_key_matches():
    """옛 형식(__n 없음)만 든 npz는 '측정 안 됨'이 아니라 '형식이 바뀜'이다.
    조용히 넘어가면 모든 패널이 not measured로 나와 그림이 빈 것처럼 보이는데
    에러는 하나도 없다."""
    with pytest.raises(ValueError, match="model__condition"):
        check_map_key_format(["deit_s__natural", "vim_s__noise"])


def test_check_map_key_format_accepts_valid_keys():
    check_map_key_format(["deit_s__natural__n16"])  # 예외가 나면 실패


def test_check_map_key_format_accepts_an_empty_npz():
    """키가 아예 없는 것은 형식 문제가 아니라 '아직 아무것도 측정되지 않음'이다."""
    check_map_key_format([])  # 예외가 나면 실패


# --- plot_erf ------------------------------------------------------------


def _write_sample(tmp_path, n=256):
    csv = tmp_path / "erf_metrics.csv"
    pd.DataFrame([_row(model=m, condition=c)
                  for m in ("deit_s", "cmt_s", "vim_s")
                  for c in ("natural", "noise", "random_init")]).to_csv(csv, index=False)
    npz = tmp_path / "erf_maps.npz"
    np.savez_compressed(npz, **{
        f"{m}__{c}__n{n}": np.random.rand(224, 224)
        for m in ("deit_s", "cmt_s", "vim_s")
        for c in ("natural", "noise", "random_init")
    })
    return csv, npz


def test_writes_a_png(tmp_path):
    csv, npz = _write_sample(tmp_path)

    out = plot_erf(csv, npz, tmp_path / "e2.png")

    assert out.exists() and out.stat().st_size > 0


def test_plot_erf_uses_the_largest_n_map_when_several_are_present(tmp_path):
    """CSV가 final_metrics로 가장 큰 N을 고르는 것과 일관되게, 히트맵도 가장
    큰 N의 맵을 그려야 한다."""
    csv = tmp_path / "erf_metrics.csv"
    pd.DataFrame([
        _row(model="deit_s", condition="natural", n_images=16),
        _row(model="deit_s", condition="natural", n_images=256),
    ]).to_csv(csv, index=False)
    npz = tmp_path / "erf_maps.npz"
    small = np.random.rand(224, 224)
    large = np.random.rand(224, 224)
    np.savez_compressed(npz, **{
        "deit_s__natural__n16": small,
        "deit_s__natural__n256": large,
    })

    out = plot_erf(csv, npz, tmp_path / "e2.png")

    assert out.exists() and out.stat().st_size > 0


def test_plot_erf_renders_a_placeholder_for_a_genuinely_missing_cell(tmp_path):
    """한 셀만 npz에서 빠졌을 뿐 나머지 형식은 멀쩡하면, 그 셀만 '측정되지
    않음'으로 그리고 나머지는 정상적으로 그린다."""
    csv = tmp_path / "erf_metrics.csv"
    pd.DataFrame([
        _row(model="deit_s", condition="natural"),
        _row(model="deit_s", condition="noise"),
    ]).to_csv(csv, index=False)
    npz = tmp_path / "erf_maps.npz"
    np.savez_compressed(npz, **{
        "deit_s__natural__n256": np.random.rand(224, 224),
        # noise 조건의 맵은 일부러 빼둔다 — 진짜 미측정 셀
    })

    out = plot_erf(csv, npz, tmp_path / "e2.png")

    assert out.exists() and out.stat().st_size > 0


def test_plot_erf_raises_loudly_on_a_key_format_mismatch(tmp_path):
    """npz 키가 통째로 옛 형식이면(model__condition, N 없음) 모든 패널이
    조용히 not measured로 나오는 대신 크게 실패해야 한다."""
    csv = tmp_path / "erf_metrics.csv"
    pd.DataFrame([_row(model="deit_s", condition="natural")]).to_csv(csv, index=False)
    npz = tmp_path / "erf_maps.npz"
    np.savez_compressed(npz, **{"deit_s__natural": np.random.rand(224, 224)})

    with pytest.raises(ValueError, match="model__condition"):
        plot_erf(csv, npz, tmp_path / "e2.png")
