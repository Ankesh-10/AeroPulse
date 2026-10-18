"""Unit tier: demo_lib rules on tiny synthetic frames (never reads data_raw/ or models/).

Covers the slider and row rules for short engines (1, 2, 5, 9 cycles), the warm-up and
capped-value labels, the missing-file helpers and the curve figure.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

import demo_lib as D
import rul_lib as L
from conftest import SENS, SUBSETS, make_runs

pytestmark = pytest.mark.unit


def _engine(n, start=1):
    """One engine of n cycles with a synthetic 'pred' column."""
    df = make_runs(lengths=(n,))
    df["cycle"] = np.arange(start, start + n)
    df["pred"] = np.linspace(100.0, 5.0, n)
    return df


def _tree(tmp_path, subset):
    for p in D.required_files(subset, tmp_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


# ---------------------------------------------------------------- slider and rows
def test_slider_bounds_single_cycle_is_fixed():
    assert D.slider_bounds(1) == (1, 1, True)


@pytest.mark.parametrize("n", [2, 5, 9])
def test_slider_bounds_short_engines_get_a_slider(n):
    assert D.slider_bounds(n) == (1, n, False)


@pytest.mark.parametrize("bad", [0, -3])
def test_slider_bounds_rejects_empty_engine(bad):
    with pytest.raises(ValueError):
        D.slider_bounds(bad)


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_row_at_every_cycle_in_range_returns_that_row(n):
    eng = _engine(n)
    for c in range(1, n + 1):
        assert D.row_at(eng, c)["cycle"] == c


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_row_at_clamps_to_first_and_last_row(n):
    eng = _engine(n)
    assert D.row_at(eng, 0)["cycle"] == 1
    assert D.row_at(eng, n + 5)["cycle"] == n


def test_row_at_takes_last_row_at_or_before_the_cycle():
    eng = _engine(3)
    eng["cycle"] = [1, 2, 5]
    assert D.row_at(eng, 4)["cycle"] == 2
    assert D.row_at(eng, 5)["cycle"] == 5


# ---------------------------------------------------------------- warm-up and labels
def test_warmup_flag_covers_cycles_one_to_nine():
    assert D.WARMUP_MAX == L.WINDOW - 1
    assert all(D.is_warmup(c) for c in range(1, 10))
    assert not D.is_warmup(10)


def test_predicted_label_caps_at_120():
    assert D.predicted_label(119.4) == "119"
    assert D.predicted_label(120.0) == "125+ (model capped)"
    assert D.predicted_label(0.0) == "0"


def test_true_rul_at_reconstructs_from_the_official_final_rul():
    assert D.true_rul_at(112.0, 31, 31) == 112.0
    assert D.true_rul_at(112.0, 31, 1) == 142.0


def test_true_rul_note_at_last_cycle_is_official_and_flags_the_cap():
    text = "; ".join(D.true_rul_note(130.0, at_last=True))
    assert "official" in text and "reconstructed" not in text
    assert "above the 125 training cap" in text


def test_true_rul_note_before_last_cycle_is_reconstructed():
    text = "; ".join(D.true_rul_note(100.0, at_last=False))
    assert "reconstructed from the official final RUL" in text
    assert "training cap" not in text


# ---------------------------------------------------------------- files
def test_missing_files_complete_tree_is_empty(tmp_path):
    _tree(tmp_path, "FD002")
    assert D.missing_files("FD002", tmp_path) == []


def test_missing_files_names_exactly_the_removed_path(tmp_path):
    _tree(tmp_path, "FD001")
    (tmp_path / "data_raw" / "test_FD001.txt").unlink()
    assert D.missing_files("FD001", tmp_path) == ["data_raw/test_FD001.txt"]


def test_missing_files_reads_the_module_root_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ROOT", tmp_path)
    assert len(D.missing_files("FD003")) == 6


def test_missing_files_rejects_unknown_subset(tmp_path):
    with pytest.raises(ValueError):
        D.missing_files("FD009", tmp_path)
    with pytest.raises(ValueError):
        D.required_files("../etc", tmp_path)


@pytest.mark.parametrize("subset", SUBSETS)
def test_restore_command_regenerates_or_restores_the_exact_path(subset):
    assert D.restore_command("data_raw/test_FD001.txt", subset) == "git checkout -- data_raw"
    assert D.restore_command("outputs/results_baseline.json", subset) == (
        "git checkout -- outputs/results_baseline.json"
    )
    expected = f"python src/run_experiments.py {subset}"
    assert D.restore_command(f"models/xgb_{subset}.json", subset) == expected
    assert D.restore_command(f"models/prep_{subset}.joblib", subset) == expected


@pytest.mark.parametrize("path", ["outputs/results.json", "app.py", "src/rul_lib.py"])
def test_restore_command_rejects_other_paths(path):
    with pytest.raises(ValueError):
        D.restore_command(path, "FD001")


# ---------------------------------------------------------------- truncated engines
@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_truncated_engine_features_are_finite_and_equal_the_full_prefix(n):
    full = make_runs(lengths=(20,))
    f_full = L.make_features(full, SENS, rolling=True)
    f_cut = L.make_features(full.iloc[:n], SENS, rolling=True)
    assert np.isfinite(f_cut.to_numpy()).all()
    pd.testing.assert_frame_equal(f_cut, f_full.iloc[:n])


# ---------------------------------------------------------------- curve figure
@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_curve_figure_short_engines_have_one_axes_and_a_warmup_band(n):
    before = plt.get_fignums()
    eng = _engine(n)
    fig = D.curve_figure(eng, n, true_final=40.0, warn_rul=30)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    band = [p for p in fig.axes[0].patches if to_hex(p.get_facecolor()).upper() == "#F7F6F3"]
    assert band, "warm-up band in canvas-warm is missing"
    assert plt.get_fignums() == before
