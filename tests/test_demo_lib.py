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


# ---------------------------------------------------------------- worst-case engines
def _worst_bundle(errors, truths):
    """Synthetic Bundle: engine u has 3 cycles and last-cycle pred = truth + error."""
    n = len(errors)
    unit = np.repeat(np.arange(1, n + 1), 3)
    cycle = np.tile(np.arange(1, 4), n)
    pred = np.zeros(3 * n)
    pred[2::3] = np.asarray(truths, float) + np.asarray(errors, float)
    te = pd.DataFrame({"unit": unit, "cycle": cycle, "pred": pred})
    return D.Bundle(None, None, te, None, np.asarray(truths, float), None)


def test_worst_cases_ranking_labels_and_cap_tag():
    errors = [5.0, -20.0, 41.0, 3.0, 20.0, -2.0, -12.0, 0.0, 8.0, -8.0, 1.0, 30.0]
    truths = [50.0] * 12
    truths[6] = 130.0  # engine 7: above the cap
    b = _worst_bundle(errors, truths)
    cases = D.worst_cases(b)
    assert len(cases) == 10
    # |err|: 41 (u3), 30 (u12), 20 (u2 before u5: tie by unit), 20 (u5), 12 (u7), 8 (u9), 8 (u10), 5, 3, 2
    assert [c.unit for c in cases] == [3, 12, 2, 5, 7, 9, 10, 1, 4, 6]
    assert cases[0].label == "Engine 3: +41 late"
    assert cases[4].label == "Engine 7: -12 early (true RUL above the 125 cap)"
    assert cases[4].above_cap and not cases[0].above_cap
    assert cases[0].error == 41.0 and cases[0].true_rul == 50.0
    assert [c.error for c in D.worst_cases(b, n=3)] == [41.0, 30.0, -20.0]


def test_worst_cases_with_few_engines_lists_all_and_labels_exact():
    b = _worst_bundle([0.0, 4.0, -4.0], [30.0, 30.0, 30.0])
    cases = D.worst_cases(b)
    assert [c.unit for c in cases] == [2, 3, 1]
    assert cases[2].label == "Engine 1: +0 exact"


def test_error_word():
    assert D.error_word(0.1) == "late"
    assert D.error_word(-0.1) == "early"
    assert D.error_word(0.0) == "exact"


# ---------------------------------------------------------------- NASA panel, alert, model card
def test_nasa_point_uses_rul_lib():
    import math

    d, cost = D.nasa_point(50.0, 60.0)
    assert d == 10.0 and cost == pytest.approx(math.exp(1.0) - 1.0, abs=1e-12)
    d, cost = D.nasa_point(60.0, 50.0)
    assert d == -10.0 and cost == pytest.approx(math.exp(10 / 13) - 1.0, abs=1e-12)
    assert D.nasa_point(40.0, 40.0)[1] == 0.0


def test_nasa_curve_symmetric_and_covers_marker():
    d, cost = D.nasa_curve()
    assert len(d) % 2 == 1 and 0.0 in d
    assert np.allclose(d, -d[::-1])
    for x, c in zip(d, cost):
        assert c == L.nasa_score([0.0], [x])
    for x in (5, 20, 50):
        i, j = int(np.flatnonzero(d == x)[0]), int(np.flatnonzero(d == -x)[0])
        assert cost[i] > cost[j]
    d2, _ = D.nasa_curve(-96.0)
    assert d2.min() < -96 and d2.max() > 96


def test_nasa_examples_late_costs_more_than_early():
    late, early = D.nasa_examples(10)
    assert late == D.nasa_point(50.0, 60.0)[1] and early == D.nasa_point(60.0, 50.0)[1]
    assert late > early


def test_nasa_figure_colours():
    before = plt.get_fignums()
    d, cost = D.nasa_point(60.0, 48.0)
    fig = D.nasa_figure(d, cost)
    try:
        assert isinstance(fig, Figure) and plt.get_fignums() == before
        ax = fig.axes[0]
        inks = {to_hex(ln.get_color()).upper() for ln in ax.lines}
        assert D.COLORS["early"].upper() in inks and D.COLORS["late"].upper() in inks
        assert D.COLORS["ink_strong"].upper() in inks
        marker = [ln for ln in ax.lines if ln.get_marker() not in (None, "None", "")][0]
        y = float(marker.get_ydata()[0])
        assert ax.get_ylim()[0] <= y <= ax.get_ylim()[1]
        assert float(marker.get_xdata()[0]) == d
    finally:
        fig.clear()


def test_alert_status_flips_at_threshold():
    assert D.alert_status(29.9, 30) == ("SCHEDULE MAINTENANCE", "red")
    assert D.alert_status(30.0, 30) == ("HEALTHY", "green")


def _card_file(tmp_path):
    import json

    (tmp_path / "outputs").mkdir()
    res = {
        "FD009": {
            "test_engines": 77,
            "models": {
                D.MODEL_KEY: {
                    "rmse": {"mean": 12.3456, "std": 0.2},
                    "nasa": {"mean": 456.7, "std": 12.4},
                }
            },
        }
    }
    (tmp_path / "outputs" / "results_baseline.json").write_text(json.dumps(res), encoding="utf-8")


def test_model_card_reads_json(tmp_path):
    _card_file(tmp_path)
    card = D.model_card("FD009", root=tmp_path)
    assert card == {
        "rmse_mean": 12.3456,
        "rmse_std": 0.2,
        "nasa_mean": 456.7,
        "nasa_std": 12.4,
        "test_engines": 77,
    }
    with pytest.raises(KeyError):
        D.model_card("FD001", root=tmp_path)


def test_model_card_line_formats(tmp_path):
    _card_file(tmp_path)
    line = D.model_card_line(D.model_card("FD009", root=tmp_path))
    assert "12.35" in line and "0.20" in line and "457" in line and "12" in line
    assert "77 test engines" in line and "3 seeds" in line and "±" in line


# ---------------------------------------------------------------- tiny-model SHAP checks
@pytest.fixture(scope="module")
def tiny_bundle():
    import shap
    import xgboost

    X = L.make_features(make_runs(), SENS, rolling=True)
    y = np.random.default_rng(0).uniform(0, 125, len(X))
    model = xgboost.XGBRegressor(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
    return D.Bundle(None, model, None, X, None, shap.TreeExplainer(model))


def _style_snapshot(style):
    """StyleConfig holds numpy arrays, so == is ambiguous: compare plain lists."""
    import dataclasses

    return {k: np.asarray(v).tolist() for k, v in dataclasses.asdict(style).items()}


def test_tiny_model_waterfall_colours_rows_and_style_restored(tiny_bundle):
    from shap.plots import _style

    assert tiny_bundle.X.shape[1] == 13
    before = _style_snapshot(_style.get_style())
    fig = D.waterfall_figure(tiny_bundle, 5)
    try:
        assert tuple(fig.get_size_inches()) == pytest.approx(D.FIG_SIZE)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("3 other features" in t for t in labels), labels
        bar_colours = {to_hex(p.get_facecolor()).upper() for p in ax.patches}
        allowed = {D.COLORS["early"].upper(), D.COLORS["late"].upper()}
        assert bar_colours and bar_colours <= allowed, bar_colours
        assert not ({"#FF0051", "#008BFB"} & bar_colours)
        assert _style_snapshot(_style.get_style()) == before
    finally:
        plt.close(fig)


def test_tiny_model_shap_is_additive(tiny_bundle):
    for label in (0, 10, 30):
        x = tiny_bundle.X.loc[[label]].to_numpy(np.float32)
        vals = tiny_bundle.explainer.shap_values(x)[0]
        base = float(np.ravel(tiny_bundle.explainer.expected_value)[0])
        raw = D.raw_prediction(tiny_bundle, label)
        assert float(vals.sum()) + base == pytest.approx(raw, abs=1e-3)
        expl = D.shap_explanation(tiny_bundle, label)
        assert float(expl.values.sum() + expl.base_values) == pytest.approx(raw, abs=1e-3)
