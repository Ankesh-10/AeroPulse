"""Artifacts and data tiers: drive the real app.py headlessly with AppTest.

Needs models/, data_raw/ and outputs/results_baseline.json (skipped when absent).
The parity tests prove the demo serves the Phase 2 baseline model with no train/serve skew.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import demo_lib
import rul_lib as L
from conftest import ROOT, SUBSETS

pytestmark = [
    pytest.mark.artifacts,
    pytest.mark.data,
    pytest.mark.skipif(
        not (ROOT / "outputs" / "results_baseline.json").is_file(),
        reason="outputs/results_baseline.json absent",
    ),
]

APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def _fresh_cache():
    import matplotlib.pyplot as plt

    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    plt.close("all")


@pytest.fixture(scope="module")
def bundles():
    return {s: demo_lib.load_bundle(s) for s in SUBSETS}


def run_app(**state):
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


@pytest.mark.parametrize("subset", SUBSETS)
def test_parity_last_cycle_prediction_equals_saved_xgb_pred(subset, bundles):
    b = bundles[subset]
    last = L.last_cycle_rows(b.te)
    ref = pd.read_csv(ROOT / "outputs" / f"pred_{subset}.csv")
    assert (b.te.loc[last, "unit"].to_numpy() == ref["unit"].to_numpy()).all()
    assert (b.te.loc[last, "cycle"].to_numpy() == ref["cycles_observed"].to_numpy()).all()
    np.testing.assert_allclose(
        b.te.loc[last, "pred"].to_numpy(), ref["xgb_pred"].to_numpy(), rtol=0, atol=1e-4
    )
    assert "cycle" in b.prep["feature_names"]


def test_default_page_renders(bundles):
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Predicted RUL (cycles)" in labels and "True RUL (cycles)" in labels
    b = bundles["FD001"]
    last = L.last_cycle_rows(b.te)
    expected = demo_lib.predicted_label(float(b.te.loc[last[0], "pred"]))
    assert _metric(at, "Predicted RUL (cycles)").value == expected
    md = [m.value for m in at.markdown]
    badges = [v for v in md if ":green-badge[HEALTHY]" in v or ":red-badge[SCHEDULE MAINTENANCE]" in v]
    assert len(badges) == 1
    assert any(demo_lib.DISCLAIMER in c.value for c in at.caption)


def images(at):
    """Count figure nodes (st.pyplot renders as Image); iteration over at.main is recursive."""
    return sum(1 for e in at.main if type(e).__name__ == "Image")


def test_curve_and_waterfall_render_at_last_and_first_cycle(bundles):
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert images(at) >= 2

    at.sidebar.slider(key="cycle-FD001-1").set_value(1).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert images(at) >= 2
    b = bundles["FD001"]
    first = demo_lib.row_at(demo_lib.engine_frame(b, 1), 1)
    assert _metric(at, "Predicted RUL (cycles)").value == demo_lib.predicted_label(float(first["pred"]))


def _assert_clean(at, expect_info=None):
    """No exception or error, two metrics, both figures, the disclaimer; the warm-up note only when expected."""
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.metric) == 2
    assert images(at) >= 2
    assert any(demo_lib.DISCLAIMER in c.value for c in at.caption)
    if expect_info is True:
        assert len(at.info) == 1 and "WARM-UP" in at.info[0].value
    elif expect_info is False:
        assert len(at.info) == 0


@pytest.mark.parametrize("subset", SUBSETS)
def test_sweep_first_last_shortest_engine(subset, bundles):
    b = bundles[subset]
    lengths = b.te.groupby("unit")["cycle"].max()
    units = sorted(lengths.index)
    targets = [units[0], units[-1], int(lengths.idxmin())]
    at = run_app(subset=subset)
    _assert_clean(at, expect_info=False)
    for u in targets:
        u = int(u)
        at.sidebar.selectbox(key="unit").set_value(u).run()
        key = f"cycle-{subset}-{u}"
        top = int(lengths[u])
        at.sidebar.slider(key=key).set_value(1).run()
        _assert_clean(at, expect_info=True)
        at.sidebar.slider(key=key).set_value(top).run()
        _assert_clean(at, expect_info=False)


def _truncated(b, n):
    """FD001 unit 1 cut to its first n cycles (same index labels in te and X)."""
    full = demo_lib.engine_frame(b, 1)
    max_c = int(full["cycle"].max())
    true_final = demo_lib.true_final_rul(b, 1)
    te = full[full["cycle"] <= n]
    return dataclasses.replace(
        b, te=te, X=b.X.loc[te.index], rul=np.array([true_final + (max_c - n)])
    )


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_truncated_engine_opens_without_errors(n, bundles, monkeypatch):
    cut = _truncated(bundles["FD001"], n)
    monkeypatch.setattr(demo_lib, "load_bundle", lambda subset: cut)
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert len(at.info) == 1 and "WARM-UP" in at.info[0].value
    keys = [s.key for s in at.sidebar.slider]
    if n == 1:
        assert "cycle-FD001-1" not in keys
        assert any("1 of 1" in c.value for c in at.sidebar.caption)
    else:
        slider = at.sidebar.slider(key="cycle-FD001-1")
        assert slider.max == n
        slider.set_value(1).run()
        assert not at.exception and not at.error
        at.sidebar.slider(key="cycle-FD001-1").set_value(n).run()
        assert not at.exception and not at.error


def _fingerprint(b):
    return (
        int(pd.util.hash_pandas_object(b.te, index=True).sum()),
        int(pd.util.hash_pandas_object(b.X, index=True).sum()),
        b.rul.tobytes(),
    )


def test_cache_keeps_one_subset_and_never_mutates_it(bundles, monkeypatch):
    calls = []
    real = demo_lib.load_bundle

    def recording(subset):
        calls.append(subset)
        return real(subset)

    monkeypatch.setattr(demo_lib, "load_bundle", recording)
    at = run_app()
    at.sidebar.selectbox(key="subset").set_value("FD003").run()
    assert not at.exception
    at.sidebar.selectbox(key="subset").set_value("FD001").run()
    assert not at.exception
    at.run()
    assert calls == ["FD001", "FD003", "FD001"]

    # Cached frames are shared by every session: renders must leave them byte-identical.
    st.cache_resource.clear()
    fixed = bundles["FD001"]
    before = _fingerprint(fixed)
    monkeypatch.setattr(demo_lib, "load_bundle", lambda subset: fixed)
    at = run_app()
    at.sidebar.selectbox(key="unit").set_value(2).run()
    at.sidebar.slider(key="cycle-FD001-2").set_value(3).run()
    at.sidebar.slider(key="warn_rul").set_value(60).run()
    assert not at.exception
    assert _fingerprint(fixed) == before


class _SpyExplainer:
    def __init__(self, inner):
        self.inner = inner
        self.expected_value = inner.expected_value
        self.shapes = []

    def shap_values(self, x):
        self.shapes.append(tuple(x.shape))
        return self.inner.shap_values(x)


def test_shap_runs_on_one_row_only(bundles, monkeypatch):
    spy = _SpyExplainer(bundles["FD001"].explainer)
    spied = dataclasses.replace(bundles["FD001"], explainer=spy)
    monkeypatch.setattr(demo_lib, "load_bundle", lambda subset: spied)
    at = run_app()
    at.sidebar.slider(key="cycle-FD001-1").set_value(5).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(spy.shapes) >= 2
    assert all(s == (1, spied.X.shape[1]) for s in spy.shapes)


def test_missing_file_shows_error_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_lib, "ROOT", tmp_path)
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.error) == 1
    text = at.error[0].value
    for needle in (
        "models/prep_FD001.joblib",
        "data_raw/test_FD001.txt",
        "outputs/results_baseline.json",
        "python src/run_experiments.py FD001",
        "git checkout -- data_raw",
        "git checkout -- outputs/results_baseline.json",
    ):
        assert needle in text, needle
    assert len(at.metric) == 0
    assert any(demo_lib.DISCLAIMER in c.value for c in at.caption)


def test_stale_selection_after_subset_switch(bundles):
    at = run_app(subset="FD002", unit=259)
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.selectbox(key="unit").value == 259
    at.sidebar.selectbox(key="subset").set_value("FD001").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert at.sidebar.selectbox(key="unit").value == 1
    b = bundles["FD001"]
    last = L.last_cycle_rows(b.te)
    expected = demo_lib.predicted_label(float(b.te.loc[last[0], "pred"]))
    assert _metric(at, "Predicted RUL (cycles)").value == expected


def test_capped_prediction_and_above_cap_truth_labels(bundles):
    b = bundles["FD001"]
    last = L.last_cycle_rows(b.te)
    preds = b.te.loc[last, "pred"].to_numpy()
    units = sorted(b.te["unit"].unique())
    capped = int(units[int(np.flatnonzero(preds >= demo_lib.CAP_LABEL_AT)[0])])
    above = int(units[int(np.flatnonzero(b.rul > L.RUL_CAP)[0])])

    at = run_app(unit=capped)
    _assert_clean(at)
    assert _metric(at, "Predicted RUL (cycles)").value == "125+ (model capped)"
    assert any(":yellow-badge[125+ CAPPED]" in m.value for m in at.markdown)

    at = run_app(unit=above)
    _assert_clean(at)
    notes = [c.value for c in at.caption if "above the 125 training cap" in c.value]
    assert len(notes) == 1 and "official" in notes[0]


# ---------------------------------------------------------------- worst-case picker (D-11)
def _reference_ranking(subset):
    ref = pd.read_csv(ROOT / "outputs" / f"pred_{subset}.csv")
    ref["err"] = ref["xgb_pred"] - ref["true_rul"]
    ref["abs_err"] = ref["err"].abs()
    return ref.sort_values(["abs_err", "unit"], ascending=[False, True], kind="stable").head(10)


@pytest.mark.parametrize("subset", SUBSETS)
def test_worst_cases_match_saved_predictions(subset, bundles):
    ref = _reference_ranking(subset)
    cases = demo_lib.worst_cases(bundles[subset])
    assert [c.unit for c in cases] == ref["unit"].tolist()
    assert [np.sign(c.error) for c in cases] == [np.sign(e) for e in ref["err"]]
    np.testing.assert_allclose([c.error for c in cases], ref["err"].to_numpy(), atol=1e-3)
    for c, t in zip(cases, ref["true_rul"]):
        assert c.above_cap == (t > L.RUL_CAP)


@pytest.mark.parametrize("subset", SUBSETS)
def test_worst_case_pick_selects_engine(subset, bundles):
    b = bundles[subset]
    first = demo_lib.worst_cases(b)[0]
    at = run_app(subset=subset)
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.selectbox(key="unit").value == 1
    at.sidebar.selectbox(key=f"worst-{subset}").set_value(first.unit).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert at.sidebar.selectbox(key="unit").value == first.unit
    assert at.sidebar.selectbox(key=f"worst-{subset}").value is None
    eng = demo_lib.engine_frame(b, first.unit)
    expected = demo_lib.predicted_label(float(eng["pred"].iloc[-1]))
    assert _metric(at, "Predicted RUL (cycles)").value == expected
