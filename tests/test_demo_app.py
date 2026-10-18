"""Artifacts and data tiers: drive the real app.py headlessly with AppTest.

Needs models/, data_raw/ and outputs/results_baseline.json (skipped when absent).
The parity tests prove the demo serves the Phase 2 baseline model with no train/serve skew.
"""
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
