"""SETUP-04: the smoke wrapper writes to smoke/ only and reproduces the default-XGBoost canary.

Also checks RERUN-04 on the smoke output: real feature names in the gain importance and the
additive seed-mean prediction columns.
"""
import json
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

pytestmark = [pytest.mark.slow, pytest.mark.data]

PRED_COLUMNS = ["unit", "cycles_observed", "true_rul", "xgb_pred", "rf_pred", "xgb_pred_mean", "rf_pred_mean"]


def _snapshot():
    snap = {}
    for d in ("outputs", "models"):
        for p in sorted((ROOT / d).rglob("*")) if (ROOT / d).exists() else []:
            if p.is_file():
                st = p.stat()
                snap[str(p.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns)
    return snap


def _assert_rerun04(res, pred, feature_names):
    """RERUN-04 on a one-seed (smoke) run: `res` is results["FD001"], `pred` the pred CSV frame."""
    assert list(pred.columns) == PRED_COLUMNS, list(pred.columns)
    # One smoke seed: the seed-mean column equals the seed-0 column. A tolerance, not equality,
    # because the CSV round-trips float32 predictions.
    for mean_col, col in (("xgb_pred_mean", "xgb_pred"), ("rf_pred_mean", "rf_pred")):
        assert np.allclose(pred[mean_col], pred[col], rtol=0, atol=1e-4), mean_col
    keys = list(res["top_gain_features"])
    assert 1 <= len(keys) <= 15, keys
    assert not [k for k in keys if re.fullmatch(r"f\d+", k)], keys
    assert set(keys) <= set(feature_names), sorted(set(keys) - set(feature_names))


def test_smoke_run_isolated_and_reproduces_default_xgb_canary():
    before = _snapshot()
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_run.py"), "FD001"],
                          cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert _snapshot() == before, "smoke run touched outputs/ or models/"
    res = json.loads((ROOT / "smoke" / "outputs" / "results.json").read_text())
    d = res["FD001"]["models"]["XGBoost default params (rolling features)"]
    assert round(d["rmse"]["mean"], 2) == 20.08 and round(d["nasa"]["mean"]) == 786
    assert (ROOT / "smoke" / "models" / "xgb_FD001.json").is_file()
    prep = joblib.load(ROOT / "smoke" / "models" / "prep_FD001.joblib")
    pred = pd.read_csv(ROOT / "smoke" / "outputs" / "pred_FD001.csv")
    _assert_rerun04(res["FD001"], pred, prep["feature_names"])
