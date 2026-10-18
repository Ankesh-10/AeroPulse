"""Artifacts tier: only runs when models/ and outputs/ hold real run output (skipped otherwise)."""
import json
import re

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from conftest import ROOT, SUBSETS

pytestmark = pytest.mark.artifacts


@pytest.mark.parametrize("name", SUBSETS)
def test_prep_and_model_agree_on_feature_names(name):
    prep = joblib.load(ROOT / "models" / f"prep_{name}.joblib")
    assert type(prep["normaliser"]).__module__ == "rul_lib"
    model = xgb.XGBRegressor()
    model.load_model(str(ROOT / "models" / f"xgb_{name}.json"))
    assert model.n_features_in_ == len(prep["feature_names"])


def test_results_json_has_every_subset():
    res = json.loads((ROOT / "outputs" / "results.json").read_text())
    assert set(res) >= set(SUBSETS)


@pytest.mark.parametrize("name", SUBSETS)
def test_gain_importance_uses_real_feature_names(name):
    prep = joblib.load(ROOT / "models" / f"prep_{name}.joblib")
    res = json.loads((ROOT / "outputs" / "results.json").read_text())[name]
    keys = list(res["top_gain_features"])
    assert 1 <= len(keys) <= 15
    assert not [k for k in keys if re.fullmatch(r"f\d+", k)], keys
    assert set(keys) <= set(prep["feature_names"])


@pytest.mark.parametrize("name", SUBSETS)
def test_pred_csv_has_additive_seed_mean_columns(name):
    df = pd.read_csv(ROOT / "outputs" / f"pred_{name}.csv")
    res = json.loads((ROOT / "outputs" / "results.json").read_text())[name]
    assert list(df.columns) == ["unit", "cycles_observed", "true_rul", "xgb_pred", "rf_pred",
                                "xgb_pred_mean", "rf_pred_mean"]
    assert len(df) == res["test_engines"] and df["unit"].is_unique
    assert not df.isna().any().any() and (df[["xgb_pred_mean", "rf_pred_mean"]] >= 0).all().all()
    for col, model in (("xgb_pred_mean", "XGBoost (rolling features)"),
                       ("rf_pred_mean", "Random Forest (rolling features)")):
        rmse = float(np.sqrt(np.mean((df[col] - df["true_rul"]) ** 2)))
        mean_of_rmse = res["models"][model]["rmse"]["mean"]
        # Minkowski: RMSE of the seed-mean prediction <= mean of per-seed RMSEs (CSV float32 round-trip slack)
        assert rmse <= mean_of_rmse + 1e-3
        assert rmse >= mean_of_rmse - 1.0  # and not wildly better (sanity)
