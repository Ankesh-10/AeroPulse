"""Artifacts tier: only runs when models/ and outputs/ hold real run output (skipped otherwise)."""
import json

import joblib
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
