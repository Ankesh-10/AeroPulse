"""Demo seam for the AeroPulse Streamlit app (D-13).

All demo logic lives here so app.py stays a thin renderer and the rules can be unit
tested without Streamlit. This module is not frozen. It imports rul_lib at top level
because models/prep_*.joblib pickles rul_lib.ConditionNormaliser, and it never imports
the experiment runner or the tuning library. joblib, shap, xgboost and
matplotlib.pyplot are imported lazily inside the functions that need them.
"""
import matplotlib

matplotlib.use("Agg")

import dataclasses  # noqa: E402
import json  # noqa: E402,F401
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rul_lib as L  # noqa: E402

ROOT = L.ROOT
SUBSETS = tuple(L.SUBSETS)
BASELINE = "outputs/results_baseline.json"
MODEL_KEY = "XGBoost (rolling features)"
CAP_LABEL_AT = 120
CAPPED_LABEL = "125+ (model capped)"
DISCLAIMER = (
    "This is simulated NASA C-MAPSS turbofan data and a demo of a machine-learning model. "
    "It is not a maintenance or safety tool."
)


@dataclasses.dataclass(frozen=True)
class Bundle:
    """Everything the page needs for one subset (cached, never mutated)."""

    prep: dict
    model: object
    te: pd.DataFrame  # test frame plus a clipped 'pred' column, original index kept
    X: pd.DataFrame  # features, same index as te
    rul: np.ndarray  # official final RUL per engine, unit order
    explainer: object


def required_files(subset, root=None):
    """The six files the page needs for one subset (T-03-01: only fixed keys build paths)."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    base = Path(root if root is not None else ROOT)
    return [
        base / "models" / f"prep_{subset}.joblib",
        base / "models" / f"xgb_{subset}.json",
        base / "data_raw" / f"train_{subset}.txt",
        base / "data_raw" / f"test_{subset}.txt",
        base / "data_raw" / f"RUL_{subset}.txt",
        base / BASELINE,
    ]


def missing_files(subset, root=None):
    """Repo-relative POSIX paths of required files that are absent (empty list: all present)."""
    base = Path(root if root is not None else ROOT)
    return [p.relative_to(base).as_posix() for p in required_files(subset, root) if not p.is_file()]


def restore_command(rel_path, subset):
    """A command that restores or regenerates exactly this path (D-03)."""
    rel = str(rel_path).replace("\\", "/")
    if rel.startswith("data_raw/"):
        return "git checkout -- data_raw"
    if rel == BASELINE:
        # Tracked Phase 2 archive: run_experiments.py writes only outputs/results.json.
        return "git checkout -- outputs/results_baseline.json"
    if rel.startswith("models/"):
        return f"python src/run_experiments.py {subset}"
    raise ValueError(f"no restore command for {rel_path!r}")


def load_bundle(subset):
    """Load model, data and features for one subset: the frozen D-00d sequence."""
    import joblib
    import shap
    import xgboost as xgb

    prep = joblib.load(ROOT / "models" / f"prep_{subset}.joblib")
    model = xgb.XGBRegressor()
    model.load_model(str(ROOT / "models" / f"xgb_{subset}.json"))
    _, te, rul = L.load_subset(subset)
    units = sorted(te["unit"].unique())
    if units != list(range(1, len(units) + 1)) or len(rul) != len(units):
        raise ValueError(f"{subset}: test units are not 1..N or the RUL file length differs")
    X = L.make_features(prep["normaliser"].transform(te), prep["sensors"], rolling=True)[
        prep["feature_names"]
    ]
    pred = np.clip(model.predict(X.to_numpy(np.float32)), 0, None)
    return Bundle(prep, model, te.assign(pred=pred), X, rul, shap.TreeExplainer(model))


def engine_frame(bundle, unit):
    """Rows of one engine sorted by cycle (a new frame; the cached te is never mutated)."""
    eng = bundle.te[bundle.te["unit"] == unit]
    return eng.sort_values("cycle", kind="stable")


def true_final_rul(bundle, unit):
    """Official RUL at the last observed cycle of this engine."""
    units = sorted(bundle.te["unit"].unique())
    return float(bundle.rul[units.index(unit)])


def slider_bounds(max_c):
    """(lo, hi, fixed): fixed is True for a one-cycle engine, which gets no slider."""
    if max_c < 1:
        raise ValueError("an engine needs at least one cycle")
    return 1, max_c, max_c == 1


def row_at(eng, cycle):
    """Last row at or before the requested cycle (causal), clamped to the first and last row."""
    cycles = eng["cycle"].to_numpy()
    i = int(np.searchsorted(cycles, cycle, side="right")) - 1
    return eng.iloc[int(np.clip(i, 0, len(eng) - 1))]


def predicted_label(pred):
    """Predictions at or above 120 cycles read as capped, since training capped RUL at 125 (D-09)."""
    return CAPPED_LABEL if pred >= CAP_LABEL_AT else f"{pred:.0f}"
