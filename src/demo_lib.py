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
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

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

FIG_SIZE = (6.0, 4.2)
# DESIGN.md colour tokens (the only colours used in figures).
COLORS = {
    "canvas": "#FFFFFF",
    "canvas_warm": "#F7F6F3",
    "surface": "#FBFBFA",
    "ink": "#2F3437",
    "ink_strong": "#111111",
    "ink_mute": "#787774",
    "ink_faint": "#B4B4B0",
    "hairline": "#EAEAEA",
    "late": "#9F2F2D",
    "late_bg": "#FDEBEC",
    "early": "#1F6C9F",
    "early_bg": "#E1F3FE",
    "healthy": "#346538",
    "healthy_bg": "#EDF3EC",
    "capped": "#956400",
    "capped_bg": "#FBF3DB",
}
# Lighter tints of the early and late inks for the waterfall's secondary text.
SHAP_TINTS = {"positive": "#8FB6D0", "negative": "#CF9796"}
# shap.plots.waterfall uses pyplot's current figure and a global style shared across sessions.
_SHAP_LOCK = threading.Lock()


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


def plot_style():
    """Shared matplotlib rc dict for every figure (D-19, DESIGN.md Charts)."""
    c = COLORS
    return {
        "figure.facecolor": c["canvas"],
        "axes.facecolor": c["canvas"],
        "axes.edgecolor": c["hairline"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": c["hairline"],
        "grid.linewidth": 0.8,
        "xtick.color": c["ink_mute"],
        "ytick.color": c["ink_mute"],
        "axes.labelcolor": c["ink"],
        "text.color": c["ink"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": c["ink_mute"],
    }


def curve_figure(eng, cycle, true_final, warn_rul):
    """Predicted-vs-true RUL curve. Built on Figure (no pyplot state) so sessions do not share it."""
    import matplotlib

    c = COLORS
    max_c = int(eng["cycle"].max())
    cycles = eng["cycle"].to_numpy()
    seen = eng[eng["cycle"] <= cycle]
    true_curve = np.minimum(true_final + (max_c - cycles), L.RUL_CAP)
    now = row_at(eng, cycle)
    with matplotlib.rc_context(plot_style()):
        fig = Figure(figsize=FIG_SIZE)
        ax = fig.subplots()
        # Warm-up band under the lines (D-10): cycles with a part-filled rolling window.
        ax.axvspan(
            0.5,
            WARMUP_MAX + 0.5,
            color=c["canvas_warm"],
            zorder=0,
            lw=0,
            label=f"Warm-up: fewer than {L.WINDOW} cycles of history",
        )
        ax.plot(seen["cycle"], seen["pred"], color=c["ink"], lw=2, label="Predicted RUL")
        ax.plot(
            cycles,
            true_curve,
            color=c["ink_mute"],
            ls="--",
            lw=1.5,
            label=f"True RUL (reconstructed from the official final RUL, capped at {L.RUL_CAP})",
        )
        ax.axhline(warn_rul, color=c["late"], ls=":", lw=1, label="Alert threshold")
        ax.plot([now["cycle"]], [now["pred"]], "o", color=c["ink_strong"], ms=5, label="_nolegend_")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL (cycles)")
        # Below the axes so the long true-RUL label never covers the curves.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig


def waterfall_figure(bundle, row_label):
    """SHAP waterfall for one row: raises-RUL contributions blue, lowers-RUL red (D-05, D-19)."""
    import matplotlib.pyplot as plt
    import shap
    from shap.plots import _style  # private but present in the pinned shap 0.53.0

    x = bundle.X.loc[[row_label]].to_numpy(np.float32)
    vals = bundle.explainer.shap_values(x)[0]
    base = float(np.ravel(bundle.explainer.expected_value)[0])
    expl = shap.Explanation(
        values=vals, base_values=base, data=x[0], feature_names=list(bundle.X.columns)
    )
    with _SHAP_LOCK:
        old = _style.get_style()
        try:
            _style.set_style(
                primary_color_positive=COLORS["early"],
                primary_color_negative=COLORS["late"],
                secondary_color_positive=SHAP_TINTS["positive"],
                secondary_color_negative=SHAP_TINTS["negative"],
                hlines_color=COLORS["hairline"],
                vlines_color=COLORS["ink_faint"],
                tick_labels_color=COLORS["ink_mute"],
            )
            with plt.rc_context(plot_style()):
                fig = plt.figure(figsize=FIG_SIZE)
                shap.plots.waterfall(expl, max_display=11, show=False)
                fig.set_size_inches(*FIG_SIZE)  # waterfall resizes the figure; restore
        finally:
            _style.set_style(old)
    return fig


# ---------------------------------------------------------------- warm-up and truth labels
WARMUP_MAX = L.WINDOW - 1  # cycles 1..WINDOW-1 have fewer than WINDOW rows in the rolling window
WARMUP_NOTE = (
    f"The rolling features at this cycle are built from fewer than {L.WINDOW} cycles, "
    "so the prediction is less stable."
)


def is_warmup(cycle):
    """True while the rolling window is not yet full (D-10)."""
    return cycle <= WARMUP_MAX


def true_rul_at(true_final, max_c, cycle):
    """Uncapped RUL at an earlier cycle, reconstructed from the official final RUL."""
    return true_final + (max_c - cycle)


def above_cap(true_rul):
    """True when the real RUL exceeds the training cap (D-09)."""
    return true_rul > L.RUL_CAP


def true_rul_note(true_rul, at_last):
    """Short caption parts for the True RUL tile (D-09)."""
    notes = ["official" if at_last else "reconstructed from the official final RUL"]
    if above_cap(true_rul):
        notes.append(f"above the {L.RUL_CAP} training cap")
    return notes


# ---------------------------------------------------------------- worst-case engines (D-11)
WORST_N = 10
CAP_TAG = f" (true RUL above the {L.RUL_CAP} cap)"


def error_word(err):
    """'late' when the model predicts more RUL than the engine has, 'early' for less, else 'exact'."""
    if err > 0:
        return "late"
    if err < 0:
        return "early"
    return "exact"


@dataclasses.dataclass(frozen=True)
class WorstCase:
    unit: int
    error: float  # predicted minus official RUL at the last observed cycle
    pred: float
    true_rul: float
    above_cap: bool
    label: str


def worst_case_label(unit, err, true_rul):
    """Picker label, for example 'Engine 34: +41 late'; tagged when the label cap drives the error."""
    tag = CAP_TAG if above_cap(true_rul) else ""
    return f"Engine {unit}: {err:+.0f} {error_word(err)}{tag}"


def worst_cases(bundle, n=WORST_N):
    """The n engines with the largest absolute last-cycle error, ties broken by unit (D-11).

    Ranked against the official uncapped RUL, the project's evaluation rule; the cap tag
    keeps the capped subsets honest without re-ranking or filtering.
    """
    last = L.last_cycle_rows(bundle.te)
    rows = bundle.te.loc[last]
    units = sorted(int(u) for u in bundle.te["unit"].unique())
    truth = {u: float(t) for u, t in zip(units, np.asarray(bundle.rul, dtype=float))}
    cases = []
    for u, p in zip(rows["unit"].to_numpy(), rows["pred"].to_numpy()):
        u, p = int(u), float(p)
        err = p - truth[u]
        cases.append(
            WorstCase(u, err, p, truth[u], above_cap(truth[u]), worst_case_label(u, err, truth[u]))
        )
    cases.sort(key=lambda c: (-abs(c.error), c.unit))
    return cases[:n]
