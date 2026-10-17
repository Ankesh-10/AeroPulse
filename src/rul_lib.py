"""Shared utilities for the C-MAPSS Remaining Useful Life (RUL) project.

Data loading, operating-condition normalisation, causal rolling-window feature
engineering and the evaluation metrics (RMSE, MAE, R2 and the NASA asymmetric score).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw"

RUL_CAP = 125          # piece-wise linear RUL target (common practice for C-MAPSS)
WINDOW = 10            # rolling window length in cycles
SEED = 42

SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + SETTINGS + SENSORS

# sub-dataset -> (operating conditions, fault modes)
SUBSETS = {
    "FD001": (1, 1),
    "FD002": (6, 1),
    "FD003": (1, 2),
    "FD004": (6, 2),
}


def load_subset(name):
    """Return (train_df, test_df, true_rul_for_test_engines)."""
    tr = pd.read_csv(DATA / f"train_{name}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(DATA / f"test_{name}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(DATA / f"RUL_{name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return tr, te, rul


def add_rul(df, cap=RUL_CAP):
    """Add the piece-wise linear RUL label to a run-to-failure frame."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = max_cycle - out["cycle"]
    out["rul"] = out["rul_raw"].clip(upper=cap)
    return out


def informative_sensors(train_df, tol=1e-6):
    """Sensors whose (condition-normalised) spread is non-negligible."""
    std = train_df[SENSORS].std()
    return [s for s in SENSORS if std[s] > tol]


class ConditionNormaliser:
    """Standardise sensors per operating condition (needed for FD002 / FD004).

    Operating conditions are discovered with k-means on the three setting
    columns; for single-condition subsets k=1 so this reduces to z-scoring.
    """

    def __init__(self, n_conditions, sensors):
        self.k = n_conditions
        self.sensors = sensors

    def fit(self, train_df):
        X = train_df[SETTINGS].to_numpy()
        self.km = KMeans(self.k, n_init=10, random_state=SEED).fit(X) if self.k > 1 else None
        c = self._cond(train_df)
        g = train_df[self.sensors].groupby(c)
        self.mu, self.sd = g.mean(), g.std().replace(0, 1.0).fillna(1.0)
        return self

    def _cond(self, df):
        if self.km is None:
            return np.zeros(len(df), dtype=int)
        return self.km.predict(df[SETTINGS].to_numpy())

    def transform(self, df):
        c = self._cond(df)
        out = df.copy()
        mu = self.mu.loc[c].to_numpy()
        sd = self.sd.loc[c].to_numpy()
        out[self.sensors] = (df[self.sensors].to_numpy() - mu) / sd
        return out


def _slope(a):
    n = len(a)
    if n < 2:
        return 0.0
    x = np.arange(n) - (n - 1) / 2.0
    return float((x * (a - a.mean())).sum() / (x * x).sum())


def make_features(df, sensors, window=WINDOW, rolling=True):
    """Causal per-engine features. Only past and current cycles are used.

    raw sensors (+ cycle index); if rolling: rolling mean, std and linear slope.
    """
    base = df[["unit", "cycle"] + sensors].copy()
    feats = {"cycle": base["cycle"].to_numpy()}
    for s in sensors:
        feats[s] = base[s].to_numpy()
    if rolling:
        g = base.groupby("unit")
        for s in sensors:
            r = g[s].rolling(window, min_periods=1)
            feats[f"{s}_mean"] = r.mean().reset_index(level=0, drop=True).sort_index().to_numpy()
            feats[f"{s}_std"] = (
                r.std().reset_index(level=0, drop=True).sort_index().fillna(0.0).to_numpy()
            )
            feats[f"{s}_slope"] = (
                r.apply(_slope, raw=True).reset_index(level=0, drop=True).sort_index().to_numpy()
            )
    return pd.DataFrame(feats, index=base.index)


# ------------------------------------------------------------------ metrics
def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot,
        "nasa": nasa_score(y_true, y_pred),
    }


def last_cycle_rows(df):
    """Index of the final observed cycle of each engine (the test-time query)."""
    return df.groupby("unit")["cycle"].idxmax().to_numpy()
