"""TEST-02: whole-engine folds, one last-cycle row per engine, evaluate() protocol."""
import numpy as np
import pandas as pd
import pytest
from conftest import SENS, make_runs

import rul_lib as L
import run_experiments as R   # imports optuna/xgboost; creates empty outputs/ and models/ (harmless)

pytestmark = pytest.mark.unit


def _assert_whole_engines(units, folds, k):
    assert len(folds) == len(units) and set(np.unique(folds)) <= set(range(k))
    per_engine = pd.DataFrame({"u": units, "f": folds}).groupby("u")["f"].nunique()
    assert (per_engine == 1).all(), "an engine is split across folds"


@pytest.mark.parametrize("k", [2, 3, 5])
def test_engine_folds_disjoint_cover_and_balanced(k):
    df = make_runs(lengths=tuple(range(8, 8 + 10)))          # 10 engines, uneven lengths
    units = df["unit"].to_numpy()
    folds = R.engine_folds(units, k)
    _assert_whole_engines(units, folds, k)
    sets = [set(np.unique(units[folds == f])) for f in range(k)]
    assert sum(len(s) for s in sets) == 10 and set().union(*sets) == set(range(1, 11))
    assert all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1:])
    sizes = [len(s) for s in sets]
    assert max(sizes) - min(sizes) <= 1 and min(sizes) >= 1


def test_engine_folds_deterministic_and_row_order_independent():
    df = make_runs()
    a = R.engine_folds(df["unit"].to_numpy(), 3)
    assert (a == R.engine_folds(df["unit"].to_numpy(), 3)).all()
    shuf = df.sample(frac=1.0, random_state=0)
    b = R.engine_folds(shuf["unit"].to_numpy(), 3)
    assert dict(zip(shuf["unit"], b)) == dict(zip(df["unit"], a))   # same engine -> same fold


def test_whole_engine_check_rejects_a_row_level_split():
    df = make_runs()
    row_split = np.arange(len(df)) % 3                       # what KFold-on-rows would do
    with pytest.raises(AssertionError):
        _assert_whole_engines(df["unit"].to_numpy(), row_split, 3)


def test_last_cycle_rows_one_max_row_per_engine():
    df = make_runs(lengths=(12, 15, 20))
    last = L.last_cycle_rows(df)
    assert len(last) == df["unit"].nunique() == 3
    rows = df.iloc[last]
    assert list(rows["unit"]) == [1, 2, 3] and list(rows["cycle"]) == [12, 15, 20]
    assert (df.loc[last, "cycle"].to_numpy() == df.groupby("unit")["cycle"].max().to_numpy()).all()


def test_last_cycle_rows_survives_row_shuffle_with_fresh_rangeindex():
    df = make_runs().sample(frac=1.0, random_state=3).reset_index(drop=True)
    last = L.last_cycle_rows(df)
    assert len(last) == 3 and sorted(df.iloc[last]["cycle"]) == [12, 15, 20]


def test_last_cycle_rows_returns_labels_so_loc_is_always_right():
    df = make_runs()
    df.index = df.index + 1000                               # labels != positions
    last = L.last_cycle_rows(df)
    assert list(df.loc[last, "cycle"]) == [12, 15, 20]       # .loc fine; .iloc[last] would be wrong


def test_features_at_last_rows_line_up_with_labels():
    df = make_runs()
    f = L.make_features(df, SENS)
    last = L.last_cycle_rows(df)
    assert (f.iloc[last]["cycle"].to_numpy() == df.loc[last, "cycle"].to_numpy()).all()


class _Probe:
    """Predicts its first input feature; remembers the seed it was built with."""
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


def test_evaluate_scores_last_cycle_against_uncapped_and_capped_truth():
    Xte = np.array([[150.0], [90.0], [-5.0]])                # clipped to 150, 90, 0
    true = np.array([200.0, 100.0, 10.0])
    agg, first, pmean, p0 = R.evaluate(_Probe, np.zeros((4, 1)), np.zeros(4), Xte, true, seeds=[0, 1, 2])
    assert agg["rmse"]["mean"] == pytest.approx(30.0)                    # sqrt((50^2+10^2+10^2)/3)
    assert agg["rmse_capped_truth"]["mean"] == pytest.approx(np.sqrt(275.0))  # truth capped at 125
    assert agg["rmse"]["std"] == 0.0 and first.seed == 0
    assert np.allclose(p0, [150, 90, 0]) and np.allclose(pmean, [150, 90, 0])
    assert {"fit_seconds", "predict_seconds_100_engines"} <= set(agg)
