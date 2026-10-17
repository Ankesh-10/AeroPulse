"""TEST-01: make_features is causal (truncation / perturbation invariance)."""
import numpy as np
import pandas as pd
import pytest
from conftest import SENS, make_runs

import rul_lib as L

pytestmark = pytest.mark.unit


def _future(df, unit, t):
    return (df["unit"] == unit) & (df["cycle"] > t)


def _assert_prefix_invariant(featurize, df, unit, t):
    """Features at cycles <= t must not change when everything after t is removed or scrambled."""
    full = featurize(df)
    fut = _future(df, unit, t)
    kept = df.index[~fut]
    # (a) drop the future rows
    pd.testing.assert_frame_equal(full.loc[kept], featurize(df[~fut]), check_exact=True)
    # (b) keep the rows but scramble the future sensor values
    pert = df.copy()
    rng = np.random.default_rng(1)
    pert.loc[fut, SENS] = rng.normal(1e6, 1e3, size=(int(fut.sum()), len(SENS)))
    pd.testing.assert_frame_equal(full.loc[kept], featurize(pert).loc[kept], check_exact=True)


@pytest.mark.parametrize("rolling", [True, False])
@pytest.mark.parametrize("unit,t", [(1, 1), (1, 5), (2, 9), (2, 14), (3, 19)])
def test_prefix_of_features_equals_features_of_prefix(unit, t, rolling):
    df = make_runs()
    _assert_prefix_invariant(lambda d: L.make_features(d, SENS, rolling=rolling), df, unit, t)


def test_truncation_test_has_power_against_a_leaky_featuriser():
    """The same check must FAIL for a centred rolling window (uses future cycles)."""
    def leaky(d):
        out = pd.DataFrame(index=d.index)
        for s in SENS:
            out[s + "_mean"] = d.groupby("unit")[s].transform(
                lambda x: x.rolling(5, center=True, min_periods=1).mean())
        return out
    with pytest.raises(AssertionError):
        _assert_prefix_invariant(leaky, make_runs(), 2, 7)


def test_engines_do_not_leak_into_each_other():
    df = make_runs()
    base = L.make_features(df, SENS)
    pert = df.copy()
    pert.loc[pert["unit"] == 1, SENS] += 1e3
    other = df["unit"] != 1
    pd.testing.assert_frame_equal(base[other], L.make_features(pert, SENS)[other], check_exact=True)


def test_rolling_values_match_hand_computation():
    df = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 13)})
    df["s1"] = df["cycle"].astype(float)          # linear ramp 1..12
    f = L.make_features(df, ["s1"], window=10)
    assert f["s1_mean"].iloc[0] == 1.0 and f["s1_mean"].iloc[4] == 3.0   # mean(1..5)
    assert f["s1_mean"].iloc[11] == pytest.approx(7.5)                    # mean(3..12)
    assert f["s1_std"].iloc[0] == 0.0                                    # NaN -> 0 at cycle 1
    assert f["s1_std"].iloc[1] == pytest.approx(np.sqrt(0.5))            # std([1,2], ddof=1)
    assert f["s1_slope"].iloc[0] == 0.0                                   # n < 2
    assert np.allclose(f["s1_slope"].iloc[1:], 1.0)                      # ramp slope = 1
    assert (f["cycle"].to_numpy() == df["cycle"].to_numpy()).all()
    assert (f["s1"].to_numpy() == df["s1"].to_numpy()).all()


def test_columns_order_count_and_alignment(runs):
    f = L.make_features(runs, SENS)
    expected = ["cycle"] + SENS + [f"{s}_{k}" for s in SENS for k in ("mean", "std", "slope")]
    assert list(f.columns) == expected and f.shape[1] == 1 + 4 * len(SENS)
    assert f.index.equals(runs.index) and not f.isna().any().any()
    raw = L.make_features(runs, SENS, rolling=False)
    assert list(raw.columns) == ["cycle"] + SENS


def test_features_follow_original_labels_when_index_has_gaps(runs):
    sub = runs.drop(index=[3, 4, 20])                 # increasing index with holes
    f = L.make_features(sub, SENS)
    assert f.index.equals(sub.index)
    assert (f["cycle"].to_numpy() == sub["cycle"].to_numpy()).all()
