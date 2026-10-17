"""TEST-04: ConditionNormaliser fits on train only; joblib round-trip."""
import joblib
import numpy as np
import pandas as pd
import pytest

import rul_lib as L

pytestmark = pytest.mark.unit
SENS = ["s1", "s2"]
CENTRES = np.array([[0.0, 0.0, 100.0], [20.0, 0.7, 100.0], [42.0, 0.84, 60.0]])


def _frame(n, k, seed, shift=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    cond = rng.integers(0, k, size=n)
    df = pd.DataFrame(CENTRES[cond] + rng.normal(scale=0.01, size=(n, 3)), columns=L.SETTINGS)
    for j, s in enumerate(SENS):
        df[s] = 10.0 * (cond + 1) * (j + 1) + shift + scale * rng.normal(size=n)
    return df, cond


@pytest.mark.parametrize("k", [1, 3])
def test_fit_uses_train_statistics_only(k):
    tr, _ = _frame(300, k, 0)
    te, _ = _frame(120, k, 1, shift=50.0, scale=3.0)             # very different distribution
    norm = L.ConditionNormaliser(k, SENS).fit(tr)
    mu0, sd0 = norm.mu.copy(), norm.sd.copy()
    centres0 = None if norm.km is None else norm.km.cluster_centers_.copy()
    tr_t, te_t = norm.transform(tr), norm.transform(te)
    pd.testing.assert_frame_equal(norm.mu, mu0)
    pd.testing.assert_frame_equal(norm.sd, sd0)
    if centres0 is not None:
        assert np.array_equal(norm.km.cluster_centers_, centres0)
    c = norm._cond(tr)
    for cl in np.unique(c):
        z = tr_t.loc[c == cl, SENS]
        assert np.allclose(z.mean(), 0.0, atol=1e-9) and np.allclose(z.std(), 1.0)
    assert te_t[SENS].mean().abs().min() > 1.0                    # test is NOT re-centred on itself
    ct = norm._cond(te)
    exp = (te[SENS].to_numpy() - mu0.loc[ct].to_numpy()) / sd0.loc[ct].to_numpy()
    assert np.allclose(te_t[SENS].to_numpy(), exp)


def test_kmeans_path_assigns_conditions_and_leaves_inputs_untouched():
    tr, cond = _frame(300, 3, 0)
    snap = tr.copy()
    norm = L.ConditionNormaliser(3, SENS).fit(tr)
    norm.transform(tr)
    pd.testing.assert_frame_equal(tr, snap)                      # fit/transform do not mutate input
    c = norm._cond(tr)
    assert len(np.unique(c)) == 3
    assert len(set(zip(cond, c))) == 3                            # one-to-one with true conditions
    assert (L.ConditionNormaliser(1, SENS).fit(tr)._cond(tr) == 0).all()


def test_joblib_round_trip_is_bit_identical_and_module_is_rul_lib(tmp_path):
    tr, _ = _frame(300, 3, 0)
    te, _ = _frame(100, 3, 2, shift=5.0)
    norm = L.ConditionNormaliser(3, SENS).fit(tr)
    path = tmp_path / "prep.joblib"
    joblib.dump({"normaliser": norm, "sensors": SENS, "feature_names": ["a", "b"]}, path)
    back = joblib.load(path)["normaliser"]
    pd.testing.assert_frame_equal(back.transform(te), norm.transform(te), check_exact=True)
    assert type(norm).__module__ == "rul_lib"
    blob = path.read_bytes()
    assert b"rul_lib" in blob and b"src.rul_lib" not in blob     # importable as rul_lib, not src.rul_lib
