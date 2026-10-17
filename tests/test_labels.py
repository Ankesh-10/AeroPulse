"""Training target: piece-wise linear RUL = min(T - t, 125); the raw RUL is kept."""
import numpy as np
import pytest
from conftest import make_runs

import rul_lib as L

pytestmark = pytest.mark.unit


def test_add_rul_caps_at_125_and_keeps_raw_without_mutating_input():
    df = make_runs(lengths=(200, 30))
    snap = df.copy()
    out = L.add_rul(df)
    assert df.equals(snap) and "rul" not in df.columns
    e1 = out[out["unit"] == 1]
    assert e1["rul_raw"].iloc[0] == 199 and e1["rul_raw"].iloc[-1] == 0
    assert e1["rul"].max() == L.RUL_CAP == 125
    assert np.array_equal(e1["rul"].to_numpy(), np.minimum(e1["rul_raw"].to_numpy(), 125))
    e2 = out[out["unit"] == 2]
    assert np.array_equal(e2["rul"].to_numpy(), e2["rul_raw"].to_numpy()) and e2["rul"].max() == 29
