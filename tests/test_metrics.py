"""TEST-03: metrics against hand-computed values (d = pred - true)."""
import math

import pandas as pd
import pytest

import rul_lib as L

pytestmark = pytest.mark.unit


def test_nasa_score_boundary_cases():
    assert L.nasa_score([100], [87]) == pytest.approx(math.e - 1)    # d = -13 (early)
    assert L.nasa_score([100], [110]) == pytest.approx(math.e - 1)   # d = +10 (late)
    assert L.nasa_score([100], [100]) == 0.0
    assert L.nasa_score([100], [110]) > L.nasa_score([100], [90])    # late costs more at |d| = 10
    assert L.nasa_score([100], [90]) == pytest.approx(math.exp(10 / 13) - 1)


def test_metrics_hand_computed():
    y, p = [10, 20, 30, 40], [12, 18, 33, 36]                        # d = +2, -2, +3, -4
    m = L.metrics(y, p)
    assert m["rmse"] == pytest.approx(math.sqrt(33 / 4))             # (4+4+9+16)/4 = 8.25
    assert m["mae"] == pytest.approx(2.75)
    assert m["r2"] == pytest.approx(1 - 33 / 500)                    # ss_tot = 225+25+25+225
    assert m["nasa"] == pytest.approx(
        (math.exp(2 / 10) - 1) + (math.exp(2 / 13) - 1) + (math.exp(3 / 10) - 1) + (math.exp(4 / 13) - 1))
    assert set(m) == {"rmse", "mae", "r2", "nasa"}


def test_metrics_perfect_and_accepts_lists_and_series():
    m = L.metrics(pd.Series([5.0, 9.0, 2.0]), [5, 9, 2])
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["r2"] == 1.0 and m["nasa"] == 0.0
