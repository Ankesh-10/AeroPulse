"""Shared fixtures and tier-skip logic (unit / data / artifacts / slow)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_FILES = [f"{k}_{s}.txt" for s in SUBSETS for k in ("train", "test", "RUL")]
ARTIFACT_FILES = (
    [f"models/xgb_{s}.json" for s in SUBSETS]
    + [f"models/prep_{s}.joblib" for s in SUBSETS]
    + [f"outputs/pred_{s}.csv" for s in SUBSETS]
    + ["outputs/results.json"]
)


def _missing(rel_paths):
    # Check FILES, not directories: importing run_experiments creates empty outputs/ and models/.
    return [p for p in rel_paths if not (ROOT / p).is_file()]


def pytest_collection_modifyitems(config, items):
    miss_data = _missing([f"data_raw/{f}" for f in DATA_FILES])
    miss_art = _missing(ARTIFACT_FILES)
    for item in items:
        if miss_data and item.get_closest_marker("data"):
            item.add_marker(pytest.mark.skip(reason=f"data_raw/ incomplete ({len(miss_data)} files missing)"))
        if miss_art and item.get_closest_marker("artifacts"):
            item.add_marker(pytest.mark.skip(reason=f"models/ or outputs/ artefacts missing ({len(miss_art)} files)"))


SENS = ["s1", "s2", "s3"]


def make_runs(lengths=(12, 15, 20), sensors=SENS, seed=0):
    """Tiny synthetic run-to-failure frame: default RangeIndex, sorted by (unit, cycle)."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    cycle = np.concatenate([np.arange(1, n + 1) for n in lengths])
    n = len(unit)
    df = pd.DataFrame({"unit": unit, "cycle": cycle})
    for i in range(3):
        df[f"op{i + 1}"] = rng.normal(size=n)
    for s in sensors:
        df[s] = rng.normal(size=n)
    return df


@pytest.fixture
def runs():
    return make_runs()
