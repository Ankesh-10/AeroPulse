"""SETUP-04: the smoke wrapper writes to smoke/ only and reproduces the default-XGBoost canary."""
import json
import subprocess
import sys

import pytest
from conftest import ROOT

pytestmark = [pytest.mark.slow, pytest.mark.data]


def _snapshot():
    snap = {}
    for d in ("outputs", "models"):
        for p in sorted((ROOT / d).rglob("*")) if (ROOT / d).exists() else []:
            if p.is_file():
                st = p.stat()
                snap[str(p.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns)
    return snap


def test_smoke_run_isolated_and_reproduces_default_xgb_canary():
    before = _snapshot()
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_run.py"), "FD001"],
                          cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert _snapshot() == before, "smoke run touched outputs/ or models/"
    res = json.loads((ROOT / "smoke" / "outputs" / "results.json").read_text())
    d = res["FD001"]["models"]["XGBoost default params (rolling features)"]
    assert round(d["rmse"]["mean"], 2) == 20.08 and round(d["nasa"]["mean"]) == 786
    assert (ROOT / "smoke" / "models" / "xgb_FD001.json").is_file()
