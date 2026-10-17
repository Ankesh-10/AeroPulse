"""Smoke run: reduced tuning/seeds, written to smoke/ only (never outputs/ or models/).

Usage (from the repo root, inside the project venv):
    python scripts/smoke_run.py            # FD001
    python scripts/smoke_run.py FD001 FD003
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments as R  # noqa: E402  (import creates empty outputs/ and models/ -- see D-02)

SMOKE = ROOT / "smoke"
SMOKE_OUT, SMOKE_MODELS = SMOKE / "outputs", SMOKE / "models"
SEEDS = [0]

SMOKE_OUT.mkdir(parents=True, exist_ok=True)
SMOKE_MODELS.mkdir(parents=True, exist_ok=True)

# rebind module globals that run_subset / tune_* read at call time
R.OUT, R.MODELS = SMOKE_OUT, SMOKE_MODELS
R.XGB_TRIALS, R.RF_TRIALS, R.FINAL_SEEDS = 2, 1, list(SEEDS)
# evaluate(..., seeds=FINAL_SEEDS) captured the list at definition time
R.evaluate.__defaults__ = (list(SEEDS),)

assert R.OUT.resolve() != (ROOT / "outputs").resolve()
assert R.MODELS.resolve() != (ROOT / "models").resolve()
assert R.evaluate.__defaults__ == ([0],)

if __name__ == "__main__":
    names = sys.argv[1:] or ["FD001"]
    results = {}
    for n in names:
        R.run_subset(n, results)
    (SMOKE_OUT / "results.json").write_text(json.dumps(results, indent=2))
    d = results[names[0]]["models"].get("XGBoost default params (rolling features)")
    if d:
        print("CANARY default XGB", round(d["rmse"]["mean"], 2), round(d["nasa"]["mean"]), "(expect 20.08 / 786)")
