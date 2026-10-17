# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Remaining Useful Life (RUL) prediction for turbofan engines on NASA C-MAPSS (FD001–FD004): tuned XGBoost vs a tuned Random Forest baseline, with FD001 studied in depth. College mini project (Advanced AI, CSE 3139, MIT Bengaluru). Authors: Ankesh Jaiswal (245890002), Bhavy Choudhary (245890192).

`Claude_Code_Handoff.pdf` is the source of truth for status, design decisions, expected results and caveats. If `src/rul_lib.py`, `src/run_experiments.py` or `app.py` are missing, rebuild them verbatim from its Appendices A, B and C. The written reports (.docx and IEEE LaTeX) live outside this repo.

## Layout

- `src/rul_lib.py` — data loading, per-condition normalisation, causal rolling features, metrics
- `src/run_experiments.py` — Optuna tuning, multi-seed evaluation, FD001 ablations; writes `outputs/` and `models/`
- `app.py` — Streamlit demo (needs `models/` from a run); belongs at the project root
- `data_raw/` — `train_FD00N.txt`, `test_FD00N.txt`, `RUL_FD00N.txt` for N = 1..4 (never edit)
- `outputs/results.json`, `outputs/pred_FD00N.csv`, `models/{xgb_FD00N.json, prep_FD00N.joblib, rf_FD00N.joblib}` — generated

## Commands

Setup (Windows; on POSIX use `python3` and `source .venv/bin/activate`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install xgboost scikit-learn pandas numpy optuna shap streamlit matplotlib joblib
```

Original versions: Python 3.13, xgboost 3.4.1, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, optuna 5.0.0, shap 0.53.0, matplotlib 3.11.2, streamlit 1.65.0.

Data (prefer NASA PCoE `CMAPSSData.zip`; this GitHub mirror also works):

```bash
mkdir -p data_raw && cd data_raw && for n in 1 2 3 4; do for k in train test RUL; do curl -sSL -O "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/${k}_FD00${n}.txt"; done; done
```

Sanity check — train engines/rows: FD001 100/20,631, FD002 260/53,759, FD003 100/24,720, FD004 249/61,249; test engines 100/259/100/248.

Run (from the project root; `run_experiments.py` imports `rul_lib` from its own directory):

- `python src/run_experiments.py FD001` — one subset, ~4 min on 2 cores (the quickest end-to-end check)
- `python src/run_experiments.py FD001 FD002 FD003 FD004` — all subsets, ~36 min; no args also means all four
- `streamlit run app.py`

There is no test suite or linter configured yet. For a fast smoke run, temporarily lower `XGB_TRIALS`, `RF_TRIALS` and `FINAL_SEEDS` in `run_experiments.py`; only the default-parameter XGBoost result is then comparable (FD001: RMSE 20.08, NASA 786 — it reproduces exactly).

## Architecture

Pipeline per subset in `run_subset()`:

1. Load, then label with piece-wise linear RUL `min(T - t, 125)` (`add_rul`; keeps `rul_raw` too).
2. Drop sensors that are constant within a condition, or (single-condition subsets only) take ≤ 2 values → 14/17/16/17 kept.
3. `ConditionNormaliser`: for FD002/FD004 k-means (k = 6) on `op1..op3`, then z-score each sensor within its cluster; for FD001/FD003 k = 1, i.e. plain z-scoring. Fit on train only.
4. `make_features`: per engine, raw sensors + `cycle` + causal 10-cycle rolling mean/std/least-squares slope (`min_periods=1`) → 57/69/65/69 features. `rolling=False` gives the raw-sensor ablation set.
5. `engine_folds`: 3 folds of whole engines. Optuna TPE: 25 XGBoost trials with early stopping 50 (final `n_estimators = max(50, 1.1 × mean best iteration)`), 6 RF trials at 80 trees (final RF uses 300).
6. `evaluate`: seeds 0/1/2, predict only at the last observed cycle of each test engine (`last_cycle_rows`), clip at 0, score against the official **uncapped** RUL plus `rmse_capped_truth`. Metrics: RMSE, MAE, R², asymmetric NASA score (late predictions penalised more).
7. FD001 only: ablations (default-param XGBoost, raw-sensor XGBoost, raw-sensor RF).
8. Persist seed-0 models, `prep_*.joblib` (`normaliser`, `sensors`, `feature_names`), predictions CSV, top-15 gain importance. `results.json` is read, updated per subset key, and rewritten — rerunning one subset overwrites only that subset.

`app.py` reloads `prep_*.joblib` + `xgb_*.json`, rebuilds features for the whole test set, reindexes by `feature_names`, and explains with SHAP `TreeExplainer`. The joblib file pickles a `rul_lib.ConditionNormaliser`, so renaming or moving `rul_lib` (or changing its class) breaks loading of existing models — retrain after such changes. Any change to feature construction must stay identical between training and the app.

## Rules that must not be broken

- Never split rows randomly. Folds are whole engines (consecutive cycles leak).
- Features are causal: past and current cycles only.
- Evaluate at the last cycle of each test engine against the official UNCAPPED RUL; also report RMSE against capped truth. Training target is `min(T - t, 125)`.
- Do not change the evaluation protocol or the contents of `data_raw/`.
- Report numbers must come from `outputs/results.json`, never typed by hand.
- Report claims must match the data: say when a difference is within noise (test sets are 100–259 engines; differences under ~1 cycle are noise).

## Reference results (original run, test set, mean ± SD over 3 seeds)

| Subset | XGBoost RMSE | RF RMSE | XGBoost NASA | RF NASA |
|---|---|---|---|---|
| FD001 | 17.77 ± 0.05 | 18.18 ± 0.08 | 582 ± 19 | 590 ± 3 |
| FD002 | 26.85 ± 0.04 | 27.19 ± 0.03 | 8618 ± 286 | 8996 ± 150 |
| FD003 | 17.36 ± 0.10 | 18.85 ± 0.13 | 617 ± 12 | 778 ± 12 |
| FD004 | 28.36 ± 0.06 | 28.49 ± 0.04 | 5667 ± 56 | 5574 ± 31 |

FD001 ablations: default-param XGBoost 20.08 / NASA 786; raw-sensor XGBoost 17.78; raw-sensor RF 17.88. A re-run should land close; if drift is large, stop and investigate before building on it. If numbers change, the written reports no longer match — regenerate their tables or keep the original `results.json` as the source of truth.

## Open work (priority order, from the handoff)

1. Re-run all four subsets and compare against the table above.
2. Harden the demo: test `app.py` on FD002–FD004 (only FD001 was tested), guard engines with very few cycles, add a README.
3. Rolling-feature ablation on FD002–FD004 (never run).
4. Give the RF the same 25-trial tuning budget.
5. Reduce late bias (quantile or NASA-cost objective) under the same protocol.
6. Replace or drop the `cycle` feature (top SHAP driver, so part of the skill is age-based) and measure the loss.

## Caveats

- Published deep-learning comparison numbers in the reports (Babu 2016, Zheng 2017, Li 2018) were written from memory and are unverified — check against the papers before submission.
- Data came from a GitHub mirror, verified by counts only, not checksum.
- CV RMSE is slightly optimistic (early stopping uses the validation fold); seeds vary the model, not the split.
- `within_sd` in `run_experiments.py` is unused and harmless. `n_jobs=2` is hard-coded, which the runtime estimates assume.
