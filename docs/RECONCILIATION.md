# Reconciliation: baseline re-run vs handoff reference

The four-subset baseline re-run is judged by `scripts/check_reference.py` against the reference table in section 4 of `Claude_Code_Handoff.pdf`. The bands live as data in that script and were committed to git before the full run started; the commit order and `outputs/run_manifest.json` are the proof (D-05). Every number in the generated block below comes from the script and none is typed by hand. After a passing run, `outputs/results_baseline.json` is the source of truth for every later phase, and the handoff reference survives only as the comparison column (D-09).

## Drift procedure (D-08)

1. Any FAIL halts the phase: no archive and no freeze.
2. Investigate in this order: transcription of the kernel against Appendix B, data checksums, k-means cluster sizes on FD002 and FD004 (six roughly balanced clusters), library pins in the venv, tuning RNG.
3. Write the findings into the Findings section below.
4. If no bug is found, the user decides between accepting the result with a documented explanation and digging further.
5. A band is never widened after results are seen without saying so in this file.

<!-- BEGIN GENERATED: check_reference.py -->

### Bands (declared before the full run)

| Tier | Checks | Band | Effect on exit code |
|---|---|---|---|
| FAIL | Counts per subset: train_engines, train_rows, test_engines, len(kept_sensors), n_features | exact | exit 1 |
| FAIL | FD001 default-parameter XGBoost mean RMSE (golden canary) | exact (2 dp) | exit 1 |
| FAIL | FD001 default-parameter XGBoost mean NASA (golden canary) | exact (0 dp) | exit 1 |
| FAIL | Tuned XGBoost and RF mean RMSE on every subset; FD001 raw-sensor XGBoost and RF mean RMSE | +-1.0 cycle | exit 1 |
| FAIL | Tuned XGBoost and RF mean NASA on every subset | +-25% relative | exit 1 |
| FAIL | A missing subset, model or metric key (never a silent skip) | present | exit 1 |
| WARN | XGB vs RF RMSE ordering flips on a subset | XGB < RF expected; gaps under ~1 cycle are noise | none (exit 0) |
| WARN | RMSE seed SD of any model | SD <= 0.3 | none (exit 0) |
| WARN | rmse_capped_truth of a tuned model | +-1.0 cycle of the handoff value | none (exit 0) |
| WARN | None of the top-3 gain features is a SHAP leader | one of cycle, s15_mean, s4_mean, s2_mean, s3_mean, s11_mean | none (exit 0) |
| WARN | FD001 tuned XGBoost RMSE above raw-sensor XGBoost RMSE | gap <= 1.0 cycle | none (exit 0) |
| INFO | rmse_capped_truth inside its band, top-5 gain features, FD001 ablation NASA, FD001 xgb_n_estimators, CV RMSE | informational | none (exit 0) |

### Reference values (Claude_Code_Handoff.pdf section 4)

Comparison targets, not reported results.

Counts per subset (sections 2 and 5.2):

| Subset | Train engines | Train rows | Test engines | Kept sensors | n_features |
|---|---|---|---|---|---|
| FD001 | 100 | 20631 | 100 | 14 | 57 |
| FD002 | 260 | 53759 | 259 | 17 | 69 |
| FD003 | 100 | 24720 | 100 | 16 | 65 |
| FD004 | 249 | 61249 | 248 | 17 | 69 |

Tuned models, mean over three seeds (section 4.2):

| Subset | Model | RMSE | RMSE band | NASA | NASA band | RMSE vs capped truth |
|---|---|---|---|---|---|---|
| FD001 | XGBoost (rolling features) | 17.77 | [16.77, 18.77] | 582 | [436.5, 727.5] | 16.60 |
| FD001 | Random Forest (rolling features) | 18.18 | [17.18, 19.18] | 590 | [442.5, 737.5] | 17.01 |
| FD002 | XGBoost (rolling features) | 26.85 | [25.85, 27.85] | 8618 | [6463.5, 10772.5] | 14.82 |
| FD002 | Random Forest (rolling features) | 27.19 | [26.19, 28.19] | 8996 | [6747.0, 11245.0] | 15.20 |
| FD003 | XGBoost (rolling features) | 17.36 | [16.36, 18.36] | 617 | [462.8, 771.2] | 15.96 |
| FD003 | Random Forest (rolling features) | 18.85 | [17.85, 19.85] | 778 | [583.5, 972.5] | 17.39 |
| FD004 | XGBoost (rolling features) | 28.36 | [27.36, 29.36] | 5667 | [4250.2, 7083.8] | 17.18 |
| FD004 | Random Forest (rolling features) | 28.49 | [27.49, 29.49] | 5574 | [4180.5, 6967.5] | 17.26 |

FD001 default-parameter XGBoost (canary) and raw-sensor ablations (section 4.1):

| Model | RMSE | RMSE band | NASA | NASA band |
|---|---|---|---|---|
| XGBoost default params (rolling features) | 20.08 | exact (2 dp) | 786 | exact (0 dp) |
| XGBoost (raw sensors only) | 17.78 | +-1.0 cycle | 566 | informational |
| Random Forest (raw sensors only) | 17.88 | +-1.0 cycle | 526 | informational |

Capped-target CV RMSE (section 4.4, informational):

| Subset | XGBoost | Random Forest |
|---|---|---|
| FD001 | 16.25 | 16.45 |
| FD002 | 16.17 | 16.45 |
| FD003 | 14.69 | 15.06 |
| FD004 | 16.11 | 16.30 |

FD001 tuned XGBoost trees (section 4.1, informational): 255

SHAP leaders on FD001 (section 4.4): cycle, s15_mean, s4_mean, s2_mean, s3_mean, s11_mean

### Comparison

Source: `outputs/results_baseline.json`

SHA256 of the source with CR bytes removed: `14df54aa91ce34e89548cd0511063fefc77e8dc2467fcea9dc828553aa28d70d`

| Level | Subset | Check | Value | Reference | Band |
|---|---|---|---|---|---|
| PASS | FD001 | train_engines | 100 | 100 | exact |
| PASS | FD001 | train_rows | 20631 | 20631 | exact |
| PASS | FD001 | test_engines | 100 | 100 | exact |
| PASS | FD001 | kept_sensors | 14 | 14 | exact |
| PASS | FD001 | n_features | 57 | 57 | exact |
| PASS | FD001 | XGBoost (rolling features) rmse | 17.7704 | 17.77 | +-1.0 cycle |
| PASS | FD001 | XGBoost (rolling features) nasa | 593.8493 | 582 | +-25% |
| PASS | FD001 | Random Forest (rolling features) rmse | 18.1812 | 18.18 | +-1.0 cycle |
| PASS | FD001 | Random Forest (rolling features) nasa | 589.5370 | 590 | +-25% |
| PASS | FD001 | XGBoost default params (rolling features) rmse | 20.0782 | 20.08 | exact (2 dp) |
| PASS | FD001 | XGBoost default params (rolling features) nasa | 785.9896 | 786 | exact (0 dp) |
| PASS | FD001 | XGBoost (raw sensors only) rmse | 17.6645 | 17.78 | +-1.0 cycle |
| PASS | FD001 | Random Forest (raw sensors only) rmse | 17.8808 | 17.88 | +-1.0 cycle |
| INFO | FD001 | XGBoost (rolling features) rmse_capped_truth | 16.6132 | 16.60 | +-1.0 cycle |
| INFO | FD001 | Random Forest (rolling features) rmse_capped_truth | 17.0108 | 17.01 | +-1.0 cycle |
| INFO | FD001 | ordering | XGB 17.7704 RF 18.1812 | XGB < RF | XGB < RF expected; gaps under ~1 cycle are noise |
| INFO | FD001 | top-5 gain | s4_mean, s15_mean, s2_mean, s11_mean, cycle | cycle, s15_mean, s4_mean, s2_mean, s3_mean, s11_mean | informational |
| INFO | FD001 | xgb_cv_rmse | 16.2460 | 16.25 | informational |
| INFO | FD001 | rf_cv_rmse | 16.4473 | 16.45 | informational |
| INFO | FD001 | XGBoost (raw sensors only) nasa | 570.3642 | 566 | informational |
| INFO | FD001 | Random Forest (raw sensors only) nasa | 525.7689 | 526 | informational |
| INFO | FD001 | raw-sensor gap | 0.1059 | 0.00 | tuned XGB above raw-sensor XGB by <= 1.0 expected |
| INFO | FD001 | xgb_n_estimators | 183 | 255 | informational |
| PASS | FD002 | train_engines | 260 | 260 | exact |
| PASS | FD002 | train_rows | 53759 | 53759 | exact |
| PASS | FD002 | test_engines | 259 | 259 | exact |
| PASS | FD002 | kept_sensors | 17 | 17 | exact |
| PASS | FD002 | n_features | 69 | 69 | exact |
| PASS | FD002 | XGBoost (rolling features) rmse | 26.8042 | 26.85 | +-1.0 cycle |
| PASS | FD002 | XGBoost (rolling features) nasa | 8437.3390 | 8618 | +-25% |
| PASS | FD002 | Random Forest (rolling features) rmse | 27.1872 | 27.19 | +-1.0 cycle |
| PASS | FD002 | Random Forest (rolling features) nasa | 8995.9314 | 8996 | +-25% |
| INFO | FD002 | XGBoost (rolling features) rmse_capped_truth | 14.7849 | 14.82 | +-1.0 cycle |
| INFO | FD002 | Random Forest (rolling features) rmse_capped_truth | 15.2042 | 15.20 | +-1.0 cycle |
| INFO | FD002 | ordering | XGB 26.8042 RF 27.1872 | XGB < RF | XGB < RF expected; gaps under ~1 cycle are noise |
| INFO | FD002 | top-5 gain | s11_mean, s4_mean, s15_mean, s2_mean, s17_mean | cycle, s15_mean, s4_mean, s2_mean, s3_mean, s11_mean | informational |
| INFO | FD002 | xgb_cv_rmse | 16.1660 | 16.17 | informational |
| INFO | FD002 | rf_cv_rmse | 16.4519 | 16.45 | informational |
| PASS | FD003 | train_engines | 100 | 100 | exact |
| PASS | FD003 | train_rows | 24720 | 24720 | exact |
| PASS | FD003 | test_engines | 100 | 100 | exact |
| PASS | FD003 | kept_sensors | 16 | 16 | exact |
| PASS | FD003 | n_features | 65 | 65 | exact |
| PASS | FD003 | XGBoost (rolling features) rmse | 17.1654 | 17.36 | +-1.0 cycle |
| PASS | FD003 | XGBoost (rolling features) nasa | 581.9053 | 617 | +-25% |
| PASS | FD003 | Random Forest (rolling features) rmse | 18.8496 | 18.85 | +-1.0 cycle |
| PASS | FD003 | Random Forest (rolling features) nasa | 777.9373 | 778 | +-25% |
| INFO | FD003 | XGBoost (rolling features) rmse_capped_truth | 15.7432 | 15.96 | +-1.0 cycle |
| INFO | FD003 | Random Forest (rolling features) rmse_capped_truth | 17.3878 | 17.39 | +-1.0 cycle |
| INFO | FD003 | ordering | XGB 17.1654 RF 18.8496 | XGB < RF | XGB < RF expected; gaps under ~1 cycle are noise |
| INFO | FD003 | top-5 gain | s3_mean, s4, s11_mean, s11, s17_mean | cycle, s15_mean, s4_mean, s2_mean, s3_mean, s11_mean | informational |
| INFO | FD003 | xgb_cv_rmse | 14.6855 | 14.69 | informational |
| INFO | FD003 | rf_cv_rmse | 15.0581 | 15.06 | informational |
| PASS | FD004 | train_engines | 249 | 249 | exact |
| PASS | FD004 | train_rows | 61249 | 61249 | exact |
| PASS | FD004 | test_engines | 248 | 248 | exact |
| PASS | FD004 | kept_sensors | 17 | 17 | exact |
| PASS | FD004 | n_features | 69 | 69 | exact |
| PASS | FD004 | XGBoost (rolling features) rmse | 28.2032 | 28.36 | +-1.0 cycle |
| PASS | FD004 | XGBoost (rolling features) nasa | 5621.9775 | 5667 | +-25% |
| PASS | FD004 | Random Forest (rolling features) rmse | 28.4916 | 28.49 | +-1.0 cycle |
| PASS | FD004 | Random Forest (rolling features) nasa | 5573.9734 | 5574 | +-25% |
| INFO | FD004 | XGBoost (rolling features) rmse_capped_truth | 16.9481 | 17.18 | +-1.0 cycle |
| INFO | FD004 | Random Forest (rolling features) rmse_capped_truth | 17.2600 | 17.26 | +-1.0 cycle |
| INFO | FD004 | ordering | XGB 28.2032 RF 28.4916 | XGB < RF | XGB < RF expected; gaps under ~1 cycle are noise |
| INFO | FD004 | top-5 gain | s3_mean, s11_mean, s17_mean, s8_mean, s4_mean | cycle, s15_mean, s4_mean, s2_mean, s3_mean, s11_mean | informational |
| INFO | FD004 | xgb_cv_rmse | 16.1052 | 16.11 | informational |
| INFO | FD004 | rf_cv_rmse | 16.3004 | 16.30 | informational |

RESULT: PASS (40 checks, 0 warnings)

<!-- END GENERATED: check_reference.py -->

## Findings

The four-subset run was made from commit `f83a7be` with the bands declared in commit `b7c78b8`, which is an ancestor of the run commit and was committed before the run started. Timings, interpreter, platform and package versions are in `outputs/run_manifest.json`; the exact environment is in `outputs/pip_freeze.txt`.

Every FAIL-band check passes (see the RESULT line in the generated block), including the exact counts per subset and the FD001 default-parameter XGBoost canary. The generated block carries no WARN row, so no check needs a named exception here.

Reading the comparison, with no number typed by hand:

- The tuned results sit close to the handoff values on every subset. Remaining differences are small next to the 100 to 259 engines in the test sets; differences under about one cycle are noise, and the NASA score moves more than RMSE because late predictions are penalised heavily.
- The XGBoost-below-RF RMSE ordering holds on all four subsets. On FD001, FD002 and FD004 the gap is small, so the ordering there should not be read as a firm win; the FD003 gap is the widest of the four.
- Two INFO rows differ from the handoff and are worth knowing about. The FD001 tuned tree count (`xgb_n_estimators`) is not the handoff's, so the Optuna search landed on a different configuration even though the test RMSE matches; tuning has several near-equivalent optima. The top gain-importance features differ from the handoff's SHAP leaders, as expected since gain and SHAP rank features differently, and the handoff list was a SHAP ranking. Neither affects any band.

Per D-09, `outputs/results_baseline.json` (archived in the next plan) is the source of truth for Phases 3 to 6. The handoff reference survives only as the comparison column of the generated block, and `CLAUDE.md` is unchanged.
