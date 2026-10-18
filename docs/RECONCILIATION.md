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

<!-- END GENERATED: check_reference.py -->

## Findings

The full four-subset run has not been made yet; findings are written here after scripts/check_reference.py outputs/results.json runs.
