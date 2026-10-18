# Frozen kernel and baseline

After the golden canary (plan 02-01) and the reference check (plan 02-03) passed, `src/rul_lib.py` and `src/run_experiments.py` are frozen, and `outputs/results_baseline.json` is the source of truth for every later phase (D-09, D-12). New experiments live in new files and never change these bytes. `tests/test_frozen.py` runs in the unit tier and fails when a recorded hash stops matching.

Archive commit: c18cb79 chore(02-04): archive the first full four-subset run as the frozen baseline (RERUN-02)
Freeze date: 2026-10-18
Freeze commit: the commit that adds this file (git log --diff-filter=A --format=%h -- docs/FROZEN.md).

## Kernel files

The hashes are of the raw bytes. `.gitattributes` keeps `*.py` at LF, so they hold on every platform. Check from the repository root with `grep -E '^[0-9a-f]{64}  src/' docs/FROZEN.md | tr -d '\r' | sha256sum -c -`.

```
1f48cf7020e52612243e476cb8d9167443faaae249ac914bfc70a5f47a772044  src/rul_lib.py
adc344a33a57a32ae19e26d512d807d251867b8eb67552b0f85c73c59eb2b753  src/run_experiments.py
```

## Baseline results

The hash is the SHA256 after removing every CR byte. Windows writes the file with CRLF while git stores LF, so the CR-removed hash equals the git blob content hash on every platform. Check with `tr -d '\r' < outputs/results_baseline.json | sha256sum`.

```
14df54aa91ce34e89548cd0511063fefc77e8dc2467fcea9dc828553aa28d70d  outputs/results_baseline.json
```

## The only change from Appendix B (RERUN-04, D-01)

The only change to the rebuilt kernel is commit c0cb307 (`feat(02-01): name gain-importance features and add seed-mean prediction columns (RERUN-04)`). It touches the persist block of `run_subset` in `src/run_experiments.py` and nothing else: gain-importance keys are mapped to feature names, and two additive seed-mean prediction columns are written to `pred_FD00N.csv`. `src/rul_lib.py` is byte-identical to the verbatim rebuild.

The verbatim bytes stay checkable: `git show 1c3469a:src/run_experiments.py | sha256sum` prints the `src/run_experiments.py` hash line of `docs/REBUILD_CHECK.md`.

Full diff (`git diff 1c3469a c0cb307 -- src/run_experiments.py`):

```diff
diff --git a/src/run_experiments.py b/src/run_experiments.py
index eeafebe..a2e6a87 100644
--- a/src/run_experiments.py
+++ b/src/run_experiments.py
@@ -194,14 +194,16 @@ def run_subset(name, results):
     # persist artefacts
     units = te.loc[last, "unit"].to_numpy()
     pd.DataFrame({"unit": units, "cycles_observed": te.loc[last, "cycle"].to_numpy(), "true_rul": rul,
-                  "xgb_pred": xpred0, "rf_pred": rpred0}).to_csv(OUT / f"pred_{name}.csv", index=False)
+                  "xgb_pred": xpred0, "rf_pred": rpred0,
+                  "xgb_pred_mean": xpred_mean, "rf_pred_mean": rpred_mean}).to_csv(OUT / f"pred_{name}.csv", index=False)
     xmodel.save_model(str(MODELS / f"xgb_{name}.json"))
     joblib.dump({"normaliser": norm, "sensors": keep, "feature_names": list(Xtr_full.columns)},
                 MODELS / f"prep_{name}.joblib")
     joblib.dump(rmodel, MODELS / f"rf_{name}.joblib", compress=3)
-    # booster gain importance for the report
+    # booster gain importance for the report (booster keys are f0..f{n-1}: training used NumPy arrays)
+    feature_names = list(Xtr_full.columns)
     imp = pd.Series(xmodel.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
-    res["top_gain_features"] = {k: float(v) for k, v in imp.head(15).items()}
+    res["top_gain_features"] = {feature_names[int(k[1:])]: float(v) for k, v in imp.head(15).items()}
     results[name] = res
 
 
```
