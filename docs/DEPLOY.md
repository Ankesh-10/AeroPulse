# Deploying the demo to Streamlit Community Cloud

This is the checklist for putting the demo on Streamlit Community Cloud. The push to GitHub and the Cloud deploy are done manually by the repository owner; nothing in this repository pushes or deploys. The app serves the original with-cycle XGBoost models (DEMO-08), loaded from the committed `models/` and `outputs/` files.

- Repository: `Ankesh-10/AeroPulse`
- Entrypoint: `app.py`
- Python: 3.13, chosen in Advanced settings when the app is created
- Dependencies: `requirements.txt` only (no other dependency file, no `packages.txt`)

## Checklist

1. Confirm sizes and ignores from the repository root:
   - `du -h models/* data_raw/*` (no file near GitHub's 100 MiB limit)
   - `git check-ignore -q models/rf_FD001.joblib && echo ignored` (the random forest pickles stay out of git)
   - `git ls-files .streamlit` (only `.streamlit/config.toml`; the secrets file is ignored)
   - `.venv/Scripts/python.exe -m pytest tests/test_demo_hygiene.py -q` (pins and forbidden dependency files)
2. Push the branch to GitHub yourself.
3. On Community Cloud choose New app and select the repository `Ankesh-10/AeroPulse`, the branch you pushed and the entrypoint `app.py`.
4. Open Advanced settings and choose Python 3.13 before deploying. Cloud ignores `runtime.txt` and `.python-version`, and changing the version later means deleting the app and deploying it again.
5. After the build, check that the build log's Python version line says 3.13.
6. Open FD001, FD002, FD003 and FD004 and note the Cloud cold-start time of each first visit (a spinner shows while the subset loads). This Cloud number is separate from the local Windows timings in the generated block below, which were measured on this machine.
7. Check that the theme is applied (bone sidebar, serif title, pastel badges). If it is not, confirm that `.streamlit/config.toml` is on the deployed branch.
8. Try a worst-case engine from the picker, and cycle 1, on one subset.
9. Paste the app URL into the live-demo placeholder of the README (Phase 6).

## If something goes wrong

- **Cloud built on another Python despite the dropdown.** All pins have wheels for 3.12 to 3.14, and the generated block below shows both the Linux x86_64 resolve and the local install passing on 3.14, so the app should still run. To get 3.13, record the subdomain, branch and entrypoint, delete the app and deploy it again with 3.13 (or 3.12) chosen in Advanced settings.
- **The SHAP install or `TreeExplainer` fails.** The fallback is XGBoost's own contributions, `booster.predict(DMatrix, pred_contribs=True)`, which give identical values; `test_real_model_shap_is_additive_and_matches_pred_contribs` proves it on all four models. If the xgboost wheel is the problem, use `xgboost-cpu==3.4.1` in `requirements.txt` (same `import xgboost`) and run the check again.
- **A startup TypeError from starlette.** Pin `starlette` to the version listed in the generated block with one added line in `requirements.txt`, or bump Streamlit, then run the check again.
- **Restarts or memory pressure.** One subset is cached at a time and SHAP runs on a single row. The cache is process-global, so two visitors on different subsets evict each other and each pays a cold load.
- **The app is asleep.** Cloud puts it to sleep after 12 hours without traffic; any visitor can wake it.

## Local measurements (D-01, D-15)

The block below is written by `scripts/check_clean_env.py`. It resolves `requirements.txt` for Cloud's Linux x86_64 platform without installing anything, builds throwaway venvs from `requirements.txt` only on this Windows machine, runs the three demo test files (which render the app and so cover model loading) and a headless start (which proves only that the server and script start), and times a cold load per subset locally. The Linux part is a resolve only, so the real Linux install is still checked at deploy (checklist step 5), and the local timings are not the Cloud cold start (checklist step 6). D-01's precompute threshold is 30 s.

<!-- BEGIN GENERATED: check_clean_env.py -->

### Clean environments: local Windows run

| Requested Python | Python | Install from requirements.txt | Demo tests | Headless start |
|---|---|---|---|---|
| 3.13 | 3.13.12 | OK | 109 passed / 0 failed / 0 skipped | OK |
| 3.14 | 3.14.3 | OK | 109 passed / 0 failed / 0 skipped | OK |

Package versions installed from requirements.txt:

| Package | Python 3.13 | Python 3.14 |
|---|---|---|
| streamlit | 1.65.0 | 1.65.0 |
| shap | 0.53.0 | 0.53.0 |
| xgboost | 3.4.1 | 3.4.1 |
| scikit-learn | 1.9.1 | 1.9.1 |
| pandas | 3.0.5 | 3.0.5 |
| numpy | 2.5.3 | 2.5.3 |
| matplotlib | 3.11.2 | 3.11.2 |
| joblib | 1.6.0 | 1.6.0 |
| scipy | 1.18.1 | 1.18.1 |
| starlette | 1.9.0 | 1.9.0 |

### Streamlit Cloud target: Linux x86_64 resolve (no install)

| Requested Python | Platform | Resolve | Resolved packages |
|---|---|---|---|
| 3.13 | x86_64-manylinux_2_31 | OK | 52 |
| 3.14 | x86_64-manylinux_2_31 | OK | 52 |

This resolve proves that every pin and its dependencies have a Linux x86_64 wheel for that Python. Nothing was installed or run on Linux.

### Cold load in seconds, local Windows measurement (this machine, not the Streamlit Cloud cold start)

| Subset | Python 3.13 | Python 3.14 |
|---|---|---|
| FD001 | 1.28 | 1.57 |
| FD002 | 3.73 | 4.75 |
| FD003 | 1.75 | 2.07 |
| FD004 | 4.49 | 5.45 |
| imports (once per process) | 4.84 | 5.15 |

D-01: worst local cold load 5.45 s (threshold 30 s): precompute not triggered

Measured on Windows 11, 20 logical CPUs, 2026-10-18

RESULT: PASS

<!-- END GENERATED: check_clean_env.py -->

## Fonts

Fonts load in the visitor's browser from Google Fonts. Metric values use the body font, because no theme key sets a metric font family. Offline, or when Google Fonts is blocked, Streamlit's bundled Source Sans and Source Code Pro are used and the title loses its serif (see the deviations table in `DESIGN.md`).

## Re-run the check

```
.venv/Scripts/python.exe scripts/check_clean_env.py --markdown docs/DEPLOY.md
```

Prerequisites: uv on the PATH and local Python 3.13 and 3.14 (see `uv python list`). Exit codes: 0 every interpreter passed, 1 a check failed, 2 the preflight failed. Only the generated block is rewritten. The slow test `pytest -m slow tests/test_check_clean_env.py` runs the same check without writing docs.
