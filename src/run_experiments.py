"""Train and evaluate XGBoost vs Random Forest for RUL on C-MAPSS.

Usage:  python3 src/run_experiments.py FD001 FD002 FD003 FD004
Outputs: outputs/results.json, outputs/pred_<subset>.csv, models/*
"""
import json
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import rul_lib as L

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = L.ROOT / "outputs"
MODELS = L.ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
N_FOLDS = 3
XGB_TRIALS = 25
RF_TRIALS = 6
FINAL_SEEDS = [0, 1, 2]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def engine_folds(units, k, seed=L.SEED):
    """Assign whole engines to folds (no leakage between cycles of one engine)."""
    u = np.unique(units)
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    fold_of = {e: i % k for i, e in enumerate(u)}
    return np.array([fold_of[e] for e in units])


def xgb_params_from(trial):
    return dict(
        max_depth=trial.suggest_int("max_depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 2.0),
    )


def make_xgb(params, n_estimators, seed=0, early=None):
    return xgb.XGBRegressor(
        n_estimators=n_estimators, tree_method="hist", n_jobs=2, random_state=seed,
        early_stopping_rounds=early, **params,
    )


def tune_xgb(X, y, folds):
    best_iters = {}

    def objective(trial):
        p = xgb_params_from(trial)
        rmses, its = [], []
        for f in range(N_FOLDS):
            tr, va = folds != f, folds == f
            m = make_xgb(p, 2000, early=50)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            pred = m.predict(X[va])
            rmses.append(np.sqrt(np.mean((pred - y[va]) ** 2)))
            its.append(m.best_iteration + 1)
        best_iters[trial.number] = int(np.mean(its))
        return float(np.mean(rmses))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=L.SEED))
    st.optimize(objective, n_trials=XGB_TRIALS)
    return st.best_params, best_iters[st.best_trial.number], st.best_value


def tune_rf(X, y, folds):
    def objective(trial):
        p = dict(
            max_depth=trial.suggest_int("max_depth", 10, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.2, 0.6),
            max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        )
        rm = []
        for f in range(N_FOLDS):
            tr, va = folds != f, folds == f
            m = RandomForestRegressor(n_estimators=80, n_jobs=2, random_state=0, **p)
            m.fit(X[tr], y[tr])
            rm.append(np.sqrt(np.mean((m.predict(X[va]) - y[va]) ** 2)))
        return float(np.mean(rm))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=L.SEED))
    st.optimize(objective, n_trials=RF_TRIALS)
    return st.best_params, st.best_value


def evaluate(model_fn, Xtr, ytr, Xte_last, true_rul, seeds=FINAL_SEEDS):
    """Fit with several seeds; return mean/std metrics and the seed-0 model + preds."""
    runs, preds, fit_t, pred_t, first = [], [], [], [], None
    for s in seeds:
        m = model_fn(s)
        t0 = time.time()
        m.fit(Xtr, ytr)
        fit_t.append(time.time() - t0)
        t0 = time.time()
        p = np.clip(m.predict(Xte_last), 0, None)
        pred_t.append(time.time() - t0)
        preds.append(p)
        runs.append(L.metrics(true_rul, p))
        r_cap = L.metrics(np.minimum(true_rul, L.RUL_CAP), p)
        runs[-1]["rmse_capped_truth"] = r_cap["rmse"]
        if first is None:
            first = m
    agg = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
           for k in runs[0]}
    agg["fit_seconds"] = float(np.mean(fit_t))
    agg["predict_seconds_100_engines"] = float(np.mean(pred_t))
    return agg, first, np.mean(preds, axis=0), preds[0]


def run_subset(name, results):
    n_cond, n_fault = L.SUBSETS[name]
    log(f"=== {name}: conditions={n_cond} fault_modes={n_fault}")
    tr, te, rul = L.load_subset(name)
    tr = L.add_rul(tr)

    # constant / near-constant sensors
    nunique = tr[L.SENSORS].nunique()
    norm_all = L.ConditionNormaliser(n_cond, L.SENSORS).fit(tr)
    within_sd = norm_all.sd  # replaced zeros by 1, so recompute exact zeros below
    cond = norm_all._cond(tr)
    raw_sd = tr[L.SENSORS].groupby(cond).std().max()
    keep = [s for s in L.SENSORS if raw_sd[s] > 1e-6 and not (n_cond == 1 and nunique[s] <= 2)]
    dropped = [s for s in L.SENSORS if s not in keep]
    log("kept", len(keep), "dropped", dropped)

    norm = L.ConditionNormaliser(n_cond, keep).fit(tr)
    trn, ten = norm.transform(tr), norm.transform(te)

    Xtr_full = L.make_features(trn, keep, rolling=True)
    Xtr_raw = L.make_features(trn, keep, rolling=False)
    Xte_full = L.make_features(ten, keep, rolling=True)
    Xte_raw = L.make_features(ten, keep, rolling=False)
    last = L.last_cycle_rows(te)
    y = tr["rul"].to_numpy()
    folds = engine_folds(tr["unit"].to_numpy(), N_FOLDS)
    A, Aq = Xtr_full.to_numpy(np.float32), Xte_full.iloc[last].to_numpy(np.float32)
    R, Rq = Xtr_raw.to_numpy(np.float32), Xte_raw.iloc[last].to_numpy(np.float32)
    log("features", A.shape[1], "train rows", A.shape[0])

    t0 = time.time()
    xp, xiters, xcv = tune_xgb(A, y, folds)
    log(f"XGB tuned: cv_rmse={xcv:.3f} iters={xiters} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    rp, rcv = tune_rf(A, y, folds)
    log(f"RF tuned: cv_rmse={rcv:.3f} ({time.time()-t0:.0f}s)")

    n_est = max(50, int(xiters * 1.1))
    xgb_fn = lambda s: make_xgb(xp, n_est, seed=s)
    rf_fn = lambda s: RandomForestRegressor(n_estimators=300, n_jobs=2, random_state=s, **rp)
    xgbdef_fn = lambda s: xgb.XGBRegressor(n_estimators=300, tree_method="hist", n_jobs=2, random_state=s)

    res = {"n_conditions": n_cond, "n_fault_modes": n_fault, "kept_sensors": keep, "dropped_sensors": dropped,
           "n_features": int(A.shape[1]), "train_rows": int(A.shape[0]), "train_engines": int(tr["unit"].nunique()),
           "test_engines": int(te["unit"].nunique()),
           "xgb_params": xp, "xgb_n_estimators": n_est, "xgb_cv_rmse": xcv,
           "rf_params": rp, "rf_cv_rmse": rcv, "models": {}}

    agg, xmodel, xpred_mean, xpred0 = evaluate(xgb_fn, A, y, Aq, rul)
    res["models"]["XGBoost (rolling features)"] = agg
    log("XGB   ", {k: round(v["mean"], 3) for k, v in agg.items() if isinstance(v, dict)})
    agg, rmodel, rpred_mean, rpred0 = evaluate(rf_fn, A, y, Aq, rul)
    res["models"]["Random Forest (rolling features)"] = agg
    log("RF    ", {k: round(v["mean"], 3) for k, v in agg.items() if isinstance(v, dict)})

    if name == "FD001":  # ablations
        agg, _, _, _ = evaluate(xgbdef_fn, A, y, Aq, rul)
        res["models"]["XGBoost default params (rolling features)"] = agg
        agg, _, _, _ = evaluate(xgb_fn, R, y, Rq, rul)
        res["models"]["XGBoost (raw sensors only)"] = agg
        agg, _, _, _ = evaluate(rf_fn, R, y, Rq, rul)
        res["models"]["Random Forest (raw sensors only)"] = agg
        for k, v in res["models"].items():
            log("  ablation", k, round(v["rmse"]["mean"], 3), round(v["nasa"]["mean"], 1))

    # persist artefacts
    units = te.loc[last, "unit"].to_numpy()
    pd.DataFrame({"unit": units, "cycles_observed": te.loc[last, "cycle"].to_numpy(), "true_rul": rul,
                  "xgb_pred": xpred0, "rf_pred": rpred0}).to_csv(OUT / f"pred_{name}.csv", index=False)
    xmodel.save_model(str(MODELS / f"xgb_{name}.json"))
    joblib.dump({"normaliser": norm, "sensors": keep, "feature_names": list(Xtr_full.columns)},
                MODELS / f"prep_{name}.joblib")
    joblib.dump(rmodel, MODELS / f"rf_{name}.joblib", compress=3)
    # booster gain importance for the report
    imp = pd.Series(xmodel.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    res["top_gain_features"] = {k: float(v) for k, v in imp.head(15).items()}
    results[name] = res


if __name__ == "__main__":
    names = sys.argv[1:] or list(L.SUBSETS)
    rpath = OUT / "results.json"
    results = json.loads(rpath.read_text()) if rpath.exists() else {}
    for n in names:
        run_subset(n, results)
        rpath.write_text(json.dumps(results, indent=2))
        log("saved", n)
