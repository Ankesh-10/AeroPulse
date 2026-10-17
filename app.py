"""Streamlit demo: engine health and Remaining Useful Life (RUL) from C-MAPSS.

Run:  streamlit run app.py
Needs the models in models/ (created by src/run_experiments.py).
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import rul_lib as L  # noqa: E402

st.set_page_config(page_title="Turbofan RUL", layout="wide")
st.title("Predictive maintenance: turbofan Remaining Useful Life")
st.caption("XGBoost on rolling-window sensor features, trained on NASA C-MAPSS")

subset = st.sidebar.selectbox("C-MAPSS subset", list(L.SUBSETS), index=0)
warn_rul = st.sidebar.slider("Maintenance alert when predicted RUL is below (cycles)", 10, 80, 30)


@st.cache_resource
def load(subset):
    prep = joblib.load(L.ROOT / "models" / f"prep_{subset}.joblib")
    model = xgb.XGBRegressor()
    model.load_model(str(L.ROOT / "models" / f"xgb_{subset}.json"))
    _, te, rul = L.load_subset(subset)
    ten = prep["normaliser"].transform(te)
    X = L.make_features(ten, prep["sensors"], rolling=True)[prep["feature_names"]]
    pred = np.clip(model.predict(X.to_numpy(np.float32)), 0, None)
    te = te.assign(pred=pred)
    return prep, model, te, X, rul, shap.TreeExplainer(model)


prep, model, te, X, rul, explainer = load(subset)
units = sorted(te["unit"].unique())
unit = st.sidebar.selectbox("Engine (test set)", units, index=0)
eng = te[te["unit"] == unit]
max_c = int(eng["cycle"].max())
cycle = st.sidebar.slider("Observed up to cycle", 1, max_c, max_c)
true_final = float(rul[units.index(unit)])

row = eng[eng["cycle"] == cycle].iloc[0]
pred_now = float(row["pred"])
c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", f"{pred_now:.0f}")
c2.metric("Cycles observed", cycle)
c3.metric("Status", "SCHEDULE MAINTENANCE" if pred_now < warn_rul else "Healthy")
if cycle == max_c:
    st.write(f"Official RUL at the last observed cycle: **{true_final:.0f}** cycles "
             f"(error {pred_now - true_final:+.1f}).")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    d = eng[eng["cycle"] <= cycle]
    ax.plot(d["cycle"], d["pred"], color="#1f6feb", lw=2, label="Predicted RUL")
    true_curve = np.minimum(true_final + (max_c - eng["cycle"]), L.RUL_CAP)
    ax.plot(eng["cycle"], true_curve, color="#888", ls="--", lw=1.5, label="True RUL (capped at 125)")
    ax.axhline(warn_rul, color="#d1242f", lw=1, ls=":", label="Alert threshold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
with right:
    xi = X.loc[[row.name]]
    sv = explainer.shap_values(xi.to_numpy(np.float32))[0]
    top = pd.Series(sv, index=X.columns).sort_values(key=np.abs, ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(top.index, top.values, color=["#d1242f" if v < 0 else "#1f6feb" for v in top.values])
    ax.set_xlabel("SHAP contribution to predicted RUL (cycles)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.caption("Blue pushes the prediction up (healthier), red pulls it down (closer to failure).")

with st.expander("Sensor trends for this engine"):
    s_pick = st.multiselect("Sensors", prep["sensors"], default=prep["sensors"][:4])
    if s_pick:
        st.line_chart(eng.set_index("cycle")[s_pick])
