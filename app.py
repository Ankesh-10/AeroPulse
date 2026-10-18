"""AeroPulse demo: engine health and Remaining Useful Life (RUL) on NASA C-MAPSS.

Run:  streamlit run app.py

This file replaces the handoff Appendix C app by design: the logic lives in
src/demo_lib.py and this module only renders. It needs the committed models/,
data_raw/ and outputs/results_baseline.json.
"""
import sys
from pathlib import Path

import streamlit as st

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import rul_lib as L  # noqa: E402,F401
import demo_lib  # noqa: E402  (forces the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402

st.set_page_config(page_title="AeroPulse: turbofan RUL", page_icon=":material/monitoring:", layout="wide")
st.title("AeroPulse: turbofan Remaining Useful Life")
st.caption("XGBoost on causal rolling-window features, NASA C-MAPSS FD001-FD004")
with st.container(border=True):
    st.caption(":material/info: " + demo_lib.DISCLAIMER)  # always visible (D-08)

subset = st.sidebar.selectbox("C-MAPSS subset", list(demo_lib.SUBSETS), key="subset")

# Missing-file guard before any load (D-03): repo-relative paths and a restore command, no traceback.
_missing = demo_lib.missing_files(subset)
if _missing:
    _lines = [f"- `{p}`: run `{demo_lib.restore_command(p, subset)}`" for p in _missing]
    st.error("Required files are missing:\n" + "\n".join(_lines))
    st.stop()


@st.cache_resource(max_entries=1, show_spinner=False)
def _load(subset):
    return demo_lib.load_bundle(subset)


_failure = None
try:
    with st.spinner(f"Loading {subset} (first load of a subset takes a few seconds)"):
        bundle = _load(subset)
except Exception as exc:  # reduce to the type name only (T-03-02)
    _failure = type(exc).__name__
if _failure is not None:
    st.error(
        f"Could not load {subset} ({_failure}). To regenerate the models run "
        f"`{demo_lib.restore_command(f'models/xgb_{subset}.json', subset)}`."
    )
    st.stop()

units = sorted(bundle.te["unit"].unique())
if st.session_state.get("unit") not in units:
    st.session_state["unit"] = units[0]  # stale selection after a subset switch resets cleanly
unit = st.sidebar.selectbox("Engine (test set)", units, key="unit")
warn_rul = st.sidebar.slider(
    "Maintenance alert when predicted RUL is below (cycles)", 10, 80, 30, key="warn_rul"
)

eng = demo_lib.engine_frame(bundle, unit)
max_c = int(eng["cycle"].max())
lo, hi, fixed = demo_lib.slider_bounds(max_c)
if fixed:
    st.sidebar.caption("Observed up to cycle: 1 of 1")
    cycle = 1
else:
    cycle = st.sidebar.slider("Observed up to cycle", lo, hi, hi, key=f"cycle-{subset}-{unit}")

row = demo_lib.row_at(eng, cycle)
pred_now = float(row["pred"])
true_final = demo_lib.true_final_rul(bundle, unit)
true_now = demo_lib.true_rul_at(true_final, max_c, cycle)
pred_label = demo_lib.predicted_label(pred_now)

if demo_lib.is_warmup(cycle):  # one quiet note, D-10
    st.info(":yellow-badge[WARM-UP] " + demo_lib.WARMUP_NOTE)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", pred_label, border=True)
if pred_label == demo_lib.CAPPED_LABEL:
    c1.badge("125+ CAPPED", color="yellow")
c2.metric("True RUL (cycles)", f"{true_now:.0f}", border=True)
_notes = [
    f":yellow[{n}]" if n.startswith("above the") else n
    for n in demo_lib.true_rul_note(true_now, at_last=cycle == max_c)
]
c2.caption("; ".join(_notes))
with c3.container(border=True):
    st.caption("Status")
    if pred_now < warn_rul:
        st.badge("SCHEDULE MAINTENANCE", color="red")
    else:
        st.badge("HEALTHY", color="green")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Predicted vs true RUL**")
        fig = demo_lib.curve_figure(eng, cycle, true_final, warn_rul)
        st.pyplot(fig)
        plt.close(fig)
with right:
    with st.container(border=True):
        st.markdown("**What drives this prediction (SHAP)**")
        fig = demo_lib.waterfall_figure(bundle, row.name)
        st.pyplot(fig)
        plt.close(fig)

with st.expander("Sensor trends"):
    sensors = bundle.prep["sensors"]
    picked = st.multiselect("Sensors", sensors, default=sensors[:4], key=f"sensors-{subset}")
    if picked:
        st.line_chart(eng.set_index("cycle")[picked])
