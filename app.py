"""AeroPulse demo: engine health and Remaining Useful Life (RUL) on NASA C-MAPSS.

Run:  streamlit run app.py

This file replaces the handoff Appendix C app by design: the logic lives in
src/demo_lib.py and this module only renders. It needs the committed models/,
data_raw/ and outputs/results_baseline.json.
"""
import logging
import sys
from pathlib import Path

import streamlit as st

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import rul_lib as L  # noqa: E402,F401
import demo_lib  # noqa: E402  (forces the Agg backend)
import matplotlib.pyplot as plt  # noqa: E402


def _mute(text):
    """Small text in the ink-mute token, for st.markdown. st.caption is not used: its container has
    opacity 0.6, which takes any mid-gray below WCAG AA (charcoal fades to 3.7:1 on white)."""
    return f":gray[:small[{text}]]"


st.set_page_config(page_title="AeroPulse: turbofan RUL", page_icon=":material/monitoring:", layout="wide")
st.title("AeroPulse: turbofan Remaining Useful Life")
st.markdown(_mute("XGBoost on causal rolling-window features, NASA C-MAPSS FD001-FD004"))
with st.container(border=True):
    st.markdown(_mute(":material/info: " + demo_lib.DISCLAIMER))  # always visible (D-08)

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
except Exception as exc:  # the page shows the type name only (T-03-02); the traceback goes to the server log
    logging.getLogger("aeropulse").exception("load_bundle(%s) failed", subset)
    _failure = type(exc).__name__
if _failure is not None:
    st.error(
        f"Could not load {subset} ({_failure}). To restore the committed model files run "
        f"`git checkout -- models/xgb_{subset}.json models/prep_{subset}.joblib`."
    )
    st.stop()

units = sorted(bundle.te["unit"].unique())
if st.session_state.get("unit") not in units:
    st.session_state["unit"] = units[0]  # stale selection after a subset switch resets cleanly
unit = st.sidebar.selectbox("Engine (test set)", units, key="unit")


def _pick_worst(key):
    """Selectbox callback: jump to the chosen engine, then reset the picker (runs before the rerun)."""
    chosen = st.session_state[key]
    if chosen is not None:
        st.session_state["unit"] = chosen
        st.session_state[key] = None


wkey = f"worst-{subset}"
cases = demo_lib.worst_cases(bundle)
_labels = {c.unit: c.label for c in cases}
st.sidebar.selectbox(
    "Worst-case engines (largest last-cycle errors)",
    [None, *[c.unit for c in cases]],
    key=wkey,
    format_func=lambda u: "Choose an engine..." if u is None else _labels[u],
    on_change=_pick_worst,
    args=(wkey,),
)
st.sidebar.markdown(_mute(
    "The engines this model gets most wrong at their last observed cycle (predicted minus official "
    "RUL), listed so the demo is not cherry-picked."
))
warn_rul = st.sidebar.slider(
    "Maintenance alert when predicted RUL is below (cycles)", 10, 80, 30, key="warn_rul"
)

eng = demo_lib.engine_frame(bundle, unit)
max_c = int(eng["cycle"].max())
lo, hi, fixed = demo_lib.slider_bounds(max_c)
if fixed:
    st.sidebar.markdown(_mute("Observed up to cycle: 1 of 1"))
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
c2.markdown(_mute("; ".join(_notes)))
with c3.container(border=True):
    st.markdown(_mute("Status"))
    _status, _colour = demo_lib.alert_status(pred_now, warn_rul)
    st.badge(_status, color=_colour)
    if cycle == max_c:
        _err = pred_now - true_final
        st.markdown(_mute(f"Last-cycle error: {_err:+.1f} cycles ({demo_lib.error_word(_err)})"))
    else:
        st.markdown(_mute("The official RUL is known only at the last observed cycle."))

# Fixed-width panels in a wrapping row: side by side on a wide screen, one per row below that, and
# never so narrow that the chart text shrinks (stretch columns would squeeze them to ~230 px).
PANEL_W = 460
with st.container(horizontal=True, gap="medium"):
    with st.container(border=True, width=PANEL_W):
        st.markdown("**Predicted vs true RUL**")
        _curve_text = demo_lib.curve_text(eng, cycle, true_final)
        fig = demo_lib.curve_figure(eng, cycle, true_final, warn_rul)
        st.pyplot(fig, alt=_curve_text)
        plt.close(fig)
        st.markdown(_mute(_curve_text))
    with st.container(border=True, width=PANEL_W):
        st.markdown("**What drives this prediction (SHAP)**")
        _wf_text = demo_lib.waterfall_text(bundle, row.name)
        fig = demo_lib.waterfall_figure(bundle, row.name)
        st.pyplot(fig, alt=_wf_text)
        plt.close(fig)
        st.markdown(_mute(_wf_text))
        st.markdown(_mute(demo_lib.WATERFALL_NOTE))
        st.markdown(_mute(demo_lib.FEATURE_LEGEND))
        if demo_lib.raw_prediction(bundle, row.name) < 0:
            st.markdown(_mute(
                "The model's raw output for this cycle is below 0; the app shows predictions clipped at 0."
            ))

with st.expander("Why late predictions cost more (NASA score)"):
    st.markdown(demo_lib.NASA_EXPLANATION)
    _late, _early = demo_lib.nasa_examples()
    st.markdown(
        f"Example: an engine predicted 10 cycles late costs {_late:.2f}; "
        f"predicted 10 cycles early it costs {_early:.2f}."
    )
    _d, _cost = demo_lib.nasa_point(true_final, float(eng["pred"].iloc[-1]))
    _nasa_text = (
        f"Engine {unit} at its last observed cycle, where the official RUL is known: "
        f"error {_d:+.1f} cycles ({demo_lib.error_word(_d)}), NASA cost {_cost:.2f}."
    )
    fig = demo_lib.nasa_figure(_d, _cost)
    st.pyplot(fig, width=PANEL_W, alt=_nasa_text)  # capped so the full-width expander doesn't blow it up
    plt.close(fig)
    st.markdown(_mute(_nasa_text))

with st.expander("Sensor trends"):
    sensors = bundle.prep["sensors"]
    picked = st.multiselect("Sensors", sensors, default=sensors[:4], key=f"sensors-{subset}")
    if picked:
        st.line_chart(eng.set_index("cycle")[picked])

with st.expander("How to read this"):
    for _line in (
        "- **Predicted RUL** is the model's estimate of the cycles left. It was trained on a target "
        f"capped at {L.RUL_CAP}, so high values read as 125+ with a CAPPED badge.",
        "- **True RUL** is the official value at the last observed cycle; at earlier cycles it is "
        "reconstructed from that value.",
        "- **Status** compares the predicted RUL with the alert threshold in the sidebar.",
        f"- **Warm-up**: for cycles 1-{demo_lib.WARMUP_MAX} the rolling features use fewer than "
        f"{L.WINDOW} cycles, so predictions are less stable.",
        "- **Worst-case engines** lists the engines this model gets most wrong, so the demo is not "
        "cherry-picked.",
        "- **Waterfall**: blue bars raise the predicted RUL, red bars lower it.",
    ):
        st.markdown(_line)
    try:
        st.markdown(demo_lib.model_card_line(demo_lib.model_card(subset)))
        st.markdown(_mute(demo_lib.NOISE_NOTE))
    except (KeyError, ValueError):
        st.markdown(_mute("The model card could not be read from the saved results."))
    st.markdown(_mute(
        "The demo serves the original XGBoost model, which includes the cycle feature (the engine's "
        "age, a strong driver, so part of its skill is age-based). A version without it is reported "
        "only as an ablation."
    ))
