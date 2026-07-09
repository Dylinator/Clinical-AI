"""
dashboard.py — Phase 5, the clinician-facing view (presentation only, Rule 9).

Reads the persisted pipeline + chosen threshold and renders ONE held-out patient's
risk trajectory, vitals, and interpretation. It never retrains and never rebuilds
features by hand — it calls the same trajectory_engine as the rest of the pipeline,
so the screen shows exactly what the model produced.

Run it (after `python run_pipeline.py` has written the artifacts/ folder):

    pip install streamlit
    streamlit run dashboard.py
"""

from __future__ import annotations
import json
import os

import pandas as pd
import altair as alt
import streamlit as st

import config
import model as model_mod
import splits
from synthetic import generate
from timeline_engine import to_frame
from trajectory_engine import trajectory, top_drivers


st.set_page_config(page_title="Clinical Trajectory MVP", layout="wide")


@st.cache_resource
def load_all():
    """Fitted pipeline, chosen threshold, and the HELD-OUT patients only.

    Restricting to test patients matters: showing a training patient would flatter
    the model (it has already seen them). We reuse the exact split saved by
    run_pipeline, or recompute the same seeded split as a fallback.
    """
    est = model_mod.load()

    thr = 0.5
    thr_path = f"{config.ARTIFACT_DIR}/threshold.json"
    if os.path.exists(thr_path):
        thr = json.load(open(thr_path)).get("threshold", 0.5)

    patients = generate()
    ids_path = f"{config.ARTIFACT_DIR}/test_ids.json"
    if os.path.exists(ids_path):
        test_ids = set(json.load(open(ids_path)))
    else:                                      # fallback: same seeded split
        _, test_ids = splits.split_patients([p.id for p in patients])
    held_out = [p for p in patients if p.id in test_ids] or patients
    return est, thr, held_out


est, threshold, patients = load_all()

st.title("AI Clinical Trajectory — risk over time")
st.caption("Held-out synthetic patients. Demonstrates the system and its reasoning, "
           "not clinical validity.")


# ---- patient picker (default to a treated sepsis case so the story shows) ----
def _label(p) -> str:
    tag = "sepsis" if p.outcome == "sepsis" else "stable"
    if p.interventions:
        tag += " · treated"
    return f"#{p.id}  ({tag})"


options = {_label(p): p for p in patients}
ordered = sorted(options, key=lambda k: (0 if "treated" in k else 1 if "sepsis" in k else 2, k))
choice = st.selectbox("Patient (held-out test set)", ordered)
patient = options[choice]

# ---- score the whole trajectory + attach medication context ----
frame = to_frame([patient])
traj = trajectory(frame, est).merge(frame[["time"] + config.MED_FEATURES], on="time", how="left")
latest = float(traj["risk"].iloc[-1])
alerting = latest >= threshold

left, right = st.columns([2, 1])

with left:
    st.subheader("Risk trajectory")
    risk_line = (
        alt.Chart(traj)
        .mark_line(point=True, color="#c0392b")
        .encode(
            x=alt.X("time:Q", title="minutes since ED arrival"),
            y=alt.Y("risk:Q", title="deterioration risk",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("time:Q", title="min"),
                     alt.Tooltip("risk:Q", title="risk", format=".1%")],
        )
    )
    thr_rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])))
    )
    layers = [risk_line, thr_rule]
    if patient.interventions:
        iv = pd.DataFrame([{"time": i.time, "intervention": i.type} for i in patient.interventions])
        iv_rules = (
            alt.Chart(iv)
            .mark_rule(color="#7f8c8d", strokeDash=[2, 2])
            .encode(x="time:Q", tooltip=["intervention", "time"])
        )
        layers.append(iv_rules)
    st.altair_chart(alt.layer(*layers).properties(height=300), use_container_width=True)
    st.caption(
        f"Dashed red = alert threshold ({threshold:.0%}, chosen for "
        f"~{config.TARGET_ALERTS_PER_DAY:.0f} alert/patient-day). "
        "Grey lines = interventions (hover for the drug)."
    )

    st.subheader("Vitals")
    st.line_chart(frame.set_index("time")[config.VITALS])

with right:
    st.subheader("Now")
    st.metric("Current risk", f"{latest:.0%}")
    if alerting:
        st.error(f"ALERT — at/above the {threshold:.0%} threshold")
    else:
        st.success(f"Below the {threshold:.0%} threshold")

    active = [m.replace("on_", "").replace("recent_", "recent ").replace("_", " ")
              for m in config.MED_FEATURES if traj[m].iloc[-1] > 0]
    if active:
        st.write("**Drugs on board:** " + ", ".join(active))
    if traj["on_vasopressor"].iloc[-1] > 0:
        st.warning("Blood pressure is vasopressor-supported — a normal number here is "
                   "not recovery. The model keeps risk high because it has that context.")

    st.write("**Top risk drivers** (importance stand-in for SHAP):")
    for name, imp in top_drivers(est, k=6):
        st.write(f"- {name}  ({imp:.2f})")

    if patient.deterioration_onset is not None:
        st.write(f"**True onset:** {patient.deterioration_onset:.0f} min "
                 "(ground truth, hidden from the model)")
    else:
        st.write("**Ground truth:** stable — no deterioration")
