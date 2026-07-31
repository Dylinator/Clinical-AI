"""
dashboard.py — Phase 5 clinician view, now covering the whole Track-A build.

Two tabs:
  1. Patient trajectory — pick a held-out patient and see the deterioration risk
     over time from BOTH models overlaid (RandomForest and the encoder-only
     Transformer), the vitals, the sparse labs, and an interpretation panel.
  2. Model benchmark — the honest head-to-head (RF vs Transformer vs NEWS2/qSOFA)
     read from the last `benchmark.py --save` run, on synthetic OR real data.

It is presentation-only (Rule 9): it LOADS saved artifacts and calls the shared
trajectory engine; it never retrains or rebuilds features by hand, so the screen
shows exactly what the models produced. The data source, cohort size, and metrics
all come from artifacts/benchmark.json, so the dashboard follows whatever the last
benchmark run trained — synthetic or the real PhysioNet-2019 cohort.

Prepare artifacts first, then run:
    python benchmark.py --save                 # synthetic (fast)
    python benchmark.py --real --save          # real PhysioNet (slower)
    python -m streamlit run dashboard.py
If no benchmark.json exists it falls back to the RandomForest-only run_pipeline
artifacts on synthetic data.
"""

from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import config
import model as model_mod
import splits
from timeline_engine import to_frame
from trajectory_engine import trajectory, trajectory_transformer, top_drivers

st.set_page_config(page_title="Clinical Trajectory — RF vs Transformer", layout="wide")
ART = config.ARTIFACT_DIR


def _bootstrap_if_missing():
    """A fresh deploy clones the repo WITHOUT the git-ignored artifacts/ folder, so
    there is no trained model and the app would error with 'No trained model found'.
    Build the synthetic artifacts once, in-process -- seeded, so they are identical to
    a local `python run_pipeline.py`. A no-op once the model exists; because it runs
    inside the cached loader it executes at most once per deployed container."""
    if os.path.exists(config.PIPELINE_PATH):
        return
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    # Modest cohort so a cold-start deploy trains fast, while keeping enough positive
    # cases for stable metrics; also matches the committed local artifacts (same seed).
    config.GEN.n_patients = 700
    with st.spinner("First launch: training the model (runs once, ~1-3 min)..."):
        import run_pipeline
        run_pipeline.main()


@st.cache_resource
def load_everything():
    """Reconstruct the exact cohort the models were trained on (from the manifest)
    and load both models + threshold + held-out ids. Cached for the session."""
    _bootstrap_if_missing()
    manifest = None
    mpath = f"{ART}/benchmark.json"
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    source = manifest["source"] if manifest else "synthetic"

    # Rebuild patients from the SAME source + settings the benchmark used.
    if source in ("real", "physionet"):          # "real" = legacy manifest value
        config.GRID_INTERVAL_MIN = manifest.get("grid_interval_min", 60)
        from ingest import physionet2019
        patients = physionet2019.load(limit=manifest.get("limit"),
                                      max_hours=manifest.get("max_hours", 72),
                                      download_if_missing=False)
    elif source == "eicu":
        config.GRID_INTERVAL_MIN = manifest.get("grid_interval_min", 60)
        from ingest import eicu
        patients = eicu.load(limit=manifest.get("limit"),
                             max_hours=manifest.get("max_hours", 72))
    else:
        from synthetic import generate
        if manifest and manifest.get("n_patients"):
            config.GEN.n_patients = manifest["n_patients"]
        patients = generate()

    rf = model_mod.load() if os.path.exists(config.PIPELINE_PATH) else None

    tmodel = None
    tpath = f"{ART}/transformer.joblib"
    if os.path.exists(tpath):
        try:
            import model_transformer as mt
            tmodel = mt.load(tpath)
        except Exception as e:               # torch missing or shape drift
            st.warning(f"Transformer artifact present but could not load ({e}).")

    thr = 0.5
    if os.path.exists(f"{ART}/threshold.json"):
        thr = json.load(open(f"{ART}/threshold.json")).get("threshold", 0.5)

    test_ids = None
    if os.path.exists(f"{ART}/test_ids.json"):
        test_ids = set(json.load(open(f"{ART}/test_ids.json")))

    return manifest, source, patients, rf, tmodel, thr, test_ids


manifest, source, patients, rf, tmodel, threshold, test_ids = load_everything()

st.title("AI Clinical Trajectory — risk over time")
src_label = {"real": "real PhysioNet-2019 ICU data",
             "physionet": "real PhysioNet-2019 ICU data",
             "eicu": "real eICU multi-center ICU data"}.get(source, "synthetic patients")
st.caption(f"Held-out {src_label}. Demonstrates the system and its reasoning; "
           "on synthetic data it is not evidence of clinical validity.")

if rf is None:
    st.error("No trained model found. Run `python benchmark.py --save` "
             "(or `python run_pipeline.py`) first.")
    st.stop()

tab_traj, tab_bench = st.tabs(["Patient trajectory", "Model benchmark"])

# =========================================================================== #
# TAB 1 — patient trajectory (RF + Transformer overlaid)
# =========================================================================== #
with tab_traj:
    held = [p for p in patients if (test_ids is None or p.id in test_ids)] or patients

    def _label(p) -> str:
        tag = "sepsis" if p.outcome == "sepsis" else "stable"
        if p.interventions:
            tag += " · treated"
        return f"#{p.id}  ({tag})"

    options = {_label(p): p for p in held}
    ordered = sorted(options, key=lambda k: (0 if "treated" in k else 1 if "sepsis" in k else 2, k))
    left_top, right_top = st.columns([3, 1])
    with left_top:
        choice = st.selectbox("Patient (held-out test set)", ordered)
    with right_top:
        model_pick = st.multiselect(
            "Models", ["RandomForest", "Transformer"] if tmodel else ["RandomForest"],
            default=["RandomForest", "Transformer"] if tmodel else ["RandomForest"])
    patient = options[choice]

    frame = to_frame([patient])

    # Score the trajectory with each chosen model.
    curves = []
    latest = {}
    if "RandomForest" in model_pick:
        tr_rf = trajectory(frame, rf).rename(columns={"risk": "value"})
        tr_rf["model"] = "RandomForest"
        curves.append(tr_rf[["time", "value", "model"]])
        latest["RandomForest"] = float(tr_rf["value"].iloc[-1])
    if tmodel is not None and "Transformer" in model_pick:
        tr_tf = trajectory_transformer(frame, tmodel).rename(columns={"risk": "value"})
        tr_tf["model"] = "Transformer"
        curves.append(tr_tf[["time", "value", "model"]])
        latest["Transformer"] = float(tr_tf["value"].iloc[-1])

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Risk trajectory")
        if curves:
            traj_df = pd.concat(curves, ignore_index=True)
            risk_line = (
                alt.Chart(traj_df).mark_line(point=True).encode(
                    x=alt.X("time:Q", title="minutes since arrival"),
                    y=alt.Y("value:Q", title="deterioration risk",
                            scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                    color=alt.Color("model:N", title="model",
                                    scale=alt.Scale(domain=["RandomForest", "Transformer"],
                                                    range=["#c0392b", "#2980b9"])),
                    tooltip=["model:N", alt.Tooltip("time:Q", title="min"),
                             alt.Tooltip("value:Q", title="risk", format=".1%")],
                )
            )
            thr_rule = (alt.Chart(pd.DataFrame({"y": [threshold]}))
                        .mark_rule(color="gray", strokeDash=[4, 4])
                        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]))))
            layers = [risk_line, thr_rule]
            if patient.interventions:
                iv = pd.DataFrame([{"time": i.time, "intervention": i.type}
                                   for i in patient.interventions])
                layers.append(alt.Chart(iv).mark_rule(color="#7f8c8d", strokeDash=[2, 2])
                              .encode(x="time:Q", tooltip=["intervention", "time"]))
            st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
            st.caption(f"Dashed gray = RF alert threshold ({threshold:.0%}). "
                       "Red = RandomForest, blue = Transformer. The two models can "
                       "disagree — that disagreement is the point of showing both.")
        else:
            st.info("Select at least one model.")

        st.subheader("Vitals")
        st.line_chart(frame.set_index("time")[config.VITALS])

        st.subheader("Labs (sparse — points are actual draws; gaps = not measured)")
        labs_long = (frame[["time"] + config.LABS]
                     .melt("time", var_name="lab", value_name="value")
                     .dropna(subset=["value"]))
        if labs_long.empty:
            st.caption("No labs drawn for this patient — a missing lab is itself signal.")
        else:
            st.altair_chart(
                alt.Chart(labs_long).mark_line(point=True).encode(
                    x=alt.X("time:Q", title="minutes since arrival"),
                    y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                    color=alt.Color("lab:N", legend=None),
                    tooltip=["lab:N", alt.Tooltip("time:Q", title="min"),
                             alt.Tooltip("value:Q", format=".1f")])
                .properties(height=110)
                .facet(row=alt.Row("lab:N", title=None, sort=config.LABS))
                .resolve_scale(y="independent"),
                use_container_width=True)

    with right:
        st.subheader("Now")
        for name, val in latest.items():
            st.metric(f"Current risk — {name}", f"{val:.0%}")
        if latest:
            worst = max(latest.values())
            if worst >= threshold:
                st.error(f"ALERT — at/above the {threshold:.0%} threshold")
            else:
                st.success(f"Below the {threshold:.0%} threshold")

        active = [m.replace("on_", "").replace("recent_", "recent ").replace("_", " ")
                  for m in config.MED_FEATURES if frame[m].iloc[-1] > 0]
        if active:
            st.write("**Drugs on board:** " + ", ".join(active))
            if frame["on_vasopressor"].iloc[-1] > 0:
                st.warning("BP is vasopressor-supported — a normal number here isn't "
                           "recovery; the model keeps risk high because it has that context.")

        st.write("**Top RF risk drivers** (importance stand-in for SHAP):")
        for name, imp in top_drivers(rf, k=6):
            st.write(f"- {name}  ({imp:.2f})")

        if patient.deterioration_onset is not None:
            st.write(f"**True onset:** {patient.deterioration_onset:.0f} min "
                     "(ground truth, hidden from the models)")
        else:
            st.write("**Ground truth:** stable — no deterioration")

# =========================================================================== #
# TAB 2 — model benchmark
# =========================================================================== #
with tab_bench:
    st.subheader("Head-to-head — same patients, same split")
    if not manifest:
        st.info("No benchmark.json yet. Run `python benchmark.py --save` "
                "(add `--real` for the real cohort) to populate this tab.")
    else:
        m = manifest["metrics"]
        rows = []
        for name, r in m.items():
            rows.append({
                "model": name,
                "AUPRC": r.get("AUPRC"),
                "AUROC": r.get("AUROC"),
                "Brier": r.get("Brier"),
                "sensitivity": r.get("sensitivity"),
                "lead (min)": r.get("lead_min"),
                "FA/day": r.get("false_alarms_per_day"),
            })
        df = pd.DataFrame(rows).set_index("model")
        cap = (f"Source: **{manifest['source']}**  ·  patients: {manifest.get('n_patients')}"
               f"  ·  test positive rate: {manifest.get('test_positive_rate', 0):.2%}"
               f"  ·  grid: {manifest.get('grid_interval_min')} min")
        st.caption(cap)
        st.dataframe(df.style.format({
            "AUPRC": "{:.3f}", "AUROC": "{:.3f}", "Brier": "{:.3f}",
            "sensitivity": "{:.0%}", "lead (min)": "{:.0f}", "FA/day": "{:.2f}"
        }, na_rep="—"), use_container_width=True)

        bar_df = df.reset_index()[["model", "AUPRC"]].dropna()
        st.altair_chart(
            alt.Chart(bar_df).mark_bar().encode(
                x=alt.X("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("model:N", sort="-x"),
                color=alt.Color("model:N", legend=None),
                tooltip=["model:N", alt.Tooltip("AUPRC:Q", format=".3f")])
            .properties(height=160, title="AUPRC by model (higher = better; leads because events are rare)"),
            use_container_width=True)

        st.markdown(
            "**Reading it honestly.** AUPRC is the headline (deterioration is rare, so "
            "AUROC flatters). Both learned models should beat NEWS2/qSOFA by a wide "
            "margin. Whether the **Transformer** beats the **RandomForest** depends on "
            "scale: at MVP size the calibrated tree usually wins; the transformer's "
            "advantages (time-aware attention, self-supervised pretraining) grow with "
            "data — which is what the real-data run is for (notes, Section 7).")
