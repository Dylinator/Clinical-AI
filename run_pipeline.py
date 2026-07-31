"""
run_pipeline.py — end-to-end entry point.

    python run_pipeline.py

Generate -> resample -> label -> featurize -> split (by patient) -> train ->
evaluate vs NEWS2/qSOFA -> pick an operating point -> save artifacts -> show one
risk trajectory. Everything is seeded, so re-running gives identical numbers.
"""

from __future__ import annotations
import json
import os
import sys

# The report below uses a few non-ASCII characters (→, ≈, ~). On Windows the
# default console encoding is often cp1252, which crashes on them; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import config
import synthetic
import timeline_engine
import labeling
import features
import splits
import model as model_mod
import baselines
import evaluate
import trajectory_engine


def main():
    # 1. Data (seeded, saved once) -------------------------------------------
    patients = synthetic.generate()
    full = timeline_engine.to_frame(patients)
    onsets = timeline_engine.onset_map(patients)
    labeled = labeling.add_labels(full, onsets)

    # 2. Features + patient-level split --------------------------------------
    X, y, meta = features.build_training_matrix(full, labeled)
    train_ids, test_ids = splits.split_patients(meta["patient_id"])
    tr, te = splits.mask_for(meta, train_ids), splits.mask_for(meta, test_ids)

    print(f"patients={len(patients)}  rows={len(X)}  positives={y.mean():.1%}")
    print(f"train rows={tr.sum()}  test rows={te.sum()}\n")

    # 3. Train + persist ------------------------------------------------------
    est = model_mod.fit(X[tr], y[tr])
    model_mod.save(est)
    full.to_csv(config.DATASET_PATH, index=False)
    json.dump(sorted(int(i) for i in test_ids),
              open(f"{config.ARTIFACT_DIR}/test_ids.json", "w"))   # dashboard shows only these

    # 4. Evaluate model vs baselines (identical test split) ------------------
    risk_te = model_mod.predict_risk(est, X[te])
    news2, qsofa = baselines.score_matrix(X[te])

    print(f"{'model':<18}{'AUPRC':>8}{'AUROC':>8}{'Brier':>8}")
    for name, prob in [("our model", risk_te), ("NEWS2", news2), ("qSOFA", qsofa)]:
        m = evaluate.metrics(y[te], prob)
        brier = f"{m['Brier']:.3f}" if "Brier" in m else "   -"
        print(f"{name:<18}{m['AUPRC']:>8.3f}{m['AUROC']:>8.3f}{brier:>8}")
    print(f"(test positive rate = {y[te].mean():.1%} — AUPRC floor for a coin flip)\n")

    # 5. Operating point — model vs baselines at a matched alert budget ------
    #    Section 4c: the result that counts is "beat NEWS2 on lead time at equal
    #    false-alarm rate", not "higher AUC". Each score is thresholded on its
    #    OWN scale at the cutoff closest to the budget, then judged on the same
    #    time-aware metrics (lead time, false alarms/day) over the same patients.
    meta_te = meta[te].reset_index(drop=True)
    print(f"operating point, each near ~{config.TARGET_ALERTS_PER_DAY:.0f} "
          f"alert/patient-day (same test split):")
    print(f"  {'method':<11}{'thresh':>8}{'alerts/d':>10}{'sens':>7}"
          f"{'lead(min)':>11}{'FA/day':>8}")
    ops = {}
    for name, scores, grid in [("our model", risk_te, None),
                               ("NEWS2", news2, np.unique(news2)),
                               ("qSOFA", qsofa, np.unique(qsofa))]:
        thr = evaluate.choose_threshold(scores, meta_te, grid=grid)
        op = evaluate.operating_point(scores, meta_te, onsets, thr)
        ops[name] = op
        print(f"  {name:<11}{op['threshold']:>8.2f}{op['alerts_per_day']:>10.2f}"
              f"{op['sensitivity']:>7.0%}{op['mean_lead_time_min']:>11.0f}"
              f"{op['false_alarms_per_day']:>8.2f}")
    m_op = ops["our model"]
    print(f"  ({m_op['n_cases']} cases / {m_op['n_controls']} controls in test)")
    json.dump({"threshold": m_op["threshold"]},
              open(f"{config.ARTIFACT_DIR}/threshold.json", "w"))
    print()

    # 6. Fairness audit across SES strata (Section 10) -----------------------
    #    Low SES was generated to BOTH raise risk AND degrade data quality (more
    #    missing readings). At the same threshold, watch whether the low stratum
    #    is detected less often — the harm the audit exists to catch.
    def _stratum(s):
        return "low" if s <= 3 else "high" if s >= 8 else "mid"
    ses_group = {p.id: _stratum(p.ses) for p in patients}
    groups = evaluate.group_report(y[te], risk_te, meta_te, onsets, ses_group,
                                    m_op["threshold"])
    print("fairness audit — performance by SES stratum (same threshold):")
    print(f"  {'stratum':<8}{'cases':>7}{'AUPRC':>8}{'sens':>7}{'FA/day':>8}{'Brier':>8}")
    for name in ["low", "mid", "high"]:
        r = groups.get(name)
        if r:
            print(f"  {name:<8}{r['n_cases']:>7}{r['AUPRC']:>8.3f}{r['sensitivity']:>7.0%}"
                  f"{r['false_alarms_per_day']:>8.2f}{r['Brier']:>8.3f}")
    print(f"  (SES used as a model feature: {config.USE_SES_AS_FEATURE} — flip "
          f"USE_SES_AS_FEATURE in config.py and re-run to compare)\n")

    # 7. One trajectory — prefer a treated patient so the med context shows -----
    cases = [p for p in patients if p.id in test_ids and p.outcome == "sepsis"]
    treated = [p for p in cases if any(iv.type == "vasopressor" for iv in p.interventions)]
    demo = (treated or cases)[0]
    dframe = timeline_engine.to_frame([demo])
    traj = trajectory_engine.trajectory(dframe, est)
    on_vaso = dframe.set_index("time")["on_vasopressor"].to_dict()
    print(f"example test patient #{demo.id} (onset {demo.deterioration_onset:.0f} min):")
    if demo.interventions:
        seq = sorted(demo.interventions, key=lambda iv: iv.time)
        print("  treatment: " + ", ".join(f"{iv.type}@{iv.time:.0f}" for iv in seq))
    for r in traj.itertuples(index=False):
        mark = "   <- BP is drug-supported (pressor)" if on_vaso.get(r.time) else ""
        print(f"  t={r.time:3.0f}  risk={r.risk:5.1%}  {'#' * int(r.risk * 40)}{mark}")

    print(f"\nartifacts written to {config.ARTIFACT_DIR}/  →  python -m streamlit run dashboard.py")


if __name__ == "__main__":
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    main()
