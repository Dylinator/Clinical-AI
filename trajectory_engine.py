"""
trajectory_engine.py — Phase 4, the core idea: risk as a function of time.

Instead of one prediction, walk a patient's timeline and score each step using
the SAME featurize function the model trained on (Rule 1) and the ONE persisted
pipeline (Rule 10). Because the label was defined per-time-point and
forward-looking (labeling.py), the risk is allowed to legitimately move over time.

`top_drivers` is a lightweight stand-in for per-step SHAP (global RF importances);
swapping in real SHAP here is the Phase 5 upgrade.
"""

from __future__ import annotations
import pandas as pd

import config
import model as model_mod
from features import featurize_at


def trajectory(patient_df: pd.DataFrame, est) -> pd.DataFrame:
    """Return a [time, risk] frame over one patient's whole grid."""
    patient_df = patient_df.sort_values("time")
    feats = [featurize_at(patient_df, t) for t in patient_df["time"]]
    X = pd.DataFrame(feats).reindex(columns=config.FEATURES)
    risk = est.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "patient_id": patient_df["patient_id"].to_numpy(),
        "time": patient_df["time"].to_numpy(),
        "risk": risk,
    })


def top_drivers(est, k: int = 3):
    """Top-k contributing features (global importances, best-effort)."""
    return model_mod.global_importances(est)[:k]


def trajectory_transformer(patient_df: pd.DataFrame, tmodel) -> pd.DataFrame:
    """Risk-over-time from the encoder-only transformer. Same shape as
    `trajectory` (RF), but scores the SEQUENCE up to each t — so the dashboard can
    overlay the two models on one patient. Past-only by the same slice (time <= t).
    """
    import numpy as np
    import sequences as seqmod
    import model_transformer as mt

    pdf = patient_df.sort_values("time")
    times_all = pdf["time"].to_numpy()
    seqs, tvecs, stats = [], [], []
    for t in times_all:
        hist = pdf[pdf["time"] <= t]
        if len(hist) > mt.MAX_LEN:
            hist = hist.iloc[-mt.MAX_LEN:]
        s, tm = seqmod._one_sequence(hist)
        seqs.append(s)
        tvecs.append(tm)
        stats.append(seqmod._static_vector(hist))
    stat = (np.vstack(stats) if stats else np.zeros((0, seqmod.N_STATIC))).astype("float32")
    risk = mt.predict_risk(tmodel, seqs, tvecs, stat)
    return pd.DataFrame({
        "patient_id": pdf["patient_id"].to_numpy(),
        "time": times_all,
        "risk": risk,
    })


if __name__ == "__main__":
    from synthetic import generate
    from timeline_engine import to_frame

    est = model_mod.load()
    ps = generate()
    demo = next(p for p in ps if p.outcome == "sepsis")
    df = to_frame([demo])
    traj = trajectory(df, est)
    print(f"patient {demo.id} (onset {demo.deterioration_onset:.0f} min)")
    for r in traj.itertuples(index=False):
        bar = "#" * int(r.risk * 40)
        print(f"  t={r.time:3.0f}  risk={r.risk:5.1%}  {bar}")
    print("top drivers:", ", ".join(f"{n}" for n, _ in top_drivers(est)))
