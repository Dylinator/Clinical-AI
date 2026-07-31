"""
features.py — the ONE feature builder (Rule 1, the most important rule).

`featurize_history` is a pure, past-only function. Training (build_training_matrix)
and live scoring (trajectory_engine) both call it, so the model can never be fed
one thing at fit time and a different thing at predict time (train/serve skew).

Feature vector (order fixed in config.FEATURES), over every CHANNEL (vitals + labs):
    <channel>_now  : last observed value, carried forward (BASELINE_FILL if never seen)
    <channel>_delta: change since the previous observation
    <channel>_slope: slope over the last SLOPE_WINDOW_MIN minutes, per minute
    <lab>_missing  : 1 if that lab is unmeasured at t (informative missingness),
                     one flag per lab in config.LABS

Imputation policy lives here and nowhere else (see the notes, Data model):
carry-forward the last observation, fall back to a baseline only before the first
reading. The model pipeline therefore receives a fully-populated vector.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config


def featurize_history(hist: pd.DataFrame) -> dict:
    """Rows for ONE patient with time <= t (sorted). Returns the feature dict."""
    t_now = float(hist["time"].to_numpy()[-1])
    times = hist["time"].to_numpy(dtype=float)
    feats: dict[str, float] = {}

    for v in config.CHANNELS:                     # vitals + labs, same treatment
        col = hist[v].to_numpy(dtype=float)
        seen = ~np.isnan(col)
        obs_t, obs_v = times[seen], col[seen]

        if obs_v.size == 0:                       # never measured yet
            now, delta, slope = config.BASELINE_FILL[v], 0.0, 0.0
        else:
            now = obs_v[-1]                        # carry-forward
            delta = obs_v[-1] - obs_v[-2] if obs_v.size >= 2 else 0.0
            win = obs_t >= (t_now - config.SLOPE_WINDOW_MIN)
            wt, wv = obs_t[win], obs_v[win]
            slope = (wv[-1] - wv[0]) / (wt[-1] - wt[0]) if wv.size >= 2 and wt[-1] > wt[0] else 0.0

        feats[f"{v}_now"] = float(now)
        feats[f"{v}_delta"] = float(delta)
        feats[f"{v}_slope"] = float(slope)

    # One informative-missingness indicator per lab (labs are ordered, not
    # continuously monitored — whether a lab exists at t is itself signal).
    for lab in config.LABS:
        feats[f"{lab}_missing"] = float(np.isnan(hist[lab].to_numpy(dtype=float)[-1]))

    # Static / demographic / history features — constant per patient, so they
    # can't leak; included only when config turns them on.
    for name in config.STATIC_ALL:
        if name in config.FEATURES:
            feats[name] = float(hist[name].to_numpy()[-1])

    # Medication-context features — time-varying flags. Past-only: the row at t
    # already reflects only drugs given at or before t (timeline_engine._med_flags).
    if config.USE_MED_FEATURES:
        for name in config.MED_FEATURES:
            feats[name] = float(hist[name].to_numpy()[-1])

    return feats


def featurize_at(patient_df: pd.DataFrame, t: float) -> dict:
    """Slice a patient's frame to rows <= t, then featurize. Past-only by design.
    (The training hot loop in build_training_matrix does this slice by binary
    search; this stays the simple, obviously-correct form for the trajectory
    engine and the leakage test, which are not performance-critical.)"""
    return featurize_history(patient_df[patient_df["time"] <= t])


def build_training_matrix(full_df: pd.DataFrame, labeled_df: pd.DataFrame):
    """
    full_df    : every observation (with NaNs) — the history to featurize from.
    labeled_df : the (patient_id, time, label) rows to actually produce.

    Returns (X, y, meta) with X columns in config.FEATURES order.
    """
    # Cache each patient's sorted frame AND its time array once, so the per-row
    # loop below slices by binary search instead of re-masking the whole frame at
    # every t (the O(N^2) constant factor the config comment flags). The feature
    # computation still goes through the ONE shared featurize_history (Rule 1).
    by_patient = {}
    for pid, g in full_df.groupby("patient_id"):
        g = g.sort_values("time")
        by_patient[pid] = (g, g["time"].to_numpy())

    feat_rows = []
    for r in labeled_df.itertuples(index=False):
        g, times = by_patient[r.patient_id]
        cut = int(np.searchsorted(times, r.time, side="right"))
        feat_rows.append(featurize_history(g.iloc[:cut]))

    X = pd.DataFrame(feat_rows).reindex(columns=config.FEATURES)
    y = labeled_df["label"].reset_index(drop=True)
    meta = labeled_df[["patient_id", "time"]].reset_index(drop=True)
    return X, y, meta


if __name__ == "__main__":
    from synthetic import generate
    from timeline_engine import to_frame, onset_map
    from labeling import add_labels
    ps = generate()
    full = to_frame(ps)
    lab = add_labels(full, onset_map(ps))
    X, y, meta = build_training_matrix(full, lab)
    print(f"X shape={X.shape}  positives={y.mean():.1%}")
    print(X.head().to_string(index=False))
