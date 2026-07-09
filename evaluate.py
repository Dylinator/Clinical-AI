"""
evaluate.py — metrics, threshold selection, and alert-level scoring (Section 4b).

Leads with AUPRC (deterioration is rare, so AUROC flatters), reports calibration
via Brier, and — the part that actually decides ICU adoption — turns a probability
into an operating point: pick the threshold that lands near an alert *budget*, then
measure sensitivity, lead time, and false alarms per patient-day at that point.

The threshold is chosen here, not hardcoded anywhere (Rule 8).

Alerts are counted as EPISODES: consecutive alerting rows collapse into one, so
the alert-rate budget and false-alarm numbers stay stable when you change the grid
resolution (GRID_INTERVAL_MIN). Sensitivity is per-patient (did any pre-onset
alert fire), so it is grid-stable too.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import config


def _first_row_mask(patient_ids) -> np.ndarray:
    """True at the first row of each patient. Assumes rows are grouped by patient
    and time-sorted (which build_training_matrix guarantees)."""
    pid = np.asarray(patient_ids)
    first = np.ones(len(pid), dtype=bool)
    first[1:] = pid[1:] != pid[:-1]
    return first


def _count_episode_starts(alert: np.ndarray, is_first: np.ndarray) -> int:
    """Number of rising edges (False->True) — one per alert episode — not counting
    carries across patient boundaries."""
    alert = np.asarray(alert, dtype=bool)
    if alert.size == 0:
        return 0
    prev = np.empty(alert.shape, dtype=bool)
    prev[0] = False
    prev[1:] = alert[:-1]
    prev = prev & ~is_first          # a patient's first row has no 'previous' row
    return int((alert & ~prev).sum())


def metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "positive_rate": float(y_true.mean()),
    }
    if 0.0 <= y_prob.min() and y_prob.max() <= 1.0:
        out["Brier"] = float(brier_score_loss(y_true, y_prob))
    return out


def choose_threshold(scores, meta,
                     target: float = config.TARGET_ALERTS_PER_DAY,
                     grid=None) -> float:
    """Pick the threshold whose alert-EPISODE rate is closest to `target` per
    patient-day. `meta` supplies patient_id/time so consecutive alerting rows
    collapse into one episode — that's what makes the budget grid-stable.

    `grid` defaults to a fine [0,1] probability grid (model scores). For an
    integer-scale score like NEWS2/qSOFA, pass the score's own values
    (e.g. `np.unique(news2)`) so the cutoff lands on an achievable value.
    """
    scores = np.asarray(scores, dtype=float)
    is_first = _first_row_mask(meta["patient_id"].to_numpy())
    patient_days = (len(scores) * config.GRID_INTERVAL_MIN) / 1440.0
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    grid = np.asarray(grid, dtype=float)
    rates = np.array([_count_episode_starts(scores >= thr, is_first) / patient_days
                      for thr in grid])
    return float(grid[np.argmin(np.abs(rates - target))])


def operating_point(y_prob, meta, onsets, thr) -> dict:
    """Alert-level metrics at threshold `thr`. `onsets`: patient_id -> onset|None.
    Alerts are counted as EPISODES (consecutive alerting rows = one alert)."""
    df = meta.copy()
    df["prob"] = np.asarray(y_prob, dtype=float)
    df["onset"] = df["patient_id"].map(onsets)
    df["alert"] = df["prob"] >= thr

    case_alerted, leads = [], []
    control_episodes, control_rows, n_controls, total_episodes = 0, 0, 0, 0
    for _, g in df.groupby("patient_id"):
        alert = g["alert"].to_numpy()
        starts = int(alert[0]) + int(np.sum(alert[1:] & ~alert[:-1])) if alert.size else 0
        total_episodes += starts
        onset = g["onset"].iloc[0]
        if pd.notna(onset):
            pre = g[g["time"] < onset]
            hit = bool(pre["alert"].any())
            case_alerted.append(hit)
            if hit:
                leads.append(float(onset - pre.loc[pre["alert"], "time"].min()))
        else:
            n_controls += 1
            control_episodes += starts
            control_rows += len(g)

    control_days = (control_rows * config.GRID_INTERVAL_MIN) / 1440.0
    total_days = (len(df) * config.GRID_INTERVAL_MIN) / 1440.0
    return {
        "threshold": float(thr),
        # episode-based alert rate over everyone — verifies two methods are
        # compared at (roughly) the same budget, independent of grid resolution.
        "alerts_per_day": (total_episodes / total_days) if total_days else float("nan"),
        "sensitivity": float(np.mean(case_alerted)) if case_alerted else float("nan"),
        "mean_lead_time_min": float(np.mean(leads)) if leads else float("nan"),
        "false_alarms_per_day": (control_episodes / control_days) if control_days else float("nan"),
        "n_cases": len(case_alerted),
        "n_controls": n_controls,
    }


def group_report(y, scores, meta, onsets, patient_group, thr) -> dict:
    """The fairness audit: performance per group (Section 10 of the notes).

    `patient_group` maps patient_id -> group label. Everything is evaluated at
    the SAME threshold `thr`, so a lower sensitivity for one group means the model
    genuinely detects that group's events less well — not that it was scored
    differently. Returns {group: {AUPRC, sensitivity, false_alarms_per_day, ...}}.
    """
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    g = meta["patient_id"].map(patient_group).to_numpy()

    out = {}
    for name in pd.unique(g):
        m = (g == name)
        if m.sum() == 0:
            continue
        mm = metrics(y[m], scores[m])
        op = operating_point(scores[m], meta[m].reset_index(drop=True), onsets, thr)
        out[name] = {
            "AUPRC": mm.get("AUPRC", float("nan")),
            "Brier": mm.get("Brier", float("nan")),
            "sensitivity": op["sensitivity"],
            "false_alarms_per_day": op["false_alarms_per_day"],
            "n_cases": op["n_cases"],
        }
    return out
