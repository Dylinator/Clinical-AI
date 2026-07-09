"""
baselines.py — the scores you must beat (Rule 5, Section 4c).

NEWS2 and qSOFA are deterministic clinical scores, not models. If a trained model
can't beat these computed from the *same* vitals, it isn't adding value. They're
kept separate from model.py so "did we beat NEWS2?" is a first-class, repeatable
comparison, evaluated on the identical patient split.

Simplifications (fine for a synthetic MVP, worth stating):
  * NEWS2 assumes the patient is alert (consciousness = 0) and on room air.
  * qSOFA drops the mentation component (we don't simulate GCS), so it ranges 0–2.
Scores are computed from the current ("_now") vitals in the feature matrix.
"""

from __future__ import annotations
import numpy as np


def news2_row(hr, rr, sbp, o2, temp) -> int:
    s = 0
    s += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    s += 0 if o2 >= 96 else 1 if o2 >= 94 else 2 if o2 >= 92 else 3
    s += 3 if temp <= 35.0 else 1 if temp <= 36.0 else 0 if temp <= 38.0 else 1 if temp <= 39.0 else 2
    s += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    s += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    return s


def qsofa_row(rr, sbp) -> int:
    return int(rr >= 22) + int(sbp <= 100)


def score_matrix(X):
    """Return (news2, qsofa) arrays aligned to the rows of feature matrix X."""
    hr, rr, sbp, o2, temp = (X[f"{v}_now"].to_numpy()
                             for v in ["hr", "rr", "sbp", "o2", "temp"])
    news2 = np.array([news2_row(*t) for t in zip(hr, rr, sbp, o2, temp)], dtype=float)
    qsofa = np.array([qsofa_row(a, b) for a, b in zip(rr, sbp)], dtype=float)
    return news2, qsofa
