"""
test_labeling.py — the forward-looking window is correct (Rule 4).

Checks that, for a known onset, the label is 1 exactly in the window
(onset - gap - horizon, onset - gap], that the blanking window and post-onset
rows are dropped, and that far-future rows are negatives.
Run: `pytest` or `python test_labeling.py`.
"""

import numpy as np
import pandas as pd

from labeling import add_labels


def _frame(onset=200):
    times = np.arange(0, 481, 30)
    df = pd.DataFrame({"patient_id": 1, "time": times})
    for c in ["hr", "rr", "sbp", "o2", "temp", "lactate"]:
        df[c] = 1.0
    return df, {1: onset}


def test_forward_window_and_blanking():
    horizon, gap, onset = 120, 30, 200
    df, onsets = _frame(onset)
    out = add_labels(df, onsets, horizon=horizon, gap=gap)

    kept = set(out["time"])
    positives = set(out.loc[out["label"] == 1, "time"])

    # onset - t in (gap, gap+horizon]  ->  t in [50, 170)  ->  grid {60,90,120,150}
    assert positives == {60, 90, 120, 150}, positives
    # blanking window (delta <= gap) and at/after onset are dropped
    assert 180 not in kept and 210 not in kept and 200 not in kept
    # far-from-onset rows are negatives, not dropped
    assert 0 in kept and 30 in kept and out.loc[out["time"] == 0, "label"].iloc[0] == 0


def test_stable_patient_all_negative():
    times = np.arange(0, 481, 30)
    df = pd.DataFrame({"patient_id": 2, "time": times})
    for c in ["hr", "rr", "sbp", "o2", "temp", "lactate"]:
        df[c] = 1.0
    out = add_labels(df, {2: None})
    assert len(out) == len(times) and out["label"].sum() == 0


if __name__ == "__main__":
    test_forward_window_and_blanking()
    test_stable_patient_all_negative()
    print("labeling tests passed")
