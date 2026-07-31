"""
test_leakage.py — the past-only guarantee (Rule 1).

The single most important invariant in the whole project: features at time t must
depend ONLY on rows at or before t. If a future measurement can change the vector,
the model is peeking at the answer. Run: `pytest` or `python test_leakage.py`.
"""

import numpy as np
import pandas as pd

import config
from features import featurize_at


def _patient_frame():
    times = np.arange(0, 121, 30)
    return pd.DataFrame({
        "patient_id": 1,
        "time": times,
        "hr":   [80, 88, 96, 104, 112],
        "rr":   [16, 17, 18, 20, 22],
        "sbp":  [120, 116, 110, 104, 98],
        "o2":   [98, 98, 97, 96, 95],
        "temp": [36.8, 36.9, 37.1, 37.4, 37.8],
        # labs — sparse (NaN where not drawn), same as real ordering
        "lactate":    [1.2, np.nan, 2.1, np.nan, 3.4],
        "creatinine": [1.0, np.nan, np.nan, 1.3, np.nan],
        "wbc":        [8.0, np.nan, np.nan, 12.0, np.nan],
        "platelets":  [250.0, np.nan, np.nan, 180.0, np.nan],
        # static attributes are constant per patient
        "age": 72.0, "sex": 1,
        "comorbidity_count": 3, "prior_complications": 1, "ses": 4,
        # medication-context flags (constant here, all off)
        "on_vasopressor": 0, "on_antibiotics": 0, "recent_fluid": 0, "on_antipyretic": 0,
    })


def test_future_rows_do_not_change_features():
    df = _patient_frame()
    t = 60
    before = featurize_at(df, t)

    tampered = df.copy()
    future = tampered["time"] > t
    for v in config.CHANNELS:               # blow up every future vital AND lab
        tampered.loc[future, v] = 999.0
    after = featurize_at(tampered, t)

    assert before == after, "features at t changed when future rows changed — leakage!"


def test_features_are_complete_and_finite():
    feats = featurize_at(_patient_frame(), 120)
    assert set(feats.keys()) == set(config.FEATURES)
    assert all(np.isfinite(list(feats.values())))


if __name__ == "__main__":
    test_future_rows_do_not_change_features()
    test_features_are_complete_and_finite()
    print("leakage tests passed")
