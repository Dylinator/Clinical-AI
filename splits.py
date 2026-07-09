"""
splits.py — patient-level train/test split (Rule 2, Section 4b).

Split by PATIENT, never by row. The same patient's rows are highly correlated
(they're the same person minutes apart); putting some in train and some in test
leaks information and inflates every score. This is the most common way a
deterioration model looks great in a notebook and fails in reality.
"""

from __future__ import annotations
import numpy as np

import config


def split_patients(patient_ids, test_frac: float = 0.25, seed: int = config.SEED):
    """Return (train_ids, test_ids) as sets, partitioned by unique patient."""
    uniq = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    test_ids = set(uniq[:n_test].tolist())
    train_ids = set(uniq[n_test:].tolist())
    return train_ids, test_ids


def mask_for(meta, ids):
    """Boolean mask selecting rows whose patient_id is in `ids`."""
    return meta["patient_id"].isin(ids).to_numpy()
