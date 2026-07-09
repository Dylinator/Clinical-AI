"""
labeling.py — Phase 2, the forward-looking target (Rules 4 and Section 4a).

The label at time t answers: "will this patient cross into deterioration within
the next HORIZON minutes, after a GAP blanking window, using only data up to t?"

This is what makes the whole "risk as a function of time" idea coherent, and it
is where clinical-ML projects most often go subtly wrong. Two things it does on
purpose:

  * GAP (blanking window): rows within GAP minutes of onset are DROPPED, not
    labelled. They are ambiguous ("is it already happening?") and predicting an
    event that has effectively begun is not prediction.
  * rows at/after onset are DROPPED: you don't predict the present or the past.

For a stable patient (no onset) every row is a negative.
"""

from __future__ import annotations
import pandas as pd

import config


def add_labels(
    df: pd.DataFrame,
    onsets: dict[int, float | None],
    horizon: int = config.HORIZON_MIN,
    gap: int = config.GAP_MIN,
) -> pd.DataFrame:
    """Return df with a `label` column added and blanked/past rows removed."""
    onset = df["patient_id"].map(onsets)          # NaN for stable patients
    delta = onset - df["time"]                    # minutes until onset (NaN if stable)
    is_case = onset.notna()

    # Drop rows too close to / after onset (delta <= gap, includes delta <= 0).
    drop = is_case & (delta <= gap)
    label = (is_case & (delta > gap) & (delta <= gap + horizon)).astype(int)

    out = df.loc[~drop].copy()
    out["label"] = label.loc[~drop].values
    return out.reset_index(drop=True)


if __name__ == "__main__":
    from synthetic import generate
    from timeline_engine import to_frame, onset_map
    ps = generate()
    lab = add_labels(to_frame(ps), onset_map(ps))
    print(f"rows kept={len(lab)}  positives={lab.label.mean():.1%}  "
          f"(row-level rate is much lower than the {config.GEN.sepsis_rate:.0%} "
          f"patient-level rate — that imbalance is why AUPRC matters)")
