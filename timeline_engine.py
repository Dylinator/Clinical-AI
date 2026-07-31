"""
timeline_engine.py — Phase 2, resampling ONLY (Rule 4).

Turns a list[Patient] into one tidy DataFrame on a uniform time grid, keyed by
(patient_id, time). It does NOT build labels (that's labeling.py) and it does NOT
impute missing values (that's features.py / the model pipeline). Keeping those
concerns out of here is deliberate: the label is the most error-prone thing in
the project, and mixing it into resampling makes it hard to change and test.

Missing values stay as NaN on purpose — the missingness itself is signal.
"""

from __future__ import annotations
import pandas as pd

from schema import Patient
import config


def _med_flags(interventions, t) -> dict:
    """Medication state at time t, from interventions given at or before t
    (past-only by construction)."""
    return {
        "on_vasopressor": int(any(iv.type == "vasopressor" and t >= iv.time for iv in interventions)),
        "on_antibiotics": int(any(iv.type == "antibiotics" and t >= iv.time for iv in interventions)),
        "recent_fluid":   int(any(iv.type == "fluids" and 0 <= t - iv.time <= 60 for iv in interventions)),
        "on_antipyretic": int(any(iv.type == "antipyretic" and 0 <= t - iv.time <= 120 for iv in interventions)),
    }


def to_frame(patients: list[Patient]) -> pd.DataFrame:
    """list[Patient] -> tidy frame with columns [patient_id, time, *VITALS]."""
    rows = []
    for p in patients:
        for o in p.observations:
            row = {
                "patient_id": p.id,
                "time": o.time,
                # vitals
                "hr": o.hr, "rr": o.rr, "sbp": o.sbp, "o2": o.o2, "temp": o.temp,
                # labs (lactate + SOFA-core)
                "lactate": o.lactate, "creatinine": o.creatinine,
                "wbc": o.wbc, "platelets": o.platelets,
                # static
                "age": p.age, "sex": p.sex,
                "comorbidity_count": p.comorbidity_count,
                "prior_complications": p.prior_complications,
                "ses": p.ses,
            }
            row.update(_med_flags(p.interventions, o.time))
            rows.append(row)
    cols = ["patient_id", "time"] + config.CHANNELS + config.STATIC_ALL + config.MED_FEATURES
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["patient_id", "time"]).reset_index(drop=True)


def onset_map(patients: list[Patient]) -> dict[int, float | None]:
    """patient_id -> deterioration onset time (or None). Used by labeling."""
    return {p.id: p.deterioration_onset for p in patients}


if __name__ == "__main__":
    from synthetic import generate
    df = to_frame(generate())
    print(df.head(10).to_string(index=False))
    print(f"\nrows={len(df)}  patients={df.patient_id.nunique()}  "
          f"lactate missing={df.lactate.isna().mean():.0%}")
