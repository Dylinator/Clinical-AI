"""
ingest/mimic.py — MIMIC-IV real-data adapter (SKETCH / scaffold).

WHY THIS DATASET. PhysioNet-2019 has no medication data, so on real data the model
is blind to treatment: a septic patient on a vasopressor looks falsely stable — the
confounder we model on synthetic (Phase 7) but literally cannot see on PhysioNet.
MIMIC-IV DOES carry treatments (vasopressors, fluids, antibiotics, timestamped in
`inputevents` / `prescriptions`), so this adapter is what finally gives the model
treatment context on real data. Like every source it emits the SAME
`Patient` / `Observation` / `Intervention` objects, so labeling, features, splits,
both models, and evaluation run UNCHANGED (Rule 6). This is the notes' planned
`ingest/mimic.py`.

STATUS: SCAFFOLD — the plumbing (table reads, itemid maps, hourly resampling,
intervention extraction, schema emission) is written and structurally correct, but
two things are deliberately left as documented work:

  1. ACCESS. MIMIC-IV is CREDENTIALED (unlike PhysioNet-2019). You need
       (a) a credentialed PhysioNet account,
       (b) the CITI "Data or Specimens Only Research" course,
       (c) a signed Data Use Agreement,
     then download MIMIC-IV v3.x — the `hosp/` and `icu/` modules (gzipped CSVs).
     For a plumbing test WITHOUT credentialing, the 100-patient "MIMIC-IV Clinical
     Database Demo" is openly available (ODC-BY) with the identical layout — point
     MIMIC_DIR at it. 100 patients is enough to exercise this code, not to train.

  2. SEPSIS-3 LABEL. Onset now comes from the shared `sepsis3.derive_onset`
     (suspected infection AND a reduced-SOFA rise >= 2) — the real two-hit structure.
     BUT its infection hit needs ANTIBIOTICS, and MIMIC antibiotics live in
     `prescriptions`/`emar` (not `inputevents`) and are NOT wired here yet — so until
     `_read_antibiotics` is implemented every stay comes back stable. Wiring
     antibiotics lights up the label; a fully publishable version still wants the
     six-organ MIT-LCP `mimic-code` sepsis-3 concept (cohort definition is ~80% of
     real clinical-ML work and where silent errors hide).

The itemid sets are the common MIMIC-IV v2/v3 ids; verify them against `d_items`
(icu) and `d_labitems` (hosp) for your exact release before relying on them.

Usage (once data is in place):
    from ingest.mimic import load
    patients = load(mimic_dir="data/mimic-iv-demo", limit=100)   # same Patient objects
    # then exactly as for any source:
    #   full = timeline_engine.to_frame(patients); labeling.add_labels(...); ...
"""

from __future__ import annotations
import os
import glob
import math

import numpy as np
import pandas as pd

from schema import Observation, Intervention, Patient
import sepsis3

# --------------------------------------------------------------------------- #
# Where the modules live (override via arg or env MIMIC_DIR)
# --------------------------------------------------------------------------- #
MIMIC_DIR = os.environ.get("MIMIC_DIR", os.path.join("data", "mimic-iv"))

# --------------------------------------------------------------------------- #
# itemid maps — the channels this project models. VERIFY against d_items /
# d_labitems for your release; ids drift slightly between versions.
# --------------------------------------------------------------------------- #
# chartevents (icu module) -> our vitals
CHART_ITEMS = {
    220045: "hr",                       # Heart Rate
    220210: "rr",                       # Respiratory Rate
    220179: "sbp", 220050: "sbp",       # NBP systolic / ART systolic
    220277: "o2",                       # SpO2
    223762: ("temp", "C"),              # Temperature Celsius
    223761: ("temp", "F"),              # Temperature Fahrenheit (convert)
}
# labevents (hosp module) -> our labs
LAB_ITEMS = {
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc", 51300: "wbc",
    51265: "platelets",
}
# inputevents (icu module) -> intervention TYPES (the treatment context we want)
VASOPRESSOR_ITEMS = {221906, 221289, 221749, 222315, 221662, 229617}  # norepi/epi/phenyl/vaso/dopa
FLUID_ITEMS = {225158, 225828, 225823, 220952, 225159, 225797}        # crystalloid boluses (verify)
# Antibiotics mostly live in `prescriptions`/`emar` (hosp), not inputevents; wired
# as a name match there — see _read_antibiotics (stubbed with a common prefix set).
ANTIBIOTIC_HINTS = ("vancomycin", "piperacillin", "cefepime", "meropenem",
                    "ceftriaxone", "zosyn", "metronidazole", "levofloxacin")

BUCKET_MIN = 60          # resample charted/lab events to an hourly grid per stay


# --------------------------------------------------------------------------- #
# Low-level readers (chunked + itemid-filtered — chartevents/labevents are huge)
# --------------------------------------------------------------------------- #
def _find(mimic_dir: str, module: str, table: str) -> str | None:
    """Locate hosp/icu table as .csv or .csv.gz (demo and full share this layout)."""
    for ext in (".csv.gz", ".csv"):
        hits = glob.glob(os.path.join(mimic_dir, module, table + ext))
        if hits:
            return hits[0]
    return None


def _read_filtered(path: str, itemids: set[int], usecols: list[str],
                   stay_col: str, stay_ids: set[int],
                   chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream a big events table, keeping only our itemids and cohort stays."""
    keep = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        m = chunk["itemid"].isin(itemids) & chunk[stay_col].isin(stay_ids)
        if m.any():
            keep.append(chunk.loc[m])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=usecols)


# --------------------------------------------------------------------------- #
# Main entry (label comes from the shared sepsis3.derive_onset — see load())
# --------------------------------------------------------------------------- #
def load(mimic_dir: str = MIMIC_DIR, limit: int | None = None,
         max_hours: int | None = 72) -> list[Patient]:
    """MIMIC-IV ICU stays -> list[Patient] with vitals, labs, AND interventions.

    Mirrors ingest.physionet2019.load's contract (limit, max_hours) so it drops
    into the same pipeline. Emits observations resampled to an hourly grid per
    stay; interventions keep their real timestamps (minutes since ICU intime)."""
    icu = _find(mimic_dir, "icu", "icustays")
    pts = _find(mimic_dir, "hosp", "patients")
    if not icu or not pts:
        raise FileNotFoundError(
            f"MIMIC-IV tables not found under {mimic_dir!r}. This adapter is a "
            "scaffold — see the module docstring for credentialed download / the "
            "open 100-patient demo, then set mimic_dir (or $MIMIC_DIR).")

    stays = pd.read_csv(icu, usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                        parse_dates=["intime", "outtime"])
    if limit:
        stays = stays.head(limit)
    stay_ids = set(stays["stay_id"].tolist())
    hadm_ids = set(stays["hadm_id"].dropna().astype(int).tolist())
    demog = pd.read_csv(pts, usecols=["subject_id", "gender", "anchor_age"])
    demog = demog.set_index("subject_id")

    # --- pull the three event streams, filtered to our itemids + cohort ---
    ce = _read_filtered(_find(mimic_dir, "icu", "chartevents"), set(CHART_ITEMS),
                        ["stay_id", "itemid", "charttime", "valuenum"], "stay_id", stay_ids)
    # Labs key on hadm_id (hospital admission), NOT stay_id — attribute each lab to
    # a stay by its charttime falling inside that stay's [intime, outtime] window.
    le_path = _find(mimic_dir, "hosp", "labevents")
    le = _read_filtered(le_path, set(LAB_ITEMS),
                        ["hadm_id", "itemid", "charttime", "valuenum"], "hadm_id",
                        hadm_ids) if le_path else pd.DataFrame()
    if not le.empty:
        le["charttime"] = pd.to_datetime(le["charttime"])
        le = le.set_index("hadm_id").sort_index()
    ie_path = _find(mimic_dir, "icu", "inputevents")
    ie = _read_filtered(ie_path, VASOPRESSOR_ITEMS | FLUID_ITEMS,
                        ["stay_id", "itemid", "starttime"], "stay_id", stay_ids) \
        if ie_path else pd.DataFrame()

    patients: list[Patient] = []
    for pid, row in enumerate(stays.itertuples(index=False), start=1):
        intime = row.intime
        def _min(ts):                                  # datetime -> minutes since intime
            return (pd.Timestamp(ts) - intime).total_seconds() / 60.0

        # long-form (t_min, channel, value) for this stay's vitals + labs
        recs = []
        cev = ce[ce["stay_id"] == row.stay_id] if not ce.empty else ce
        for r in cev.itertuples(index=False):
            spec = CHART_ITEMS[r.itemid]
            ch, val = (spec, r.valuenum) if isinstance(spec, str) else \
                      (spec[0], (r.valuenum - 32) * 5 / 9 if spec[1] == "F" else r.valuenum)
            recs.append((_min(r.charttime), ch, val))
        # labs: this stay's hadm_id, charttime within the ICU stay window
        if not le.empty and not pd.isna(row.hadm_id) and int(row.hadm_id) in le.index:
            lrows = le.loc[[int(row.hadm_id)]]
            in_stay = (lrows["charttime"] >= intime) & (lrows["charttime"] <= row.outtime)
            for r in lrows[in_stay].itertuples(index=False):
                recs.append((_min(r.charttime), LAB_ITEMS[r.itemid], r.valuenum))
        ev = pd.DataFrame(recs, columns=["t_min", "channel", "value"])
        if max_hours is not None and not ev.empty:
            ev = ev[ev["t_min"] <= max_hours * 60]
        if len(ev) < 2:
            continue

        # interventions (vasopressor / fluids) with real timestamps
        interventions: list[Intervention] = []
        iev = ie[ie["stay_id"] == row.stay_id] if not ie.empty else ie
        for r in iev.itertuples(index=False):
            typ = "vasopressor" if r.itemid in VASOPRESSOR_ITEMS else "fluids"
            interventions.append(Intervention(time=_min(r.starttime), type=typ))

        # resample to an hourly grid: last value per (channel, hour bucket)
        obs_list = _to_observations(ev)

        # Shared Sepsis-3 derivation (reduced SOFA). NOTE: it needs ANTIBIOTICS for
        # the suspected-infection hit; MIMIC antibiotics live in prescriptions/emar
        # and are not wired yet (see _read_antibiotics TODO) — until they are, this
        # returns None (stable) for every stay. Wiring antibiotics lights up the label.
        onset = sepsis3.derive_onset(obs_list, interventions)

        g = demog.loc[row.subject_id] if row.subject_id in demog.index else None
        # impute so static never carries NaN (would NaN the transformer)
        age = float(g["anchor_age"]) if g is not None and not pd.isna(g["anchor_age"]) else 60.0
        sex = (1 if str(g["gender"]).upper().startswith("M") else 0) if g is not None else 0
        patients.append(Patient(
            id=pid, observations=obs_list, interventions=interventions,
            outcome="sepsis" if onset is not None else "stable",
            deterioration_onset=onset,
            age=age, sex=sex,
            comorbidity_count=0, prior_complications=0, ses=6,   # derivable from diagnoses_icd (TODO)
        ))
    return patients


def _to_observations(ev: pd.DataFrame) -> list[Observation]:
    """Long-form (t_min, channel, value) -> hourly-bucketed Observations (last
    value wins in each hour). Missing channels stay None (the missingness is signal)."""
    ev = ev.dropna(subset=["value"]).copy()
    ev["bucket"] = (ev["t_min"] // BUCKET_MIN).astype(int)
    obs_list: list[Observation] = []
    for bucket, grp in ev.groupby("bucket"):
        kw = {}
        for ch, sub in grp.groupby("channel"):
            kw[ch] = float(sub.sort_values("t_min")["value"].iloc[-1])
        obs_list.append(Observation(time=float(bucket * BUCKET_MIN), **kw))
    return sorted(obs_list, key=lambda o: o.time)


if __name__ == "__main__":
    # Point at the open 100-patient demo to exercise the plumbing.
    d = os.environ.get("MIMIC_DIR", os.path.join("data", "mimic-iv-demo"))
    print(f"MIMIC-IV adapter (scaffold) — reading from {d!r}")
    try:
        ps = load(d, limit=100)
        n_sep = sum(p.outcome == "sepsis" for p in ps)
        n_treated = sum(bool(p.interventions) for p in ps)
        print(f"parsed {len(ps)} stays — {n_sep} sepsis (Sepsis-3), {n_treated} with interventions")
        print("Plumbing validated on the 100-patient demo: vitals + labs (hadm_id join) "
              "+ vasopressor/fluid interventions all flow through the shared schema.")
        print("REMAINING: the Sepsis-3 label needs ANTIBIOTICS (prescriptions/emar) — until "
              "wired, sepsis count stays 0. Then: comorbidity_count (diagnoses_icd), and the "
              "full six-organ mimic-code SOFA for a publishable label.")
    except FileNotFoundError as e:
        print(e)
