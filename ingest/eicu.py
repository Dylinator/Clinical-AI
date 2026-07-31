"""
ingest/eicu.py — eICU Collaborative Research Database adapter (real, OPEN data).

WHY THIS ONE. The eICU-CRD **Demo** is openly downloadable — NO credentialing — and,
unlike PhysioNet-2019, it carries TREATMENTS (vasopressors, fluids, antibiotics in
the `medication` and `treatment` tables). So it's the real dataset a student can
actually get today that also lets the model see treatment context — the confounder
we model on synthetic but couldn't observe on PhysioNet. It emits the SAME
`Patient` / `Observation` / `Intervention` objects as every other source, so
labeling, features, both models, and evaluation run unchanged (Rule 6).

BONUS over MIMIC: eICU is **multi-center** (200+ hospitals), so the same pipeline
supports a cross-site generalization experiment (train on some hospitals, test on
others) — the "external validation" the notes (Section 4b/6) call the real bar.

SCOPE (honest):
  * The DEMO is ~2,500 unit stays — real and treatment-carrying, but small: enough
    to demonstrate the system on real multi-center data, not to train a definitive
    model. The FULL eICU is credentialed (like MIMIC); this adapter reads either.
  * SEPSIS-3 LABEL now comes from the shared `sepsis3.derive_onset` (suspected
    infection via antibiotics AND a SOFA rise >= 2 in the infection window) — the
    real two-hit structure, but over a REDUCED SOFA (coagulation + renal +
    cardiovascular; see sepsis3.py for the omitted organs and the cultures gap). A
    fully publishable label still wants the six-organ mimic-code concept + eICU
    `microLab` cultures + `diagnosis`.
  * SPLIT CAVEAT: one `uniquepid` can have several unit stays. This scaffold makes
    one Patient per stay; for the full set, split by `uniquepid` (not stay) to keep
    a patient out of both train and test.

Schema facts (verified against the demo, not memory):
  * All times are OFFSETS IN MINUTES from unit admission (t=0 = unit admit).
  * `patient.age` is a string, ">89" for the elderly; `gender` is Male/Female.
  * eICU temperature is already Celsius. SBP: noninvasive (`vitalAperiodic`) is the
    common source; arterial (`vitalPeriodic.systemicsystolic`) fills gaps.

Usage:
    python -m ingest.eicu                      # reads data/eicu-crd-demo, previews
    from ingest.eicu import load
    patients = load("data/eicu-crd-demo", limit=2500)   # same Patient objects
"""

from __future__ import annotations
import os
import re

import numpy as np
import pandas as pd

from schema import Observation, Intervention, Patient
import sepsis3

EICU_DIR = os.environ.get("EICU_DIR", os.path.join("data", "eicu-crd-demo"))
BUCKET_MIN = 60          # resample the 5-min vitals to an hourly grid per stay

# --- channel maps (source column -> our schema field) ----------------------- #
# NOTE: `vitalPeriodic.temperature` is charted sparsely in eICU (~5% of hours);
# fuller temperature lives in `nurseCharting` — wiring that is a TODO if the NEWS2
# baseline (which needs temp) matters. Values that ARE present are correct Celsius.
VP_CHANNELS = {"heartrate": "hr", "respiration": "rr", "sao2": "o2",
               "temperature": "temp", "systemicsystolic": "sbp"}
VA_CHANNELS = {"noninvasivesystolic": "sbp"}
# lab.labname (EXACT) -> our lab; excludes "urinary creatinine", "WBC's in urine", …
LAB_NAMES = {"lactate": "lactate", "creatinine": "creatinine",
             "WBC x 1000": "wbc", "platelets x 1000": "platelets"}

# --- intervention keyword matchers (drug names / treatment strings) --------- #
_VASO = re.compile(r"norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine|"
                   r"levophed|vasopressor", re.I)
_ABX = re.compile(r"vancomycin|piperacillin|cefepime|meropenem|ceftriaxone|zosyn|"
                  r"levofloxacin|metronidazole|ciprofloxacin|azithromycin|"
                  r"ampicillin|gentamicin|aztreonam|antibiotic", re.I)
_FLUID = re.compile(r"\bbolus\b|sodium chloride 0\.9|lactated ringer|normal saline|"
                    r"intravenous fluid|fluid resuscitation", re.I)


def _read(name: str, usecols=None) -> pd.DataFrame:
    path = os.path.join(EICU_DIR_CUR[0], f"{name}.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(EICU_DIR_CUR[0], f"{name}.csv")
    return pd.read_csv(path, usecols=usecols)


# module-level current dir (set in load) so _read can find files
EICU_DIR_CUR = [EICU_DIR]


DEFAULT_AGE = 60.0     # impute missing age so static features never carry NaN
                       # (real data has missing ages; unlike vitals, static isn't
                       # carry-forward imputed, and a NaN here NaNs the transformer).


def _parse_age(a) -> float:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return DEFAULT_AGE
    s = str(a).strip()
    if s.startswith(">"):          # ">89"
        return 90.0
    try:
        return float(s)
    except ValueError:
        return DEFAULT_AGE


def _classify(text: str) -> str | None:
    """Map a drug name / treatment string to an intervention TYPE (vaso > abx >
    fluid priority, so a pressor named in a fluid line isn't miscounted)."""
    if not isinstance(text, str):
        return None
    if _VASO.search(text):
        return "vasopressor"
    if _ABX.search(text):
        return "antibiotics"
    if _FLUID.search(text):
        return "fluids"
    return None


def _vitals_long(stay_ids: set[int]) -> pd.DataFrame:
    """vitalPeriodic + vitalAperiodic -> long [patientunitstayid, t_min, channel, value]."""
    frames = []
    vp = _read("vitalPeriodic",
               usecols=["patientunitstayid", "observationoffset"] + list(VP_CHANNELS))
    vp = vp[vp["patientunitstayid"].isin(stay_ids)]
    frames.append(vp.melt(["patientunitstayid", "observationoffset"],
                          var_name="src", value_name="value"))
    va = _read("vitalAperiodic",
               usecols=["patientunitstayid", "observationoffset"] + list(VA_CHANNELS))
    va = va[va["patientunitstayid"].isin(stay_ids)]
    frames.append(va.melt(["patientunitstayid", "observationoffset"],
                          var_name="src", value_name="value"))
    long = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    chan = {**VP_CHANNELS, **VA_CHANNELS}
    long["channel"] = long["src"].map(chan)
    long = long.rename(columns={"observationoffset": "t_min"})
    return long[["patientunitstayid", "t_min", "channel", "value"]]


def _label_extras(stay_ids: set[int]) -> dict:
    """Per-stay label-ONLY series for the Sepsis-3 SOFA (NOT model features, so they
    add label fidelity without adding circularity): bilirubin (liver) and MAP
    (cardiovascular). Returns {sid: {"bilirubin": (times, vals), "map": (times, vals)}}
    with each pair sorted by time (for sepsis3's binary-search carry-forward)."""
    extras: dict = {sid: {} for sid in stay_ids}
    lab = _read("lab", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab = lab[lab["patientunitstayid"].isin(stay_ids) & (lab["labname"] == "total bilirubin")]
    for sid, g in lab.dropna(subset=["labresult"]).sort_values("labresultoffset").groupby("patientunitstayid"):
        extras[sid]["bilirubin"] = (g["labresultoffset"].tolist(), g["labresult"].tolist())
    va = _read("vitalAperiodic", usecols=["patientunitstayid", "observationoffset", "noninvasivemean"])
    va = va[va["patientunitstayid"].isin(stay_ids)].dropna(subset=["noninvasivemean"])
    for sid, g in va.sort_values("observationoffset").groupby("patientunitstayid"):
        extras[sid]["map"] = (g["observationoffset"].tolist(), g["noninvasivemean"].tolist())
    return extras


def _labs_long(stay_ids: set[int]) -> pd.DataFrame:
    lab = _read("lab", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab = lab[lab["patientunitstayid"].isin(stay_ids) & lab["labname"].isin(LAB_NAMES)]
    lab = lab.dropna(subset=["labresult"]).copy()
    lab["channel"] = lab["labname"].map(LAB_NAMES)
    return lab.rename(columns={"labresultoffset": "t_min", "labresult": "value"})[
        ["patientunitstayid", "t_min", "channel", "value"]]


def _interventions_long(stay_ids: set[int]) -> pd.DataFrame:
    """medication + treatment -> long [patientunitstayid, t_min, type]."""
    rows = []
    med = _read("medication", usecols=["patientunitstayid", "drugstartoffset", "drugname"])
    med = med[med["patientunitstayid"].isin(stay_ids)]
    med["type"] = med["drugname"].map(_classify)
    rows.append(med.dropna(subset=["type"]).rename(columns={"drugstartoffset": "t_min"})
                [["patientunitstayid", "t_min", "type"]])
    tr = _read("treatment", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])
    tr = tr[tr["patientunitstayid"].isin(stay_ids)]
    tr["type"] = tr["treatmentstring"].map(_classify)
    rows.append(tr.dropna(subset=["type"]).rename(columns={"treatmentoffset": "t_min"})
                [["patientunitstayid", "t_min", "type"]])
    return pd.concat(rows, ignore_index=True)


def _bucket_wide(events: pd.DataFrame, max_hours: int | None) -> pd.DataFrame:
    """Bucket ALL stays at once (one global groupby, not one per stay — the
    difference between seconds and minutes on the 5-min eICU vitals).

    Long [patientunitstayid, t_min, channel, value] -> wide frame indexed by
    (patientunitstayid, bucket) with one column per channel, last value per hour."""
    ev = events[events["t_min"] >= 0]
    if max_hours is not None:
        ev = ev[ev["t_min"] <= max_hours * 60]
    ev = ev.dropna(subset=["value"]).sort_values("t_min")
    ev = ev.assign(bucket=(ev["t_min"] // BUCKET_MIN).astype(int))
    wide = (ev.groupby(["patientunitstayid", "bucket", "channel"])["value"].last()
              .unstack("channel"))
    return wide


def load(eicu_dir: str = EICU_DIR, limit: int | None = None,
         max_hours: int | None = 72) -> list[Patient]:
    """eICU unit stays -> list[Patient] with vitals, labs, AND interventions
    (vasopressors/fluids/antibiotics). Same (limit, max_hours) contract as the
    other adapters, so it drops straight into the pipeline."""
    EICU_DIR_CUR[0] = eicu_dir
    ppath = os.path.join(eicu_dir, "patient.csv.gz")
    if not os.path.exists(ppath) and not os.path.exists(ppath[:-3]):
        raise FileNotFoundError(
            f"eICU tables not found under {eicu_dir!r}. Download the open demo "
            "(physionet.org/content/eicu-crd-demo) into this dir, or set $EICU_DIR.")

    pt = _read("patient", usecols=["patientunitstayid", "uniquepid", "gender", "age",
                                   "hospitalid", "unitdischargeoffset"])
    if limit:
        pt = pt.head(limit)
    stay_ids = set(pt["patientunitstayid"].tolist())

    vitals = _vitals_long(stay_ids)
    labs = _labs_long(stay_ids)
    events = pd.concat([vitals, labs], ignore_index=True)
    ivs = _interventions_long(stay_ids)
    ivs = ivs[ivs["t_min"].notna() & (ivs["t_min"] >= 0)]

    # ONE global bucketing pass -> wide (stayid, bucket) x channel; then per-stay
    # slices are cheap dict-building (no pandas groupby inside the 2.5k loop).
    wide = _bucket_wide(events, max_hours)
    wide_by_stay = {sid: sub.reset_index(level=0, drop=True)
                    for sid, sub in wide.groupby(level="patientunitstayid")}
    iv_by_stay = dict(tuple(ivs.groupby("patientunitstayid")))
    extras_by_stay = _label_extras(stay_ids)     # bilirubin + MAP for the SOFA label

    patients: list[Patient] = []
    for pid, row in enumerate(pt.itertuples(index=False), start=1):
        sid = row.patientunitstayid
        sub = wide_by_stay.get(sid)
        if sub is None or len(sub) < 2:
            continue

        # wide rows -> Observations (one per hour bucket; drop NaN channels)
        obs_list = []
        for bucket, vals in zip(sub.index.to_numpy(), sub.to_dict("records")):
            kw = {ch: float(v) for ch, v in vals.items() if pd.notna(v)}
            obs_list.append(Observation(time=float(int(bucket) * BUCKET_MIN), **kw))
        obs_list.sort(key=lambda o: o.time)

        interventions = []
        ivg = iv_by_stay.get(sid)
        if ivg is not None:
            interventions = [Intervention(time=float(r.t_min), type=r.type)
                             for r in ivg.itertuples(index=False)]

        onset = sepsis3.derive_onset(obs_list, interventions,   # 4-organ Sepsis-3
                                     extra=extras_by_stay.get(sid))

        patients.append(Patient(
            id=pid, observations=obs_list, interventions=interventions,
            outcome="sepsis" if onset is not None else "stable",
            deterioration_onset=onset,
            age=_parse_age(row.age),
            sex=(1 if str(row.gender).lower().startswith("m") else 0)
                 if isinstance(row.gender, str) else 0,     # default so static isn't NaN
            comorbidity_count=0, prior_complications=0, ses=6,   # derivable from diagnosis (TODO)
        ))
    return patients


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    d = os.environ.get("EICU_DIR", os.path.join("data", "eicu-crd-demo"))
    print(f"eICU adapter — reading from {d!r}")
    try:
        ps = load(d)
        n_sep = sum(p.outcome == "sepsis" for p in ps)
        n_treated = sum(bool(p.interventions) for p in ps)
        by_type = {}
        for p in ps:
            for iv in p.interventions:
                by_type[iv.type] = by_type.get(iv.type, 0) + 1
        print(f"parsed {len(ps)} stays — {n_sep} sepsis(proxy), {n_treated} with interventions")
        print(f"intervention counts by type: {by_type}")
        demo = next((p for p in ps if p.outcome == "sepsis" and p.interventions), ps[0])
        print(f"example stay #{demo.id}: onset={demo.deterioration_onset} age={demo.age} "
              f"sex={demo.sex} n_obs={len(demo.observations)} n_iv={len(demo.interventions)}")
        print("NOTE: onset is a PLACEHOLDER proxy; split by uniquepid for the full set; "
              "swap in a validated Sepsis-3 label before trusting results.")
    except FileNotFoundError as e:
        print(e)
