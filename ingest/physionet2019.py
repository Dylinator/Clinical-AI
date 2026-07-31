"""
ingest/physionet2019.py — the REAL-data adapter (Rule 6, the notes' `ingest/mimic.py` idea).

The PhysioNet/CinC Challenge 2019 sepsis dataset is real, de-identified ICU
time-series — one row per hour of vitals + labs + a forward-looking Sepsis-3
label — and, unlike MIMIC-IV / eICU proper, it is openly downloadable WITHOUT
credentialing. It contains exactly the channels this project models (the SOFA-core
labs creatinine / WBC / platelets, plus lactate), which is why it's the natural
first real dataset here.

This module does the one job an adapter is allowed to do: turn the source files
into the SAME `Patient` / `Observation` objects the synthetic generator emits, so
everything downstream (labeling, features, splits, model, evaluation, the
transformer) runs UNCHANGED on real data. Nothing here builds features or labels
— that stays in features.py / labeling.py (one inference path, Rule 1).

Two source-specific decisions, documented so they can't be silently wrong:

  * TIME. The source is hourly (column ICULOS). We emit `time` in MINUTES so the
    same config horizons apply, and start each stay at t=0. Because rows are
    hourly, a real run sets config.GRID_INTERVAL_MIN = 60 (see run_real.py) so the
    alert-rate / false-alarms-per-day math is correct.
  * ONSET. The challenge's `SepsisLabel` is 1 starting 6 h BEFORE the Sepsis-3
    onset. We reconstruct the onset itself ( first hour with SepsisLabel==1, + 6 h )
    and hand THAT to our own labeling.py, so the label is built one way for both
    synthetic and real data — with our horizon/gap, not the challenge's.

Fields the source does NOT provide (comorbidity_count, prior_complications, ses,
interventions) are set to neutral constants / empty, documented as carrying no
signal for real patients — they exist only so the synthetic-tuned feature list
still populates without NaNs. Age and sex ARE real.

Usage:
    python -m ingest.physionet2019            # download a subset + preview
    from ingest.physionet2019 import download, load
    paths = download(n=800)                    # cached under data/physionet2019/
    patients = load(limit=800)                 # -> list[Patient]
"""

from __future__ import annotations
import os
import math
import concurrent.futures as cf
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

from schema import Observation, Patient

# ---- source layout ---------------------------------------------------------- #
_BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
_SET_PREFIX = {"A": ("training_setA", 1), "B": ("training_setB", 100001)}
DATA_DIR = os.path.join("data", "physionet2019")

# Source column -> our schema field (only the channels we model; the rest ignored).
_VITAL_MAP = {"HR": "hr", "Resp": "rr", "SBP": "sbp", "O2Sat": "o2", "Temp": "temp"}
_LAB_MAP = {"Lactate": "lactate", "Creatinine": "creatinine",
            "WBC": "wbc", "Platelets": "platelets"}

# Only these columns are parsed (of the source's 41) — a big read speedup at scale.
_USECOLS = set(_VITAL_MAP) | set(_LAB_MAP) | {"Age", "Gender", "ICULOS", "SepsisLabel"}

# The challenge marks SepsisLabel=1 starting this many hours BEFORE true onset.
_LABEL_LEAD_HOURS = 6


# --------------------------------------------------------------------------- #
# Download (cached, resumable, polite)
# --------------------------------------------------------------------------- #
def _pid_name(set_key: str, i: int) -> str:
    _, start = _SET_PREFIX[set_key]
    return f"p{start + i - 1:06d}.psv"


def _fetch_one(set_key: str, i: int, dest_dir: str, timeout: float = 30.0):
    """Download one patient file if not already cached. Returns local path or None
    (None = the id doesn't exist, i.e. a 404 — expected past the end of a set)."""
    folder, _ = _SET_PREFIX[set_key]
    name = _pid_name(set_key, i)
    local = os.path.join(dest_dir, folder, name)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    url = f"{_BASE}/{folder}/{name}"
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        if not data:
            return None
        with open(local, "wb") as f:
            f.write(data)
        return local
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None


def download(n: int = 800, set_key: str = "A", dest_dir: str = DATA_DIR,
             workers: int = 10) -> list[str]:
    """Download up to `n` patient files from training set A (or B), cached under
    dest_dir. Idempotent: re-running skips files already present. Returns the list
    of local .psv paths actually available (existing ids only)."""
    # Try a bit more than n ids, since a few may not exist near a set's end.
    attempts = list(range(1, int(n * 1.15) + 2))
    paths: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, set_key, i, dest_dir): i for i in attempts}
        for fut in cf.as_completed(futs):
            p = fut.result()
            if p:
                paths.append(p)
            if len(paths) >= n:
                break
        for fut in futs:            # stop any stragglers once we have enough
            fut.cancel()
    paths.sort()
    return paths[:n]


# --------------------------------------------------------------------------- #
# Parse -> schema objects
# --------------------------------------------------------------------------- #
def _to_none(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def _parse_psv(path: str, pid: int, max_hours: int | None = None) -> Patient | None:
    """One .psv (pipe-separated, hourly) -> one Patient of schema Observations.

    `max_hours` caps the stay to its first N hours. Real ICU stays run to hundreds
    of hours; since featurization is O(stay^2), an uncapped cohort is intractable.
    Capping is also clinically apt — the project targets the FIRST 24–48 h of
    critical care (notes, Section 0). The onset stays the true (uncapped) value, so
    labels remain correct: a patient whose onset falls beyond the cap simply
    contributes its early, pre-onset rows as negatives (never a mislabel)."""
    df = pd.read_csv(path, sep="|", usecols=lambda c: c in _USECOLS)
    if df.empty or "ICULOS" not in df.columns:
        return None
    if max_hours is not None and len(df) > max_hours:
        df = df.iloc[:max_hours]

    # Relative minutes since the stay's first row (source is hourly).
    iculos0 = float(df["ICULOS"].iloc[0])
    t_min = (df["ICULOS"].astype(float) - iculos0) * 60.0

    obs_list: list[Observation] = []
    for j in range(len(df)):
        kw = {}
        for src, dst in {**_VITAL_MAP, **_LAB_MAP}.items():
            kw[dst] = _to_none(df[src].iloc[j]) if src in df.columns else None
        obs_list.append(Observation(time=float(t_min.iloc[j]), **kw))

    # Onset from SepsisLabel: 1 begins _LABEL_LEAD_HOURS before true onset, so
    # reconstruct the onset and let OUR labeling.py apply OUR horizon/gap.
    onset = None
    outcome = "stable"
    if "SepsisLabel" in df.columns:
        pos = np.flatnonzero(df["SepsisLabel"].to_numpy() == 1)
        if pos.size:
            outcome = "sepsis"
            onset = float(t_min.iloc[int(pos[0])]) + _LABEL_LEAD_HOURS * 60.0

    age = _to_none(df["Age"].iloc[0]) if "Age" in df.columns else None
    sex = df["Gender"].iloc[0] if "Gender" in df.columns else None
    sex = int(sex) if (sex is not None and not (isinstance(sex, float) and math.isnan(sex))) else None

    return Patient(
        id=pid,
        observations=obs_list,
        interventions=[],                 # not in this source (Phase-7 flags -> 0)
        outcome=outcome,
        deterioration_onset=onset,
        age=age, sex=sex,
        # Not in the source; neutral constants so the synthetic-tuned static
        # feature list still populates without NaNs (they carry no real signal).
        comorbidity_count=0, prior_complications=0, ses=6,
    )


def load(data_dir: str = DATA_DIR, limit: int | None = None,
         download_if_missing: bool = True, max_hours: int | None = 72) -> list[Patient]:
    """Read cached .psv files under data_dir into a list[Patient]. If nothing is
    cached and download_if_missing, fetch a subset first. `max_hours` caps each
    stay (default 72 h — see _parse_psv; pass None for the full, slow stay)."""
    psvs: list[str] = []
    for root, _, files in os.walk(data_dir):
        psvs += [os.path.join(root, f) for f in files if f.endswith(".psv")]
    psvs.sort()
    if not psvs and download_if_missing:
        download(n=limit or 800, dest_dir=data_dir)
        return load(data_dir, limit=limit, download_if_missing=False, max_hours=max_hours)
    if limit:
        psvs = psvs[:limit]

    patients: list[Patient] = []
    for pid, path in enumerate(psvs, start=1):
        try:
            p = _parse_psv(path, pid, max_hours=max_hours)
        except Exception:
            p = None
        if p is not None and len(p.observations) >= 2:   # need a trend
            patients.append(p)
    return patients


if __name__ == "__main__":
    import config
    try:
        sys_stdout = __import__("sys").stdout
        sys_stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("downloading a subset of PhysioNet 2019 (cached under data/) ...")
    paths = download(n=int(os.environ.get("N", "300")))
    print(f"  {len(paths)} patient files available")
    patients = load(limit=len(paths), download_if_missing=False)
    n_sep = sum(p.outcome == "sepsis" for p in patients)
    print(f"parsed {len(patients)} patients — {n_sep} sepsis "
          f"({n_sep / max(1, len(patients)):.1%}), {len(patients) - n_sep} stable")
    demo = next((p for p in patients if p.outcome == "sepsis"), patients[0])
    print(f"\nexample patient id={demo.id} outcome={demo.outcome} "
          f"onset={demo.deterioration_onset} age={demo.age} sex={demo.sex} "
          f"n_obs={len(demo.observations)}")
    for o in demo.observations[:6]:
        def s(x): return "  --" if x is None else f"{x:6.1f}"
        print(f"  t={o.time:5.0f}  hr={s(o.hr)} sbp={s(o.sbp)} o2={s(o.o2)} "
              f"temp={s(o.temp)} lac={s(o.lactate)} creat={s(o.creatinine)} "
              f"wbc={s(o.wbc)} plt={s(o.platelets)}")
