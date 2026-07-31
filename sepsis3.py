"""
sepsis3.py — a Sepsis-3 onset derivation over the shared schema (Rule 6).

Replaces the placeholder onset proxy ("first vasopressor OR lactate>2") the real
adapters shipped with. Sepsis-3 (Singer et al., JAMA 2016) is a TWO-hit definition:

    sepsis = suspected infection  AND  acute organ dysfunction (SOFA rise >= 2)

The proxy captured neither — it fired on a single value, so it both over-called
(a vasopressor for cardiogenic shock ≠ sepsis) and leaked (its trigger was a
feature). This derivation implements the real structure, operationalized the way
the PhysioNet/CinC-2019 challenge and the MIT-LCP `mimic-code` concept do:

  1. Suspected infection (SOI): time of the first ANTIBIOTIC. (Full SOI also
     requires a body-fluid culture within a window of the antibiotic; our schema
     carries antibiotics but not cultures, so antibiotics alone are the SOI proxy —
     documented, and the single biggest fidelity gap here.)
  2. Organ dysfunction: SOFA computed over time; onset is where SOFA rises >= 2
     from baseline within the window [SOI - 48 h, SOI + 24 h].
  3. onset = min(SOI, time-of-SOFA-rise), per the challenge's t_sepsis convention.

HONEST SCOPE — this is a REDUCED-SOFA Sepsis-3. Full SOFA scores six organ systems.
From the schema alone it scores THREE; when an adapter supplies label-only `extra`
series (bilirubin, MAP — NOT model features, so no added circularity) it scores FOUR:
    * Coagulation    — platelets                     (full fidelity)
    * Renal          — creatinine                    (full fidelity)
    * Cardiovascular — vasopressor present, else MAP<70  (reduced: no pressor DOSE)
    * Liver          — bilirubin                     (via `extra`; full fidelity)
Still omitted, because eICU keeps them in tables we don't pull (each is a real
cohort-engineering lift): Respiration (PaO2/FiO2 — FiO2 lives in respiratoryCharting),
CNS (GCS — nurseCharting), and culture-based SOI (antibiotics alone stand in). Baseline
SOFA is assumed 0 (previously-healthy convention). So this is the honest middle: the
correct STRUCTURE over four of six organs — much better than the proxy, not yet the
publishable six-organ `mimic-code` concept with cultures.

Because labeling.py is forward-looking (predict onset with a gap+horizon, past-only
features), using pressor/abx times inside the onset definition does NOT leak: the
labelled rows sit BEFORE onset, where those flags are still 0.
"""

from __future__ import annotations
import bisect

# SOFA sub-scores (standard thresholds; Vincent et al. 1996 / Singer 2016).
def _sofa_coag(platelets: float | None) -> int:
    if platelets is None:
        return 0
    if platelets >= 150: return 0
    if platelets >= 100: return 1
    if platelets >= 50:  return 2
    if platelets >= 20:  return 3
    return 4


def _sofa_renal(creatinine: float | None) -> int:
    if creatinine is None:
        return 0
    if creatinine < 1.2: return 0
    if creatinine < 2.0: return 1
    if creatinine < 3.5: return 2
    if creatinine < 5.0: return 3
    return 4


def _sofa_cardio(vasopressor_active: bool, mean_ap: float | None = None) -> int:
    # Any vasopressor -> 2 (the "on pressors" SOFA threshold; we lack reliable dose
    # to separate 2/3/4). Otherwise MAP < 70 -> 1 when a MAP series is supplied
    # (label-only; not a model feature). MAP refines the old SBP-less version.
    if vasopressor_active:
        return 2
    if mean_ap is not None and mean_ap < 70:
        return 1
    return 0


def _sofa_liver(bilirubin: float | None) -> int:
    if bilirubin is None:
        return 0
    if bilirubin < 1.2: return 0
    if bilirubin < 2.0: return 1
    if bilirubin < 6.0: return 2
    if bilirubin < 12.0: return 3
    return 4


# Sepsis-3 windows (minutes), matching the common operationalization.
SOI_WINDOW_BEFORE = 48 * 60      # SOFA rise counts from 48 h before suspicion …
SOI_WINDOW_AFTER = 24 * 60       # … to 24 h after
SOFA_RISE = 2                    # organ-dysfunction threshold


def _carry_forward_latest(obs_sorted, attr, upto_t):
    """Last non-None value of `attr` at or before upto_t (carry-forward)."""
    val = None
    for o in obs_sorted:
        if o.time > upto_t:
            break
        v = getattr(o, attr, None)
        if v is not None:
            val = v
    return val


def _cf_series(series, upto_t):
    """Carry-forward from an `extra` label-only series (times, vals) — a sorted
    parallel-list pair — via binary search. None before the first value."""
    if not series:
        return None
    times, vals = series
    i = bisect.bisect_right(times, upto_t) - 1
    return vals[i] if i >= 0 else None


def sofa_at(obs_sorted, interventions, t: float, extra: dict | None = None) -> int:
    """Reduced SOFA at time t. Always scores coagulation (platelets) + renal
    (creatinine) + cardiovascular (vasopressor). If `extra` supplies label-only
    series it also scores liver (bilirubin) and refines cardiovascular with MAP —
    a 4-organ SOFA. `extra` = {"bilirubin": (times, vals), "map": (times, vals)}."""
    extra = extra or {}
    plt = _carry_forward_latest(obs_sorted, "platelets", t)
    cr = _carry_forward_latest(obs_sorted, "creatinine", t)
    vaso = any(iv.type == "vasopressor" and iv.time <= t for iv in interventions)
    mean_ap = _cf_series(extra.get("map"), t)
    bili = _cf_series(extra.get("bilirubin"), t)
    return (_sofa_coag(plt) + _sofa_renal(cr)
            + _sofa_cardio(vaso, mean_ap) + _sofa_liver(bili))


def derive_onset(observations, interventions, extra: dict | None = None) -> float | None:
    """Sepsis-3 onset (minutes since arrival) for one patient, or None if the
    patient never meets suspected-infection AND SOFA-rise-in-window.

    `observations`: list[Observation] (schema), `interventions`: list[Intervention].
    `extra`: optional label-only series {"bilirubin": (times, vals), "map": (...)}
    an adapter can supply to score liver + MAP-refined CV (4-organ SOFA). Omit for
    the 3-organ schema-only SOFA.
    """
    abx = sorted(iv.time for iv in interventions if iv.type == "antibiotics")
    if not abx:
        return None                      # no suspected infection -> not sepsis here
    t_soi = abx[0]

    obs = sorted(observations, key=lambda o: o.time)
    if len(obs) < 1:
        return None

    baseline = 0                         # previously-healthy convention (see docstring)
    lo, hi = t_soi - SOI_WINDOW_BEFORE, t_soi + SOI_WINDOW_AFTER
    for o in obs:
        if o.time < lo or o.time > hi:
            continue
        if sofa_at(obs, interventions, o.time, extra) - baseline >= SOFA_RISE:
            return float(min(t_soi, o.time))   # t_sepsis = min(suspicion, dysfunction)
    return None


if __name__ == "__main__":
    # tiny self-check with fake schema-like objects
    from dataclasses import dataclass
    from typing import Optional
    @dataclass
    class O:
        time: float
        platelets: Optional[float] = None
        creatinine: Optional[float] = None
    @dataclass
    class IV:
        time: float
        type: str
    obs = [O(0, 250, 1.0), O(120, 90, 1.0), O(240, 90, 2.5)]   # plt drops, cr rises
    iv = [IV(60, "antibiotics")]
    print("onset (expect ~120, SOFA hits 2 when plt<100):", derive_onset(obs, iv))
    print("no abx -> None:", derive_onset(obs, []))
