"""
synthetic.py — Phase 1 synthetic patient generator (the "synthetic adapter").

Replaces the earlier one-hardcoded-patient stub. This emits many patients as
schema objects, with the four things that make the later ML *meaningful* rather
than circular (see "The synthetic-data circularity trap" in the notes):

    - realistic measurement noise
    - heterogeneity (patients deteriorate at different rates / onsets)
    - red herrings + class overlap (so the classes are not trivially separable)
    - sparse, slightly-informative missingness on the lactate lab

It also emits a `deterioration_onset` time, which is what labeling.py needs to
build a forward-looking target.

Reminder (bottom of the old stub was right): data from here is for building the
pipeline/dashboard and demonstrating system design. It is NOT clinical evidence —
the generator knows the answer by construction.

Run `python synthetic.py` to eyeball a few patients.
"""

from __future__ import annotations
import math
import numpy as np

import config
from schema import Observation, Intervention, Patient


def _grid() -> np.ndarray:
    return np.arange(0, config.STAY_MINUTES + 1, config.GRID_INTERVAL_MIN)


def _clip(name: str, value: float) -> float:
    lo, hi = {
        "hr": (30, 220), "rr": (4, 60), "sbp": (40, 260),
        "o2": (70, 100), "temp": (33.0, 42.0),
        "lactate": (0.2, 20.0), "creatinine": (0.2, 15.0),
        "wbc": (0.5, 60.0), "platelets": (10.0, 700.0),
    }[name]
    return float(np.clip(value, lo, hi))


def _concern_score(row: dict, gen: "config.GenConfig") -> int:
    """How many concerning vital signs are currently OBSERVABLE — fever,
    tachycardia, hypotension. This is the picture that makes a clinician order a
    lactate. Unmeasured (None) vitals can't be reacted to."""
    score = 0
    temp, hr, sbp = row.get("temp"), row.get("hr"), row.get("sbp")
    if temp is not None and temp >= gen.concern_temp:
        score += 1
    if hr is not None and hr >= gen.concern_hr:
        score += 1
    if sbp is not None and sbp <= gen.concern_sbp:
        score += 1
    return score


def generate(gen: config.GenConfig = config.GEN, seed: int = config.SEED) -> list[Patient]:
    """Return a reproducible list of synthetic Patients."""
    rng = np.random.default_rng(seed)
    times = _grid()
    patients: list[Patient] = []

    # Logistic-risk intercept, set so the overall positive rate lands near
    # gen.sepsis_rate for an average patient (median SES, mean age/comorbidity,
    # 50/50 sex). Each factor's *mean* contribution is subtracted here, so adding
    # a new factor shifts individuals' risk without moving the base rate.
    intercept = (math.log(gen.sepsis_rate / (1 - gen.sepsis_rate))
                 - gen.risk_comorbidity * gen.comorbidity_lambda
                 - gen.risk_prior * gen.prior_lambda
                 - gen.risk_sex * 0.5)

    for pid in range(1, gen.n_patients + 1):
        # Static / demographic / history attributes.
        age = float(np.clip(rng.normal(gen.age_mean, gen.age_sd), 18, 95))
        sex = int(rng.random() < 0.5)
        comorbidity = int(min(rng.poisson(gen.comorbidity_lambda), 8))
        prior = int(min(rng.poisson(gen.prior_lambda), 6))
        ses = int(rng.integers(1, 11))                 # 1..10 decile, 1 = lowest

        # Each patient's sepsis chance scales from the base intercept by a sum of
        # risk-factor contributions. Every variable gets its own labelled line —
        # this dict is the "room" for more: add a line here + a coefficient in
        # config.GenConfig and the new variable is live.
        risk_terms = {
            "age":         gen.risk_age * (age - gen.age_mean) / 10.0,   # per decade
            "comorbidity": gen.risk_comorbidity * comorbidity,
            "prior":       gen.risk_prior * prior,
            "sex":         gen.risk_sex * sex,            # male; modest, evidence mixed
            "ses":         gen.risk_ses * (5.5 - ses),    # low SES -> higher
        }
        logit = intercept + sum(risk_terms.values())
        septic = rng.random() < 1.0 / (1.0 + math.exp(-logit))
        onset = float(rng.uniform(*gen.onset_range_min)) if septic else None

        # Lower SES -> less monitoring -> higher chance any reading is missing.
        miss_p = float(np.clip(gen.missing_base + gen.missing_ses * max(0.0, 5.5 - ses),
                               0.0, 0.85))

        # Treatment sequence (Phase 7): treated septic patients get fluids as early
        # signs appear (in the prodrome), then antibiotics + an antipyretic, and a
        # vasopressor if they progress to shock. These times are stored as
        # interventions AND used below to MASK specific vitals.
        interventions = []
        t_fluids = t_abx = t_antipyretic = t_vaso = None
        if septic and rng.random() < gen.treat_rate:
            t_fluids = max(0.0, onset - float(rng.uniform(30, 90)))
            t_abx = t_fluids + float(rng.uniform(10, 40))
            t_antipyretic = max(0.0, t_abx + float(rng.uniform(-10, 20)))
            interventions += [
                Intervention(time=t_fluids, type="fluids"),
                Intervention(time=t_abx, type="antibiotics"),
                Intervention(time=t_antipyretic, type="antipyretic"),
            ]
            if rng.random() < gen.vasopressor_rate:
                t_vaso = onset + float(rng.uniform(0, 60))
                interventions.append(Intervention(time=t_vaso, type="vasopressor"))

        # Per-patient healthy baseline.
        base = {v: rng.normal(mu, sd) for v, (mu, sd) in gen.baseline.items()}

        # Per-patient drift slopes (heterogeneous), sign preserved.
        if septic:
            drift = {}
            for v, (mu, sd) in gen.drift_per_min.items():
                s = rng.normal(mu, sd)
                s = abs(s) if mu > 0 else -abs(s)   # keep the clinical direction
                drift[v] = s
        else:
            drift = {v: 0.0 for v in gen.baseline}

        # Deterioration begins subtly `prodrome_min` before the overt onset. That
        # early trend is the signal a trend-based model can catch and a single-
        # timepoint score cannot ("early deterioration is visible in trends").
        # Labs LEAD — their derangement starts even earlier (lab_prodrome_min).
        prod_start = (onset - gen.prodrome_min) if septic else None
        lab_prod_start = (onset - gen.lab_prodrome_min) if septic else None

        # Some controls transiently look bad (red herrings -> false-alarm pressure).
        herring = (not septic) and (rng.random() < gen.red_herring_rate)
        h_center = float(rng.uniform(120, 360)) if herring else None
        h_width = 60.0

        obs_list: list[Observation] = []
        for t in times:
            row = {}
            # Vitals use the standard prodrome; labs LEAD with an earlier one, so
            # lab trends carry the earliest pre-onset signal the model can catch.
            vital_effect_t = max(0.0, t - prod_start) if septic else 0.0
            lab_effect_t = max(0.0, t - lab_prod_start) if septic else 0.0
            for v in gen.baseline:
                val = base[v]
                if septic:
                    effect_t = lab_effect_t if v in config.LABS else vital_effect_t
                    val = val + drift[v] * effect_t
                if herring:
                    bump = np.exp(-((t - h_center) ** 2) / (2 * h_width ** 2))
                    # transient tachycardia / tachypnea / mild hypotension
                    val += {"hr": 18, "rr": 5, "sbp": -12}.get(v, 0.0) * bump
                # Medication MASKING: move a specific vital without touching the
                # illness — a propped-up number, not real recovery.
                if v == "sbp":
                    if t_fluids is not None and t >= t_fluids:
                        val += gen.fluid_sbp_boost * math.exp(-(t - t_fluids) / gen.fluid_decay_min)
                    if t_vaso is not None and t >= t_vaso:
                        val += gen.vaso_sbp_boost
                elif v == "temp":
                    if t_antipyretic is not None and t >= t_antipyretic:
                        val -= gen.antipyretic_temp_drop * math.exp(-(t - t_antipyretic) / gen.antipyretic_decay_min)
                val += rng.normal(0, gen.meas_noise[v])   # measurement noise
                row[v] = _clip(v, val)

            # Monitoring gaps: any VITAL reading may be missing; lower SES -> more
            # gaps. Labs are handled separately below (they are ordered, not
            # continuously monitored).
            for vit in config.VITALS:
                if rng.random() < miss_p:
                    row[vit] = None

            # Labs: a slow per-lab routine schedule PLUS a reactive draw when the
            # OBSERVED vitals look concerning — the doctor sees fever/tachycardia/
            # hypotension and orders labs. Keyed to what's observable, not the
            # hidden onset, so draws fire during the symptomatic prodrome (and
            # sometimes on a red-herring control). SES under-monitoring thins them.
            # A missing lab is therefore itself informative (see config / Sec. 6).
            concern = _concern_score(row, gen)
            for lab in config.LABS:
                drawn = (t % gen.lab_every_min[lab] == 0)
                if concern > 0:
                    drawn = drawn or (rng.random() < gen.lab_order_prob[concern])
                if drawn and rng.random() < miss_p:
                    drawn = False
                if not drawn:
                    row[lab] = None   # missing -> informative (see config)

            obs_list.append(Observation(time=float(t), **row))

        patients.append(Patient(
            id=pid,
            observations=obs_list,
            interventions=interventions,
            outcome="sepsis" if septic else "stable",
            deterioration_onset=onset,
            age=age, sex=sex, comorbidity_count=comorbidity,
            prior_complications=prior, ses=ses,
        ))

    return patients


if __name__ == "__main__":
    ps = generate()
    n_sep = sum(p.outcome == "sepsis" for p in ps)
    print(f"generated {len(ps)} patients — {n_sep} sepsis, {len(ps) - n_sep} stable")
    demo = next(p for p in ps if p.outcome == "sepsis")
    print(f"\nexample septic patient id={demo.id}, onset={demo.deterioration_onset:.0f} min")
    for o in demo.observations[:6]:
        lac = "  --" if o.lactate is None else f"{o.lactate:4.1f}"
        print(f"  t={o.time:3.0f}  hr={o.hr:5.1f}  rr={o.rr:4.1f}  "
              f"sbp={o.sbp:5.1f}  o2={o.o2:4.1f}  temp={o.temp:4.1f}  lactate={lac}")
