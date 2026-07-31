"""
schema.py — the canonical data model (Rules 2 and 6).

Everything in the pipeline speaks in these objects. The synthetic generator emits
them today; a future MIMIC-IV loader would emit the *same* objects, and nothing
downstream would change (that's the "data source is an adapter" idea).

Field names and units are the ones pinned in config.py. Access is consistent:
attributes, e.g. `patient.observations`, never a mix of dicts and attributes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Observation:
    """One timestamped set of vitals + labs. Any channel may be None (not yet
    measured). Vitals are dense (bedside monitor); labs are sparse and irregular
    (a clinician has to order them) — the missingness itself is signal."""
    time: float                 # minutes since ED arrival
    # Vitals (dense, monitored)
    hr: Optional[float] = None
    rr: Optional[float] = None
    sbp: Optional[float] = None
    o2: Optional[float] = None
    temp: Optional[float] = None
    # Labs (sparse, ordered) — lactate + SOFA-core (creatinine=kidney,
    # wbc=infection, platelets=coagulation). Same shape as vitals so they flow
    # through the identical schema + feature path; distinguished only by being
    # sparse and carrying a missing-indicator (see config.LABS).
    lactate: Optional[float] = None
    creatinine: Optional[float] = None
    wbc: Optional[float] = None
    platelets: Optional[float] = None


@dataclass
class Intervention:
    """A timestamped clinical action. Same event shape as an Observation so the
    Phase 7 treatment-response layer is an addition, not a rewrite (Rule 7)."""
    time: float
    type: str                   # e.g. "fluids", "vasopressors"


@dataclass
class Patient:
    id: int
    observations: list = field(default_factory=list)      # list[Observation]
    interventions: list = field(default_factory=list)     # list[Intervention]
    outcome: str = "stable"                               # "sepsis" | "stable"
    # When the patient first meets deterioration criteria (minutes), or None.
    # This is what lets labeling build a *forward-looking* target — a static
    # "sepsis: yes/no" string cannot (see the notes, Data model).
    deterioration_onset: Optional[float] = None

    # Static / demographic / history attributes (constant over the stay). These
    # are legitimate risk factors AND the basis of the fairness audit. `ses` is a
    # 1–10 socioeconomic decile (1 = lowest); see the notes' Ethics section for
    # why it is modelled as affecting both risk and data quality.
    age: Optional[float] = None
    sex: Optional[int] = None                # 0 / 1
    comorbidity_count: Optional[int] = None
    prior_complications: Optional[int] = None
    ses: Optional[int] = None
