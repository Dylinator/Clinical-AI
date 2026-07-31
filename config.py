"""
config.py — one home for every tunable (Rule 3).

If you find yourself typing a magic number (a 0.60 threshold, a 30-minute grid)
into any other file, it belongs here instead. Everything downstream imports from
this module, so an experiment is a one-line change here, not a hunt across files.

Units are fixed and documented once, here, so the generator and a future
real-data loader can't quietly disagree (see schema.py). These units match the
PhysioNet/CinC-2019 real-data adapter (ingest/physionet2019.py) so synthetic and
real patients are interchangeable downstream:
    time       : minutes since ED arrival (t=0)
    hr         : beats/min
    rr         : breaths/min
    sbp        : systolic blood pressure, mmHg   (NOT mean arterial pressure)
    o2         : SpO2, %
    temp       : degrees Celsius                 (NOT Fahrenheit — NEWS2 needs C)
    lactate    : mmol/L
    creatinine : mg/dL                           (kidney; SOFA renal component)
    wbc        : 10^3 cells/uL                   (infection; leukocytosis in sepsis)
    platelets  : 10^3 cells/uL                   (coagulation; SOFA drops in sepsis)
"""

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


# --------------------------------------------------------------------------- #
# The prediction task (Section 4a of the notes)
# --------------------------------------------------------------------------- #
GRID_INTERVAL_MIN = 10      # one row every 10 minutes (was 30; finer = more rows/slower)
STAY_MINUTES = 480          # 8-hour window per patient -> grid t = 0,10,...,480
HORIZON_MIN = 120           # predict deterioration within the next 2 hours
GAP_MIN = 30                # blanking window: never predict the *current* step

# The alert operating point is CHOSEN, not hardcoded (Rule 8). evaluate.py picks
# the probability threshold that lands closest to this alert budget.
TARGET_ALERTS_PER_DAY = 1.0


# --------------------------------------------------------------------------- #
# The canonical feature vector (built only in features.py — Rule 1)
# --------------------------------------------------------------------------- #
# Vitals are DENSE (bedside monitor); labs are SPARSE (a clinician orders them),
# so labs additionally carry a `_missing` indicator — whether a lab was drawn is
# itself signal (informative missingness, Section 6). Both kinds flow through the
# identical feature path; CHANNELS is the union the feature builder iterates.
VITALS = ["hr", "rr", "sbp", "o2", "temp"]
LABS = ["lactate", "creatinine", "wbc", "platelets"]   # lactate + SOFA-core
CHANNELS = VITALS + LABS

SLOPE_WINDOW_MIN = 60       # rolling-slope lookback

# --- Static / demographic / history features (constant per patient) --------- #
# Legitimate risk factors that travel through the SAME schema + feature path as
# the vitals. Extend this list as you add labs, meds, more history, etc.
USE_STATIC_FEATURES = True      # age, sex, comorbidities, prior complications
USE_SES_AS_FEATURE = False      # fairness experiment: does feeding SES to the
#                                 model help low-SES patients, hurt them, or just
#                                 move who gets missed? Flip this and re-run.

STATIC_FEATURES = ["age", "sex", "comorbidity_count", "prior_complications"]
SES_FEATURES = ["ses"]
STATIC_ALL = STATIC_FEATURES + SES_FEATURES    # every static column the frame carries

# --- Medication / intervention context (Phase 7) ----------------------------- #
# Meds MASK specific vitals — a vasopressor props up blood pressure, an antipyretic
# hides a fever — WITHOUT changing the underlying illness (so the label is
# unchanged). A normal-looking number can therefore be drug-supported rather than
# recovery. These flags tell the model which drugs are on board so it can read the
# vitals in context. Toggle off to watch it get fooled by masked vitals (and see
# the causal caveat, Section 6 of the notes). We deliberately do NOT model whether
# a drug actually *works* — that's the causal-inference problem the notes defer.
USE_MED_FEATURES = True
MED_FEATURES = ["on_vasopressor", "on_antibiotics", "recent_fluid", "on_antipyretic"]

# Order matters: this is the exact column order the model trains on and the
# trajectory engine scores on. Keep it in one place so the two can't drift.
_SIGNAL_FEATURES = (
    [f"{c}_now" for c in CHANNELS]        # last observed value at time t
    + [f"{c}_delta" for c in CHANNELS]    # change since the previous observation
    + [f"{c}_slope" for c in CHANNELS]    # slope over the last SLOPE_WINDOW_MIN
    + [f"{lab}_missing" for lab in LABS]  # informative-missingness, one per lab
)

FEATURES = (
    _SIGNAL_FEATURES
    + (STATIC_FEATURES if USE_STATIC_FEATURES else [])
    + (SES_FEATURES if USE_SES_AS_FEATURE else [])
    + (MED_FEATURES if USE_MED_FEATURES else [])
)

# Fallback values used only when a channel has never been observed yet (start of
# a stay). Carry-forward handles everything after the first observation.
BASELINE_FILL = {
    "hr": 80, "rr": 16, "sbp": 120, "o2": 98, "temp": 36.8,
    "lactate": 1.2, "creatinine": 1.0, "wbc": 8.0, "platelets": 250.0,
}


# --------------------------------------------------------------------------- #
# Synthetic generator parameters (Phase 1)
# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    n_patients: int = 1200      # bigger cohort so a low prevalence still yields
    #                             enough sepsis cases; raise for a steadier fairness
    #                             audit (costs runtime — the feature builder is O(N^2)).
    # Base sepsis prevalence for an *average* patient (median SES, mean age/
    # comorbidity). Grounded in real epidemiology: sepsis is present in ~6% of
    # adult hospitalizations (Rhee et al., JAMA 2017). Each patient's actual chance
    # scales UP or DOWN from this base via the logistic model in
    # synthetic.generate() — higher for older / comorbid / prior-complication /
    # LOW-SES patients. Realized rate runs ~4% (young/high-SES) to ~15% (elderly/
    # low-SES), averaging ~8% across this risk-skewed cohort.
    sepsis_rate: float = 0.06

    # Healthy baseline (mean, sd) for each channel (vitals + labs).
    baseline: dict = field(default_factory=lambda: {
        "hr":   (80, 8),
        "rr":   (16, 2),
        "sbp":  (120, 10),
        "o2":   (98, 1.0),
        "temp": (36.8, 0.3),
        "lactate":    (1.2, 0.3),      # mmol/L
        "creatinine": (1.0, 0.2),      # mg/dL  (normal 0.6–1.2)
        "wbc":        (8.0, 2.0),      # 10^3/uL (normal ~4–11)
        "platelets":  (250.0, 50.0),   # 10^3/uL (normal ~150–400)
    })

    # Per-minute drift AFTER deterioration onset (mean, sd across patients).
    # Heterogeneous: some patients crash fast, some slow (Rule: overlap/realism).
    # Labs move in their clinically correct direction during sepsis: creatinine
    # rises (AKI), WBC rises (leukocytosis), platelets fall (thrombocytopenia).
    drift_per_min: dict = field(default_factory=lambda: {
        "hr":   (+0.09, 0.03),
        "rr":   (+0.020, 0.008),
        "sbp":  (-0.07, 0.03),
        "o2":   (-0.014, 0.005),
        "temp": (+0.006, 0.002),
        "lactate":    (+0.015, 0.005),
        "creatinine": (+0.0045, 0.0018),
        "wbc":        (+0.030, 0.012),
        "platelets":  (-0.42, 0.16),
    })

    # Deterioration ramps up over this many minutes BEFORE the labelled onset
    # (a prodrome). This is what puts subtle, catchable signal into the pre-onset
    # prediction window — real early-warning depends on the trend, not the level.
    prodrome_min: int = 180
    # Labs LEAD: metabolic/organ derangement begins before the vitals visibly
    # move, so labs start drifting earlier. This is what makes the model lean on
    # lab trends for the *earliest* pre-onset signal — a stronger lab signal that
    # is still leakage-safe (the label is the forward-looking onset, never a lab
    # value; features stay past-only). Set == prodrome_min to disable the lead.
    lab_prodrome_min: int = 300
    onset_range_min: tuple = (180, 420)   # overt-onset window
    meas_noise: dict = field(default_factory=lambda: {
        "hr": 3.0, "rr": 1.0, "sbp": 4.0, "o2": 0.7, "temp": 0.15,
        "lactate": 0.15, "creatinine": 0.05, "wbc": 0.5, "platelets": 10.0,
    })

    # Labs are sparse and irregular. Beyond a slow routine schedule (per lab),
    # a clinician REACTIVELY orders labs when the OBSERVED vitals look concerning
    # (fever / tachycardia / hypotension) — not on the hidden onset. So the
    # PRESENCE of a lab is itself a signal the patient looked sick then
    # (informative missingness — see Section 6 of the notes). Lactate is drawn
    # most often; the CBC/CMP panel (creatinine/wbc/platelets) slightly less.
    lab_every_min: dict = field(default_factory=lambda: {
        "lactate": 120, "creatinine": 180, "wbc": 180, "platelets": 180,
    })
    concern_temp: float = 37.5   # degC — fever
    concern_hr: float = 95.0     # bpm — tachycardia
    concern_sbp: float = 110.0   # mmHg — hypotension (order if BELOW)
    # P(a lab is ordered reactively this step) by number of concerning signs (0..3):
    lab_order_prob: tuple = (0.0, 0.30, 0.65, 0.90)

    # A fraction of controls transiently look bad (red herrings) so the classes
    # are not linearly separable.
    red_herring_rate: float = 0.20

    # --- Static risk factors: how age / comorbidity / prior / SES drive the
    #     probability of deteriorating (a simple logistic risk model). ---------
    age_mean: float = 60.0
    age_sd: float = 18.0
    comorbidity_lambda: float = 1.5      # mean comorbidity count (Poisson)
    prior_lambda: float = 0.7            # mean prior-complication count (Poisson)
    #   To add a new risk variable: give it a coefficient here and one line in
    #   synthetic.generate()'s `risk_terms` dict. That's the whole "add a variable".
    risk_age: float = 0.30               # logit per decade above the mean
    risk_comorbidity: float = 0.35       # logit per comorbidity
    risk_prior: float = 0.45             # logit per prior complication
    risk_sex: float = 0.20               # logit for male (modest; the evidence is mixed)
    risk_ses: float = 0.14               # logit per SES step BELOW median (low SES -> higher risk)

    # --- Data-quality penalty: lower SES -> less monitoring -> more missing data.
    #     THIS is the fairness mechanism. The model gets thinner data on low-SES
    #     patients, so it detects them later / less often unless corrected for. --
    missing_base: float = 0.05           # baseline chance any given reading is missing
    missing_ses: float = 0.045           # extra missingness per SES step below median

    # --- Treatment (Phase 7): septic patients get treated as they deteriorate.
    #     Effects are cosmetic MASKS on specific vitals (they move the monitor
    #     number, not the illness or the label) — which is exactly why the model
    #     needs to know a drug is on board. --------------------------------------
    treat_rate: float = 0.85             # fraction of septic patients treated at all
    vasopressor_rate: float = 0.5        # fraction of treated who progress to a pressor
    fluid_sbp_boost: float = 18.0        # transient systolic bump from a fluid bolus (mmHg)
    fluid_decay_min: float = 45.0        # bolus effect decay constant (min)
    vaso_sbp_boost: float = 25.0         # sustained systolic support while a pressor runs
    antipyretic_temp_drop: float = 0.9   # temperature reduction from an antipyretic (deg C)
    antipyretic_decay_min: float = 90.0


GEN = GenConfig()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Anchor artifacts to THIS file's directory so paths resolve no matter where a
# script or `streamlit run` is launched from (not the current working directory).
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
PIPELINE_PATH = f"{ARTIFACT_DIR}/pipeline.joblib"
DATASET_PATH = f"{ARTIFACT_DIR}/dataset.csv"   # CSV keeps deps minimal; swap to
#                                                .parquet if you install pyarrow
