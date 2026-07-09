# Better Minds AI — Clinical Trajectory Platform

## Project Notes & MVP Plan

A design document for an AI system that models patient state continuously —
from Emergency Department (ED) arrival, through MICU/ICU admission, and across
the first 24–48 hours of critical care — updating predictions of deterioration,
organ failure, and treatment response over time from time-series clinical data.

---

## 0. System Definition (Core Idea)

An AI clinical trajectory system that continuously models patient state from
**ED arrival through MICU/ICU admission and the first 24–48 hours of critical
care**, updating predictions of deterioration, organ failure, and treatment
response over time using time-series clinical data.

The central idea, and the project's main differentiator, is not a one-time
prediction but a **continuously updating understanding of patient progression** —
and specifically **how that understanding changes after clinical events and
interventions.**

---

## 1. Problem Space (Why this exists)

- ED and ICU systems are fragmented and not connected as a continuous model.
- Clinicians must manually integrate large amounts of time-based data (vitals, labs, interventions).
- Most current tools give **static predictions** (single time snapshot).
- Patient condition is dynamic, especially in critical care settings.
- Early deterioration is often visible in **trends**, not single values.

Framing the problem precisely matters. Experienced clinicians are already very
good at gestalt trend reading, so the goal is not to compensate for humans who
"can't integrate the data." The goal is **earlier, more consistent, less
alarm-fatiguing detection** than a good nurse plus a simple early-warning score
already achieve. That comparison — beating the existing bar — is the real target
(see section 4c, Baselines).

---

## 2. Competitive Landscape

This market moves fast (acquisitions, new entrants, FDA clearances). The list
below is a snapshot to re-verify before relying on it, not a fixed fact.

### Imaging AI

- **Aidoc** — stroke, PE, trauma imaging triage. Strong in radiology, not patient-wide physiological modeling.

### Symptom / Triage Systems

- **Ada Health, Infermedica, K Health** — intake and symptom checking, mostly pre-ED / early triage. Not built for ICU-level monitoring.

### ICU / Clinical Deterioration AI

- **CLEW Medical** — predicts ICU deterioration and sepsis; focused on post-admission ICU monitoring.
- **Epic's Sepsis Model / "Deterioration Index"** is the most important case study here — not a competitor to copy, but a cautionary tale about validation and label design (see sections 4a and 4b). **Bayesian Health** and the broader sepsis-prediction literature are also worth tracking.

### Care-Coordination / Alerting

- **Viz.ai** — detection plus care-team communication (stroke/PE pathways). Its real moat is *workflow and notification*, not just the model — a lesson that applies here too.

### Hospital Platforms (Integrated EHR)

- **Epic Systems, Oracle Health (Cerner)** — EHR platforms with embedded decision support. Strong adoption because they're already the system of record.

### Where the gap is (the core differentiator)

- ED tools are snapshot-based.
- ICU tools are post-admission focused.
- No system clearly connects **ED → ICU as one continuous trajectory model.**

The second differentiator is the shape of the output. Most deployed tools output
a *level* ("risk is high"). This system outputs a **trajectory and its response
to events** ("risk was 25%, fluids were given, it's now 18% and flattening").
That is the part worth defending.

---

## 3. Proposed System (Data → Model → Interface)

1. **ED Intake** — symptoms, vitals, initial labs, imaging orders, triage category.
2. **Continuous ED Monitoring** — repeated vitals, early lab updates, initial treatment response, notes.
3. **ICU / MICU Admission** — ventilation, vasopressors, organ support, continuous monitoring.
4. **ICU Data Stream** — hourly vitals, lab trends, medication changes, nursing assessments, device data.
5. **Patient Timeline Engine** — unifies everything into a single time-series representation with a maintained chronological state history.
6. **Trajectory Modeling Engine (ML core)** — time-series prediction, deterioration detection, multi-outcome risk, treatment-response modeling.
7. **Output Layer** — current risk, 3–6h forecast, trend direction (↑ ↓ →), top contributing factors, treatment-response indicator, **confidence/calibration**.
8. **Clinician Dashboard** — visual timeline, risk-trend graphs, contributing factors, intervention tracking.

### System Modeling Inspirations

| Component | Inspiration |
| --- | --- |
| Data integration | Epic / Oracle Health |
| ICU prediction | CLEW Medical |
| ED triage concepts | Infermedica / Ada |
| Explainability | SHAP-style feature attributions |
| Alerting + communication | Viz.ai |

Explainability uses **SHAP** (feature attributions), optionally with a
counterfactual "what drove the change" framing so the dashboard can show *why*
the risk moved, not just that it moved.

---

## 4. Key Features

### Prediction Targets (long-term vision)

- Septic shock
- Respiratory failure
- Acute kidney injury (AKI)
- Cardiac arrest
- Mortality risk
- ICU length of stay (LOS)

The MVP targets **one** of these — **sepsis / septic shock** — chosen for its
rich literature, a standard label definition (Sepsis-3), and ready-made cohort
code for MIMIC-IV. The remaining outcomes stay on the roadmap as future work. A
model that reliably predicts one outcome is more valuable than one that predicts
everything, so the six-outcome list is the destination, not the starting point.

### Modeling Capabilities

- Time-series trajectory modeling
- Multi-modal fusion (vitals, labs, meds, notes) — notes/NLP is a whole project on its own and belongs in a later phase, not the MVP.
- Treatment-response tracking (with the causal caveat in section 6)
- Trend-based risk updates
- **Static & history features** (age, sex, comorbidities, prior-complication history, socioeconomic status, …) alongside the vitals. They are constant per patient, carry genuine risk signal, and flow through the *same* schema and feature path as the time-series — so adding one is extending a list, not re-plumbing. Sensitive attributes (SES, and anything correlated with race) additionally require the equity audit in section 10.

### Explainability

- Why risk changed over time
- Key contributing variables (via SHAP)
- **Calibration/confidence** (not just a point score)
- Intervention-impact visualization, framed as *association*, not proven *effect*

---

## 4a. Prediction Task Specification

This is the single most important thing to nail down. Most clinical-ML projects
that fail, fail here — in the definition of the task, not in the model. Before
writing any model code, fix the answers to these six questions:

1. **Prediction time (t):** at what moments does the system produce a prediction? (e.g., every hour, or at each new observation.)
2. **Outcome definition:** what exactly is the label? A single patient-level "sepsis: yes/no" is too coarse for a trajectory system. The right form is a **time-varying label**: *"Will this patient meet [criteria] within the next N hours, given only data up to time t?"*
3. **Prediction horizon (N):** how far ahead? (e.g., 6 hours is common for deterioration.)
4. **Gap / blanking window:** leave a buffer between t and the outcome window so the system is predicting the *future*, not the present. Predicting an event that's already begun is not prediction.
5. **Feature availability:** at time t, use **only information that exists at or before t.** No peeking.
6. **No mechanical containment:** the outcome must not be built from features that define it (see the label-leakage note in Phase 2).

Getting these six right turns "risk as a function of time" from a slogan into a
well-posed problem. It is also what makes the trajectory engine (Phase 4)
coherent: because the target is defined at each time point, risk can legitimately
move over time.

---

## 4b. Evaluation Strategy

The system can't be judged without this, and for a product whose whole value is
*the number*, evaluation matters more than the model choice.

**Discrimination**

- **AUPRC (precision–recall)** is more informative than AUROC here, because deterioration is **rare** — class imbalance makes AUROC look strong while the model is still useless in practice. Report both; lead with AUPRC.

**Calibration (the most important property here)**

- A score that says "70%" should be right about 70% of the time. Use **calibration curves** and the **Brier score**. If the dashboard shows "risk 72% ↑," that number has to mean something, or the trajectory story collapses.

**Clinically-grounded / alert metrics (what actually drives adoption)**

- **Sensitivity at a fixed alert rate** (e.g., "at 1 alert per patient-day, we catch X% of events").
- **Lead time:** how far *before* the event did the alert fire? A model that's accurate but fires 10 minutes early is worthless; hours of lead time is the value.
- **False alarms per patient-day** — tied directly to alarm fatigue. This is the metric that gets a system thrown out of an ICU.

**Validation design**

- **Split by patient, never by time-point.** The same patient in train and test means leakage and inflated scores.
- Eventually validate **across time periods** and **across sites (external validation)** — these models notoriously fail to generalize.

---

## 4c. Baselines to Beat

Any deterioration model has to be compared against the simple tools clinicians
already use. If a random forest can't beat these computed from the *same* vitals,
it isn't adding value:

- **NEWS / NEWS2** (National Early Warning Score)
- **MEWS** (Modified Early Warning Score)
- **qSOFA** and **SOFA** (for sepsis / organ dysfunction)
- **APACHE II** (ICU severity, mainly for benchmarking)

The MVP includes at least **NEWS2 and qSOFA** as baselines. It makes the project
dramatically more credible and sets the real bar. "We beat NEWS2 on lead time at
equal false-alarm rate" is a real result; "our AUC was 0.9 on synthetic data" is
not.

---

## 5. Key Concept

> The value of the system is not the prediction itself, but the **change in
> prediction over time after interventions.**

This is the heart of the project. On real data, though, attributing that change
to the intervention is a hard causal problem, not a supervised-learning problem
(see section 6).

---

## 6. Challenges

The AI is the easy part. The hard parts:

- **Data access & quality** — large, high-quality, labeled ICU data is scarce and messy.
- **Integration** — with EHRs and bedside monitors; real-time pipelines are hard.
- **Alarm fatigue** — too many false alarms and clinicians ignore the system. This is a *design and evaluation* constraint, not an afterthought (see section 4b).
- **Validation** — retrospective plus prospective; models that look great in-sample routinely fail externally.
- **Regulation** — see section 9.

The challenges that most often sink these projects:

- **Treatment-response is a causal-inference problem, not prediction.** Intervention response is the project's crown jewel, which also makes it the hardest problem in the field. On observational ICU data, **confounding by indication** is brutal: the sickest patients get the most aggressive treatment, so a naive model "learns" that vasopressors *cause* death (they don't — severity does). Supervised learning on this data gets it backwards. For the MVP, the honest claim is *temporal association of the risk curve with intervention timing*, not a causal effect. Going to real data would require causal methods (propensity / marginal-structural-model-style thinking) or, at minimum, sober caveats. The system cannot claim "the model shows fluids reduce risk" from this kind of data alone.
- **Informative missingness.** In real EHR data, *whether and when a lab was ordered* is itself a signal — clinicians draw a lactate when they're worried. Missingness is not random, and it can **leak the label** (dying patients get labs drawn more often), so it has to be handled carefully. *The generator now models this directly: a lactate is ordered reactively when the observed vitals look concerning (fever/tachycardia/hypotension), so `lactate_missing` carries genuine signal — realistic, and a reminder to watch that this kind of signal doesn't quietly encode bias.*
- **Irregular sampling.** Monitor vitals are dense; labs are sparse and irregular. A clean "every 30 min" grid hides this — acceptable for the MVP, as long as the simplification is understood.
- **Dataset shift / generalization.** Different hospitals chart differently and serve different populations and case mixes. A model tuned on one ICU can degrade badly elsewhere.

---

## 7. Suggested Tech Stack

Simulate first — no hospital systems needed:

- **Python**, **Pandas / NumPy** (time-series)
- **XGBoost / gradient boosting** — the workhorse. For tabular time-series with limited data, GBMs usually **match or beat** deep learning and are far simpler to get right.
- **Scikit-learn** for baselines, calibration (`CalibratedClassifierCV`), metrics.
- **SHAP** for explainability.
- **Streamlit** (fastest) or React for the dashboard.
- **Synthetic patient generator** (see MVP Phase 1, including the circularity fix explained there).

Optional / later:

- **LSTM / Transformer / time-aware models** — useful only once there's a lot of real data and irregular-sampling structure to exploit. Not part of the MVP, where they add complexity and usually *lose* to XGBoost at small scale.
- **Survival analysis** (Cox, or discrete-time survival) — a strong fit for time-to-event framing of deterioration / LOS / mortality; answers "when," not just "whether."

Moving to real data means **MIMIC-IV** (best first choice; requires credentialed
access plus a short human-subjects training course) and the **eICU Collaborative
Research Database** (multi-center, good for testing generalization). Budget
serious time for **cohort definition** — deciding who's included, when the clock
starts, and how outcomes are ascertained is roughly 80% of real clinical-ML work
and where most silent errors hide.

---

## 8. Build Plan (Roadmap)

- **Phase 1** — Synthetic patient generator + basic risk scoring.
- **Phase 2** — Time-series tracking + patient timeline visualization.
- **Phase 3** — One-outcome prediction model + deterioration detection.
- **Phase 4** — Trajectory simulation (the core).
- **Phase 5** — Explainability + evaluation + full dashboard.
- **Phase 6** — ED → ICU transition (as visualization).
- **Phase 7 (optional)** — Treatment-response layer (with honest framing).

---

# MVP Implementation Plan (Python)

## Goal of the MVP

The first version is **not** a medical-grade AI. It is:

> A simulated ED → ICU patient trajectory system that updates risk predictions
> over time using synthetic patient data.

It demonstrates time-series ML thinking, clinical-workflow modeling, trajectory
prediction, explainable output, and system-design ability. It does **not**
provide evidence of clinical validity — and the reason it can't is the
circularity trap described in Phase 1.

---

## Code Architecture

The seven phases below describe *behavior*. This section pins the *module
boundaries*, because most of the ways a pipeline like this goes wrong are at the
seams between modules, not inside them. The file names matter less than a handful
of structural rules that keep the pieces from drifting apart.

### Module layout

```
config.py            # horizon N, gap, grid interval, alert-rate target, seed, feature list
schema.py            # Patient / Observation / Intervention dataclasses; canonical names + units
ingest/
  synthetic.py       # the current generator — emits schema objects
  mimic.py           # later: real data, emitting the SAME schema objects
timeline_engine.py   # resample raw events onto a uniform grid (only)
labeling.py          # forward-looking labels (horizon / gap / criteria)
features.py          # featurize(history, t) -> vector   (pure, past-only, shared)
splits.py            # patient-level train / test split
baselines.py         # NEWS2 / qSOFA scoring
model.py             # Pipeline(preprocess + estimator); fit / predict / persist
evaluate.py          # AUPRC, calibration, Brier, lead time, false-alarms/day, threshold selection
explain.py           # SHAP over the fitted pipeline
trajectory_engine.py # risk (+ explanations) as a function of time
dashboard.py         # presentation only — reads saved artifacts
artifacts/           # generated dataset, fitted pipeline, trajectory + SHAP outputs
tests/               # leakage + labeling tests
```

### Structural rules (these are what keep it correct)

1. **One inference path.** Feature construction lives in a single pure function, `featurize(history, t) -> vector`, imported by both training and the trajectory engine; preprocessing (imputation, scaling) and the estimator are wrapped in one scikit-learn `Pipeline` that is fitted once and persisted. Training and live scoring must never build the input two different ways — when they do, the risk curve silently stops matching the trained model (train/serve skew), with no error to catch it. This is the most important rule here.

2. **Carry identity through the whole pipeline.** Every row keeps a `(patient_id, time)` index from the generator to the final prediction. Without `patient_id`, "split by patient, not by row" can't be enforced and "false alarms per patient-day" can't be computed; without `time`, a prediction can't be mapped back into a trajectory or turned into a lead-time. Predictions travel as an indexed table, never as a bare array.

3. **One home for tunables.** Horizon N, gap, grid interval, the alert operating point, the RNG seed, and the canonical feature list all live in `config.py`. These are the knobs the whole project exists to turn; scattered across files as magic numbers (a `0.60` here, a `30 min` there) they make runs irreproducible and every experiment a multi-file edit.

4. **Resampling and labeling are separate concerns.** Putting raw events on a uniform grid is data-shaping; defining the forward-looking label (with horizon and gap) is task-definition, and the label is the single most error-prone thing in the project. Keeping `labeling.py` apart from `timeline_engine.py` makes the part most likely to be wrong the easiest to change and test.

5. **Model, baselines, and evaluation are three modules, not one.** NEWS2 and qSOFA are deterministic clinical scores, not models; the metrics (AUPRC, calibration, lead time, false-alarm rate) are reused across the model and every baseline and must run on the identical patient splits. Separating `baselines.py` and `evaluate.py` from `model.py` makes "did we actually beat NEWS2?" a first-class, repeatable step rather than something bolted onto training.

6. **Data source is an adapter, so synthetic and real look identical downstream.** The generator and a future MIMIC-IV loader both emit the same `Patient` / `Observation` objects; everything after ingestion is source-agnostic. This is what lets a pipeline built entirely on synthetic data later run on real data by swapping one module, instead of a rewrite.

7. **Model events for interventions from the start.** An intervention (fluids, vasopressors) is just another timestamped event. Giving observations and interventions one shared event schema now means the Phase 7 treatment-response layer is an addition, not a retrofit that touches every module.

8. **The alert threshold is chosen, not hardcoded.** The `0.60` in the ED→ICU rule is a placeholder. The real operating point is selected on a validation set to hit a target alert rate (e.g., 1 alert per patient-day), produced by `evaluate.py`, and only then handed to the dashboard and the transition rule.

9. **Explanations ride the same inference path.** Per-timestep SHAP values are computed from the one fitted pipeline and stored alongside the risk series in the trajectory artifact. The dashboard renders them; it never re-derives features or re-runs the model, which would open a second path that can diverge from the first.

10. **Persist between stages.** The synthetic dataset is generated once (seeded) and saved; the model reads that fixed file; the fitted pipeline and the trajectory + SHAP outputs are saved artifacts the dashboard loads. This makes iteration reproducible — model changes are compared on the same data — and keeps the dashboard from re-running training.

### Data model

Use small dataclasses (`Patient`, `Observation`, `Intervention`) rather than bare
dicts, and access them one consistent way — the phase snippets below otherwise
drift between `patient["timeline"]` (dict) and `patient.timeline` (attribute).
Pin canonical field names and units in the schema — decide whether `bp` means
systolic or mean arterial pressure, whether `o2` is SpO2 or something
FiO2-adjusted — so the generator and a later real-data loader can't quietly
disagree.

One consequence for the generator: a static `outcome: "sepsis"` string is not
enough to build a forward-looking label. Labeling needs a **deterioration onset
time** (or criteria that can be evaluated from the emitted vitals at each step),
so the generator must emit *when* a patient crosses into deterioration, not just
*whether* they do.

A related decision that needs an owner: **missingness handling.** Whether a gap
is filled by carry-forward, imputed, or flagged with a missing-indicator is a
policy, and (per section 6) that indicator can itself be informative or leak the
label. Put that policy in `features.py` / `config.py`, applied identically at
train and inference time, rather than improvising it in each module.

### Testing

The two riskiest pieces deserve tiny unit tests: one asserting `featurize(history, t)`
never reads a row later than `t` (the past-only, no-leakage guarantee), and one
checking that the label at `t` reflects the correct future window with the gap
applied. These are three-line tests that catch the errors most likely to
invalidate every downstream number.

---

## PHASE 1 — Synthetic Patient Generator

**File:** `ingest/synthetic.py` (the synthetic adapter — see rule 6 above)

Generate patients with a time-series, a deterioration onset time, and an outcome, e.g.:

```python
patient = {
    "id": 1,
    "timeline": [
        {"time": 0,  "hr": 95,  "bp": 120, "o2": 98, "lactate": 1.2},
        {"time": 30, "hr": 110, "bp": 105, "o2": 96, "lactate": 1.8},
        {"time": 60, "hr": 125, "bp": 90,  "o2": 93, "lactate": 3.2},
    ],
    "outcome": "sepsis",
    "deterioration_onset": 75,   # minute criteria are first met — lets labeling build a forward-looking target
}
```

Start with simple rules (random baseline vitals + deterioration patterns):

```
if patient_is_septic:
    HR increases over time
    BP decreases
    lactate increases
```

### The synthetic-data circularity trap

This is the most important conceptual point in the whole MVP.

If data is generated with hardcoded rules ("septic ⇒ HR↑, BP↓, lactate↑") and a
model is then trained to predict the outcome, **the model is just learning those
rules back.** The impressive-looking risk curve is an artifact of the
*generator*, not a discovery about patients. Taken to its logical end, if the
rules fully determine the outcome, no ML is needed at all — the rules could just
be applied directly.

The ML step only becomes meaningful when the generator includes things the model
has to *see through*:

- **Realistic noise** on every measurement.
- **Missing / irregularly-sampled** values, with missingness made slightly informative.
- **Heterogeneity** — not all septic patients look alike; some deteriorate fast, some slow, some atypically.
- **Confounders and red herrings** — non-septic patients who transiently look bad; septic patients who look fine early.
- **Overlap** — the classes should *not* be perfectly separable, or the metrics are meaningless.

Synthetic data is excellent for building the *pipeline, timeline engine, and
dashboard* and for demonstrating system design. It is **not** evidence the
approach works clinically, and any accuracy number on it carries that caveat
explicitly. Real evaluation waits for MIMIC-IV / eICU.

**Output:** 100–1000 synthetic patients, each with time-series data, a
deterioration onset time, and an outcome label — with noise, missingness, and
non-trivial overlap built in. Generation is seeded (rule 10) so the dataset is
reproducible, and saved once to `artifacts/` for every downstream stage to read.

---

## PHASE 2 — Patient Timeline Engine

**File:** `timeline_engine.py` (resampling only; the label is built in `labeling.py` — rule 4)

Convert raw events → a clean, structured matrix on uniform time steps, keyed by `(patient_id, time)`:

| patient_id | time | hr | bp | o2 | lactate | label |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 95 | 120 | 98 | 1.2 | 0 |
| 1 | 30 | 110 | 105 | 96 | 1.8 | 0 |
| 1 | 60 | 125 | 90 | 93 | 3.2 | 1 |

The `patient_id` column isn't decoration: without it, the patient-level split in
Phase 3 and the per-patient alert metrics can't be computed (rule 2).

### Label semantics (avoiding leakage)

In the table above the label flips to 1 at t=60 — but `lactate = 3.2` at t=60 is
itself part of how sepsis is *defined*. Using the concurrent lactate to
"predict" a lactate-defined label is **label leakage**: that's not prediction,
it's restatement. The label has to be **forward-looking** instead: at each row t,
the label is *"does the patient meet the deterioration criteria within the next N
minutes (with a gap), using only rows ≤ t?"* This labeling logic lives in
`labeling.py`, and the **trend features** (deltas, rolling slopes) — the whole
thesis — are built in the shared `features.py`, so training and the trajectory
engine compute them identically (rule 1).

**Output:** a clean, leakage-safe dataset where `X = past-only time-series features` and `y = future-outcome-within-horizon`.

---

## PHASE 3 — First Prediction Model (Baseline ML)

**File:** `model.py` (fit/predict only; baselines in `baselines.py`, metrics and threshold selection in `evaluate.py` — rule 5)

Start simple: **RandomForest** first, then **XGBoost**. The feature vector below is
built by `features.py` — the same function the trajectory engine calls (rule 1) —
not re-assembled by hand here.

Flatten current state + trends:

```python
X = [
    hr_now, bp_now, o2_now, lactate_now,
    delta_hr, delta_bp, delta_lactate,      # trends — the point of the project
    slope_hr_30min, slope_lactate_30min,    # optional rolling trends
]

risk_score = model.predict_proba(X)[:, 1]
```

Before trusting any number:

1. **Split by patient** (not by row).
2. **Calibrate** the probabilities (`CalibratedClassifierCV`) — the dashboard shows a %, so it has to be honest.
3. **Compare against NEWS2 / qSOFA** computed from the same vitals (section 4c).
4. **Report AUPRC + calibration + lead time**, not just accuracy (section 4b).

**Output:** a working, *evaluated*, *calibrated* baseline that can be honestly
compared to a standard clinical score.

---

## PHASE 4 — Trajectory Simulation Engine (CORE IDEA)

**File:** `trajectory_engine.py`

Instead of one prediction, produce a *sequence* over the patient's timeline:

```python
pipeline = load("artifacts/pipeline.joblib")   # the ONE fitted object from Phase 3

for t in patient.timeline:
    X = featurize(history_up_to_t)      # SAME function used in training (rule 1)
    risk = pipeline.predict_proba(X)[:, 1]
    shap_t = explain(pipeline, X)       # top drivers at this t
    store(patient_id, t, risk, shap_t)  # keep the index (rule 2)
```

Output:

```
Time   →  Risk
0 min  →  12%
30 min →  25%
60 min →  44%
90 min →  70%
```

This is the **core differentiator**: risk as a function of time. It's coherent
*because* Phase 2 defined a time-varying, forward-looking label. Add a simple
visual marker of *why* risk moved between steps (the top SHAP contributors at
each t) — that's the explainable-trajectory story that student projects almost
never have. Those per-step SHAP values are written into the trajectory artifact,
so the dashboard only renders them and never recomputes them (rule 9).

---

## PHASE 5 — Visualization Dashboard

**File:** `dashboard.py` (Streamlit — `pip install streamlit`)

Include:

1. **Risk-over-time line chart** (with the alert threshold drawn on it).
2. **Vital-signs charts** (HR, BP, O2, lactate) aligned to the same time axis.
3. **Clinical interpretation panel**, e.g.:

```
Current Risk: 72% ↑   (calibrated)
Key drivers (SHAP):
  - BP dropping
  - Lactate rising
  - HR increasing
Lead time to predicted event: ~2.5 h
Prediction: elevated likelihood of septic progression
```

Add a small **calibration plot** and a **false-alarms-per-day** readout somewhere
in the UI — it signals an understanding of what makes these systems usable, not
just accurate.

---

## PHASE 6 — ED → ICU Transition Logic

A simple rule makes the concept visible:

```python
if risk > config.alert_threshold:   # chosen on validation to hit a target alert rate — not a hardcoded 0.60
    patient_status = "ICU"
```

This is a **visualization of the concept, not real logic.** Real ICU admission
is a clinician decision, not a threshold on the model's own output, and using the
score to define the transition is circular. It's a clean way to show "ED →
deterioration → ICU" on the timeline, and nothing more should be claimed for it.
The threshold itself comes from `evaluate.py`'s operating-point selection (rule 8),
not from a number typed into this rule.

---

## PHASE 7 — Treatment-Response Layer (built)

Interventions are timestamped events on the schema the pipeline already carries
(fluids, antibiotics, antipyretic, vasopressor). Two design choices make this both
honest and genuinely useful.

**Medications MASK specific vitals; they do not change the illness.** A vasopressor
props up systolic BP, an antipyretic hides a fever, a fluid bolus lifts BP
transiently. In the generator these move the *monitor number* without touching the
underlying deterioration or the label. That decoupling is the whole point: a
normal-looking vital can be drug-supported rather than genuine recovery.

**The model is told what drugs are on board** — `on_vasopressor`, `on_antibiotics`,
`recent_fluid`, `on_antipyretic` — so it can read the vitals in context, which is
what stops it being fooled. In the demo a treated patient's BP crashes at onset, a
pressor lifts it back to normal, and the risk trajectory *stays high* because the
model knows the BP is propped up. Removing the context (`USE_MED_FEATURES = False`)
measurably hurts (AUPRC ~0.70 vs ~0.73), and would hurt more on the post-onset
stretch the pre-onset label doesn't score.

**What this deliberately does NOT model: treatment efficacy.** Whether a drug
actually helps is the confounding-by-indication problem from section 6 — the
sickest patients get the most treatment, so naive supervised learning would wrongly
conclude "vasopressor → death". So the layer never claims "fluids reduce risk". It
demonstrates the subtler, correct point: interventions change what the vitals
*show*, and a system that ignores them will misread drug-supported stability as
improvement. Estimating a real treatment effect needs causal methods, not this
pipeline — which is the honest thing to be able to say.

---

## 9. Regulatory & Explainability

Software that influences clinical decisions generally requires regulatory review.
The sharper version (US-focused; this area evolves, so verify current guidance —
this is general information, not legal advice):

- Clinical Decision Support (CDS) software can sometimes fall **outside** FDA device regulation under the 21st Century Cures Act, but roughly only if it (among other criteria) provides *recommendations rather than specific directives* **and lets the clinician independently review the basis** for the recommendation.
- A system that **analyzes patterns in time-series signals** to produce a continuously-updating risk score is very likely to be regulated as a **medical device (Software as a Medical Device, SaMD)** — precisely because it analyzes signals/patterns that a clinician can't independently reconstruct.
- This is why explainability isn't just a nice-to-have — it's partly a regulatory strategy. SHAP-style attributions ("here's *why* the risk moved") support the "clinician can review the basis" posture and improve trust and adoption at the same time. The explainability work and the regulatory posture are the same conversation.

---

## 10. Ethics & Equity

Clinical models can encode and amplify existing disparities, so equity is a
first-class design concern, not a footnote. The canonical warning (Obermeyer et
al., 2019) is an algorithm that used healthcare *cost* as a proxy for *need*:
because less money is spent on equally-sick Black patients, the proxy understated
their need and under-served them. The lesson is not "never use sensitive
variables" — it is "know what your variable is actually a proxy for."

### Socioeconomic status: a worked example of the tension

Lower socioeconomic status (SES) genuinely tracks worse outcomes — it is one of
the strongest social determinants of health there is. So adding it as a predictor
is tempting, and not naive: reflexively deleting sensitive variables ("fairness
through unawareness") usually fails anyway, because the signal leaks back through
correlated features and you lose the ability to audit for the disparity. But
three things complicate SES-as-a-feature:

- **Prediction vs. use.** The same feature can help or harm depending on the decision it drives. Using SES to escalate monitoring *earlier* for a low-reserve patient helps them; using it to ration a scarce resource can entrench the disparity and can be self-fulfilling.
- **Confounding by access.** Much of the "low SES → bad outcome" signal is not the patient's physiology — it is the system under-treating and under-measuring them. A model that learns this can quietly learn to *expect* worse outcomes for low-SES patients instead of flagging them as preventable (the Obermeyer trap, one step removed).
- **Residual signal, and the individual vs. the group.** The physiological effects of low SES largely show up in the vitals and labs you already measure. The honest, testable question is whether SES adds signal *beyond* the measured physiology — often the answer is "a little" — which then has to be weighed against the ethical and regulatory cost of encoding group membership into a medical device, and against the ecological-fallacy risk of applying a group average to an individual whose physiology you are already measuring.

### The reframe: fairness is an audit, not (only) a feature choice

The core question is not "should SES be an input?" but "does the model work
equally well for everyone?" — is it as sensitive and as well-calibrated for each
group? That audit should run whether or not SES is a feature. Do it by
**stratifying every metric** (sensitivity, calibration, false-alarm rate) across
groups, not by reporting a good average.

### What the MVP now demonstrates

The disparity begins before the model even runs. Sepsis prevalence is generated to
**scale from a realistic base** (~6% of adult admissions, Rhee et al. 2017) by
socioeconomic status, so low-SES patients develop sepsis roughly 2.4× as often as
high-SES patients (~12% vs ~5% in the cohort) — mirroring the real gradient. On
top of that, low SES degrades **data quality** (less monitoring → more missing
readings). The stratified audit (`evaluate.group_report`, printed by
`run_pipeline`) then measures whether the model serves each group equally.

At realistic low prevalence the harm shows up most in **calibration and
false-alarm rate**: low-SES patients get noisier predictions (worse Brier) and
more false alarms per day, because the model has thinner data on them — and, with
a large enough cohort, in a **sensitivity gap** too. (On a small, fast cohort
there are only a handful of sepsis cases per stratum, so the sensitivity numbers
are noisy; a cleaner sensitivity-gap demo needs a bigger cohort, which is a
runtime/optimization question, not a modelling one.) Flipping `USE_SES_AS_FEATURE`
tests whether adding SES as an input reduces the gap — and the honest finding
stands: it helps only partway, because the root cause is the *data*
(under-monitoring), not the missing label. The fix is better data collection for
under-served patients, not merely a feature — and building the audit that reveals
this is worth more than any single accuracy number. (Figures illustrate the
mechanism; they are not clinical estimates — the data is synthetic.)

---

## What the MVP Demonstrates

- Time-series ML thinking
- Clinical-workflow modeling
- Trajectory prediction (rare in student projects)
- Explainable outputs
- System-design ability
- And, through the sections above: a working grasp of **evaluation, leakage, baselines, causal limits, and regulation** — the things that separate a cool demo from real clinical-ML understanding.
