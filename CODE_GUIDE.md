# Code Guide — how every file works

A plain-language walkthrough of the whole project, meant to be read next to the
code. The system is an **assembly line**: synthetic patients go in one end and a
risk-over-time curve plus an evaluation come out the other. Data changes shape as
it moves down the line:

```
list[Patient]  →  tidy table  →  labeled table  →  feature matrix (X, y)
   →  trained model  →  scores + metrics  →  risk trajectory  →  dashboard
```

Two files are shared "contracts" everything imports (`config.py`, `schema.py`).
`run_pipeline.py` is the conveyor belt that runs the stations in order.

Each function below is written as **`name(inputs) → output`**, then what it does
and why.

---

## Part 1 — the two contracts

### `config.py` — the settings panel
No logic, just every tunable in one place (so an experiment is a one-line change,
not a hunt across files). Key groups:

- **Task definition:** `GRID_INTERVAL_MIN` (10 — a row every 10 min; smaller = finer but slower), `STAY_MINUTES`
  (480 — an 8 h window), `HORIZON_MIN` (120 — predict within 2 h), `GAP_MIN` (30 —
  the blanking window), `SEED` (42 — makes every run identical).
- **Feature list:** `VITALS`, and `FEATURES`, the exact ordered column list the
  model trains on. `FEATURES` is *assembled from toggles* — `USE_STATIC_FEATURES`,
  `USE_SES_AS_FEATURE`, and `USE_MED_FEATURES` — so you can add or remove whole
  feature groups (and run the SES or medication-context experiments) by flipping
  one line and re-running.
- **`BASELINE_FILL`:** default values used only before a vital is ever measured.
- **`GenConfig`:** a dataclass holding every generator parameter — baseline vitals,
  drift rates, `prodrome_min`, the SES-linked missingness penalty (`missing_base`,
  `missing_ses`), and one **risk coefficient per variable** (`risk_age`,
  `risk_comorbidity`, `risk_prior`, `risk_sex`, `risk_ses`) setting how strongly
  each factor moves a patient's sepsis chance. Adding a risk variable is a
  coefficient here plus one line in `synthetic.py`'s `risk_terms` dict — nothing
  else. `GEN` is the single instance everything uses.
- **Paths:** where the fitted pipeline and dataset are saved.

### `schema.py` — the data shapes
Three dataclasses that everything speaks in:

- **`Observation`** — one timestamped set of vitals; any vital may be `None` (not
  yet measured).
- **`Intervention`** — a timestamped action (fluids, vasopressors). Unused until
  Phase 7, but defined now so adding it later isn't a rewrite.
- **`Patient`** — id, a list of `Observation`s, `outcome`, `deterioration_onset`
  (the minute they cross into deterioration, or `None`), and the static attributes
  `age / sex / comorbidity_count / prior_complications / ses`.

Because a future real-data (MIMIC-IV) loader would emit these same objects, every
station after ingestion is source-agnostic.

---

## Part 2 — making and shaping the data

### `synthetic.py` — generate the patients (Phase 1)
- **`_grid() → np.array`** — the time points 0, 30, …, 480.
- **`_clip(name, value) → float`** — keeps a vital inside a physiologically
  plausible range.
- **`generate() → list[Patient]`** — the heart. Per patient it:
  1. draws static attributes (age, sex, comorbidities, prior complications, SES);
  2. runs a **logistic risk model** — an explicit, extensible `risk_terms` dict —
     that scales each patient's sepsis chance up or down from a realistic base rate
     (~6% of admissions, Rhee et al. 2017) by **age, comorbidity count, prior
     complications, sex, and SES**. Each factor is one labelled line, so adding a
     variable = one line here plus a coefficient in `config`. The effects are real
     and monotonic (e.g. ~4% chance under 45 rising to ~15% at 80+; low-SES ~12% vs
     high-SES ~5%);
  3. if septic, picks an `onset` and starts vitals drifting `prodrome_min` (180)
     minutes *before* it — the subtle early trend the model can catch and a
     single-timepoint score can't;
  4. adds measurement noise, red-herring bumps on some controls (so classes
     overlap), sets `miss_p` from SES so **low-SES patients have more missing
     readings** (the fairness mechanism), and orders lactates **reactively** when
     the observed vitals look concerning (`_concern_score`) so lab *presence* is
     itself informative;
  5. for treated septic patients, generates a treatment sequence (fluids →
     antibiotics → antipyretic → vasopressor if severe) and applies **masking**
     effects — a pressor props up BP, an antipyretic lowers temp — that move the
     monitor number without changing the illness or the label (Phase 7);
  6. builds `Observation`s (+ `Intervention`s) and returns `Patient`s.

  Its `__main__` prints a few example patients so you can eyeball the data.

### `timeline_engine.py` — patients → one tidy table (Phase 2, resampling only)
- **`to_frame(patients) → DataFrame`** — flattens the `Patient` list into one row
  per `(patient_id, time)` with vitals, static, and medication-context columns.
  Missing values stay `NaN` on purpose (the missingness is signal). It does *only*
  reshaping — no labels, no imputation.
- **`_med_flags(interventions, t) → dict`** — the medication state at time t
  (`on_vasopressor`, `on_antibiotics`, `recent_fluid`, `on_antipyretic`) from drugs
  given at or before t. `to_frame` attaches these so the model knows what's on board.
- **`onset_map(patients) → {id: onset}`** — the lookup the labeler needs.

### `labeling.py` — add the answer key (Phase 2)
- **`add_labels(df, onsets, horizon, gap) → DataFrame`** — for each row computes
  `delta = onset − time` and sets `label = 1` when the onset falls in the window
  `(gap, gap+horizon]` ahead (deteriorates within 2 h, after a 30-min buffer).
  Rows inside the blanking window (`delta ≤ gap`) or after onset are **dropped**
  — you don't predict the present or the past. Stable patients are all negatives.
  This is the file that makes "risk over time" a well-posed problem.

---

## Part 3 — turning data into model inputs

### `features.py` — the ONE feature builder (Rule 1, the most important file)
- **`featurize_history(hist) → dict`** — takes one patient's rows *up to time t*
  and returns the feature dict: each vital's `_now` (last value, carried forward),
  `_delta` (change since the previous reading), `_slope` (trend over the last
  60 min), the `lactate_missing` flag, any static features, and any
  medication-context flags. It is **pure and past-only** — that guarantee is exactly what `test_leakage` verifies.
- **`featurize_at(patient_df, t) → dict`** — slices a patient's frame to rows ≤ t,
  then calls `featurize_history`. Past-only by construction.
- **`build_training_matrix(full_df, labeled_df) → (X, y, meta)`** — runs
  `featurize_at` for every labeled row to build the model matrix `X`, the labels
  `y`, and `meta` (patient_id + time, kept so predictions can be mapped back).

Training and the live trajectory both call `featurize_at`, so they can never build
the input two different ways (no train/serve skew).

### `splits.py` — train/test by patient (Rule 2)
- **`split_patients(ids) → (train_ids, test_ids)`** — shuffles the *unique
  patients* and splits them, so no patient is in both sets. Splitting by row
  instead would leak (a patient's rows minutes apart are near-duplicates) and
  inflate every score.
- **`mask_for(meta, ids) → bool array`** — turns an ID set into a row mask.

---

## Part 4 — the model and how it's judged

### `model.py` — fit / predict / persist (Rule 5)
- **`build_estimator() → estimator`** — a scikit-learn `Pipeline`
  (`StandardScaler` → `RandomForestClassifier`) wrapped in
  `CalibratedClassifierCV`. Bundling preprocessing *with* the model means training
  and serving preprocess identically. Calibration makes the output probability
  trustworthy — "72%" should be right ~72% of the time.
- **`fit(X, y)` / `predict_risk(est, X)` / `save` / `load`** — thin wrappers;
  `predict_risk` returns the positive-class probability.
- **`global_importances(est) → [(feature, importance)]`** — averages the forest's
  importances across calibration folds; a lightweight stand-in for SHAP.

Swapping RandomForest for XGBoost is a one-line change in `build_estimator`.

### `baselines.py` — the scores you must beat (Rule 5)
- **`news2_row(...) / qsofa_row(...) → int`** — the standard clinical early-warning
  scores, computed from the **current** vitals only (they see the level, not the
  trend — which is why a trend-aware model can beat them).
- **`score_matrix(X) → (news2, qsofa)`** — runs them across the same feature
  matrix, so the comparison is on identical rows.

### `evaluate.py` — metrics, threshold, fairness (Section 4b/4c/10)
- **`metrics(y_true, y_prob) → dict`** — AUPRC (leads, because deterioration is
  rare and AUROC flatters), AUROC, and Brier (calibration).
- **`choose_threshold(scores, meta, grid) → float`** — picks the cutoff whose
  alert-*episode* rate is closest to the budget (≈1 alert/patient-day). It counts
  episodes (consecutive alerting rows collapsed into one) rather than rows, which
  keeps the budget stable when you change `GRID_INTERVAL_MIN`. `grid` lets it work
  on the model's [0,1] probabilities *or* NEWS2's integer scale.
- **`operating_point(y_prob, meta, onsets, thr) → dict`** — converts a threshold
  into the metrics that decide adoption: sensitivity, mean lead time, false alarms
  per patient-day, and the achieved alert rate (all episode-based, so grid-stable).
- **`group_report(y, scores, meta, onsets, patient_group, thr) → {group: metrics}`**
  — the fairness audit: the same metrics per group at one shared threshold, so a
  lower number for a group means genuinely worse detection, not different scoring.

---

## Part 5 — the product

### `trajectory_engine.py` — risk as a function of time (Phase 4, the core idea)
- **`trajectory(patient_df, est) → DataFrame[time, risk]`** — walks one patient's
  timeline, calls `featurize_at` at each step, scores it with the loaded pipeline,
  and returns the risk curve. Because features are past-only and the label was
  forward-looking, the curve is allowed to rise *before* the event.
- **`top_drivers(est, k) → list`** — the k biggest contributing features.

### `dashboard.py` — the screen (Phase 5, presentation only)
A Streamlit app. `_load()` (cached) reads the saved pipeline + threshold and
regenerates the patients for display. The body lets you pick a patient and draws
the risk curve, the vitals, and an interpretation panel (current risk, top
drivers, true onset). It only *reads* artifacts and calls `trajectory_engine` —
it never retrains or rebuilds features, so the screen shows exactly what the model
produced. Run with `streamlit run dashboard.py` after `run_pipeline.py`.

---

## Part 6 — the conductor and the guardrails

### `run_pipeline.py` — the end-to-end entry point
`main()` runs the line in seven steps: (1) generate, (2) tidy + label + featurize
+ split, (3) train and save, (4) evaluate vs NEWS2/qSOFA, (5) the matched-budget
operating-point table, (6) the SES fairness audit, (7) one example trajectory.
Seeded, so numbers are identical every run. **This is the file you run.**

### `test_leakage.py` / `test_labeling.py` — the guardrails
- **`test_leakage`** — features a patient at t=60, then corrupts the *future* rows
  and asserts the feature vector is unchanged (proves no peeking), plus that the
  vector is complete and finite.
- **`test_labeling`** — asserts the label lands in exactly the right forward
  window, that blanking and post-onset rows are dropped, and that a stable patient
  is all negatives.

Run both with `python test_leakage.py && python test_labeling.py` (or `pytest`).

---

## How to run and experiment

```bash
pip install -r requirements.txt
python run_pipeline.py          # the whole line; writes artifacts/
python -m streamlit run dashboard.py      # the viewer
python test_leakage.py && python test_labeling.py
```

To try the fairness experiment: set `USE_SES_AS_FEATURE = True` in `config.py`
and re-run `run_pipeline.py`; compare the low-SES sensitivity row before and after.

To test the medication-context idea: set `USE_MED_FEATURES = False` and re-run —
the model loses the flags that tell it a vital is drug-supported and does slightly
worse (~0.70 vs ~0.73 AUPRC), and would do worse still on the post-onset stretch
the pre-onset label doesn't score.

To add a new risk factor (say, immunosuppression): give it a coefficient in
`config.GenConfig` and add one line to the `risk_terms` dict in
`synthetic.generate()`. That alone makes it move each patient's *true* sepsis
chance. To also let the model *see* it as an input, add it to the generated
attributes (`schema.py` + `synthetic.py`) and to `STATIC_FEATURES` in `config.py`,
and it flows through the shared feature path automatically.

### How each risk factor shapes the sepsis chance

The per-patient probability is a logistic model summed from the `risk_terms`
contributions. Measured over a large sample, each factor moves it as expected
(age is the strongest driver):

| factor | effect on sepsis chance |
| --- | --- |
| age | ~4% (under 45) → ~15% (80+) |
| comorbidity count | ~5% (0) → ~12% (3) |
| SES | ~12% (low) vs ~5% (high) |
| sex | ~7% (female) vs ~9% (male) — modest, controlled |
