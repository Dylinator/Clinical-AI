# AI Clinical Trajectory — MVP

A simulated ED → ICU patient-trajectory system that updates a **calibrated
deterioration risk over time** from synthetic time-series vitals, compares itself
against the clinical early-warning scores it has to beat (NEWS2, qSOFA), and
explains and visualises the result.

It is **not** a medical-grade model and makes no claim of clinical validity — the
synthetic generator knows the answer by construction (see the circularity caveat
in the notes). The point is a correct, honest *pipeline* and *system design*.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py          # generate → train → evaluate → save artifacts
streamlit run dashboard.py      # interactive risk-trajectory viewer
pytest                          # the two invariants that matter most
```

`run_pipeline.py` is seeded, so the numbers are identical every run. Latest
(1200 synthetic patients; sepsis prevalence ~9%, so ~2% of rows are positive):

| model      | AUPRC | AUROC |
|------------|-------|-------|
| our model  | 0.72  | 0.97  |
| NEWS2      | 0.06  | 0.72  |
| qSOFA      | 0.03  | 0.55  |

The model beats NEWS2 by a wide margin because it reads the **trend** (deltas/
slopes through the prodrome) while the clinical scores see only the current level
— the whole thesis of the project. AUPRC is lower than on an artificially balanced
cohort because deterioration is now realistically rare — which is exactly why
AUPRC, not AUROC, is the honest headline. `run_pipeline` also prints a
matched-alert-rate operating-point table and a fairness audit. Real evaluation
waits for MIMIC-IV / eICU.

### Fairness audit

Sepsis prevalence scales from a realistic base by SES, so low-SES patients develop
sepsis ~2.4× as often as high-SES (~12% vs ~5%) — the disparity starts before the
model. Low SES also degrades data quality (more missing readings), so the model
gives low-SES patients noisier predictions and more false alarms per day. Set
`USE_SES_AS_FEATURE = True` in `config.py` and re-run to test whether adding SES as
a feature helps — it only partly does, because the root cause is the data, not the
label. See section 10 of the notes.

## Module map

| File | Role |
|------|------|
| `config.py` | every tunable in one place (horizon, gap, grid, seed, feature list, threshold budget) |
| `schema.py` | `Patient` / `Observation` / `Intervention` dataclasses; canonical names + units |
| `synthetic.py` | seeded generator: noise, overlap, red herrings, a prodrome, sparse informative missingness, static risk factors (age/sex/comorbidities/SES) + an SES-linked data-quality penalty, and treatment events that *mask* specific vitals (Phase 7) |
| `timeline_engine.py` | resample events onto a uniform `(patient_id, time)` grid — resampling only |
| `labeling.py` | forward-looking label with horizon + gap (the blanking window) |
| `features.py` | `featurize_history(...)` — the ONE pure, past-only feature builder (vitals + static/history features; train == serve) |
| `splits.py` | patient-level train/test split (never by row) |
| `model.py` | Pipeline + calibration, fit / predict / persist |
| `baselines.py` | NEWS2 and qSOFA |
| `evaluate.py` | AUPRC, AUROC, Brier, scale-agnostic threshold selection, lead time, false alarms/day, per-group fairness audit (`group_report`) |
| `trajectory_engine.py` | risk as a function of time — the core differentiator |
| `dashboard.py` | Streamlit presentation, reads saved artifacts only |
| `run_pipeline.py` | wires it all together end to end |
| `test_leakage.py` / `test_labeling.py` | the past-only and forward-window invariants |

## Design rules this scaffold enforces

1. **One inference path** — `features.py` is imported by both training and the trajectory engine, so they cannot drift.
2. **Identity is carried** — every row keeps `(patient_id, time)` through to the prediction.
3. **One home for tunables** — `config.py`.
4. **Resampling ≠ labeling** — separate modules.
5. **Model / baselines / evaluation are separate** — "did we beat NEWS2?" is a first-class step.
6. **Source is an adapter** — a future `mimic.py` emits the same schema; nothing downstream changes.
8. **Threshold is chosen, not hardcoded** — `evaluate.py` picks it for an alert budget.

## Things to iterate on next (good mentor conversations)

- Swap RandomForest for XGBoost in `model.py` (one line).
- Real SHAP in place of `model.global_importances` for per-step drivers.
- Deeper equity work: the SES fairness audit is built in; next is per-group thresholds and testing whether richer low-SES data closes the gap.
- Treatment-response is built (medication **masking** of vitals + `on_*` context flags so the model isn't fooled by drug-supported numbers); the next causal step is modelling treatment *efficacy*, which needs causal methods (section 6).
- The Phase 7 treatment-response layer — honest *association*, not causal effect.
- Then real data (MIMIC-IV): write `ingest/mimic.py` emitting the same schema.
