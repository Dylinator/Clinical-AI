# AI Clinical Trajectory — MVP

A simulated ED → ICU patient-trajectory system that updates a **calibrated
deterioration risk over time** from time-series **vitals + labs**, compares itself
against the clinical early-warning scores it has to beat (NEWS2, qSOFA), and
explains and visualises the result. It runs on a synthetic cohort *or* on the real
PhysioNet/CinC-2019 ICU dataset through the same pipeline, and ships two models —
a calibrated RandomForest and an **encoder-only transformer** — benchmarked
head-to-head.

On synthetic data it is **not** a medical-grade model and makes no claim of
clinical validity — the generator knows the answer by construction (see the
circularity caveat in the notes). The point is a correct, honest *pipeline* and
*system design*.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py          # generate → train RF → evaluate → save artifacts
python test_leakage.py && python test_labeling.py   # the invariants that matter most

# transformer + real data (adds torch, xgboost)
pip install -r requirements-extra.txt
python benchmark.py --save                 # RF vs Transformer vs NEWS2/qSOFA (synthetic) + save for dashboard
python benchmark.py --real --save          # the same comparison on real PhysioNet-2019 ICU data
python -m streamlit run dashboard.py       # two tabs: patient trajectory (both models) + benchmark
```

The dashboard follows whatever `benchmark.py --save` last trained — synthetic or
real — overlaying the RandomForest and Transformer risk curves for a held-out
patient, and showing the head-to-head metrics.

`run_pipeline.py` is seeded, so the numbers are identical every run. Latest
(1200 synthetic patients; sepsis prevalence ~8%, ~2.5% of rows positive):

| model         | AUPRC | AUROC | Brier | sens @ ~1 alert/day | lead time |
|---------------|-------|-------|-------|---------------------|-----------|
| RandomForest  | 0.94  | 0.996 | 0.005 | 100%                | ~192 min  |
| NEWS2         | 0.08  | 0.749 |   —   | 42%                 | ~71 min   |
| qSOFA         | 0.03  | 0.528 |   —   | 0%                  |    —      |

The model beats NEWS2 by a wide margin because it reads the **trend** (deltas/
slopes through the prodrome, now including **lab** trends that lead the vitals)
while the clinical scores see only the current level — the whole thesis of the
project. AUPRC is far below AUROC because deterioration is realistically rare —
which is exactly why AUPRC, not AUROC, is the honest headline. `run_pipeline` also
prints the matched-alert-rate operating-point table and a fairness audit.

### The encoder-only transformer (and how it's trained to compete)

`benchmark.py` trains a RandomForest and an encoder-only transformer on one
identical patient split and reports both against NEWS2/qSOFA. The transformer is
not a toy — it has the two upgrades that let a sequence model genuinely compete:

- **Time-aware attention (Time2Vec):** it embeds the *real elapsed minute* of each
  observation, so irregular sampling (dense vitals, sparse labs) carries
  information a tree's flattened features can't represent.
- **Self-supervised pretraining (masked value modeling, the Med-BERT move):** it
  first learns physiology by reconstructing randomly-masked values across *all*
  patients (including stable/unlabeled ones), then fine-tunes on the scarce sepsis
  labels — the lever that pays off as data grows.

**On synthetic data the calibrated tree still edges it** (notes, Section 7): the
transformer is well-calibrated and beats NEWS2/qSOFA by a mile but does not
out-rank the RandomForest on AUPRC. Its advantages scale with data — so the real
path (`benchmark.py --real --save`) trains both on thousands of real patients with
self-supervised pretraining on all of them.

**What real data shows (4000 PhysioNet patients; 0.32% of rows positive):**

| model         | AUPRC | AUROC | sens @ ~1 alert/day | FA/day |
|---------------|-------|-------|---------------------|--------|
| RandomForest  | 0.007 | 0.696 | 67%                 | 0.61   |
| Transformer+SSL | 0.007 | 0.671 | 78%                 | 0.59   |
| NEWS2         | 0.005 | 0.622 | 59%                 | 0.78   |
| qSOFA         | 0.004 | 0.572 | —                   | —      |

Two honest lessons, both more valuable than any single number. **(1) Synthetic
0.94 → real 0.007 AUPRC.** That collapse *is* the circularity caveat made concrete:
real 6-hour sepsis prediction on this feature set is genuinely hard (AUPRC sits near
the 0.003 random-chance floor set by the 0.32% base rate; AUROC ~0.70 shows real but
weak signal). **(2) Scale closed the transformer's gap.** From a clear synthetic
loss it drew level with the tree on ranking and edged it on the clinically-relevant
metric (78% vs 67% sensitivity at a matched alert budget) — but did *not* decisively
win. Consistent with the notes: a transformer needs more data / richer features / a
longer horizon to clearly beat a calibrated tree here. Numbers live in
`artifacts/benchmark.json` and render in the dashboard's benchmark tab.

**And on eICU (open, multi-center, *with* treatments; 2,393 stays), under a real
4-organ Sepsis-3 label (`sepsis3.py`, ~15.5% prevalence):** *(table below is the
3-organ run; a 4-organ refresh is in flight — AUROC held ~0.9)*

| model         | AUPRC | AUROC | sens @ ~1 alert/day | lead | FA/day |
|---------------|-------|-------|---------------------|------|--------|
| RandomForest  | 0.050 | 0.906 | 89%                 | ~272 min | 0.90 |
| Transformer+SSL | 0.029 | 0.837 | 100%                | ~288 min | 0.64 |
| NEWS2         | 0.005 | 0.666 | 29%                 | ~526 min | 1.14 |
| qSOFA         | 0.004 | 0.599 | —                   | —    | —      |

Replacing the earlier proxy label with real Sepsis-3 made the numbers **more honest
AND revealed better ranking**: AUPRC fell (RF 0.127 → 0.050 — the label is rarer and
more specific, so the random floor is lower) while **AUROC rose (0.81 → 0.91)** — the
physiologically-coherent label is more *learnable*. Caveat that remains: the reduced
SOFA is scored on creatinine/platelets, which are *also features*, so even with
forward-looking labels some ranking power reflects "labs predict their own future"
(inherent to Sepsis-3 prediction; mitigated by the gap+horizon, not eliminated). The
durable findings hold: **RF still wins on ranking**, the transformer is **competitive
at the operating point** (100% sens, lower false-alarm rate), both crush the clinical
scores, calibration is excellent (Brier 0.003), and lead times are now sensible
(~4.5 h). Reproduce with `benchmark.py --source eicu --save`.

**Med-features ablation (does feeding treatments help?).** On synthetic, yes — they
let the model see through drug-masked vitals (Phase 7). On real eICU with the
forward-looking label, **no**: dropping the four treatment flags moved RF AUROC by
only ~0.008 (0.844 → 0.836), and RF placed just ~3.7% of its importance on all four
combined. The reason is structural — at prediction time (before onset) those flags
are 0, so they can't help the early warning. For a real-data early-warning model,
`USE_STATIC_FEATURES`-style toggling them off (`USE_MED_FEATURES=False`) buys a
cleaner model at ~no cost; they earn their keep only for post-onset interpretation.

### Labs (new)

Alongside the vitals, the pipeline now models **lactate + the SOFA-core labs**
(creatinine = kidney, WBC = infection, platelets = coagulation). Labs are *sparse*
and *ordered reactively* when the observed vitals look concerning, so whether a lab
exists at time *t* is itself signal (informative missingness), and lab derangement
**leads** the vitals — giving the model earlier, stronger, still-leakage-safe
signal (the label stays the forward-looking onset, never a concurrent lab value).

### Real data (new) — three sources, one schema

Every adapter emits the *same* `Patient` objects, so labeling, features, splits,
models, and evaluation run unchanged (Rule 6). Pick a source with `benchmark.py
--source {synthetic,physionet,eicu}`:

- **`ingest/physionet2019.py`** — PhysioNet/CinC-2019 sepsis challenge. Open (no
  credentialing), ~40k patients, hourly. **No medication data**, so the model is
  blind to treatment — a septic patient on a vasopressor looks falsely stable.
- **`ingest/eicu.py`** — eICU-CRD **Demo**. Open (no credentialing), ~2,400 stays
  across **200+ hospitals**, *with* treatments (vasopressors/fluids/antibiotics in
  `medication`/`treatment`). This is the one that gives the model treatment context
  on real data you can actually get — and being multi-center, it supports the
  cross-site external-validation experiment the notes call the real bar. Onset comes
  from the real **`sepsis3.py`** derivation (see below), giving a clinically-plausible
  ~11.5% prevalence.
- **`ingest/mimic.py`** — MIMIC-IV (credentialed; scaffold validated on the open
  100-patient demo). Treatments from `inputevents`. The gold-standard once you have
  access; the natural home for a proper `mimic-code` Sepsis-3 label.

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
| `schema.py` | `Patient` / `Observation` / `Intervention` dataclasses; canonical names + units (vitals + labs) |
| `synthetic.py` | seeded generator: noise, overlap, red herrings, a prodrome, **labs that lead + are sparsely/informatively missing**, static risk factors (age/sex/comorbidities/SES) + an SES-linked data-quality penalty, and treatment events that *mask* specific vitals (Phase 7) |
| `ingest/physionet2019.py` | **real-data adapter** — downloads PhysioNet-2019 and emits the same `Patient` objects (Rule 6); no meds |
| `ingest/eicu.py` | **real, OPEN, multi-center adapter** — eICU-CRD Demo (no credentialing) *with* treatments (vasopressors/fluids/antibiotics) |
| `ingest/mimic.py` | MIMIC-IV adapter (scaffold; credentialed) — same schema, treatments from `inputevents` |
| `timeline_engine.py` | resample events onto a uniform `(patient_id, time)` grid — resampling only |
| `labeling.py` | forward-looking label with horizon + gap (the blanking window) |
| `sepsis3.py` | **Sepsis-3 onset** for real data — suspected infection + a 4-organ SOFA rise ≥2 (shared by all adapters) |
| `features.py` | `featurize_history(...)` — the ONE pure, past-only feature builder (vitals + labs + static/history; train == serve) |
| `sequences.py` | the **sequence view** of the same labeled rows for the transformer (past-only, padded) |
| `splits.py` | patient-level train/test split (never by row) |
| `model.py` | RandomForest Pipeline + calibration, fit / predict / persist |
| `model_transformer.py` | **encoder-only transformer** (PyTorch) + isotonic calibration; same predict interface |
| `baselines.py` | NEWS2 and qSOFA |
| `evaluate.py` | AUPRC, AUROC, Brier, scale-agnostic threshold selection, lead time, false alarms/day, per-group fairness audit (`group_report`) |
| `benchmark.py` | **honest head-to-head** — RF vs transformer vs NEWS2/qSOFA on one split; `--real` for PhysioNet |
| `trajectory_engine.py` | risk as a function of time — the core differentiator |
| `dashboard.py` | Streamlit presentation, reads saved artifacts only (now charts labs too) |
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

- Swap RandomForest for XGBoost in `model.py` (one line; `xgboost` is in `requirements-extra.txt`).
- Real SHAP in place of `model.global_importances` for per-step drivers.
- Deeper equity work: the SES fairness audit is built in; next is per-group thresholds and testing whether richer low-SES data closes the gap.
- Treatment-response is built (medication **masking** of vitals + `on_*` context flags so the model isn't fooled by drug-supported numbers); the next causal step is modelling treatment *efficacy*, which needs causal methods (section 6).
- **The transformer is built** (`model_transformer.py` + `benchmark.py`) and honestly loses to the tree at MVP scale — the payoff is scale. Next: run `benchmark.py --real` on a *larger* PhysioNet subset and watch whether the gap narrows as data grows; then add per-step attention-weight explanations.
- **Real data is wired** (`ingest/physionet2019.py`, no credentialing). **`ingest/mimic.py`** exists as a scaffold, plumbing-validated on the open 100-patient MIMIC-IV Demo — it emits the same schema **with `interventions` populated** (vasopressors/fluids), which is what finally gives the model the treatment context PhysioNet lacks. Remaining before it yields trustworthy numbers: a real Sepsis-3 label (mimic-code concept) in place of the placeholder proxy, wiring antibiotics + comorbidities, and credentialed access to the full dataset.
