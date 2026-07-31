# MVP Demo Cheat-Sheet — AI Clinical Trajectory

**One-line pitch:** *A clinical early-warning system that predicts deterioration as a
**calibrated risk trajectory over time** — trained on synthetic data, validated on
real multi-center ICU data, and honest about exactly what it can and can't claim.*

---

## What it is (say this first)
A clinical trajectory system using a random forest model and a custom transformer model to predict symptom deterioration and give risk probability of different causative complications 2 hours before onset based on a number of patient factors, this mvp focuses on Sepsis, using qSOFA and NEWS2 (methods of detecting sepsis and other things such as organ dysfunction) as baseline accuracy and diagnoses times for the models to beat.

## Headline results

**Synthetic (the concept, cleanly) — `python run_pipeline.py`:**

| model | AUPRC | AUROC | sens @ ~1 alert/day | lead |
|-------|-------|-------|---------------------|------|
| Our model (RF) | 0.94 | 0.996 | 100% | ~192 min |
| NEWS2 | 0.08 | 0.75 | 42% | ~71 min |
| qSOFA | 0.03 | 0.53 | 0% | — |

**Real (the rigor) — eICU, 2,393 stays, real 4-organ Sepsis-3 label:**

| model | AUPRC | AUROC | sens | Brier |
|-------|-------|-------|------|-------|
| RandomForest | 0.10 | **0.85** | 92% | 0.006 |
| Transformer+SSL | 0.04 | 0.81 | 96% | 0.006 |
| NEWS2 / qSOFA | ~0.01 | ~0.64 | 39% | — |

Both beat the clinical scores; both are **well-calibrated**. On real data the tree
beats the transformer — the expected, literature-consistent result at this scale.

## Demo script (~90 sec — the dashboard is the star)
1. Open the dashboard (synthetic). **"Each patient is a risk curve over time."**
2. Pick a **treated sepsis** patient. Point at the curve climbing toward the red
   alert line **before** the true-onset marker → *"that lead time is the value."*
3. Point at the **vitals + labs** panels → *"it reads trends, not snapshots — labs
   lead the vitals."*
4. Point at the **vasopressor note** → *"BP looks normal here, but it's drug-
   supported — the model knows, so risk stays high."* (the treatment-masking story)
5. Switch to the **Model benchmark tab** → *"honestly benchmarked vs the clinical
   scores and a transformer."*

## What makes this serious (your differentiators)
- **Leakage-safe by construction** — features are past-only; a unit test proves it.
- **Forward-looking labels** with a gap+horizon — real prediction, not restatement.
- **Real Sepsis-3 derivation** (suspected infection + 4-organ SOFA rise ≥2), not a shortcut.
- **Runs on real multi-center data** through the same code (adapter pattern).
- **Calibrated probabilities** (Brier ~0.005) — "72%" means 72%.
- **Fairness audit** across SES strata; a from-scratch **time-aware transformer** with
  self-supervised pretraining, benchmarked honestly.

## Say these limitations *proactively* (honesty is the point)
- **Not clinically valid.** Synthetic data knows the answer by construction; real data
  is a limited-feature demo. No deployment claim.
- **Real AUPRC is low** because sepsis is rare (~0.6% of rows) — that's why AUROC +
  calibration are the honest headline, not AUPRC.
- **Reduced SOFA** (4 of 6 organs — no GCS/PaO2·FiO2) and antibiotics-as-infection
  proxy; a publishable label needs the full six-organ concept + cultures.
- **Tree beats transformer** at this scale — the transformer is the architecture that
  pays off with more data, not an instant win.

## Q&A prep
- *"Clinically valid?"* → No, and here's exactly why (above). That I can say precisely why is the point.
- *"Why does the tree beat the transformer?"* → Data scale; consistent with the field; deep models need volume.
- *"Why is AUPRC low?"* → Rare events floor it; AUROC 0.85 + calibration show real signal.
- *"What's the label?"* → Sepsis-3: suspected infection + a SOFA organ-dysfunction rise, forward-looking.

## Roadmap (one slide)
Clinical Trajectory **Transformer** (event-stream representation started) → merge more
databases (leave-one-source-out validation built) → more features & more onset types
(AKI, respiratory failure).

## The three commands that must work (rehearse them)
```
python run_pipeline.py                      # synthetic headline + fairness + a trajectory
python -m streamlit run dashboard.py        # the demo (two tabs)
python test_leakage.py && python test_labeling.py   # the invariants
```
Record a 60–90s screen capture of the dashboard as a live-demo fallback.
