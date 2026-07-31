# Presentation Notes — AI Clinical Trajectory (Sepsis MVP)

*10–12 min • keep this on your second monitor • the dashboard is the thing you screen-share*

**Flow:** Intro → Problem → Why → Solution → **Live Demo** → Team → Impact & Future → Close & Q&A
**Timing:** 1 · 1.5 · 1 · 2 · 3 · 1 · 1.5 · 1 min

> Fill in every `[ ]` before you present. Anything in `[brackets]` is a placeholder for you.

---

## 1. Introduction *(~1 min)*
**Say:**
- "Hi, I'm **Dylan**, and this is **my project**."
- The hook (your line): *"People often come into the Emergency Room with complications that can be time-sensitive and life-threatening — the gap between arriving and getting treated can make all the difference."*
- One sentence on what we built: *"We built an AI that watches a patient's trajectory and flags who is about to deteriorate — roughly **2 hours before it happens** — starting with sepsis."*
- Roadmap the talk: *"I'll walk through the problem, our solution, a live demo, and where this goes next."*

---

## 2. Problem Statement *(~1.5 min)*
**Say:**
- Sepsis is one of the leading causes of in-hospital death, and it moves fast — **every hour** of delayed treatment raises the risk of dying.
- The tools hospitals use today — **qSOFA** and **NEWS2** — are checklists on a *single snapshot* of vitals. They only fire once the patient has *already* crossed a danger threshold.
- **The gap:** by the time a snapshot looks bad, the patient is often already crashing. Nobody is watching the *trend* that leads up to it.

**One-liner to land it:** *"We're not trying to detect sepsis. We're trying to predict it — before the numbers ever look scary."*

---

## 3. Why This Problem *(~1 min)*
**Say:**
- **Why it matters:** [your personal reason — a story, an interest in medicine/ML, a stat that hit you]
- **Why it's a good ML problem:** it's a *prediction* problem over time-series data — exactly what a model can learn that a threshold checklist can't.
- **Why now / why us:** the impact is huge and *measurable*, and there's real public hospital data we can prove the concept on.

---

## 4. Your Solution *(~2 min)*
**Say (your description):** *"A clinical-trajectory system using a random-forest model and a custom transformer model to predict deterioration and give a risk probability ~2 hours before onset. This MVP focuses on sepsis, using qSOFA and NEWS2 as the baselines to beat."*

**Break it down (the clear version):**
- **Two models:** a **RandomForest** (the reliable workhorse) and a **from-scratch encoder-only transformer** (the architecture that pays off as data grows).
- **What makes it serious — 5 points:**
  1. **Reads trends, not snapshots** — and *labs lead the vitals*, so it sees trouble coming early.
  2. **Forward-looking labels** — it predicts ~2h ahead with a gap, so it's real prediction, not restating the present.
  3. **Leakage-safe by construction** — only past data feeds a prediction, and a **unit test proves it**.
  4. **Calibrated** — when it says "72%," it means 72% (Brier ≈ 0.005).
  5. **Runs on real multi-center hospital data** through the *same code* (an adapter pattern).

**Headline results — say "synthetic shows the concept cleanly, real data shows the rigor":**

*Synthetic (clean concept) — `python run_pipeline.py`*

| model | AUPRC | AUROC | sens @ ~1 alert/day | lead time |
|-------|-------|-------|---------------------|-----------|
| **Our model (RF)** | **0.94** | **0.996** | **100%** | **~192 min** |
| NEWS2 | 0.08 | 0.75 | 42% | ~71 min |
| qSOFA | 0.03 | 0.53 | 0% | — |

*Real (rigor) — eICU, 2,393 ICU stays, real 4-organ Sepsis-3 label*

| model | AUPRC | AUROC | sens | Brier |
|-------|-------|-------|------|-------|
| **RandomForest** | 0.10 | **0.85** | 92% | 0.006 |
| Transformer+SSL | 0.04 | 0.81 | 96% | 0.006 |
| NEWS2 / qSOFA | ~0.01 | ~0.64 | 39% | — |

**Takeaway line:** *"Both of our models beat the clinical scores, and both are well-calibrated. On real data the tree beats the transformer — which is the honest, expected result at this data scale."*

---

## 5. Live Demonstration *(~3 min — this is the star; slow down here)*
Screen-share the dashboard: `python -m streamlit run dashboard.py`

1. **Open on the synthetic Patient tab.** SAY: *"Every patient is a risk curve over time."*
2. **Pick a treated-sepsis patient.** Point at the curve **climbing toward the red alert line BEFORE** the true-onset marker. SAY: *"That lead time — that gap before onset — is the entire value."*
3. **Point at the vitals + labs panels.** SAY: *"It reads trends, not snapshots — and the labs move before the vitals do."*
4. **Point at the vasopressor note.** SAY: *"Blood pressure looks normal here — but it's drug-supported. The model knows that, so risk stays high."* *(this is the treatment-masking story — it's a strong moment)*
5. **Switch to the Model Benchmark tab.** SAY: *"And here it's benchmarked honestly against the clinical scores and the transformer."*

> **Fallback:** have a 60–90s screen recording of these 5 steps saved, in case the live app misbehaves.

---

## 6. Team Roles & My Contributions *(~1 min)*
**Say:**
- **[Your name] — [your role]:** [your individual contributions — e.g. model/pipeline design, the transformer, the dashboard, data adapters]
- **[Teammate name] — [role]:** [what they did]
- **[Teammate name] — [role]:** [what they did]
- If solo or mostly solo, say so plainly and describe the pieces you built end-to-end (data → labels → models → dashboard).

> Only claim what's true — judges reward honesty about who did what.

---

## 7. Impact & Future Scope *(~1.5 min)*
**Say (impact):**
- Earlier warning means earlier treatment — and in sepsis, earlier treatment saves lives.
- It's calibrated and leakage-safe, so the risk numbers are trustworthy, not just flashy.

**Say (roadmap — one breath each):**
- **Clinical Trajectory Transformer** — turn every chart into one shared "event stream" so any hospital's data speaks the same language *(started — see `events.py`)*.
- **Merge more databases** — train across hospitals, validate by holding one out entirely *(leave-one-source-out — already built)*.
- **More complications** — extend beyond sepsis to AKI (kidney) and respiratory failure.

**Be honest, proactively (say the limits — this is a strength):**
- **Not clinically valid yet** — synthetic data knows the answer by design; real data is a limited-feature demo. No deployment claim.
- **Real AUPRC is low** because sepsis is rare (~0.6% of rows) — that's *why* AUROC + calibration are the honest headline.
- **Reduced SOFA** (4 of 6 organs) and antibiotics used as an infection proxy — a publishable label needs the full six-organ concept + cultures.
- **Tree beats transformer** at this scale — the transformer is the bet that pays off with *more data*, not an instant win.

---

## 8. Closing & Q&A *(~1 min + questions)*
**Close:** *"So — an honest, calibrated, leakage-safe early-warning system that already beats the standard clinical scores, and a clear path from here to a real multi-hospital clinical-trajectory model. Thank you — happy to take questions."*

**Q&A prep (rehearse these):**
- *"Is it clinically valid?"* → No, and I can tell you *exactly* why (see limits). Being able to say precisely why is the point.
- *"Why does the tree beat the transformer?"* → Data scale. Deep models need volume; this is consistent with the literature.
- *"Why is AUPRC low?"* → Sepsis is rare, which floors AUPRC. AUROC 0.85 + good calibration show the signal is real.
- *"What exactly is the label?"* → Sepsis-3: suspected infection + a SOFA organ-dysfunction rise ≥2, defined *forward in time*.
- *"Is there data leakage?"* → No — features are past-only, and a unit test enforces it on every run.

---

## Commands that must work (rehearse before you present)
```
python run_pipeline.py                               # synthetic headline + fairness + a trajectory
python -m streamlit run dashboard.py                 # the live demo (two tabs)
python test_leakage.py && python test_labeling.py    # the invariants that prove it's honest
```
**Record a 60–90s screen capture of the dashboard as a live-demo fallback.**
