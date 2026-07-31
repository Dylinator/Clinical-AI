"""
benchmark.py — the honest head-to-head: transformer vs RandomForest vs NEWS2/qSOFA.

The notes are emphatic that at MVP scale a transformer usually LOSES to a good
tabular model, and that the result worth reporting is a fair comparison on
identical patients — not a cherry-picked win. So this harness:

    * builds ONE patient-level split (splits.py, seeded);
    * trains the RandomForest on the tabular feature matrix (features.py) and the
      transformer on the sequence view (sequences.py) of the SAME labeled rows;
    * scores NEWS2 / qSOFA from the same rows;
    * reports AUPRC / AUROC / Brier and a matched-alert-rate operating point for
      every method, on the identical test patients.

Data source is a flag, so the exact same comparison runs on synthetic OR on the
real PhysioNet-2019 cohort (Rule 6 — nothing downstream of ingestion cares):

    python benchmark.py                 # synthetic (config.GEN.n_patients)
    python benchmark.py --real          # PhysioNet 2019 (downloads a subset)
    python benchmark.py --real --limit 800 --grid 60

`--grid` sets GRID_INTERVAL_MIN for the run (real data is hourly -> 60); it only
affects the per-day alert arithmetic, computed consistently for all methods.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config



def _load_one(source: str, limit: int | None):
    if source == "physionet":
        from ingest import physionet2019
        return physionet2019.load(limit=limit)
    if source == "eicu":
        from ingest import eicu
        return eicu.load(limit=limit)
    import synthetic                          # synthetic
    if limit:
        config.GEN.n_patients = limit
    return synthetic.generate()


def _load_frame(sources: list[str], limit: int | None):
    """Load and MERGE one or more sources into one cohort. Because every adapter
    emits the SAME schema and its own forward-looking onset (Rule 6), merging is a
    concatenation — the elegant payoff of the adapter design. Re-ids patients so
    ids are unique across sources; returns `src_of` (patient_id -> source) so a
    leave-one-source-out split is possible. Returns (full, onsets, patients, src_of)."""
    import timeline_engine
    patients, src_of, nid = [], {}, 1
    for src in sources:
        for p in _load_one(src, limit):
            p.id = nid
            src_of[nid] = src
            patients.append(p)
            nid += 1
    full = timeline_engine.to_frame(patients)
    onsets = timeline_engine.onset_map(patients)
    return full, onsets, patients, src_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="data source, or comma-separated to MERGE "
                         "(e.g. 'eicu,physionet'); default synthetic")
    ap.add_argument("--real", action="store_true", help="alias for --source physionet")
    ap.add_argument("--eicu", action="store_true", help="alias for --source eicu")
    ap.add_argument("--holdout", default=None,
                    help="leave-one-source-out: train on the other merged sources, "
                         "TEST only on this source (external validation)")
    ap.add_argument("--limit", type=int, default=None, help="max patients per source")
    ap.add_argument("--grid", type=int, default=None,
                    help="GRID_INTERVAL_MIN for alert math (real hourly data -> 60)")
    ap.add_argument("--epochs", type=int, default=None, help="override transformer finetune epochs")
    ap.add_argument("--pretrain-epochs", type=int, default=None,
                    help="self-supervised pretraining epochs (0 disables pretraining)")
    ap.add_argument("--no-pretrain", action="store_true",
                    help="skip self-supervised pretraining (ablation)")
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="cap labeled TRAINING rows (keeps all positives, subsamples "
                         "negatives) — same rows for RF and transformer; test stays full")
    ap.add_argument("--save", action="store_true",
                    help="persist RF + transformer + metrics to artifacts/ for the dashboard")
    args = ap.parse_args()

    # resolve the source list (comma-separated --source merges; then aliases; else synthetic)
    if args.source:
        sources = [s.strip() for s in args.source.split(",") if s.strip()]
    else:
        sources = ["eicu"] if args.eicu else ["physionet"] if args.real else ["synthetic"]

    if args.grid:
        config.GRID_INTERVAL_MIN = args.grid
    elif any(s in ("physionet", "eicu") for s in sources):
        config.GRID_INTERVAL_MIN = 60           # real sources are hourly-bucketed

    import labeling, features, splits, sequences
    import model as rf_mod
    import model_transformer as mt
    import baselines, evaluate

    if args.epochs:
        mt.EPOCHS = args.epochs
    if args.pretrain_epochs is not None:
        mt.PRETRAIN_EPOCHS = args.pretrain_epochs

    src_str = "+".join(sources)
    ho = f"  (leave-one-source-out: test on {args.holdout})" if args.holdout else ""
    print(f"=== benchmark on {src_str}{ho} ===")
    full, onsets, patients, src_of = _load_frame(sources, args.limit)
    labeled = labeling.add_labels(full, onsets)

    # Tabular view + sequence view of the SAME labeled rows.
    X, y, meta = features.build_training_matrix(full, labeled)
    seqs, seq_times, stat, y2, _ = sequences.build_sequences(
        full, labeled, max_len=mt.MAX_LEN)
    assert np.array_equal(y, y2), "row alignment between views drifted"

    # Split: leave-one-source-out (train on other sources, test on --holdout) for
    # external validation, else the usual seeded patient-level split.
    if args.holdout:
        src_series = meta["patient_id"].map(src_of)
        te = (src_series == args.holdout).to_numpy()
        tr = (src_series != args.holdout).to_numpy()
        if te.sum() == 0:
            raise SystemExit(f"--holdout {args.holdout!r} matched no patients "
                             f"(sources present: {sorted(set(src_of.values()))})")
        tr_ids = set(meta["patient_id"][tr]); te_ids = set(meta["patient_id"][te])
    else:
        tr_ids, te_ids = splits.split_patients(meta["patient_id"])
        tr = splits.mask_for(meta, tr_ids)
        te = splits.mask_for(meta, te_ids)

    # Optionally subsample TRAIN rows to a cap so the transformer is tractable at
    # scale: keep every (rare) positive, randomly drop negatives. Applied to the
    # SAME rows for RF and transformer, so the comparison stays fair; the TEST set
    # is never subsampled. Pretraining still sees ALL patients (it needs no labels).
    if args.max_train_rows is not None and tr.sum() > args.max_train_rows:
        rng = np.random.default_rng(config.SEED)
        tr_pos = tr & (y == 1)
        tr_neg = tr & (y == 0)
        n_neg_keep = max(0, args.max_train_rows - int(tr_pos.sum()))
        neg_idx = np.flatnonzero(tr_neg)
        keep_neg = rng.choice(neg_idx, size=min(n_neg_keep, len(neg_idx)), replace=False)
        tr = tr_pos.copy()
        tr[keep_neg] = True

    if len(sources) > 1:
        from collections import Counter
        print("per-source patients:", dict(Counter(src_of.values())))
    print(f"patients={len(patients)}  rows={len(X)}  positives={y.mean():.2%}")
    print(f"train rows={tr.sum()}  test rows={te.sum()}  "
          f"train positives={int(y[tr].sum())}  (grid={config.GRID_INTERVAL_MIN} min)\n")

    scores = {}

    # --- RandomForest (tabular) ---
    print("training RandomForest ...")
    rf = rf_mod.fit(X[tr], y[tr])
    scores["RandomForest"] = (rf_mod.predict_risk(rf, X[te]), None)

    # --- Transformer (sequence, time-aware, optionally pretrained) ---
    do_pretrain = (not args.no_pretrain) and mt.PRETRAIN_EPOCHS > 0

    seqs_tr = [s for s, m in zip(seqs, tr) if m]
    seqs_te = [s for s, m in zip(seqs, te) if m]
    times_tr = [t for t, m in zip(seq_times, tr) if m]
    times_te = [t for t, m in zip(seq_times, te) if m]

    # Standardizer fit on TRAIN labeled sequences only (no test leakage).
    standardizer = sequences.SeqStandardizer().fit(seqs_tr)

    backbone_state = None
    if do_pretrain:
        # Self-supervised on FULL timelines of TRAIN patients (labels unused) —
        # abundant unlabeled structure the classifier then fine-tunes from.
        pre_seqs, pre_times, _ = sequences.build_pretrain_sequences(
            full[full["patient_id"].isin(tr_ids)], max_len=mt.MAX_LEN)
        print(f"self-supervised pretraining on {len(pre_seqs)} train timelines ...")
        backbone_state = mt.pretrain(pre_seqs, pre_times, standardizer, verbose=True)

    print("fine-tuning Transformer (encoder-only, time-aware) ...")
    tmodel = mt.train(seqs_tr, times_tr, stat[tr], y[tr],
                      standardizer=standardizer, init_backbone_state=backbone_state,
                      verbose=True)
    label = "Transformer+SSL" if do_pretrain else "Transformer"
    scores[label] = (mt.predict_risk(tmodel, seqs_te, times_te, stat[te]), None)

    # --- clinical baselines ---
    news2, qsofa = baselines.score_matrix(X[te])
    scores["NEWS2"] = (news2, np.unique(news2))
    scores["qSOFA"] = (qsofa, np.unique(qsofa))

    # --- discrimination + calibration ---
    results = {}       # method -> metrics (collected for the dashboard)
    print(f"\n{'method':<16}{'AUPRC':>8}{'AUROC':>8}{'Brier':>8}")
    for name, (prob, _) in scores.items():
        m = evaluate.metrics(y[te], prob)
        brier = f"{m['Brier']:.3f}" if "Brier" in m else "   -"
        print(f"{name:<16}{m['AUPRC']:>8.3f}{m['AUROC']:>8.3f}{brier:>8}")
        results[name] = {"AUPRC": m["AUPRC"], "AUROC": m["AUROC"],
                         "Brier": m.get("Brier")}
    print(f"(test positive rate = {y[te].mean():.2%})\n")

    # --- matched operating point ---
    meta_te = meta[te].reset_index(drop=True)
    rf_threshold = 0.5
    print(f"operating point (~{config.TARGET_ALERTS_PER_DAY:.0f} alert/patient-day, same test split):")
    print(f"  {'method':<14}{'thresh':>8}{'alerts/d':>10}{'sens':>7}{'lead(min)':>11}{'FA/day':>8}")
    for name, (prob, grid) in scores.items():
        thr = evaluate.choose_threshold(prob, meta_te, grid=grid)
        op = evaluate.operating_point(prob, meta_te, onsets, thr)
        results[name].update({"sensitivity": op["sensitivity"],
                              "lead_min": op["mean_lead_time_min"],
                              "false_alarms_per_day": op["false_alarms_per_day"],
                              "threshold": op["threshold"]})
        if name == "RandomForest":
            rf_threshold = op["threshold"]
        print(f"  {name:<14}{op['threshold']:>8.2f}{op['alerts_per_day']:>10.2f}"
              f"{op['sensitivity']:>7.0%}{op['mean_lead_time_min']:>11.0f}"
              f"{op['false_alarms_per_day']:>8.2f}")

    print("\nNote: at this scale the calibrated tree is a strong, honest baseline; "
          "the transformer is expected to be competitive, not automatically better "
          "(see the notes, Section 7).")

    # --- optionally persist everything the dashboard needs ---
    if args.save:
        import json
        os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
        rf_mod.save(rf)                                     # artifacts/pipeline.joblib
        mt.save(tmodel, f"{config.ARTIFACT_DIR}/transformer.joblib")
        json.dump({"threshold": rf_threshold},
                  open(f"{config.ARTIFACT_DIR}/threshold.json", "w"))
        json.dump(sorted(int(i) for i in te_ids),
                  open(f"{config.ARTIFACT_DIR}/test_ids.json", "w"))
        manifest = {
            "source": src_str,                  # single source, or "a+b" if merged
            "holdout": args.holdout,
            "limit": args.limit,
            "grid_interval_min": config.GRID_INTERVAL_MIN,
            "max_hours": 72 if any(s in ("physionet", "eicu") for s in sources) else None,
            "n_patients": len(patients),
            "transformer_label": label,
            "test_positive_rate": float(y[te].mean()),
            "metrics": results,
        }
        json.dump(manifest, open(f"{config.ARTIFACT_DIR}/benchmark.json", "w"), indent=2)
        print(f"\nsaved RF + transformer + benchmark.json to {config.ARTIFACT_DIR}/  "
              f"→  python -m streamlit run dashboard.py")


if __name__ == "__main__":
    main()
