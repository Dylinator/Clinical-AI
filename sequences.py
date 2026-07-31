"""
sequences.py — build padded, past-only SEQUENCES for the transformer.

The RandomForest consumes one flat feature vector per labeled row (features.py).
A sequence model instead consumes the ordered history of channel readings UP TO t.
This module builds those sequences from the SAME labeled rows, the SAME channels,
and the SAME past-only rule — so the transformer is trained and evaluated on an
identical task to the tabular model (a fair head-to-head, Rule 5).

For a labeled row (patient_id, t) we take every grid step <= t and, per step, emit:
    [ value_c for c in CHANNELS ]            # carry-forward imputed, then standardized
  + [ was_missing_c for c in LABS ]          # informative-missingness, per lab
  + [ dt_since_prev_step ]                    # normalized time gap (irregular-sampling aware)
Static features (age/sex/…) are constant over the stay, so they are returned once
per sequence as a separate vector and fused at the model's head, not repeated at
every timestep.

Leakage safety mirrors features.py: only rows with time <= t are ever read, and
carry-forward uses only past values. test_leakage covers featurize_at; the same
slice (df[df.time <= t]) is used here, so the guarantee carries over.

Outputs are plain numpy so model_transformer.py owns all torch specifics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config

# Per-step channel layout (order fixed here, mirrored by the model).
SEQ_VALUE_COLS = list(config.CHANNELS)
SEQ_MISSING_COLS = [f"{lab}_missing" for lab in config.LABS]
SEQ_FEATURES = SEQ_VALUE_COLS + SEQ_MISSING_COLS + ["dt"]
N_SEQ_FEATURES = len(SEQ_FEATURES)

# Static vector fused at the head (same toggles as the tabular FEATURES list).
STATIC_COLS = (
    (config.STATIC_FEATURES if config.USE_STATIC_FEATURES else [])
    + (config.SES_FEATURES if config.USE_SES_AS_FEATURE else [])
    + (config.MED_FEATURES if config.USE_MED_FEATURES else [])
)
N_STATIC = len(STATIC_COLS)


def _carry_forward(values: np.ndarray, baseline: float) -> tuple[np.ndarray, np.ndarray]:
    """Carry the last observed value forward; baseline before the first reading.
    Returns (filled, was_missing). Vectorized (no Python loop) so building
    sequences stays tractable at scale — the classic forward-fill-by-index trick."""
    v = values.astype(float)
    was_missing = np.isnan(v)
    if was_missing.all():
        return np.full(v.shape, baseline, dtype=float), was_missing
    seen = ~was_missing
    idx = np.where(seen, np.arange(v.size), 0)
    np.maximum.accumulate(idx, out=idx)          # index of last real value at/before i
    filled = v[idx]
    first = int(seen.argmax())                   # positions before the first reading
    if first > 0:
        filled = filled.copy()
        filled[:first] = baseline
    return filled, was_missing


# The value channels occupy the first len(CHANNELS) columns of each step vector;
# only those are standardized / reconstructed in pretraining (the missing flags and
# dt are already O(1) indicators).
N_VALUE = len(config.CHANNELS)


def _one_sequence(hist: pd.DataFrame):
    """One patient's rows with time <= t (sorted) ->
        (seq[T, N_SEQ_FEATURES] float32, times[T] float32 minutes).
    `times` is the absolute minute of each step, kept separate so the model can
    build a time-aware (Time2Vec) embedding from real elapsed time — not just
    sequence position (this is what a tree's flattened features cannot use)."""
    times = hist["time"].to_numpy(dtype=float)
    T = len(times)
    out = np.zeros((T, N_SEQ_FEATURES), dtype=np.float32)

    col = 0
    missing_by_lab = {}
    for c in config.CHANNELS:
        filled, was_missing = _carry_forward(
            hist[c].to_numpy(dtype=float), config.BASELINE_FILL[c])
        out[:, col] = filled
        missing_by_lab[c] = was_missing
        col += 1
    for lab in config.LABS:                      # per-lab missing indicator
        out[:, col] = missing_by_lab[lab].astype(np.float32)
        col += 1
    dt = np.zeros(T, dtype=np.float32)            # gap since previous step
    if T > 1:
        dt[1:] = np.diff(times)
    # normalize; cap so a long real-data charting gap can't feed a huge value in
    out[:, col] = np.clip(dt / float(config.SLOPE_WINDOW_MIN), 0.0, 24.0)
    return out, times.astype(np.float32)


def _static_vector(hist: pd.DataFrame) -> np.ndarray:
    if N_STATIC == 0:
        return np.zeros(0, dtype=np.float32)
    last = hist.iloc[-1]
    return np.array([float(last[c]) for c in STATIC_COLS], dtype=np.float32)


def build_sequences(full_df: pd.DataFrame, labeled_df: pd.DataFrame,
                    max_len: int | None = None):
    """Mirror of features.build_training_matrix, but for sequences.

    Returns:
        seqs   : list of (T_i, N_SEQ_FEATURES) float32 arrays (variable length)
        times  : list of (T_i,) float32 absolute minutes per step
        stat   : (N, N_STATIC) float32
        y      : (N,) int
        meta   : DataFrame[patient_id, time]  (same identity carried through)
    `max_len` optionally caps history to the most recent `max_len` steps (keeps
    memory bounded on long real-data stays); None = full history.
    """
    # Cache each patient's sorted frame + time array once, then slice by binary
    # search per labeled row (not a boolean mask over the whole frame) — the same
    # O(N^2)-constant fix used in features.build_training_matrix, essential at scale.
    by_patient = {}
    for pid, g in full_df.groupby("patient_id"):
        g = g.sort_values("time")
        by_patient[pid] = (g, g["time"].to_numpy())

    seqs, times, stats = [], [], []
    for r in labeled_df.itertuples(index=False):
        g, tarr = by_patient[r.patient_id]
        cut = int(np.searchsorted(tarr, r.time, side="right"))
        lo = max(0, cut - max_len) if max_len is not None else 0
        hist = g.iloc[lo:cut]
        s, tm = _one_sequence(hist)
        seqs.append(s)
        times.append(tm)
        stats.append(_static_vector(hist))

    # defensive: static isn't carry-forward imputed, so a missing static value
    # (e.g. an unknown real-data age) would arrive as NaN and NaN the net — zero it.
    stat = (np.nan_to_num(np.vstack(stats), nan=0.0) if stats
            else np.zeros((0, N_STATIC), dtype=np.float32))
    y = labeled_df["label"].to_numpy(dtype=int)
    meta = labeled_df[["patient_id", "time"]].reset_index(drop=True)
    return seqs, times, stat.astype(np.float32), y, meta


def build_pretrain_sequences(full_df: pd.DataFrame, max_len: int | None = None,
                             min_len: int = 4):
    """Full-timeline sequences for SELF-SUPERVISED pretraining — one per patient,
    over EVERY patient (including stable / unlabeled ones), because the masked-value
    objective needs no label. This is the extra data that pretraining exploits:
    physiology is learned from all patients before the scarce sepsis labels are
    ever seen. Returns (seqs, times, patient_ids)."""
    seqs, times, pids = [], [], []
    for pid, g in full_df.groupby("patient_id"):
        g = g.sort_values("time")
        if max_len is not None and len(g) > max_len:
            g = g.iloc[-max_len:]
        if len(g) < min_len:
            continue
        s, tm = _one_sequence(g)
        seqs.append(s)
        times.append(tm)
        pids.append(int(pid))
    return seqs, times, pids


# Bound standardized inputs so real-data OUTLIERS (data-entry errors like a lab of
# 10^6) can't blow up the neural net. Trees are immune; a transformer is not — an
# unclipped outlier drove the loss to NaN on eICU. ±8 robust-SD keeps all real
# physiology and clips only the impossible.
STD_CLIP = 8.0


class SeqStandardizer:
    """Standardize the VALUE channels of each step, fit on training sequences only.
    Missing flags and dt are already O(1) and left untouched. Uses ROBUST center/
    scale (median / IQR) so a single garbage value doesn't skew the normalization —
    on clean data this is ~identical to mean/std, so synthetic/PhysioNet are
    unaffected; on messy real data it is what keeps the encoder finite."""

    def __init__(self):
        self.mean = np.zeros(N_VALUE, dtype=np.float32)   # center (median)
        self.std = np.ones(N_VALUE, dtype=np.float32)     # scale (IQR-based)

    def fit(self, seqs: list[np.ndarray]) -> "SeqStandardizer":
        if seqs:
            allv = np.concatenate([s[:, :N_VALUE] for s in seqs], axis=0)
            self.mean = np.nanmedian(allv, axis=0).astype(np.float32)
            q75, q25 = np.nanpercentile(allv, [75, 25], axis=0)
            iqr = ((q75 - q25) / 1.349).astype(np.float32)     # ≈ SD for a normal
            self.std = np.where(iqr < 1e-6, 1.0, iqr).astype(np.float32)
        return self

    def transform(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for s in seqs:
            s2 = s.copy()
            z = (s[:, :N_VALUE] - self.mean) / self.std
            z = np.nan_to_num(z, nan=0.0, posinf=STD_CLIP, neginf=-STD_CLIP)
            np.clip(z, -STD_CLIP, STD_CLIP, out=z)
            s2[:, :N_VALUE] = z
            out.append(s2)
        return out


def pad_batch(seqs: list[np.ndarray], times: list[np.ndarray] | None = None):
    """Right-pad a list of (T_i, F) arrays to (B, T_max, F). Returns
    (X, key_pad) or (X, T_abs, key_pad) when `times` is given.
    key_pad[b, s] == True marks PADDING (what the transformer must ignore)."""
    B = len(seqs)
    T_max = max((s.shape[0] for s in seqs), default=1)
    F = N_SEQ_FEATURES
    X = np.zeros((B, T_max, F), dtype=np.float32)
    key_pad = np.ones((B, T_max), dtype=bool)      # True = pad
    T_abs = np.zeros((B, T_max), dtype=np.float32)
    for b, s in enumerate(seqs):
        T = s.shape[0]
        X[b, :T] = s
        key_pad[b, :T] = False
        if times is not None:
            T_abs[b, :T] = times[b]
    if times is None:
        return X, key_pad
    return X, T_abs, key_pad
