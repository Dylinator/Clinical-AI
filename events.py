"""
events.py — the EVENT-STREAM representation for the Clinical Trajectory Transformer.

This is the "turn different charts into the same string" primitive. Instead of the
fixed (T, F) grid the first transformer used (sequences.py), a patient becomes a
variable-length SEQUENCE OF EVENTS, each event a token:

        (channel, value, time)      e.g. (HR, 112, 15min), (lactate, 3.4, 60min)

drawn from a shared CHANNEL VOCABULARY. This is the Med-BERT / BEHRT idea adapted to
continuous ICU values, and it is what makes heterogeneous multi-database training
natural: a database that measures bilirubin emits `bilirubin` events; one that does
not simply omits them. The vocabulary is the UNION of channels across sources, so:
  * adding a database that charts new labs = new vocabulary entries, no re-plumbing;
  * adding features over time = extend EVENT_CHANNELS;
  * sparsity + irregular timing are represented natively (only measured -> a token).

Leakage safety is the SAME slice as everywhere else (events with time <= t), so the
past-only guarantee `test_leakage` protects carries over. The model
(model_ctt.py — next) embeds each event as
    channel_embedding[channel] + value_encoder(value) + time_embedding(time)
and runs a transformer encoder over the stream. This module owns only the data:
plain numpy, no torch.

v1 covers the current schema's value channels (vitals + labs). Interventions as
event-tokens and richer channel sets are the documented next extensions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config

# --- shared channel vocabulary (the union grows as databases/features are added) --
EVENT_CHANNELS = list(config.CHANNELS)                    # vitals + labs (carry a value)
CHANNEL_IDX = {c: i for i, c in enumerate(EVENT_CHANNELS)}
N_CHANNELS = len(EVENT_CHANNELS)
PAD_CHANNEL = N_CHANNELS                                  # index reserved for padding

# static vector fused at the model head (same toggles as the tabular feature list)
STATIC_COLS = (
    (config.STATIC_FEATURES if config.USE_STATIC_FEATURES else [])
    + (config.SES_FEATURES if config.USE_SES_AS_FEATURE else [])
    + (config.MED_FEATURES if config.USE_MED_FEATURES else [])
)
N_STATIC = len(STATIC_COLS)


def _patient_event_stream(g: pd.DataFrame):
    """One patient's sorted frame -> arrays (times, chan_idx, value) of every
    non-NaN measurement, time-ordered (ties broken by channel order). This is the
    full event stream; a labelled example slices it by time <= t."""
    times = g["time"].to_numpy(dtype=float)
    ev_t, ev_c, ev_v = [], [], []
    for c in EVENT_CHANNELS:
        col = g[c].to_numpy(dtype=float)
        seen = ~np.isnan(col)
        if seen.any():
            ev_t.append(times[seen])
            ev_c.append(np.full(int(seen.sum()), CHANNEL_IDX[c], dtype=np.int64))
            ev_v.append(col[seen])
    if not ev_t:
        return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0)
    t = np.concatenate(ev_t); c = np.concatenate(ev_c); v = np.concatenate(ev_v)
    order = np.lexsort((c, t))                # by time, then channel
    return t[order], c[order], v[order]


def _static_vector(g_last) -> np.ndarray:
    if N_STATIC == 0:
        return np.zeros(0, dtype=np.float32)
    return np.array([float(g_last[c]) for c in STATIC_COLS], dtype=np.float32)


def build_event_sequences(full_df: pd.DataFrame, labeled_df: pd.DataFrame,
                          max_events: int | None = 256):
    """Mirror of sequences.build_sequences, but as event streams.

    Returns:
        events : list of (E_i, 3) float arrays — columns [channel_idx, value, time_min]
        stat   : (N, N_STATIC) float32 static vectors
        y      : (N,) int labels
        meta   : DataFrame[patient_id, time]
    `max_events` caps each example to its most recent E events (bounds cost on long
    real stays). Past-only: only events with time <= the labelled t are included.
    """
    by_patient = {}
    for pid, g in full_df.groupby("patient_id"):
        g = g.sort_values("time")
        by_patient[pid] = (g, _patient_event_stream(g))

    events, stats = [], []
    for r in labeled_df.itertuples(index=False):
        g, (et, ec, ev) = by_patient[r.patient_id]
        cut = int(np.searchsorted(et, r.time, side="right"))   # events with time <= t
        lo = max(0, cut - max_events) if max_events is not None else 0
        sl = slice(lo, cut)
        arr = np.stack([ec[sl].astype(np.float32), ev[sl].astype(np.float32),
                        et[sl].astype(np.float32)], axis=1) if cut > lo \
            else np.zeros((0, 3), dtype=np.float32)
        events.append(arr)
        # static from the last grid row at/before t
        gt = g["time"].to_numpy()
        ri = int(np.searchsorted(gt, r.time, side="right")) - 1
        stats.append(_static_vector(g.iloc[max(0, ri)]))

    stat = (np.nan_to_num(np.vstack(stats), nan=0.0) if stats
            else np.zeros((0, N_STATIC), dtype=np.float32)).astype(np.float32)
    y = labeled_df["label"].to_numpy(dtype=int)
    meta = labeled_df[["patient_id", "time"]].reset_index(drop=True)
    return events, stat, y, meta


class EventValueStandardizer:
    """Per-CHANNEL z-score of event values, fit on train events only, robust
    (median/IQR) + clipped — same rationale as the sequence standardizer (real data
    has outliers that would blow up the net). Channel idx and time are untouched."""

    def __init__(self, clip: float = 8.0):
        self.mean = np.zeros(N_CHANNELS, dtype=np.float32)
        self.std = np.ones(N_CHANNELS, dtype=np.float32)
        self.clip = clip

    def fit(self, events: list[np.ndarray]) -> "EventValueStandardizer":
        allc = np.concatenate([e[:, 0] for e in events if len(e)]).astype(int) if events else np.zeros(0, int)
        allv = np.concatenate([e[:, 1] for e in events if len(e)]) if events else np.zeros(0)
        for ci in range(N_CHANNELS):
            vals = allv[allc == ci]
            if vals.size:
                self.mean[ci] = np.median(vals)
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = (q75 - q25) / 1.349
                self.std[ci] = iqr if iqr > 1e-6 else 1.0
        return self

    def transform(self, events: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        for e in events:
            if not len(e):
                out.append(e); continue
            e2 = e.copy()
            ci = e[:, 0].astype(int)
            z = (e[:, 1] - self.mean[ci]) / self.std[ci]
            e2[:, 1] = np.clip(np.nan_to_num(z, nan=0.0), -self.clip, self.clip)
            out.append(e2)
        return out


def pad_events(events: list[np.ndarray]):
    """Right-pad to (B, E_max, 3) and a key-padding mask (True = pad). Padded rows
    get channel index PAD_CHANNEL so the embedding can map them to a pad vector."""
    B = len(events)
    E_max = max((len(e) for e in events), default=1)
    chan = np.full((B, E_max), PAD_CHANNEL, dtype=np.int64)
    val = np.zeros((B, E_max), dtype=np.float32)
    tim = np.zeros((B, E_max), dtype=np.float32)
    key_pad = np.ones((B, E_max), dtype=bool)
    for b, e in enumerate(events):
        E = len(e)
        if E:
            chan[b, :E] = e[:, 0].astype(np.int64)
            val[b, :E] = e[:, 1]
            tim[b, :E] = e[:, 2]
            key_pad[b, :E] = False
    return chan, val, tim, key_pad


if __name__ == "__main__":
    config.GEN.n_patients = 40
    from synthetic import generate
    from timeline_engine import to_frame, onset_map
    from labeling import add_labels
    ps = generate(); full = to_frame(ps); lab = add_labels(full, onset_map(ps))
    ev, stat, y, meta = build_event_sequences(full, lab, max_events=128)
    lens = [len(e) for e in ev]
    print(f"examples={len(ev)}  static_dim={stat.shape[1]}  vocab={N_CHANNELS} channels")
    print(f"events/example: min={min(lens)} median={int(np.median(lens))} max={max(lens)}")
    print(f"example event stream (channel_idx, value, time) — first 6 of one row:")
    print(ev[len(ev)//2][:6])
    std = EventValueStandardizer().fit(ev)
    zev = std.transform(ev)
    allz = np.concatenate([e[:, 1] for e in zev if len(e)])
    print(f"standardized values finite={np.isfinite(allz).all()}  max|z|={np.abs(allz).max():.1f}")
    chan, val, tim, kp = pad_events(zev[:8])
    print(f"padded batch: chan{chan.shape} val{val.shape} pad_frac={kp.mean():.2f}")
