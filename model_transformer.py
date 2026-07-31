"""
model_transformer.py — an ENCODER-ONLY transformer for deterioration risk,
with TIME-AWARE attention and SELF-SUPERVISED pretraining (Track A upgrades).

Why encoder-only (not GPT-style decoder): the task is to read a patient's history
UP TO t and output ONE probability — "will they deteriorate within the horizon?".
That is sequence CLASSIFICATION, so a bidirectional encoder over the observed
window (BERT-family, like Med-BERT/BEHRT in the notes) is the right shape. There
is no autoregressive generation, so a causal decoder would only add masking
complexity for nothing. Past-only-ness is guaranteed by how the sequence is built
(sequences.py slices time <= t); within that fixed window attention is free both ways.

Two upgrades over the first version, both aimed at genuinely competing with the
calibrated tree — NOT by overfitting, but by giving the encoder advantages a tree
cannot use and data a tree does not need:

  1. TIME-AWARE ATTENTION (Time2Vec). The model embeds the REAL elapsed minute of
     each step, not just its position. Irregular sampling (dense vitals, sparse
     labs) then carries information — a tree's flattened features cannot represent
     "this lactate is 4 hours stale".

  2. SELF-SUPERVISED PRETRAINING (masked value modeling, the Med-BERT move). Before
     ever seeing the scarce sepsis labels, the encoder is pretrained to reconstruct
     randomly-masked channel values across ALL patients (including stable/unlabeled
     ones). It learns physiology and temporal structure from abundant unlabeled
     data, then fine-tunes on the labels. This is the lever that pays off with scale
     — exactly the regime (lots of real data) where the notes expect a transformer
     to earn its keep.

Honesty features carried over from model.py: pos_weight-balanced loss, isotonic
CALIBRATION of the output probabilities, and a predict interface identical to the
RandomForest so evaluate.py treats it uniformly.

CPU-friendly: small dims, seeded. Requires torch (requirements-extra.txt).
"""

from __future__ import annotations
import warnings
import numpy as np

import config
import sequences as seqmod

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    warnings.filterwarnings("ignore", message=".*nested tensor.*")
except Exception:                       # keep import-safe if torch is absent
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Hyperparameters (small on purpose; one home, like config)
# --------------------------------------------------------------------------- #
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.1
TIME2VEC_K = 16        # dimensionality of the time-aware embedding
MAX_LEN = 96           # cap history length (real ICU stays can be long)

# finetune
EPOCHS = 30
BATCH = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.15
PATIENCE = 6

# pretrain
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 5e-4
MASK_FRAC = 0.15       # fraction of real timesteps masked for reconstruction

N_VALUE = seqmod.N_VALUE       # number of value channels reconstructed


def _seed_everything(seed: int = config.SEED):
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


if _HAS_TORCH:

    class _PositionalEncoding(nn.Module):
        """Sinusoidal encoding of sequence POSITION (order), complementary to the
        Time2Vec encoding of real elapsed TIME below."""
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class _Time2Vec(nn.Module):
        """Time2Vec (Kazemi et al., 2019): map a scalar time -> a vector with one
        linear term and k-1 learnable periodic terms, so the model can represent
        absolute time AND periodicities from real elapsed minutes."""
        def __init__(self, k: int):
            super().__init__()
            self.w0 = nn.Parameter(torch.randn(1) * 0.01)
            self.b0 = nn.Parameter(torch.zeros(1))
            self.w = nn.Parameter(torch.randn(k - 1) * 0.01)
            self.b = nn.Parameter(torch.zeros(k - 1))
        def forward(self, t):                    # t: (B,T) minutes
            t = (t / 60.0).unsqueeze(-1)         # -> hours, (B,T,1)
            lin = self.w0 * t + self.b0          # (B,T,1)
            per = torch.sin(t * self.w + self.b)  # (B,T,k-1)
            return torch.cat([lin, per], dim=-1)  # (B,T,k)

    class _Backbone(nn.Module):
        """Shared encoder: per-step embed + position + time2vec -> TransformerEncoder
        -> per-step hidden states. Reused by BOTH pretraining and classification."""
        def __init__(self, n_seq_feat: int):
            super().__init__()
            self.embed = nn.Linear(n_seq_feat, D_MODEL)
            self.time2vec = _Time2Vec(TIME2VEC_K)
            self.time_proj = nn.Linear(TIME2VEC_K, D_MODEL)
            self.pos = _PositionalEncoding(D_MODEL, max_len=MAX_LEN + 4)
            self.mask_token = nn.Parameter(torch.randn(D_MODEL) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
                dropout=DROPOUT, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)

        def forward(self, x, t_abs, key_pad, mask_positions=None):
            # x:(B,T,F)  t_abs:(B,T)  key_pad:(B,T) True=pad
            tok = self.embed(x)                          # (B,T,D)
            if mask_positions is not None:               # BERT-style content mask
                tok = torch.where(mask_positions.unsqueeze(-1),
                                  self.mask_token.expand_as(tok), tok)
            tok = tok + self.time_proj(self.time2vec(t_abs))   # time-aware
            tok = self.pos(tok)                          # + order
            return self.encoder(tok, src_key_padding_mask=key_pad)

    class _PretrainModel(nn.Module):
        """Backbone + a linear head that reconstructs the standardized value
        channels at masked positions (masked value modeling)."""
        def __init__(self, n_seq_feat: int):
            super().__init__()
            self.backbone = _Backbone(n_seq_feat)
            self.recon = nn.Linear(D_MODEL, N_VALUE)
        def forward(self, x, t_abs, key_pad, mask_positions):
            h = self.backbone(x, t_abs, key_pad, mask_positions)
            return self.recon(h)                         # (B,T,N_VALUE)

    class ClinicalEncoder(nn.Module):
        """Backbone + masked mean-pool + static fusion -> one deterioration logit."""
        def __init__(self, n_seq_feat: int, n_static: int):
            super().__init__()
            self.backbone = _Backbone(n_seq_feat)
            head_in = D_MODEL + n_static
            self.head = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, D_MODEL), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(D_MODEL, 1),
            )
        def forward(self, x, t_abs, key_pad, static):
            h = self.backbone(x, t_abs, key_pad)
            keep = (~key_pad).unsqueeze(-1).float()      # (B,T,1)
            pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
            if static.shape[1] > 0:
                pooled = torch.cat([pooled, static], dim=1)
            return self.head(pooled).squeeze(-1)         # (B,) logit


# --------------------------------------------------------------------------- #
# Self-supervised pretraining
# --------------------------------------------------------------------------- #
def pretrain(seqs, times, standardizer, seed: int = config.SEED,
             epochs: int | None = None, verbose: bool = True):
    """Masked value modeling on full patient timelines (labels not used).
    Returns the trained backbone's state_dict, to initialize the classifier."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch required (pip install -r requirements-extra.txt).")
    epochs = PRETRAIN_EPOCHS if epochs is None else epochs   # read current global
    _seed_everything(seed)
    seqs = standardizer.transform(seqs)
    rng = np.random.default_rng(seed)

    model = _PretrainModel(seqmod.N_SEQ_FEATURES)
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    n = len(seqs)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot, seen = 0.0, 0
        for i in range(0, n, BATCH):
            bi = order[i:i + BATCH]
            X, T_abs, key_pad = seqmod.pad_batch([seqs[j] for j in bi],
                                                 [times[j] for j in bi])
            Xt = torch.from_numpy(X); Ta = torch.from_numpy(T_abs)
            kp = torch.from_numpy(key_pad)
            real = ~kp
            # choose masked positions among real steps
            probs = torch.rand(real.shape) * real.float()
            mask_pos = probs > (1.0 - MASK_FRAC)
            mask_pos = mask_pos & real
            if mask_pos.sum() == 0:
                continue
            target = Xt[:, :, :N_VALUE].clone()          # standardized value targets
            opt.zero_grad()
            pred = model(Xt, Ta, kp, mask_pos)           # (B,T,N_VALUE)
            diff = (pred - target)[mask_pos]             # only masked positions
            loss = (diff * diff).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(bi); seen += len(bi)
        if verbose:
            print(f"  pretrain epoch {epoch+1:2d}/{epochs}  recon_mse={tot/max(1,seen):.4f}")
    return model.backbone.state_dict()


# --------------------------------------------------------------------------- #
# Classifier wrapper (sklearn-ish interface)
# --------------------------------------------------------------------------- #
class TransformerRiskModel:
    """Trained classifier + probability calibrator + input standardizer. Interface
    mirrors model.py: scoring is predict_proba(seqs, times, static)."""

    def __init__(self, n_seq_feat: int, n_static: int, standardizer=None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required (pip install -r requirements-extra.txt).")
        self.n_seq_feat = n_seq_feat
        self.n_static = n_static
        self.net = ClinicalEncoder(n_seq_feat, n_static)
        self.calibrator = None
        self.standardizer = standardizer or seqmod.SeqStandardizer()

    def _forward_logits(self, seqs, times, static, batch=256) -> np.ndarray:
        seqs = self.standardizer.transform(seqs)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), batch):
                X, T_abs, kp = seqmod.pad_batch(seqs[i:i + batch], times[i:i + batch])
                logit = self.net(torch.from_numpy(X), torch.from_numpy(T_abs),
                                 torch.from_numpy(kp), torch.from_numpy(static[i:i + batch]))
                out.append(logit.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def predict_proba(self, seqs, times, static) -> np.ndarray:
        logits = self._forward_logits(seqs, times, static)
        if self.calibrator is not None:
            p1 = self.calibrator.predict(logits)
        else:
            p1 = 1.0 / (1.0 + np.exp(-logits))
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])


def train(seqs, times, static, y, standardizer=None, init_backbone_state=None,
          seed: int = config.SEED, epochs: int | None = None, verbose: bool = True):
    """Fine-tune the classifier (optionally from a pretrained backbone), early-stop
    on an inner split, then isotonically calibrate. Returns a TransformerRiskModel."""
    from sklearn.isotonic import IsotonicRegression
    epochs = EPOCHS if epochs is None else epochs           # read current global
    _seed_everything(seed)

    static = np.ascontiguousarray(static, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if standardizer is None:
        standardizer = seqmod.SeqStandardizer().fit(seqs)

    model = TransformerRiskModel(seqmod.N_SEQ_FEATURES, static.shape[1], standardizer)
    if init_backbone_state is not None:
        model.net.backbone.load_state_dict(init_backbone_state)
        if verbose:
            print("  (initialized encoder from pretrained backbone)")

    std_seqs = standardizer.transform(seqs)        # standardize once for training
    net = model.net
    n = len(std_seqs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRAC))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    pos = float(y[tr_idx].sum()); neg = float(len(tr_idx) - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def _logits(indices):
        net.eval()
        with torch.no_grad():
            X, T_abs, kp = seqmod.pad_batch([std_seqs[i] for i in indices],
                                            [times[i] for i in indices])
            return net(torch.from_numpy(X), torch.from_numpy(T_abs),
                       torch.from_numpy(kp), torch.from_numpy(static[indices])).cpu().numpy()

    best_val, best_state, waited = float("inf"), None, 0
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(tr_idx))
        epoch_loss = 0.0
        for i in range(0, len(order), BATCH):
            bi = tr_idx[order[i:i + BATCH]]
            X, T_abs, kp = seqmod.pad_batch([std_seqs[j] for j in bi],
                                            [times[j] for j in bi])
            opt.zero_grad()
            logit = net(torch.from_numpy(X), torch.from_numpy(T_abs),
                        torch.from_numpy(kp), torch.from_numpy(static[bi]))
            loss = loss_fn(logit, torch.from_numpy(y[bi]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(bi)

        val_logits = _logits(val_idx)
        vl = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
            torch.from_numpy(val_logits), torch.from_numpy(y[val_idx])).item()
        if verbose:
            print(f"  epoch {epoch+1:2d}/{epochs}  train_loss={epoch_loss/len(tr_idx):.4f}"
                  f"  val_loss={vl:.4f}")
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                if verbose:
                    print(f"  early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(_logits(val_idx), y[val_idx])
    model.calibrator = iso
    return model


def predict_risk(model: "TransformerRiskModel", seqs, times, static) -> np.ndarray:
    """Positive-class probability — same name/return as model.predict_risk."""
    return model.predict_proba(seqs, times, static)[:, 1]


# --------------------------------------------------------------------------- #
# Persistence (so the dashboard can load a trained transformer)
# --------------------------------------------------------------------------- #
def save(model: "TransformerRiskModel", path: str):
    """Persist net weights + calibrator + standardizer + dims to one file."""
    import io, joblib
    buf = io.BytesIO()
    torch.save(model.net.state_dict(), buf)
    joblib.dump({
        "net_state": buf.getvalue(),
        "n_seq_feat": model.n_seq_feat,
        "n_static": model.n_static,
        "calibrator": model.calibrator,
        "std_mean": model.standardizer.mean,
        "std_std": model.standardizer.std,
    }, path)


def load(path: str) -> "TransformerRiskModel":
    import io, joblib
    blob = joblib.load(path)
    model = TransformerRiskModel(blob["n_seq_feat"], blob["n_static"])
    model.net.load_state_dict(torch.load(io.BytesIO(blob["net_state"])))
    model.net.eval()
    model.calibrator = blob["calibrator"]
    model.standardizer.mean = blob["std_mean"]
    model.standardizer.std = blob["std_std"]
    return model
