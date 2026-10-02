"""The behavioural scorer, owned end to end.

This module exists because two details decide the outcome and both are easy to get wrong
when reimplementing from prose:

**1. Padding order.** A 10-second chunk is padded to a fixed 100 points. *When* the
padding happens relative to the feature transform completely changes what the padding
means:

    repeat_point : pad the raw (x, y, t) points with the last point, THEN transform.
                   Differences, dt, speed and acceleration all become 0 in the padded
                   tail -- "the cursor stopped". This is what `rlcaptcha._windows` does,
                   and it is the physically sensible reading.
    repeat_row   : transform FIRST, then repeat the last feature row. For `xy` this is
                   identical to repeat_point (a repeated position). For any DIFFERENCE
                   representation it repeats the last velocity, i.e. "the cursor kept
                   moving at its final speed forever" -- a large artificial signal
                   filling 35-96% of a typical chunk.

`expkit.features.windows` does `repeat_row`. Every Phase-0 number produced before this
module existed used it, which is why the difference representations collapsed at the
chunk operating point. Both modes are kept so the comparison is explicit rather than
accidental.

**2. Duplicate chunks.** `rlcaptcha.data._split_into_chunks` recursively replays earlier
chunks when a session is shorter than 12 windows, so every user lasts the full episode.
Across the corpus that turns 760 distinct chunks into 1681, and the duplication is
heavily skewed: NaiveBot has 192 chunks but only 19 distinct. Training or scoring on the
duplicated set silently up-weights short sessions. `build_chunk_dataset(dedup=True)` is
the default for model selection; the simulation keeps the duplicates, because that is
what the environment actually replays.

The scorer's operating point is the CHUNK, not the session: the RL controller is called
once per 10 s of session time. Selection, calibration and reporting all happen there.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .features import REPRESENTATIONS, N_FEATURES
from .paths import RESULTS

CONTEXT = 100
PADDING_MODES = ("repeat_point", "repeat_row", "mask", "zero_mask")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScorerConfig:
    representation: str = "kinematic"
    padding: str = "mask"
    lstm_units: tuple = (32,)
    dense_units: tuple = (32,)
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 64
    context: int = CONTEXT
    epochs: int = 120
    patience: int = 15
    # Early stopping neither monitors nor tracks the best epoch before this one. 0 keeps
    # round 1-2 behaviour. Without it, a fit whose learning starts late (median best epoch
    # 35 across 30 cross-fitted fits; 9 of 30 peaked at >= 45) is killed after 15 flat
    # epochs and restores near-initial weights: 1 of 30 fits collapsed that way (val AUC
    # 0.62), 0 of 30 with 30 here.
    start_from_epoch: int = 0
    class_balance: str = "window"        # "window" | "family"
    augmentation: str = "none"
    n_copies: int = 1
    magnitude: int = 1
    sigma: float = 2.0

    @property
    def n_features(self) -> int:
        n = N_FEATURES[self.representation]
        return n + 1 if self.padding in ("mask", "zero_mask") else n

    def build(self):
        from keras.layers import Dense, Dropout, Input, LSTM
        from keras.models import Sequential

        layers = [Input(shape=(self.context, self.n_features))]
        for i, u in enumerate(self.lstm_units):
            layers.append(LSTM(u, return_sequences=(i < len(self.lstm_units) - 1),
                               recurrent_dropout=self.recurrent_dropout))
            if self.dropout:
                layers.append(Dropout(self.dropout))
        for u in self.dense_units:
            layers.append(Dense(u, activation="relu"))
        layers.append(Dense(1, activation="sigmoid"))
        return Sequential(layers)


# --------------------------------------------------------------------------- #
# Windowing -- the part that must be exactly right
# --------------------------------------------------------------------------- #

def make_window(movements, cfg: ScorerConfig) -> np.ndarray:
    """One chunk -> (n_windows, context, n_features), padded per `cfg.padding`.

    A chunk longer than `context` yields several windows; the final one is padded.
    """
    rep = REPRESENTATIONS[cfg.representation]
    L = cfg.context
    n = len(movements)

    if cfg.padding == "repeat_point":
        # Pad the POINTS, then transform: diffs/dt/speed go to zero in the tail.
        pts = list(movements)
        out = []
        for i in range(n // L):
            out.append(rep(pts[i * L:(i + 1) * L]))
        tail = pts[(n // L) * L:]
        if tail or n == 0:
            padded = tail + [tail[-1] if tail else pts[-1]] * (L - len(tail))
            out.append(rep(padded))
        arr = np.asarray(out, dtype=np.float32)
        return arr

    mat = rep(movements)
    out = [mat[i * L:(i + 1) * L] for i in range(len(mat) // L)]
    tail = mat[(len(mat) // L) * L:]
    if len(tail) < L:
        k = L - len(tail)
        if cfg.padding == "zero_mask":
            pad = np.zeros((k, mat.shape[1]), dtype=mat.dtype)
        else:                                  # repeat_row and mask
            pad = np.repeat(mat[-1:], k, axis=0)
        tail = np.concatenate([tail, pad], axis=0) if len(tail) else pad
    out.append(tail)
    arr = np.asarray(out, dtype=np.float32)

    if cfg.padding in ("mask", "zero_mask"):
        # Append a channel marking real points, so padding cannot be mistaken for
        # genuine stillness (or, under repeat_row, for genuine constant motion).
        real = np.zeros(arr.shape[:2] + (1,), dtype=np.float32)
        n_rows = len(mat)
        for w in range(arr.shape[0]):
            lo, hi = w * L, min((w + 1) * L, n_rows)
            real[w, :max(0, hi - lo), 0] = 1.0
        arr = np.concatenate([arr, real], axis=2)
    return arr


def chunk_key(session_name: str, chunk) -> tuple:
    """Identity of a chunk, so replayed copies collapse to one."""
    return (session_name, len(chunk), chunk[0]["timestamp"], chunk[-1]["timestamp"])


@dataclass
class ChunkSet:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray          # session name
    families: np.ndarray
    pad_fraction: np.ndarray

    def __len__(self):
        return len(self.y)


def build_chunk_dataset(refs, cfg: ScorerConfig, dedup: bool = True,
                        augment_seed: int | None = None) -> ChunkSet:
    """Build the dataset at the CHUNK operating point -- what the controller sees.

    `dedup=True` removes chunks replayed by the 12-window top-up. Use it for model
    selection, calibration and reporting. The simulation keeps duplicates.
    """
    import random as _rnd

    from rlcaptcha.config import MAX_CHUNKS, SESSION_CHUNK_MS
    from rlcaptcha.data import _split_into_chunks

    from .features import PERTURBATIONS

    rng = _rnd.Random(augment_seed or 0)
    perturb = PERTURBATIONS[cfg.augmentation]
    kw = ({"sigma": cfg.sigma} if cfg.augmentation == "gaussian"
          else {"magnitude": cfg.magnitude} if cfg.augmentation in ("rigid", "per_move")
          else {})
    copies = cfg.n_copies if cfg.augmentation != "none" else 1

    X, y, g, fam, pf, seen = [], [], [], [], [], set()
    for ref in refs:
        raw = json.loads(Path(ref.path).read_text())
        if not raw.get("mouse_movements"):
            continue
        for chunk in _split_into_chunks(raw, SESSION_CHUNK_MS, MAX_CHUNKS):
            if not chunk:
                continue
            key = chunk_key(ref.name, chunk)
            if dedup and key in seen:
                continue
            seen.add(key)
            for _ in range(copies):
                mv = perturb(chunk, rng, **kw) if cfg.augmentation != "none" else chunk
                w = make_window(mv, cfg)
                X.append(w)
                y.append(np.full(len(w), int(ref.is_bot), dtype=np.int32))
                g.append(np.full(len(w), ref.name, dtype=object))
                fam.append(np.full(len(w), ref.family, dtype=object))
                pf.append(np.full(len(w), max(0, cfg.context - len(chunk)) / cfg.context))
    if not X:
        raise RuntimeError("no chunks built")
    return ChunkSet(np.concatenate(X).astype(np.float32), np.concatenate(y),
                    np.concatenate(g), np.concatenate(fam), np.concatenate(pf))


class Standardiser:
    """Per-feature mean/std fitted on training windows only."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": np.asarray(self.mean).tolist(),
                "std": np.asarray(self.std).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #

class HumanityScorer:
    """Output is P(bot): HIGH means bot-like. The manuscript's H is 1 - score."""

    def __init__(self, cfg: ScorerConfig, model=None, std: Standardiser | None = None,
                 temperature: float = 1.0):
        self.cfg = cfg
        self.model = model
        self.std = std
        self.temperature = float(temperature)
        # How the fit went, and what it was fitted on. Both are persisted by `save`, so a
        # cached scorer can be checked without refitting it -- see `preflight`.
        self.training_summary: dict | None = None
        self.provenance: dict | None = None

    # -- training -------------------------------------------------------- #
    def fit(self, train_refs, val_refs, seed: int = 0, verbose: int = 0):
        import tensorflow as tf
        from keras.callbacks import EarlyStopping
        from keras.optimizers import Adam

        tf.keras.utils.set_random_seed(seed)
        tr = build_chunk_dataset(train_refs, self.cfg, dedup=True, augment_seed=seed)
        va = build_chunk_dataset(val_refs, _no_aug(self.cfg), dedup=True)

        self.std = Standardiser().fit(tr.X)
        self.model = self.cfg.build()
        self.model.compile(optimizer=Adam(self.cfg.lr), loss="bce", metrics=["accuracy"])

        sw = _sample_weights(tr, self.cfg.class_balance)
        es_kw = {"start_from_epoch": self.cfg.start_from_epoch} if self.cfg.start_from_epoch else {}
        self.history = self.model.fit(
            self.std(tr.X), tr.y, sample_weight=sw,
            validation_data=(self.std(va.X), va.y),
            epochs=self.cfg.epochs, batch_size=self.cfg.batch_size, verbose=verbose,
            callbacks=[EarlyStopping(monitor="val_loss", patience=self.cfg.patience,
                                     restore_best_weights=True, **es_kw)])

        # Temperature is fitted at the operating point, on held-out data.
        from .calibration import fit_temperature, roc_auc
        pv = self.model.predict(self.std(va.X), batch_size=1024, verbose=0).ravel()
        self.temperature = float(fit_temperature(pv, va.y))

        # Early stopping restores the best epoch. If that is epoch 1, the "trained" scorer
        # is the initialisation: round 2's fold-0 scorer stopped exactly there, with raw
        # outputs in 0.41-0.64 for everyone, and nothing downstream raised an error.
        # Keras restores the best epoch AMONG MONITORED ones -- those from start_from_epoch
        # on -- so take the argmin over that window, or best_epoch would name an earlier
        # epoch whose weights are not the ones in the model.
        val_loss = [float(v) for v in self.history.history["val_loss"]]
        start = min(self.cfg.start_from_epoch, len(val_loss) - 1)
        best = start + int(np.argmin(val_loss[start:]))
        self.training_summary = {
            "epochs_run": len(val_loss),
            "start_from_epoch": int(self.cfg.start_from_epoch),
            "best_epoch": best + 1,
            "val_loss_first": val_loss[0],
            "val_loss_best": val_loss[best],
            "val_auc": roc_auc(pv, va.y),
            "n_train_windows": int(len(tr.y)),
            "n_val_windows": int(len(va.y)),
        }
        return self

    # -- scoring --------------------------------------------------------- #
    def _calibrate(self, p):
        if self.temperature == 1.0:
            return p
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        return 1.0 / (1.0 + np.exp(-np.log(pc / (1 - pc)) / self.temperature))

    def score_chunk(self, movements) -> float | None:
        if not movements:
            return None
        w = make_window(movements, self.cfg)
        p = self.model.predict(self.std(w), batch_size=256, verbose=0)
        return float(np.mean(self._calibrate(p)))

    def score_chunks(self, chunks, batch_size: int = 4096):
        """Score many chunks in ONE batched pass -- the only path the RL loop should use.

        A per-chunk `predict` costs ~59 ms of graph dispatch regardless of model size;
        batched it is ~0.33 ms amortised. A user's scores do not depend on the agent's
        actions, so a whole episode's score table can be built up front.
        """
        flat, owner = [], []
        for i, ch in enumerate(chunks):
            if not ch:
                continue
            w = make_window(ch, self.cfg)
            flat.append(w)
            owner.extend([i] * len(w))
        out = [None] * len(chunks)
        if not flat:
            return out
        p = self.model.predict(self.std(np.concatenate(flat)),
                               batch_size=batch_size, verbose=0).ravel()
        p = self._calibrate(p)
        owner = np.asarray(owner)
        for i in np.unique(owner):
            out[int(i)] = float(p[owner == i].mean())
        return out

    # -- persistence ------------------------------------------------------ #
    def save(self, directory):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        self.model.save(str(d / "model.keras"))
        (d / "scorer.json").write_text(json.dumps({
            "config": asdict(self.cfg), "temperature": self.temperature,
            "standardiser": self.std.to_dict(),
            "params": int(self.model.count_params()),
            "training": self.training_summary,
            "provenance": self.provenance}, indent=2))
        return d

    @classmethod
    def load(cls, directory):
        import tensorflow as tf
        d = Path(directory)
        meta = json.loads((d / "scorer.json").read_text())
        cfg = ScorerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in meta["config"].items()})
        scorer = cls(cfg, tf.keras.models.load_model(str(d / "model.keras")),
                     Standardiser.from_dict(meta["standardiser"]),
                     meta.get("temperature", 1.0))
        # Absent in anything saved before round 3 -- including round 2's cached fold
        # scorers, which is exactly how a stale cache is recognised.
        scorer.training_summary = meta.get("training")
        scorer.provenance = meta.get("provenance")
        return scorer


def _no_aug(cfg: ScorerConfig) -> ScorerConfig:
    import dataclasses
    return dataclasses.replace(cfg, augmentation="none", n_copies=1)


def _sample_weights(cs: ChunkSet, mode: str) -> np.ndarray:
    """`window` balances human vs bot. `family` additionally equalises the four bot
    generators, which otherwise contribute very unequally: NaiveBot has 19 distinct
    chunks against MimicBot's and FallibleBot's hundreds."""
    w = np.ones(len(cs), dtype=np.float32)
    n_pos, n_neg = int(cs.y.sum()), int((cs.y == 0).sum())
    tot = n_pos + n_neg
    w[cs.y == 1] = tot / (2 * max(n_pos, 1))
    w[cs.y == 0] = tot / (2 * max(n_neg, 1))
    if mode == "family":
        bots = cs.y == 1
        fams = np.unique(cs.families[bots])
        for f in fams:
            m = bots & (cs.families == f)
            w[m] *= (n_pos / len(fams)) / max(int(m.sum()), 1)
    return w


# --------------------------------------------------------------------------- #
# Self-test -- run `python -m expkit.humanity_scorer`
# --------------------------------------------------------------------------- #

def _selftest():
    import dataclasses

    from .partition import index_sessions

    refs = index_sessions()
    mv = json.loads(Path(refs[0].path).read_text())["mouse_movements"][:30]

    print("padding semantics on a 30-point chunk (tail row of window 0):")
    for pad in PADDING_MODES:
        cfg = ScorerConfig(representation="dxdy", padding=pad)
        w = make_window(mv, cfg)[0]
        print(f"  {pad:12s} shape={w.shape}  last real={np.round(w[29][:2], 1)}  "
              f"pad={np.round(w[-1][:2], 1)}"
              + (f"  mask={w[-1][-1]:.0f}" if pad in ('mask', 'zero_mask') else ""))
    print("\n  repeat_point -> padded deltas are 0 (cursor stopped)")
    print("  repeat_row   -> padded deltas repeat the last velocity (cursor flies on)")

    cfg = ScorerConfig(representation="kinematic", padding="mask")
    full = build_chunk_dataset(refs[:20], cfg, dedup=False)
    ded = build_chunk_dataset(refs[:20], cfg, dedup=True)
    print(f"\ndedup: {len(full)} windows -> {len(ded)} distinct "
          f"({len(full) - len(ded)} replayed by the 12-chunk top-up)")
    print(f"n_features for {cfg.representation}+{cfg.padding}: {cfg.n_features}")
    print(f"params: {cfg.build().count_params():,}")


if __name__ == "__main__":
    _selftest()
