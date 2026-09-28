"""Retrain the behavioural scorer under a clean protocol, for several input
representations and perturbation models.

Architecture is the manuscript's, verified against the published checkpoint:
LSTM(200, seq) -> LSTM(100) -> Dense(128, relu) -> Dense(64, relu) -> Dense(1, sigmoid),
304,049 parameters at 2 input features.

What differs from the published training run, deliberately:

* **Session-level split.** Windows never straddle pools (see `partition.py`).
* **Validation is carved from the training pool**, so the evaluation pool never selects
  a checkpoint. The published run passed the test set as `validation_data` together
  with `EarlyStopping(restore_best_weights=True)`.
* **Standardisation is fitted on the training pool only** and reused everywhere. The
  published pipeline min-max scaled each call over whatever array it was handed, which
  makes train and inference scales disagree; the deployed `HumanityScorer` sidesteps
  that by returning the *unadapted* array, so the published model in fact consumes raw
  pixels. Fixing this is what lets different representations be compared fairly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .features import N_FEATURES, PERTURBATIONS, REPRESENTATIONS, windows
from .paths import SCORERS

CONTEXT = 100


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #

def _load_movements(path: str):
    raw = json.loads(Path(path).read_text())
    return raw.get("mouse_movements") or []


@dataclass
class WindowSet:
    X: np.ndarray            # (n, CONTEXT, F)
    y: np.ndarray            # (n,) 0=human 1=bot
    groups: np.ndarray       # (n,) session name -- for group-aware CV and bootstrap

    def __len__(self) -> int:
        return len(self.y)


def build_windows(
    refs,
    representation: str = "xy",
    perturbation: str = "none",
    n_copies: int = 1,
    seed: int = 0,
    magnitude: int = 1,
    sigma: float = 2.0,
) -> WindowSet:
    """Turn session refs into a window dataset.

    `n_copies` > 1 with a per-movement perturbation is the augmentation path: each
    session contributes several distinct copies. With `perturbation="rigid"` the copies
    are near-duplicates, which is the point of the comparison.
    """
    rep = REPRESENTATIONS[representation]
    perturb = PERTURBATIONS[perturbation]
    rng = random.Random(seed)

    X, y, groups = [], [], []
    for ref in refs:
        movements = _load_movements(ref.path)
        if not movements:
            continue
        for _ in range(n_copies):
            kw = {"sigma": sigma} if perturbation == "gaussian" else (
                {"magnitude": magnitude} if perturbation in ("rigid", "per_move") else {})
            mv = perturb(movements, rng, **kw)
            w = windows(rep(mv), CONTEXT)
            X.append(w)
            y.append(np.full(len(w), 1 if ref.is_bot else 0))
            groups.append(np.full(len(w), ref.name, dtype=object))

    if not X:
        raise RuntimeError("no windows built -- are the session paths correct?")
    return WindowSet(
        X=np.concatenate(X).astype(np.float32),
        y=np.concatenate(y).astype(np.int32),
        groups=np.concatenate(groups),
    )


class Standardiser:
    """Per-feature mean/std fitted on training windows only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self.std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.mean = np.array(d["mean"], dtype=np.float32)
        s.std = np.array(d["std"], dtype=np.float32)
        return s


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(n_features: int, context: int = CONTEXT):
    from keras.layers import LSTM, Dense
    from keras.models import Sequential

    model = Sequential([
        LSTM(200, return_sequences=True, input_shape=(context, n_features)),
        LSTM(100),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    return model


def split_by_session(refs, val_fraction: float = 0.2, seed: int = 7):
    """Carve a validation pool out of the training pool, stratified by family."""
    from collections import defaultdict

    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in refs:
        strata[r.family].append(r)

    train, val = [], []
    for fam in sorted(strata):
        members = sorted(strata[fam], key=lambda r: r.name)
        rng.shuffle(members)
        n_val = max(1, round(len(members) * val_fraction)) if len(members) > 2 else 0
        val.extend(members[:n_val])
        train.extend(members[n_val:])
    return train, val


@dataclass
class ScorerResult:
    name: str
    representation: str
    perturbation: str
    n_copies: int
    n_train_windows: int
    n_val_windows: int
    epochs_run: int
    params: int
    model_path: str
    standardiser: dict


def train_scorer(
    train_refs,
    representation: str = "xy",
    perturbation: str = "none",
    n_copies: int = 1,
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 12,
    seed: int = 0,
    name: str | None = None,
    verbose: int = 0,
) -> ScorerResult:
    """Train one scorer variant. Returns where it was saved plus its standardiser."""
    import tensorflow as tf
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam

    tf.keras.utils.set_random_seed(seed)
    name = name or f"{representation}__{perturbation}__x{n_copies}"

    tr_refs, va_refs = split_by_session(train_refs, seed=seed + 7)
    tr = build_windows(tr_refs, representation, perturbation, n_copies, seed=seed)
    va = build_windows(va_refs, representation, "none", 1, seed=seed + 1)

    std = Standardiser().fit(tr.X)
    Xtr, Xva = std(tr.X), std(va.X)

    model = build_model(N_FEATURES[representation])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="bce", metrics=["accuracy"])

    # Bots outnumber humans ~2.5:1 at session level; reweight so the scorer is not
    # simply biased toward the majority class.
    n_pos = int(tr.y.sum())
    n_neg = int(len(tr.y) - n_pos)
    total = n_pos + n_neg
    class_weight = {0: total / (2 * max(n_neg, 1)), 1: total / (2 * max(n_pos, 1))}

    stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(
        Xtr, tr.y,
        validation_data=(Xva, va.y),
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        class_weight=class_weight, callbacks=[stop],
    )

    path = SCORERS / f"{name}.keras"
    model.save(str(path))
    meta = ScorerResult(
        name=name,
        representation=representation,
        perturbation=perturbation,
        n_copies=n_copies,
        n_train_windows=len(tr),
        n_val_windows=len(va),
        epochs_run=len(hist.history["loss"]),
        params=int(model.count_params()),
        model_path=str(path),
        standardiser=std.to_dict(),
    )
    (SCORERS / f"{name}.json").write_text(json.dumps(asdict(meta), indent=2))
    return meta


def load_scorer(name: str):
    """Return (keras model, Standardiser, meta dict)."""
    import tensorflow as tf

    meta = json.loads((SCORERS / f"{name}.json").read_text())
    model = tf.keras.models.load_model(meta["model_path"])
    return model, Standardiser.from_dict(meta["standardiser"]), meta


def predict_windows(model, std, ws: WindowSet, batch_size: int = 256) -> np.ndarray:
    return model.predict(std(ws.X), batch_size=batch_size, verbose=0).ravel()
