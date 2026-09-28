"""Architecture search and grouped cross-validation for the behavioural scorer.

Why this module exists
----------------------
The published scorer has 304,049 parameters and was fitted on 206 windows -- about
1,170 parameters per training sample. That is severe overparameterisation, and it shows:
rerunning identical training code moved window AUC from 0.910 to 0.948. Nothing built on
a single fit of that model is reproducible.

Two fixes, both here:

* **Smaller models.** `ArchSpec` parameterises depth, width, dropout and learning rate so
  the search can find a model matched to the amount of data that actually exists.
* **Grouped cross-validation.** Every window of a recording stays in one fold, so a fold
  boundary is a session boundary. With `group="campaign"` the fold boundary is a whole
  collection sitting, which is the pessimistic estimate: it asks whether the scorer
  survives a change of setup rather than a change of session.

`StratifiedGroupKFold` keeps the human/bot ratio roughly constant across folds while
respecting the grouping, which plain `GroupKFold` does not.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .calibration import expected_calibration_error, fit_temperature, apply_temperature, report
from .features import N_FEATURES
from .paths import RESULTS
from .scorer_train import CONTEXT, Standardiser, build_windows

SEARCH = RESULTS / "search"
SEARCH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArchSpec:
    """One candidate scorer.

    `lstm_units` also sets the depth: (32,) is one recurrent layer, (32, 16) is two.
    `dense_units` are the fully connected layers before the sigmoid.
    """

    name: str
    lstm_units: tuple = (32, 16)
    dense_units: tuple = (32,)
    dropout: float = 0.0            # between recurrent layers
    recurrent_dropout: float = 0.0  # inside the recurrent step
    lr: float = 1e-3
    batch_size: int = 64
    bidirectional: bool = False

    def build(self, n_features: int, context: int = CONTEXT):
        from keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
        from keras.models import Sequential

        layers = [Input(shape=(context, n_features))]
        for i, u in enumerate(self.lstm_units):
            last = i == len(self.lstm_units) - 1
            cell = LSTM(u, return_sequences=not last,
                        recurrent_dropout=self.recurrent_dropout)
            layers.append(Bidirectional(cell) if self.bidirectional else cell)
            if self.dropout and not last:
                layers.append(Dropout(self.dropout))
        if self.dropout:
            layers.append(Dropout(self.dropout))
        for u in self.dense_units:
            layers.append(Dense(u, activation="relu"))
        layers.append(Dense(1, activation="sigmoid"))
        return Sequential(layers)

    def param_count(self, n_features: int) -> int:
        return int(self.build(n_features).count_params())


#: The published architecture, for reference in every comparison.
PUBLISHED = ArchSpec(name="published", lstm_units=(200, 100), dense_units=(128, 64),
                     lr=1e-3)


# --------------------------------------------------------------------------- #
# Grouped cross-validation
# --------------------------------------------------------------------------- #

def _folds(refs, n_splits: int, group: str, seed: int):
    """Yield (train_refs, test_refs). Grouping decides what a fold boundary means."""
    from sklearn.model_selection import StratifiedGroupKFold

    if group == "campaign":
        for camp in sorted({r.campaign for r in refs}):
            tr = [r for r in refs if r.campaign != camp]
            te = [r for r in refs if r.campaign == camp]
            if tr and te:
                yield tr, te, f"hold-out campaign {camp}"
        return

    key = {"session": lambda r: r.name, "family": lambda r: r.family}[group]
    y = np.array([int(r.is_bot) for r in refs])
    g = np.array([key(r) for r in refs])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(splitter.split(np.zeros(len(refs)), y, g)):
        yield [refs[j] for j in tr], [refs[j] for j in te], f"fold {i}"


def _session_scores(p, groups, y):
    """Collapse window predictions to one score per recording (what the sim consumes)."""
    import pandas as pd

    d = pd.DataFrame({"p": p, "g": groups, "y": y}).groupby("g").agg(
        p=("p", "mean"), y=("y", "first"))
    return d.p.values, d.y.values


@dataclass
class CVResult:
    spec: str
    representation: str
    augmentation: str
    n_copies: int
    params: int
    seed: int
    group: str
    window_auc: float
    window_acc: float
    session_auc: float
    session_acc: float
    brier: float
    ece: float
    ece_after_T: float
    temperature: float
    epochs: float
    seconds: float
    n_test_windows: int
    per_fold: list = field(default_factory=list)


def cross_validate(
    refs,
    spec: ArchSpec,
    representation: str = "xy",
    augmentation: str = "none",
    n_copies: int = 1,
    magnitude: int = 1,
    sigma: float = 2.0,
    n_splits: int = 5,
    group: str = "session",
    seed: int = 0,
    epochs: int = 120,
    patience: int = 15,
    val_fraction: float = 0.2,
    verbose: bool = False,
) -> CVResult:
    """Fit `spec` across folds and pool the out-of-fold predictions.

    Pooling out-of-fold predictions means every recording is scored exactly once by a
    model that never saw it, so the whole dataset serves as the test set. That is the
    point of cross-validation here: with 156 recordings a single held-out split wastes
    two thirds of the evidence.

    Early stopping uses a slice of each fold's TRAINING refs, never the test fold.
    """
    import tensorflow as tf
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam

    t0 = time.time()
    oof_p, oof_y, oof_g = [], [], []
    val_p, val_y = [], []
    epochs_run, fold_notes = [], []

    for tr_refs, te_refs, label in _folds(list(refs), n_splits, group, seed):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tr_refs))
        n_val = max(2, int(len(tr_refs) * val_fraction))
        va_refs = [tr_refs[i] for i in idx[:n_val]]
        fit_refs = [tr_refs[i] for i in idx[n_val:]]

        tr = build_windows(fit_refs, representation, augmentation, n_copies,
                           seed=seed, magnitude=magnitude, sigma=sigma)
        va = build_windows(va_refs, representation, "none", 1, seed=seed + 1)
        te = build_windows(te_refs, representation, "none", 1, seed=seed + 2)

        std = Standardiser().fit(tr.X)
        tf.keras.utils.set_random_seed(seed)
        model = spec.build(N_FEATURES[representation])
        model.compile(optimizer=Adam(learning_rate=spec.lr), loss="bce",
                      metrics=["accuracy"])

        n_pos = int(tr.y.sum()); n_neg = len(tr.y) - n_pos
        tot = n_pos + n_neg
        cw = {0: tot / (2 * max(n_neg, 1)), 1: tot / (2 * max(n_pos, 1))}

        hist = model.fit(
            std(tr.X), tr.y, validation_data=(std(va.X), va.y),
            epochs=epochs, batch_size=spec.batch_size, verbose=0, class_weight=cw,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                     restore_best_weights=True)])
        epochs_run.append(len(hist.history["loss"]))

        p = model.predict(std(te.X), batch_size=512, verbose=0).ravel()
        oof_p.append(p); oof_y.append(te.y); oof_g.append(te.groups)
        pv = model.predict(std(va.X), batch_size=512, verbose=0).ravel()
        val_p.append(pv); val_y.append(va.y)

        sp, sy = _session_scores(p, te.groups, te.y)
        from .calibration import roc_auc
        fold_notes.append({"fold": label, "n_test": int(len(te)),
                           "window_auc": roc_auc(p, te.y),
                           "session_auc": roc_auc(sp, sy)})
        if verbose:
            print(f"    {label:22s} n={len(te):4d}  win AUC={fold_notes[-1]['window_auc']:.3f}")

    P = np.concatenate(oof_p); Y = np.concatenate(oof_y); G = np.concatenate(oof_g)
    VP = np.concatenate(val_p); VY = np.concatenate(val_y)

    T = fit_temperature(VP, VY)            # fitted on validation, never on the test folds
    win = report(P, Y)
    win_T = report(apply_temperature(P, T), Y)
    sp, sy = _session_scores(P, G, Y)
    ses = report(sp, sy)

    return CVResult(
        spec=spec.name, representation=representation, augmentation=augmentation,
        n_copies=n_copies, params=spec.param_count(N_FEATURES[representation]),
        seed=seed, group=group,
        window_auc=win["AUC"], window_acc=win["accuracy"],
        session_auc=ses["AUC"], session_acc=ses["accuracy"],
        brier=win["Brier"], ece=win["ECE_quantile"],
        ece_after_T=win_T["ECE_quantile"], temperature=T,
        epochs=float(np.mean(epochs_run)), seconds=time.time() - t0,
        n_test_windows=int(len(Y)), per_fold=fold_notes,
    )


def to_frame(results):
    import pandas as pd

    rows = []
    for r in results:
        d = asdict(r)
        d.pop("per_fold")
        rows.append(d)
    return pd.DataFrame(rows)


def save(results, name: str):
    path = SEARCH / f"{name}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return path
