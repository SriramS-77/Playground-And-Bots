"""Humanity Scorer on the web bot detection dataset of Iliou et al. (2021) -- D1 and D2.

D1 = phase-1 humans + MODERATE bots, D2 = phase-1 humans + ADVANCED bots, each with the dataset's
own session-level train/test lists (35 + 35 train, 15 + 15 test sessions). Iliou et al. report
per-SESSION precision/recall/F1/accuracy (their Table 8), so we report session level for the
comparison and chunk level (each 10 s chunk, what the RL controller sees) alongside it.

Preprocessing (fixed before any result was seen; the rules of `expkit.external` without its
120 s cap, since a session is classified as a whole):
  * coordinates and client timestamps from `mousemove_total_behaviour` / `mousemove_times`;
  * events closer than 10 ms to the previous kept event dropped (our own logger's minimum gap);
  * the session cut into consecutive 10 s chunks from its first event; chunks with < 2 events
    dropped; no top-up replay;
  * each chunk -> windows of 32 events of [dx, dy, dt, speed, accel] + mask, standardised with
    the PRE-TRAINED scorer's statistics (fine-tuning keeps the input scaling).
Scores: window P(bot) -> temperature -> mean over the chunk's windows (chunk score) -> mean over
the session's chunks (session score). Threshold 0.5.

Models, 5 seeds each:
  zero-shot   our scorer (fold 2, cross-fit seed 1 refit, the RL experiments' scorer), as is
  fine-tuned  that scorer, all layers trained on the D_k train sessions: Adam 1e-4, batch 64,
              class-balanced window weights, early stopping (patience 10) on a validation split
              of 20 % of the TRAIN sessions (stratified, by session); temperature refitted there
  scratch     the same architecture from random initialisation, same recipe, lr 3e-3
The test sessions are used only once, for the final metrics.

    python round4/webbot_finetune.py            # -> results/webbot/{webbot_runs,webbot_summary}.csv
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from expkit.humanity_scorer import HumanityScorer, Standardiser, make_window  # noqa: E402

ROOT = EXP.parents[1] / "offline" / "scoring_service" / "web_bot_detection_dataset" / "phase1"
SCORER = EXP / "results" / "round4" / "fold2" / "drift" / "scorer" / "refit"
OUT = EXP / "results" / "webbot"
DATASETS = {"D1": "humans_and_moderate_bots", "D2": "humans_and_advanced_bots"}
SEEDS = range(5)
CHUNK_MS, MIN_GAP_MS, MIN_EVENTS = 10_000, 10, 2


def load(sub):
    lab = {}
    for split in ("train", "test"):
        for line in open(ROOT / "annotations" / sub / split):
            if line.strip():
                sid, cls = line.split()
                lab[sid] = (split, cls != "human")
    sessions = []
    for d in sorted((ROOT / "data" / "mouse_movements" / sub).iterdir()):
        raw = json.loads((d / "mouse_movements.json").read_text())
        t = [int(v) for v in raw["mousemove_times"].split(",") if v.strip()]
        xy = re.findall(r"\[(-?\d+),(-?\d+)\]", raw["mousemove_total_behaviour"])
        assert len(t) == len(xy), raw["session_id"]
        moves, last = [], None
        for ti, (x, y) in zip(t, xy):
            if last is not None and ti - last < MIN_GAP_MS:
                continue
            moves.append({"x": int(x), "y": int(y), "timestamp": ti})
            last = ti
        chunks, cur, end = [], [], moves[0]["timestamp"] + CHUNK_MS
        for m in moves:
            while m["timestamp"] > end:
                chunks.append(cur)
                cur, end = [], end + CHUNK_MS
            cur.append(m)
        chunks.append(cur)
        chunks = [c for c in chunks if len(c) >= MIN_EVENTS]
        split, is_bot = lab[raw["session_id"]]
        sessions.append({"id": raw["session_id"], "split": split, "is_bot": is_bot, "chunks": chunks})
    return sessions


def windows(sessions, cfg):
    X, y, owner = [], [], []
    for si, s in enumerate(sessions):
        for ci, c in enumerate(s["chunks"]):
            w = make_window(c, cfg)
            X.append(w)
            y.append(np.full(len(w), int(s["is_bot"])))
            owner.extend([(si, ci)] * len(w))
    return np.concatenate(X).astype(np.float32), np.concatenate(y), owner


def balanced_weights(y):
    w = np.ones(len(y), np.float32)
    for c in (0, 1):
        w[y == c] = len(y) / (2.0 * max((y == c).sum(), 1))
    return w


def fit(base: HumanityScorer, train, val, seed, scratch):
    import tensorflow as tf
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam

    from expkit.calibration import fit_temperature

    tf.keras.utils.set_random_seed(seed)
    cfg = base.cfg
    Xtr, ytr, _ = windows(train, cfg)
    Xva, yva, _ = windows(val, cfg)
    if scratch:
        std, model, lr = Standardiser().fit(Xtr), cfg.build(), 3e-3
    else:
        std, lr = base.std, 1e-4
        model = cfg.build()
        model.set_weights(base.model.get_weights())
    model.compile(optimizer=Adam(lr), loss="bce")
    model.fit(std(Xtr), ytr, sample_weight=balanced_weights(ytr), validation_data=(std(Xva), yva),
              epochs=100, batch_size=64, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    pv = model.predict(std(Xva), batch_size=2048, verbose=0).ravel()
    return HumanityScorer(cfg, model, std, float(fit_temperature(pv, yva)))


def scores(scorer, sessions):
    X, _, owner = windows(sessions, scorer.cfg)
    p = scorer._calibrate(scorer.model.predict(scorer.std(X), batch_size=4096, verbose=0).ravel())
    chunk = {}
    for (si, ci), v in zip(owner, p):
        chunk.setdefault((si, ci), []).append(v)
    chunk = {k: float(np.mean(v)) for k, v in chunk.items()}
    c_true = np.array([int(sessions[si]["is_bot"]) for si, _ in chunk])
    c_score = np.array(list(chunk.values()))
    s_score = np.array([np.mean([v for (si, _), v in chunk.items() if si == i]) for i in range(len(sessions))])
    s_true = np.array([int(s["is_bot"]) for s in sessions])
    return (c_true, c_score), (s_true, s_score)


def metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    yhat = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, labels=[1, 0], zero_division=0)
    return {"n": len(y), "accuracy": accuracy_score(y, yhat), "auc": roc_auc_score(y, p),
            "bot_precision": pr[0], "bot_recall": rc[0], "bot_f1": f1[0],
            "human_precision": pr[1], "human_recall": rc[1], "human_f1": f1[1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = HumanityScorer.load(SCORER)
    rows = []
    for dname, sub in DATASETS.items():
        sess = load(sub)
        train = [s for s in sess if s["split"] == "train"]
        test = [s for s in sess if s["split"] == "test"]
        print(dname, "train", len(train), "test", len(test), "test chunks",
              sum(len(s["chunks"]) for s in test), flush=True)
        for seed in SEEDS:
            rng = random.Random(seed)
            val = []
            for cls in (False, True):
                ids = sorted(s["id"] for s in train if s["is_bot"] == cls)
                val += rng.sample(ids, round(0.2 * len(ids)))
            tr = [s for s in train if s["id"] not in val]
            va = [s for s in train if s["id"] in val]
            models = {"zero-shot": base} if seed == 0 else {}
            models["fine-tuned"] = fit(base, tr, va, seed, scratch=False)
            models["scratch"] = fit(base, tr, va, seed, scratch=True)
            for mname, sc in models.items():
                (ct, cs), (st, ss) = scores(sc, test)
                for level, (y, p) in (("chunk", (ct, cs)), ("session", (st, ss))):
                    rows.append({"dataset": dname, "model": mname, "seed": seed, "level": level,
                                 **metrics(y, p)})
                print(dname, seed, mname, {k: round(v, 3) for k, v in rows[-1].items()
                                           if k in ("accuracy", "auc", "bot_f1")}, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "webbot_runs.csv", index=False)
    runs = pd.DataFrame(rows)
    num = [c for c in runs.columns if c not in ("dataset", "model", "seed", "level")]
    summ = runs.groupby(["dataset", "model", "level"])[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(OUT / "webbot_summary.csv")
    print(runs.groupby(["dataset", "model", "level"])[["accuracy", "auc", "bot_f1", "human_f1"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
