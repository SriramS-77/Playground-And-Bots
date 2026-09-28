"""Does deduping hurt NaiveBot recall? And is NaiveBot even an LSTM problem?

Train with/without duplicates and with/without family balancing, but ALWAYS evaluate on
deduplicated chunks -- otherwise the metric moves for the wrong reason.
"""
import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import expkit
from expkit.partition import index_sessions
from expkit.scorer_search import _folds
from expkit.humanity_scorer import (ScorerConfig, Standardiser, build_chunk_dataset,
                                    _sample_weights)
from expkit.calibration import report, roc_auc
from rlcaptcha.data import _split_into_chunks
from rlcaptcha.config import SESSION_CHUNK_MS, MAX_CHUNKS

REFS = index_sessions()
CFG = ScorerConfig(representation="kinematic", padding="mask", context=32,
                   lstm_units=(32,), dense_units=(32,), epochs=80, patience=10)

# ---- 0. Can a one-feature rule do NaiveBot's job? ------------------------- #
lens, fams = [], []
for r in REFS:
    raw = json.loads(Path(r.path).read_text())
    if not raw.get("mouse_movements"):
        continue
    seen = set()
    for c in _split_into_chunks(raw, SESSION_CHUNK_MS, MAX_CHUNKS):
        if not c:
            continue
        k = (r.name, len(c), c[0]["timestamp"], c[-1]["timestamp"])
        if k in seen:
            continue
        seen.add(k)
        lens.append(len(c)); fams.append(r.family)
lens, fams = np.array(lens), np.array(fams)
hum = lens[fams == "human"]
print("movements per (deduped) chunk:")
for f in ("human", "NaiveBot", "HumanishBot", "MimicBot", "FallibleBot"):
    v = lens[fams == f]
    print(f"  {f:12s} n={len(v):4d}  min={v.min():4d}  median={int(np.median(v)):4d}  max={v.max():5d}")
for thr in (6, 8, 10, 12):
    nb = lens[fams == "NaiveBot"]
    print(f"  rule 'movements < {thr:2d}' -> NaiveBot recall {(nb < thr).mean():.3f}, "
          f"human false-positive {(hum < thr).mean():.3f}")
mask_nb = np.isin(fams, ["NaiveBot", "human"])
print(f"  movement-count alone, NaiveBot vs human AUC = "
      f"{roc_auc(-lens[mask_nb], (fams[mask_nb] == 'NaiveBot').astype(int)):.3f}")

# ---- 1. dedup / balancing sweep ------------------------------------------ #
rows = []
for dedup_train, balance in ((False, "window"), (True, "window"), (True, "family")):
    for seed in (0, 1, 2):
        P, Y, F = [], [], []
        for tr_refs, te_refs, _ in _folds(REFS, 5, "session", seed):
            rng = np.random.default_rng(seed); idx = rng.permutation(len(tr_refs))
            nv = max(2, int(len(tr_refs) * .2))
            va = [tr_refs[i] for i in idx[:nv]]; fit = [tr_refs[i] for i in idx[nv:]]
            tr = build_chunk_dataset(fit, CFG, dedup=dedup_train)
            vv = build_chunk_dataset(va, CFG, dedup=True)
            te = build_chunk_dataset(te_refs, CFG, dedup=True)      # always deduped
            std = Standardiser().fit(tr.X)
            tf.keras.utils.set_random_seed(seed)
            m = CFG.build(); m.compile(optimizer=Adam(CFG.lr), loss="bce")
            m.fit(std(tr.X), tr.y, sample_weight=_sample_weights(tr, balance),
                  validation_data=(std(vv.X), vv.y), epochs=CFG.epochs,
                  batch_size=CFG.batch_size, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=CFG.patience,
                                           restore_best_weights=True)])
            P.append(m.predict(std(te.X), batch_size=1024, verbose=0).ravel())
            Y.append(te.y); F.append(te.families)
        P, Y, F = map(np.concatenate, (P, Y, F))
        r = report(P, Y)
        rec = {f: float(((P[F == f] >= .5) if f != "human" else (P[F == f] < .5)).mean())
               for f in np.unique(F)}
        rows.append({"train_dedup": dedup_train, "balance": balance, "seed": seed,
                     "n_train_rows": len(tr), "auc": r["AUC"], "acc": r["accuracy"],
                     **{f"rec_{k}": v for k, v in rec.items()}})
        print(f"  dedup={dedup_train!s:5s} balance={balance:7s} seed={seed} "
              f"AUC={r['AUC']:.3f} NaiveBot={rec['NaiveBot']:.3f} "
              f"Humanish={rec['HumanishBot']:.3f} human={rec['human']:.3f}", flush=True)

df = pd.DataFrame(rows); df.to_csv("results/naive_dedup_balance.csv", index=False)
g = df.groupby(["train_dedup", "balance"]).agg(
    auc=("auc", "mean"), auc_sd=("auc", "std"),
    **{f"rec_{k}": (f"rec_{k}", "mean") for k in
       ("human", "NaiveBot", "HumanishBot", "MimicBot", "FallibleBot")}).round(3)
print(); print(g.to_string())
