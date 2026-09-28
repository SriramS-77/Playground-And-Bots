"""Per-family out-of-fold recall by representation. Why is NaiveBot not 100%?"""
import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import expkit
from expkit.partition import index_sessions
from expkit.scorer_search import ArchSpec, _folds
from expkit.scorer_train import build_windows, Standardiser
from expkit.features import N_FEATURES
from expkit.calibration import roc_auc

REFS = index_sessions()
FAM = {r.name: r.family for r in REFS}
SPEC = ArchSpec("lstm32", (32,), (32,))

rows = []
for rep in ("xy", "dxdy", "dxdy_dt", "kinematic"):
    P, Y, G = [], [], []
    for tr_refs, te_refs, _ in _folds(REFS, 5, "session", 0):
        rng = np.random.default_rng(0); idx = rng.permutation(len(tr_refs))
        nv = max(2, int(len(tr_refs) * .2))
        va = [tr_refs[i] for i in idx[:nv]]; fit = [tr_refs[i] for i in idx[nv:]]
        tw = build_windows(fit, rep, "none", 1, seed=0)
        vw = build_windows(va, rep, "none", 1, seed=1)
        ew = build_windows(te_refs, rep, "none", 1, seed=2)
        std = Standardiser().fit(tw.X)
        tf.keras.utils.set_random_seed(0)
        m = SPEC.build(N_FEATURES[rep]); m.compile(optimizer=Adam(SPEC.lr), loss="bce")
        npos = int(tw.y.sum()); nneg = len(tw.y) - npos; tot = npos + nneg
        m.fit(std(tw.X), tw.y, validation_data=(std(vw.X), vw.y), epochs=90,
              batch_size=64, verbose=0,
              class_weight={0: tot/(2*max(nneg,1)), 1: tot/(2*max(npos,1))},
              callbacks=[EarlyStopping(monitor="val_loss", patience=12,
                                       restore_best_weights=True)])
        P.append(m.predict(std(ew.X), batch_size=512, verbose=0).ravel())
        Y.append(ew.y); G.append(ew.groups)
    P, Y, G = np.concatenate(P), np.concatenate(Y), np.concatenate(G)
    fams = np.array([FAM[g] for g in G])
    for f in ("NaiveBot", "HumanishBot", "MimicBot", "FallibleBot", "human"):
        mk = fams == f
        if not mk.any():
            continue
        correct = (P[mk] >= .5) if f != "human" else (P[mk] < .5)
        rows.append({"representation": rep, "family": f, "n_windows": int(mk.sum()),
                     "recall": float(correct.mean()), "mean_score": float(P[mk].mean()),
                     "min_score": float(P[mk].min()), "max_score": float(P[mk].max())})
    print(f"{rep} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/per_family_by_representation.csv", index=False)
print()
print(df.pivot_table(index="family", columns="representation", values="recall").round(3).to_string())
print()
print(df.pivot_table(index="family", columns="representation", values="mean_score").round(3).to_string())
print()
print(df[df.family == "NaiveBot"].round(3).to_string(index=False))
