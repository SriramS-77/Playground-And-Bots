"""RL scores 10s CHUNKS, not whole sessions. Does the representation ranking survive?"""
import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import expkit
from expkit.partition import index_sessions
from expkit.scorer_search import ArchSpec, _folds
from expkit.scorer_train import build_windows, Standardiser
from expkit.features import N_FEATURES, REPRESENTATIONS, windows as mkwin
from expkit.calibration import roc_auc, report
from rlcaptcha.data import _split_into_chunks
from rlcaptcha.config import SESSION_CHUNK_MS, MAX_CHUNKS

REFS = index_sessions()
SPEC = ArchSpec("lstm32", (32,), (32,))
FAM = {r.name: r.family for r in REFS}

def chunk_windows(refs, rep):
    """One window per non-empty 10s chunk -- exactly what the RL simulation feeds."""
    fn = REPRESENTATIONS[rep]
    X, y, g, padfrac = [], [], [], []
    for r in refs:
        raw = json.loads(Path(r.path).read_text())
        if not raw.get("mouse_movements"):
            continue
        for ch in _split_into_chunks(raw, SESSION_CHUNK_MS, MAX_CHUNKS):
            if not ch:
                continue
            w = mkwin(fn(ch), 100)
            X.append(w); y.append(np.full(len(w), int(r.is_bot)))
            g.append(np.full(len(w), r.name, dtype=object))
            padfrac.append(np.full(len(w), max(0, 100 - len(ch)) / 100))
    return (np.concatenate(X).astype("float32"), np.concatenate(y),
            np.concatenate(g), np.concatenate(padfrac))

rows = []
for rep in ("xy", "dxdy", "dxdy_dt", "kinematic"):
    for train_on in ("session", "chunk"):
        P, Y, G, PF = [], [], [], []
        for tr_refs, te_refs, _ in _folds(REFS, 5, "session", 0):
            rng = np.random.default_rng(0); idx = rng.permutation(len(tr_refs))
            nv = max(2, int(len(tr_refs) * .2))
            va = [tr_refs[i] for i in idx[:nv]]; fit = [tr_refs[i] for i in idx[nv:]]
            if train_on == "session":
                tw = build_windows(fit, rep, "none", 1, seed=0)
                vw = build_windows(va, rep, "none", 1, seed=1)
                tX, ty, vX, vy = tw.X, tw.y, vw.X, vw.y
            else:
                tX, ty, _, _ = chunk_windows(fit, rep)
                vX, vy, _, _ = chunk_windows(va, rep)
            eX, ey, eg, epf = chunk_windows(te_refs, rep)   # ALWAYS eval on chunks
            std = Standardiser().fit(tX)
            tf.keras.utils.set_random_seed(0)
            m = SPEC.build(N_FEATURES[rep]); m.compile(optimizer=Adam(SPEC.lr), loss="bce")
            npos = int(ty.sum()); nneg = len(ty) - npos; tot = npos + nneg
            m.fit(std(tX), ty, validation_data=(std(vX), vy), epochs=90, batch_size=64,
                  verbose=0, class_weight={0: tot/(2*max(nneg,1)), 1: tot/(2*max(npos,1))},
                  callbacks=[EarlyStopping(monitor="val_loss", patience=12,
                                           restore_best_weights=True)])
            P.append(m.predict(std(eX), batch_size=512, verbose=0).ravel())
            Y.append(ey); G.append(eg); PF.append(epf)
        P, Y, G, PF = (np.concatenate(P), np.concatenate(Y),
                       np.concatenate(G), np.concatenate(PF))
        rep_ = report(P, Y)
        # chunk-level is the operating point; also collapse to session for comparison
        d = pd.DataFrame({"p": P, "g": G, "y": Y}).groupby("g").agg(p=("p","mean"), y=("y","first"))
        fams = np.array([FAM[g] for g in G])
        rows.append({"representation": rep, "trained_on": train_on,
                     "chunk_auc": rep_["AUC"], "chunk_acc": rep_["accuracy"],
                     "chunk_ece": rep_["ECE_quantile"], "session_auc": roc_auc(d.p.values, d.y.values),
                     "corr_score_padding": float(np.corrcoef(P, PF)[0, 1]),
                     **{f"recall_{f}": float(((P[fams==f] >= .5) if f != "human"
                        else (P[fams==f] < .5)).mean()) for f in
                        ("NaiveBot","HumanishBot","MimicBot","FallibleBot","human")}})
        print(f"{rep:10s} trained_on={train_on:8s} chunkAUC={rep_['AUC']:.3f} "
              f"ECE={rep_['ECE_quantile']:.3f} corr(score,padding)={rows[-1]['corr_score_padding']:+.2f}",
              flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/chunk_level_eval.csv", index=False)
print()
print(df.pivot_table(index="representation", columns="trained_on", values="chunk_auc").round(3).to_string())
print()
print(df[[c for c in df.columns if c.startswith(("representation","trained_on","recall_"))]].round(3).to_string(index=False))
