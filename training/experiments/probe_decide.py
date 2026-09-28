"""Settle: dxdy vs kinematic, padding mode, and context length -- at the CHUNK operating
point, deduplicated, grouped 5-fold CV, 2 seeds."""
import os, itertools, dataclasses
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import expkit
from expkit.partition import index_sessions
from expkit.scorer_search import _folds
from expkit.humanity_scorer import (ScorerConfig, Standardiser, build_chunk_dataset,
                                    _sample_weights)
from expkit.calibration import report, fit_temperature, apply_temperature

REFS = index_sessions()
rows = []
GRID = list(itertools.product(("dxdy", "kinematic"),
                              ("repeat_point", "mask"),
                              (100, 50, 32)))
for rep, pad, ctx in GRID:
    for seed in (0, 1):
        cfg = ScorerConfig(representation=rep, padding=pad, context=ctx,
                           lstm_units=(32,), dense_units=(32,), epochs=80, patience=10)
        P, Y, F, PF = [], [], [], []
        VP, VY = [], []
        for tr_refs, te_refs, _ in _folds(REFS, 5, "session", seed):
            rng = np.random.default_rng(seed); idx = rng.permutation(len(tr_refs))
            nv = max(2, int(len(tr_refs) * .2))
            va = [tr_refs[i] for i in idx[:nv]]; fit = [tr_refs[i] for i in idx[nv:]]
            tr = build_chunk_dataset(fit, cfg, dedup=True)
            vv = build_chunk_dataset(va, cfg, dedup=True)
            te = build_chunk_dataset(te_refs, cfg, dedup=True)
            std = Standardiser().fit(tr.X)
            tf.keras.utils.set_random_seed(seed)
            m = cfg.build(); m.compile(optimizer=Adam(cfg.lr), loss="bce")
            m.fit(std(tr.X), tr.y, sample_weight=_sample_weights(tr, "window"),
                  validation_data=(std(vv.X), vv.y), epochs=cfg.epochs,
                  batch_size=cfg.batch_size, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                           restore_best_weights=True)])
            P.append(m.predict(std(te.X), batch_size=1024, verbose=0).ravel())
            Y.append(te.y); F.append(te.families); PF.append(te.pad_fraction)
            VP.append(m.predict(std(vv.X), batch_size=1024, verbose=0).ravel()); VY.append(vv.y)
        P, Y, F, PF = map(np.concatenate, (P, Y, F, PF))
        T = fit_temperature(np.concatenate(VP), np.concatenate(VY))
        r = report(P, Y); rT = report(apply_temperature(P, T), Y)
        rec = {f: float(((P[F == f] >= .5) if f != "human" else (P[F == f] < .5)).mean())
               for f in np.unique(F)}
        rows.append({"representation": rep, "padding": pad, "context": ctx, "seed": seed,
                     "n_windows": len(Y), "chunk_auc": r["AUC"], "acc": r["accuracy"],
                     "ece_T": rT["ECE_quantile"], "corr_pad": float(np.corrcoef(P, PF)[0, 1]),
                     "mean_pad": float(PF.mean()), **{f"rec_{k}": v for k, v in rec.items()}})
        print(f"{rep:10s} {pad:12s} ctx={ctx:3d} seed={seed} AUC={r['AUC']:.3f} "
              f"ECE={rT['ECE_quantile']:.3f} corr_pad={rows[-1]['corr_pad']:+.2f} "
              f"meanpad={rows[-1]['mean_pad']:.2f}", flush=True)

df = pd.DataFrame(rows); df.to_csv("results/decide_scorer.csv", index=False)
g = df.groupby(["representation", "padding", "context"]).agg(
    auc=("chunk_auc", "mean"), auc_sd=("chunk_auc", "std"), ece=("ece_T", "mean"),
    corr_pad=("corr_pad", "mean"), mean_pad=("mean_pad", "mean"),
    rec_human=("rec_human", "mean"), rec_Mimic=("rec_MimicBot", "mean"),
    rec_Fallible=("rec_FallibleBot", "mean"), rec_Humanish=("rec_HumanishBot", "mean"),
    rec_Naive=("rec_NaiveBot", "mean")).round(3).sort_values("auc", ascending=False)
print(); print(g.to_string())
