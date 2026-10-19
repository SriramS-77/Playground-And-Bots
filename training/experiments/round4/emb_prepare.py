"""Embedding state, step 1: one scorer, its scores and its embeddings for every session (round 6).

The cross-fit scorers' penultimate layers live in different coordinate systems, and the refit saw
the RL block, so neither can supply a consistent, leakage-free embedding. Instead ONE scorer is
trained on the SCORER BLOCK ONLY (refit config, seed 1, 25% validation split) and used for
everything: q (calibrated chunk probability, mean of windows) and the chunk embedding (mean of
the windows' 32-d Dense-ReLU outputs) for the scorer, RL and eval blocks and every external pool.

Embeddings are standardised and reduced to K = 8 PCA components fitted on SCORER-BLOCK chunks only
(frozen). Tables keyed like the score cache, (session key, chunk index):
    results/round4/fold2/drift/emb/{q_table.pkl, emb_table.pkl, pca.pkl, scorer/}

Probe (no RL): logistic regression on clean RL-block chunks, using q, the embedding, or both;
chunk AUC on the eval block with covert bots replacing a share rho of bots (0, 0.5, 1).

    python round4/emb_prepare.py
"""
from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402

R, D = G.R, G.D
OUT = R.OUT / "emb"
K = 8
MAX_CHUNKS = 12


def chunk_q_emb(sc, emb_model, sessions):
    """-> {(key, i): q}, {(key, i): 32-d mean embedding} in one batched pass."""
    from expkit.humanity_scorer import make_window
    flat, owner = [], []
    for s in sessions:
        for i, ch in enumerate(s.chunks[:MAX_CHUNKS]):
            if not ch:
                continue
            w = make_window(ch, sc.cfg)
            flat.append(w)
            owner += [(s.key, i)] * len(w)
    if not flat:
        return {}, {}
    X = sc.std(np.concatenate(flat))
    p = sc._calibrate(sc.model.predict(X, batch_size=4096, verbose=0).ravel())
    e = emb_model.predict(X, batch_size=4096, verbose=0)
    q, emb = {}, {}
    owner = np.array(owner, dtype=object)
    keys = list(dict.fromkeys(map(tuple, owner)))
    idx = {}
    for j, k in enumerate(map(tuple, owner)):
        idx.setdefault(k, []).append(j)
    for k in keys:
        q[k] = float(p[idx[k]].mean())
        emb[k] = e[idx[k]].mean(axis=0)
    return q, emb


def main():
    import keras
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from expkit.humanity_scorer import HumanityScorer
    from expkit.partition import split_refs
    from expkit.simx import sessions_from_refs

    p = D.pools(False, "R6 embedding state: prepare")
    ctx = R.context(with_eval=True, reason="R6 embedding state: prepare (eval chunks)")
    OUT.mkdir(parents=True, exist_ok=True)
    sdir = OUT / "scorer"
    if (sdir / "model.keras").exists():
        sc = HumanityScorer.load(sdir)
    else:
        ref = HumanityScorer.load(R.OUT / "scorer" / "refit")
        tr, va = split_refs(list(ctx.fold.scorer), 0.25, seed=1)
        sc = HumanityScorer(dataclasses.replace(ref.cfg)).fit(tr, va, seed=1)
        sc.save(sdir)
    print("scorer-block scorer:", {k: round(v, 3) if isinstance(v, float) else v
                                   for k, v in (sc.training_summary or {}).items()}, "T", sc.temperature)
    emb_model = keras.Model(sc.model.inputs, sc.model.layers[-2].output)

    sb_h, sb_b = sessions_from_refs(list(ctx.fold.scorer))
    groups = {"scorer": sb_h + sb_b, "rl": p.train_h + p.train_b, "eval": p.ev_h + p.ev_b}
    for k, v in p.ext.items():
        groups[f"ext:{k}"] = v
    q_tab, raw = {}, {}
    for g, ss in groups.items():
        q, e = chunk_q_emb(sc, emb_model, ss)
        q_tab.update(q)
        raw.update(e)
        print(f"  {g}: {len(q)} chunks")

    sb_keys = [k for k in raw if k[0] in {s.key for s in groups["scorer"]}]
    Xs = np.stack([raw[k] for k in sb_keys])
    scaler = StandardScaler().fit(Xs)
    pca = PCA(n_components=K, random_state=0).fit(scaler.transform(Xs))
    print("PCA explained variance (scorer block):", pca.explained_variance_ratio_.round(3),
          "total", pca.explained_variance_ratio_.sum().round(3))
    keys = list(raw)
    Z = pca.transform(scaler.transform(np.stack([raw[k] for k in keys])))
    emb_tab = {k: z.astype(np.float32) for k, z in zip(keys, Z)}
    pickle.dump(q_tab, open(OUT / "q_table.pkl", "wb"))
    pickle.dump(emb_tab, open(OUT / "emb_table.pkl", "wb"))
    pickle.dump({"scaler": scaler, "pca": pca}, open(OUT / "pca.pkl", "wb"))

    # ---- probe
    def rows(sessions, y):
        ks = [(s.key, i) for s in sessions for i in range(MAX_CHUNKS) if (s.key, i) in q_tab]
        return ks, np.full(len(ks), y)

    def feats(ks, which):
        q = np.array([[q_tab[k]] for k in ks])
        e = np.stack([emb_tab[k] for k in ks])
        return {"q": q, "emb": e, "q+emb": np.c_[q, e]}[which]

    trk_h, yh = rows(p.train_h, 0)
    trk_b, yb = rows(p.train_b, 1)
    evh, _ = rows(p.ev_h, 0)
    evb, _ = rows(p.ev_b, 1)
    cov, _ = rows(p.ext["webbot_advanced_heldout"], 1)
    print("\nchunk AUC on the eval block (covert bots = held-out Phase-2 advanced web bots)")
    print(f"{'input':8s} {'rho=0':>7s} {'rho=0.5':>8s} {'rho=1':>7s}")
    for which in ("q", "emb", "q+emb"):
        if which == "q":
            score = lambda ks: feats(ks, "q").ravel()
        else:
            clf = LogisticRegression(max_iter=2000).fit(
                feats(trk_h + trk_b, which), np.r_[yh, yb])
            score = lambda ks, c=clf, w=which: c.predict_proba(feats(ks, w))[:, 1]
        sh, sb, sc_ = score(evh), score(evb), score(cov)
        out = []
        for rho in (0.0, 0.5, 1.0):
            # bots: (1 - rho) eval bots + rho covert bots, as sample weights
            s = np.r_[sh, sb, sc_]
            y = np.r_[np.zeros(len(sh)), np.ones(len(sb) + len(sc_))]
            w = np.r_[np.full(len(sh), 1 / len(sh)), np.full(len(sb), (1 - rho) / len(sb)),
                      np.full(len(sc_), rho / len(sc_))]
            out.append(roc_auc_score(y, s, sample_weight=w))
        print(f"{which:8s} {out[0]:7.3f} {out[1]:8.3f} {out[2]:7.3f}")


if __name__ == "__main__":
    main()
