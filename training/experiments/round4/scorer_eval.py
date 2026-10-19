"""Humanity Scorer on the fold-2 EVAL BLOCK: the scorer every RL result in the paper uses
(cross-fit seed 1 refit, `results/round4/fold2/drift/scorer/refit`).

Chunk level (each distinct 10 s chunk; the 12-chunk top-up replays removed) and session level
(mean of the session's chunk scores): AUC, accuracy at 0.5, confusion matrix, Brier, ECE (10
uniform bins) with and without temperature scaling, and a reliability diagram. Opens the eval
block once, logged.

    python round4/scorer_eval.py     -> results/round4/fold2/drift/scorer_eval.{json,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402

from expkit.calibration import brier, expected_calibration_error, roc_auc  # noqa: E402
from expkit.humanity_scorer import HumanityScorer, chunk_key  # noqa: E402

R = G.R
OUT = R.OUT


def main():
    ctx = R.context(with_eval=True, reason="R6 scorer metrics on the eval block")
    sc = HumanityScorer.load(OUT / "scorer" / "refit")
    T = sc.temperature
    sessions = ctx.ev_h + ctx.ev_b
    rows = []                                      # (session index, raw p, calibrated p, y)
    for i, s in enumerate(sessions):
        seen = set()
        chunks = []
        for c in s.chunks:
            if not c:
                continue
            k = chunk_key(s.key, c)
            if k in seen:
                continue
            seen.add(k)
            chunks.append(c)
        sc.temperature = 1.0
        raw = sc.score_chunks(chunks)
        sc.temperature = T
        cal = sc.score_chunks(chunks)
        rows += [(i, r, c, int(s.is_bot)) for r, c in zip(raw, cal) if r is not None]
    a = np.array(rows, dtype=float)
    out = {"n_sessions": len(sessions), "n_humans": len(ctx.ev_h), "n_bots": len(ctx.ev_b),
           "n_chunks": len(a), "temperature": T}
    for level in ("chunk", "session"):
        if level == "chunk":
            y, praw, pcal = a[:, 3], a[:, 1], a[:, 2]
        else:
            idx = np.unique(a[:, 0])
            y = np.array([a[a[:, 0] == i, 3][0] for i in idx])
            praw = np.array([a[a[:, 0] == i, 1].mean() for i in idx])
            pcal = np.array([a[a[:, 0] == i, 2].mean() for i in idx])
        yhat = (pcal >= 0.5).astype(int)
        out[level] = {
            "n": int(len(y)), "auc": roc_auc(pcal, y), "accuracy": float((yhat == y).mean()),
            "tp_bot": int(((yhat == 1) & (y == 1)).sum()), "fn_bot": int(((yhat == 0) & (y == 1)).sum()),
            "tn_human": int(((yhat == 0) & (y == 0)).sum()), "fp_human": int(((yhat == 1) & (y == 0)).sum()),
            "brier_raw": brier(praw, y), "brier_cal": brier(pcal, y),
            "ece_raw": expected_calibration_error(praw, y)["ECE"],
            "ece_cal": expected_calibration_error(pcal, y)["ECE"]}
    (OUT / "scorer_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
