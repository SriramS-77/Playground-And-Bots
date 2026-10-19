"""Phase-2 web-bot sets scored by the paper's final scorer (cross-fit seed 1 refit,
`results/round4/fold2/drift/scorer/refit`), to replace the local-scorer numbers quoted for
excluding the moderate bots. External data only: the eval block is not opened.

Per set (both halves, as `expkit.external` prepares them): mean calibrated chunk P(bot), mean
session score, share of sessions classified bot at 0.5, and session AUC against the Phase-2
humans (same source).

    python round4/webbot_moderate_rescore.py  -> results/round4/fold2/drift/webbot_moderate_rescore.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402

from expkit.calibration import roc_auc            # noqa: E402
from expkit.external import load_sessions         # noqa: E402
from expkit.humanity_scorer import HumanityScorer, chunk_key  # noqa: E402

OUT = G.R.OUT


def session_scores(sc, sessions):
    chunk_p, sess_p = [], []
    for s in sessions:
        seen, chunks = set(), []
        for c in s.chunks:
            if not c:
                continue
            k = chunk_key(s.key, c)
            if k not in seen:
                seen.add(k)
                chunks.append(c)
        p = [x for x in sc.score_chunks(chunks) if x is not None]
        if p:
            chunk_p += p
            sess_p.append(float(np.mean(p)))
    return np.array(chunk_p), np.array(sess_p)


def main():
    sc = HumanityScorer.load(OUT / "scorer" / "refit")
    ext = load_sessions()
    res = {}
    for kind in ("humans", "moderate", "advanced"):
        ss = ext[f"webbot_{kind}_inject"] + ext[f"webbot_{kind}_heldout"]
        cp, sp = session_scores(sc, ss)
        res[kind] = {"n_sessions": int(len(sp)), "n_chunks": int(len(cp)),
                     "chunk_mean": float(cp.mean()), "session_mean": float(sp.mean()),
                     "share_classified_bot": float((sp >= 0.5).mean()), "_s": sp}
    h = res["humans"]["_s"]
    for kind in ("moderate", "advanced"):
        b = res[kind]["_s"]
        res[kind]["session_auc_vs_humans"] = roc_auc(np.r_[h, b], np.r_[np.zeros(len(h)), np.ones(len(b))])
    for r in res.values():
        r.pop("_s")
    (OUT / "webbot_moderate_rescore.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
