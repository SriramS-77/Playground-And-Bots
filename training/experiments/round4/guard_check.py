"""Can a label-free guard tell bot-side drift from a change in bot share? (exploratory)

Guard: fit the current traffic's first-chunk score distribution as a mixture
pi * F_bot + (1 - pi) * F_human of the REFERENCE score distributions (scorer block, out of fold),
choosing pi to minimise the Kolmogorov-Smirnov distance; a parametric bootstrap from the fitted
mixture gives a p-value. Label shift (only pi changes) fits; a change in what bots or humans look
like should not -- unless the new class looks exactly like the other one (replay), which no
label-free test can detect.

Traffic: 100 humans + 200 bots drawn from the shifted EVAL pool, calibration-only (round 4) and
replay (AUC-degrading) shifts, 20 draws per point.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G  # noqa: E402

import numpy as np  # noqa: E402

D, C = G.D, G.C
orig_shift = D.shifted
import auc_drift  # noqa: E402  (patches D.shifted)
replay_shift = D.shifted
D.shifted = orig_shift

from expkit.posterior_reward import replay_scores  # noqa: E402
from expkit.simx import sessions_from_refs  # noqa: E402

p = D.pools(False, "R6 guard check (scores only)")
sh, sb = sessions_from_refs(p.ctx.fold.scorer)
rows = replay_scores(sh, sb, p.ctx.crossfit.cache(sh + sb))
ref_h = np.array([r[1] for r in rows if r[0] == 0 and not r[3]])
ref_b = np.array([r[1] for r in rows if r[0] == 0 and r[3]])
grid = np.linspace(0, 1, 101)
xs = np.linspace(0, 1, 401)
Fh = np.searchsorted(np.sort(ref_h), xs, side="right") / len(ref_h)
Fb = np.searchsorted(np.sort(ref_b), xs, side="right") / len(ref_b)


def fit(s):
    Fs = np.searchsorted(np.sort(s), xs, side="right") / len(s)
    ks = [np.max(np.abs(Fs - (pi * Fb + (1 - pi) * Fh))) for pi in grid]
    j = int(np.argmin(ks))
    return grid[j], ks[j]


def first_scores(sessions, rng, n):
    pick = [sessions[i] for i in rng.integers(0, len(sessions), n)]
    v = [p.cache._table.get((s.key, 0)) for s in pick]
    return np.array([0.5 if x is None else x for x in v], dtype=float)


def pval(pi, ks, n, rng, B=200):
    null = []
    for _ in range(B):
        k = rng.binomial(n, pi)
        s = np.r_[rng.choice(ref_b, k), rng.choice(ref_h, n - k)]
        null.append(fit(s)[1])
    return float(np.mean(np.array(null) >= ks))


rng = np.random.default_rng(0)
print(f"{'shift':16s}{'dir':8s}{'rho':>5s}{'true pi':>9s}{'fitted pi':>11s}{'KS':>7s}{'p-value':>9s}{'alarm@.01':>10s}")
for label, fn in (("calibration-only", orig_shift), ("replay (AUC)", replay_shift)):
    D.shifted = fn
    for d, rho in [("none", 0.0)] + [(d, r) for d in ("bots", "humans") for r in (0.25, 0.5, 1.0)]:
        h, b, _ = D.shifted(p, d, rho, held_out=True)
        res = []
        for _ in range(20):
            s = np.r_[first_scores(h, rng, 100), first_scores(b, rng, 200)]
            pi, ks = fit(s)
            res.append((pi, ks, pval(pi, ks, len(s), rng)))
        r = np.array(res)
        print(f"{label:16s}{d:8s}{rho:5.2f}{200/300:9.2f}{r[:,0].mean():11.2f}{r[:,1].mean():7.3f}"
              f"{np.median(r[:,2]):9.3f}{np.mean(r[:,2] < 0.01):10.2f}")
