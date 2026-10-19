"""How the fixed-share posterior turns a score into P(bot), at different traffic compositions.

Uses the posterior reward's own calibration (Platt on the scorer block, out of fold; no eval data).
For a user whose latest and running-mean chunk scores both equal q at step k:
    P(bot) = sigma( logit(pi) + log LR(q, q, k) )
with pi = the calibration share (fixed) or the true share of the traffic. Prints the posterior
for typical scores and the raw-score threshold at which P(bot) crosses 0.5.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G  # noqa: E402

import numpy as np  # noqa: E402

C = G.C
sys.path.insert(0, str(HERE.parent / "round3"))
import e7_proxy_reward as E  # noqa: E402

p = C.pools(final=False)
cal = E.fit_calibration(p)
print(f"\npi_cal (calibration bot share) = {cal.pi_cal:.3f}; LR bounds {cal.bounds()}")


def post(q, k, pi):
    ll = float(cal.log_lr([q], [q], [k])[0])
    return 1 / (1 + math.exp(-(math.log(pi / (1 - pi)) + ll)))


def threshold(k, pi):
    qs = np.linspace(0.001, 0.999, 999)
    above = [q for q in qs if post(q, k, pi) > 0.5]
    return above[0] if above else float("nan")


shares = {"fixed (calibration)": cal.pi_cal}
shares.update({f"true, 100 humans + {b} bots": b / (100 + b) for b in (20, 100, 200, 500, 1000)})
scores = {"typical human 0.11": 0.11, "covert bot 0.39": 0.39, "0.5": 0.5, "typical bot 0.85": 0.85}
for k in (0, 5):
    print(f"\nstep {k}: P(bot) for a user whose scores are q, and the q at which P(bot) = 0.5")
    print(f"{'bot share pi':32s}" + "".join(f"{s:>20s}" for s in scores) + f"{'threshold q*':>15s}")
    for lab, pi in shares.items():
        print(f"{lab:32s}" + "".join(f"{post(q, k, pi):20.3f}" for q in scores.values())
              + f"{threshold(k, pi):15.3f}  (pi={pi:.3f})")
