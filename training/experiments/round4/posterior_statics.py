"""Static threshold policies on the POSTERIOR P(bot | score evidence) instead of the raw score,
under round 4's calibration drift (grounded world, eval block). Exploratory -- not in the paper.

Posterior at each decision: sigma(logit(pi) + log LR(latest score, running mean, step)), with the
likelihood ratio from the Platt calibration fitted on the SCORER block (E7's `fit_calibration`,
the posterior reward's own evidence model). Two ways to set the bot share pi:

  fixed   pi = the calibration set's bot share (the "offline" policy: nothing adapts)
  EM      pi re-estimated without labels from the CURRENT traffic: EM over the first-chunk
          likelihood ratios of the users of that traffic setting (Saerens et al. 2002) --
          the label-free "online" adaptation a static policy can do

Policies: single (level 10 if posterior > 0.5 else 0) and multi (level = floor(10 * posterior)),
for the raw score (the paper's statics), the fixed-share posterior and the EM posterior.
Evaluated at rho = 0 and every shift level in both directions, 100 humans x bots {20..1000},
20 eval seeds -- the same cells as round 4's offline panel.

    python round4/posterior_statics.py --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
sys.path.insert(0, str(HERE.parent / "round3"))
OUT = G.OUT.parent / ("posterior_statics_true" if os.environ.get("PS_TRUE") else "posterior_statics")


class PosteriorStatic:
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, calib, pi, multi, name):
        self.calib, self.multi, self.name = calib, multi, name
        self.prior_logit = math.log(pi / (1 - pi))

    def posterior(self, user):
        step = max(user._chunk_index - 1, 0)
        ll = float(self.calib.log_lr([user.bot_score], [user.avg_bot_score], [step])[0])
        return 1.0 / (1.0 + math.exp(-(self.prior_logit + ll)))

    def select_action(self, user, n_active):
        q = self.posterior(user)
        return min(int(q * 10), 9) if self.multi else (10 if q > 0.5 else 0)


def points():
    return [("none", 0.0)] + [(d, r) for d in D.DIRECTIONS for r in D.LEVELS]


def tasks():
    return [{"direction": d, "rho": r} for d, r in points()]


def em_pi(calib, cache, humans, bots, n_h, n_b, seed):
    """Label-free bot share of one traffic setting: EM over first-chunk likelihood ratios of a
    draw of n_h humans and n_b bots from the (shifted) evaluation pool."""
    rng = np.random.default_rng(seed)
    pick = [humans[i] for i in rng.integers(0, len(humans), n_h)] + \
           [bots[i] for i in rng.integers(0, len(bots), n_b)]
    s = []
    for sess in pick:
        v = cache._table.get((sess.key, 0))
        s.append(0.5 if v is None else v)
    from expkit.posterior_reward import em_share
    return em_share(calib.log_lr(s, s, [0] * len(s)), pi0=calib.pi_cal)


def run_task(i):
    import e7_proxy_reward as E
    from expkit.simx import evaluate_x, run_x

    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{t['direction']}_{t['rho']}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = D.pools(False, f"R6 posterior statics {t}")
    calib = E.fit_calibration(p)
    h, b, real = D.shifted(p, t["direction"], t["rho"], held_out=True)
    rows = []
    for nb in D.SCORED_VOLUMES:
        pi_em = em_pi(calib, p.cache, h, b, 100, nb, seed=nb)
        if os.environ.get("PS_TRUE"):
            pi_t = nb / (100 + nb)
            pols = [PosteriorStatic(calib, pi_t, False, "Posterior single (true share)"),
                    PosteriorStatic(calib, pi_t, True, "Posterior multi (true share)")]
        else:
          pols = C.statics() + [
            PosteriorStatic(calib, calib.pi_cal, False, "Posterior single (fixed share)"),
            PosteriorStatic(calib, calib.pi_cal, True, "Posterior multi (fixed share)"),
            PosteriorStatic(calib, pi_em, False, "Posterior single (EM share)"),
            PosteriorStatic(calib, pi_em, True, "Posterior multi (EM share)")]
        for pol in pols:
            for s in range(C.EVAL_SEEDS):
                res, _ = run_x(pol, h, b, nb, cache=p.cache, n_humans=100, seed=s,
                               solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
                m = evaluate_x(res)
                rows.append({"policy": pol.name, "eval_direction": t["direction"],
                             "eval_rho": t["rho"], "rho_realised": real, "bots": nb, "eval_seed": s,
                             "pi_used": None if pol.name.startswith("Static") else
                             (pi_em if "EM" in pol.name else nb / (100 + nb) if "true" in pol.name else calib.pi_cal),
                             "true_share": nb / (100 + nb), "DI": m.DI,
                             "human_survival": res.surviving_humans / 100,
                             "bot_survival": res.surviving_bots / nb,
                             "friction_s": res.mean_human_friction_seconds})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{t}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T = tasks()
    done = lambda i: (OUT / f"{T[i]['direction']}_{T[i]['rho']}.csv").exists()
    todo = [i for i in range(len(T)) if not done(i)]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            log = open(OUT / "logs" / f"{i}.log", "w")
            running[i] = (subprocess.Popen([sys.executable, __file__, "--task", str(i)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env), log)
        for i, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[i]
                print(f"task {i} exit {proc.returncode}{'' if done(i) else ' NO CSV'}", flush=True)
        time.sleep(5)


def collect():
    runs = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv"))], ignore_index=True)
    runs.to_csv(OUT.parent / "posterior_statics_runs.csv", index=False)
    base = runs[runs.eval_direction == "none"]
    for d in D.DIRECTIONS:
        f = pd.concat([base.assign(eval_direction=d), runs[runs.eval_direction == d]])
        t = f.groupby(["policy", "eval_rho"]).DI.mean().unstack().round(1)
        t.to_csv(OUT.parent / f"posterior_statics_{d}.csv")
        print(f"\n{d} look {'human' if d == 'bots' else 'bot-like'}: DI by rho\n{t.to_string()}")
    em = runs[runs.policy == "Posterior single (EM share)"].groupby(
        ["eval_direction", "eval_rho", "bots"])[["pi_used", "true_share"]].mean()
    em.to_csv(OUT.parent / "posterior_statics_em_share.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.collect:
        collect()
    elif a.run_all:
        run_all(a.run_all)
    elif a.task is not None:
        run_task(a.task)


if __name__ == "__main__":
    main()
