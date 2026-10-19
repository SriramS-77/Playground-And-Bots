"""RL performance as a function of Humanity Scorer quality (round 6).

A weaker scorer is made from the real one by adding fixed Gaussian noise to each chunk's logit,
z' = logit(q) + sigma * e(chunk), e drawn once per (session, chunk) so the degraded scorer is a
deterministic function like the real one. sigma is set by bisection so the chunk-level AUC on the
RL block (training cache) hits the target; the noisy scores are then Platt-recalibrated on the RL
block, so the weaker scorer stays calibrated and only its ranking power changes. The same noise
and map are applied to the eval block's chunks.

PPO and DQN are then trained and evaluated as in Table 4 (200 episodes, 5 seeds, eval block,
fixed traffic 100 humans x 20..1000 bots, 50 eval seeds); the static NP threshold (alpha 0.1,
recomputed on the degraded RL-block human scores) is evaluated alongside.

    python round4/auc_sweep.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import hashlib
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
OUT = G.OUT.parent / "auc_sweep"
TARGETS = (0.75, 0.8, 0.85, 0.9, 0.95)
ALGOS = ("ppo", "dqn")
SEEDS = tuple(range(5))
EPS = 1e-6


def tasks():
    t = [{"auc": a, "algo": "statics", "seed": -1} for a in TARGETS]
    return t + [{"auc": a, "algo": f, "seed": s} for a in TARGETS for f in ALGOS for s in SEEDS]


def name(t):
    return f"auc{t['auc']}_{t['algo']}_s{t['seed']}"


def _e(key):
    h = int(hashlib.sha256(repr(key).encode()).hexdigest()[:12], 16)
    return float(np.random.default_rng(h).standard_normal())


def degrade(p, target):
    """Replace p.cache's scores in place by a scorer of chunk AUC `target` on the RL block."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    tab = p.cache._table
    label = {s.key: int(s.is_bot) for s in p.train_h + p.train_b + p.ev_h + p.ev_b}
    train_keys = {s.key for s in p.train_h + p.train_b}
    keys = [k for k, v in tab.items() if v is not None and k[0] in label]
    z = np.array([np.log(np.clip(tab[k], EPS, 1 - EPS) / (1 - np.clip(tab[k], EPS, 1 - EPS))) for k in keys])
    e = np.array([_e(k) for k in keys])
    y = np.array([label[k[0]] for k in keys])
    tr = np.array([k[0] in train_keys for k in keys])
    auc = lambda s: roc_auc_score(y[tr], (z + s * e)[tr])
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if auc(mid) > target else (lo, mid)
    sigma = (lo + hi) / 2
    zn = z + sigma * e
    platt = LogisticRegression(C=1e6).fit(zn[tr][:, None], y[tr])
    q = platt.predict_proba(zn[:, None])[:, 1]
    for k, v in zip(keys, q):
        tab[k] = float(v)
    ev = ~tr
    return {"sigma": sigma, "auc_rl": roc_auc_score(y[tr], q[tr]),
            "auc_eval": roc_auc_score(y[ev], q[ev]) if ev.any() and len(set(y[ev])) > 1 else np.nan,
            "human_scores_rl": q[tr & (y == 0)]}


class NP:
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, t):
        self.t, self.name = t, "Static NP threshold"

    def select_action(self, user, n_active):
        return 10 if user.bot_score > self.t else 0


def run_task(i):
    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = C.pools(final=True, smoke=False, reason=f"R6 scorer-AUC sweep {name(t)}")
    info = degrade(p, t["auc"])
    tags = {"target_auc": t["auc"], "sigma": info["sigma"], "auc_rl": info["auc_rl"],
            "auc_eval": info["auc_eval"]}
    if t["algo"] == "statics":
        # per-user NP threshold on the degraded RL-block human scores: at most 10% of human
        # chunks above it (a per-chunk version; per-user needs the scorer block, not degraded here)
        th = float(np.quantile(info["human_scores_rl"], 0.9))
        df = G.evaluate(NP(th), "Static NP threshold", p, G.EVAL_SEEDS,
                        {**tags, "family": "statics", "train_seed": -1, "t": th}, fractions=())
    else:
        spec = D.choice(False)[t["algo"]].split(":", 1)[1]
        variants, _ = R.train_candidate(t["algo"], spec, t["seed"], p.train_h, p.train_b, p.cache,
                                        R.EPISODES, solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
        (_, pol), = variants.items()
        pol.eval_mode()
        df = G.evaluate(pol, t["algo"].upper(), p, G.EVAL_SEEDS,
                        {**tags, "family": t["algo"], "train_seed": t["seed"]}, fractions=())
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(t)}: sigma {info['sigma']:.2f}, AUC rl {info['auc_rl']:.3f} eval {info['auc_eval']:.3f}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T_ = tasks()
    done = lambda i: (OUT / f"{name(T_[i])}.csv").exists()
    todo = [i for i in range(len(T_)) if not done(i)]
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
                print(f"task {i} {name(T_[i])} exit {proc.returncode}{'' if done(i) else ' NO CSV'}", flush=True)
        time.sleep(5)


def collect():
    r = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv"))], ignore_index=True)
    r.to_csv(OUT.parent / "auc_sweep_runs.csv", index=False)
    s = r.groupby(["family", "target_auc", "train_seed"]).agg(DI=("DI", "mean"), auc_eval=("auc_eval", "first"))
    g = s.groupby(["family", "target_auc"]).agg(DI=("DI", "mean"), sd=("DI", "std"), n=("DI", "size"),
                                                auc_eval=("auc_eval", "mean")).round(3)
    g.to_csv(OUT.parent / "auc_sweep.csv")
    print(g.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.list:
        for i, t in enumerate(tasks()):
            print(i, t)
    elif a.collect:
        collect()
    elif a.run_all:
        run_all(a.run_all)
    elif a.task is not None:
        run_task(a.task)


if __name__ == "__main__":
    main()
