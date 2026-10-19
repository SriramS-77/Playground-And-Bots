"""Is LinUCB's poor showing the best it can do? Tune its exploration and learning rate (round 6).

Validation only: trained on rotation 0's rl_fit, evaluated on its rl_val (the eval block stays
shut), grounded world, cross-fit seed-1 scorer, 200 episodes, the selected embedding network
(h64_e32). Grid: alpha (UCB exploration weight) x lr (embedding network), 3 seeds. Fixed traffic:
100 humans x bots 20..1000, 20 eval seeds.

    python round4/linucb_tune.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
OUT = G.OUT.parent / "linucb_tune"
ALPHAS = (0.1, 0.3, 1.0, 3.0)
LRS = (1e-3, 3e-4)
SEEDS = (0, 1, 2)


def tasks():
    return [{"alpha": a, "lr": lr, "seed": s} for a in ALPHAS for lr in LRS for s in SEEDS]


def name(t):
    return f"a{t['alpha']}_lr{t['lr']}_s{t['seed']}"


def run_task(i):
    from exp_policy_arch_search import BANDIT_ARCHS
    from expkit.bandits_x import train_bandit

    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = C.pools(final=False, smoke=True)               # rotation 0: rl_fit -> rl_val
    arch = dataclasses.replace(next(a for a in BANDIT_ARCHS if a.name == "h64_e32"),
                               alpha=t["alpha"], lr=t["lr"])
    pol, _ = train_bandit("linucb", p.train_h, p.train_b, p.cache, arch=arch, episodes=R.EPISODES,
                          seed=t["seed"], human_counts=R.TRAIN_HUMAN_RANGE,
                          bot_choices=D.BOT_VOLUMES, verbose=0, solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
    df = C.evaluate(pol, "LinUCB", p.ev_h, p.ev_b, p.cache, C.EVAL_SEEDS, solve=C.GROUNDED,
                    cfg=C.GROUNDED_CFG, alpha_ucb=t["alpha"], lr=t["lr"], train_seed=t["seed"])
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min", flush=True)


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
    runs = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv"))], ignore_index=True)
    per_seed = runs.groupby(["alpha_ucb", "lr", "train_seed"]).DI.mean()
    t = per_seed.groupby(["alpha_ucb", "lr"]).agg(["mean", "std", "min", "count"]).round(2)
    t.to_csv(OUT.parent / "linucb_tune.csv")
    print(t.sort_values("mean", ascending=False).to_string())


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
        print(len(tasks()), "tasks")
    elif a.collect:
        collect()
    elif a.run_all:
        run_all(a.run_all)
    elif a.task is not None:
        run_task(a.task)


if __name__ == "__main__":
    main()
