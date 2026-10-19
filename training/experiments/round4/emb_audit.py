"""Audit-recalibrated rewards with the embedding state (round 6, companion to emb_drift.py).

Arms audit_recal (2% of users labelled) and audit_ch (level-9 challenge audit) from np_audit.py,
trained online at rho = 1 (covert bots from episode 100), DQN and PPO, 5 seeds, for each state
variant none / both / emb. All variants use emb_drift's scorer-block scorer for q (so `none` here
differs from the np_audit runs in the paper, which use the cross-fitted cache).

    python round4/emb_audit.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("none", "both", "emb")
ALGOS = ("dqn", "ppo")
ARMS = ("audit_recal", "audit_ch")
SEEDS = tuple(range(5))
MIX_SEEDS = (0, 1, 2)
os.environ.update({"AUDIT_EPS": "0.02", "AUDIT_LEVEL": "9", "AUDIT_LEVELS": "0.5,1.0", "N_SEEDS": "5"})


EMB_ARMS = ("audit_recal_emb", "audit_ch_emb")    # recalibration on (q, embedding); embedding states only


def tasks():
    t = [{"mode": m, "algo": a, "arm": r, "seed": s}
         for m in MODES for a in ALGOS for r in ARMS for s in SEEDS]
    t += [{"mode": m, "algo": a, "arm": r, "seed": s}
          for m in ("both", "emb") for a in ALGOS for r in EMB_ARMS for s in SEEDS]
    # mixed old + covert bots after the shift: rho = 0.5, seeds 0-2 (appended)
    t += [{"mode": m, "algo": a, "arm": r, "seed": s, "rho": 0.5}
          for m in MODES for a in ALGOS for r in ARMS for s in MIX_SEEDS]
    t += [{"mode": m, "algo": a, "arm": r, "seed": s, "rho": 0.5}
          for m in ("both", "emb") for a in ALGOS for r in EMB_ARMS for s in MIX_SEEDS]
    return t


def name(t):
    return "__".join(str(v) for v in t.values())


def run_task(i):
    t = tasks()[i]
    sys.path.insert(0, str(HERE))
    import np_audit as N
    D = N.D
    N.ARMS = {a: ARMS + EMB_ARMS for a in ALGOS}
    import expkit.trainer as T
    assert (T.EMB_MODE or "none") == t["mode"], (T.EMB_MODE, t["mode"])
    N.OUT = D.R.OUT / "emb_audit" / t["mode"]
    q_tab = pickle.load(open(D.R.OUT / "emb" / "q_table.pkl", "rb"))
    real_pools = D.pools

    def pools(smoke, reason):
        p = real_pools(smoke, reason)
        p.cache._table.clear()
        p.cache._table.update(q_tab)
        return p
    D.pools = pools
    j = [k for k, x in enumerate(N.tasks()) if x.get("kind") == "online" and x["algo"] == t["algo"]
         and x["arm"] == t["arm"] and x["direction"] == "bots" and x["rho"] == t.get("rho", 1.0) and x["seed"] == t["seed"]]
    assert len(j) == 1, j
    N.run_task(j[0])


def out_path(t, root):
    return root / t["mode"] / f"online__{t['algo']}__{t['arm']}__bots__{t.get('rho', 1.0)}__{t['seed']}.csv"


def run_all(jobs):
    T_ = tasks()
    sys.path.insert(0, str(HERE))
    import grounded_headline as G
    root = G.D.R.OUT / "emb_audit"
    done = lambda i: out_path(T_[i], root).exists()
    todo = sorted((i for i in range(len(T_)) if not done(i)), key=lambda i: T_[i]["algo"] != "dqn")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "RLC_EMB": "" if T_[i]["mode"] == "none" else T_[i]["mode"],
                   "RLC_EMB_DIR": str(G.D.R.OUT / "emb")}
            log = open(root / "logs" / f"{i}.log", "w")
            running[i] = (subprocess.Popen([sys.executable, __file__, "--task", str(i)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env), log)
        for i, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[i]
                print(f"task {i} {name(T_[i])} exit {proc.returncode}{'' if done(i) else ' NO CSV'}", flush=True)
        time.sleep(5)


def collect():
    import pandas as pd
    sys.path.insert(0, str(HERE))
    import grounded_headline as G
    root = G.D.R.OUT / "emb_audit"
    r = pd.concat([pd.read_csv(f).assign(state_mode=m) for m in MODES
                   for f in sorted((root / m).glob("*.csv"))], ignore_index=True)
    r.to_csv(root.parent / "emb_audit_runs.csv", index=False)
    r = r[(r.eval_direction == "bots") & (r.eval_rho == 1.0)]
    c = ["DI", "human_survival", "bot_survival", "friction_s"]
    s = r.groupby(["algo", "arm", "state_mode", "train_seed"])[c].mean()
    g = s.groupby(level=[0, 1, 2]).agg(DI=("DI", "mean"), sd=("DI", "std"), hs=("human_survival", "mean"),
                                       bs=("bot_survival", "mean"), fr=("friction_s", "mean"), n=("DI", "size"))
    print(g.round(3).to_string())


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
