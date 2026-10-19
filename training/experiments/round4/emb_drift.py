"""Embedding state under covert-bot drift: online training, DQN and PPO (round 6).

All three state variants use ONE scorer, trained on the scorer block only (round4/emb_prepare.py),
for q, for the embedding and for the score-only reward, so they differ in the state alone:
  none   the 5-scalar state (q, mean q, passed, last level, active users) -- the baseline
  both   + PCA-8 embedding of the current chunk and its session mean (21 inputs)
  emb    the same 21 stored inputs with q and mean q masked at the network input
Protocol: round 4's online training (r4_fold2.run_trained: 300 episodes, covert bots from
episode 100), evaluated on the eval block. Rewards: score-only (deployable) at control (no shift),
rho = 0.5 and 1; label-based (oracle) at rho = 1 as a diagnostic. 5 seeds.

    python round4/emb_drift.py --list | --task N | --run-all J | --collect
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
OUT_ROOT = None
MODES = ("none", "both", "emb")
ALGOS = ("dqn", "ppo")
SEEDS = tuple(range(5))
MIX_SEEDS = (0, 1, 2)


def tasks():
    t = []
    for m in MODES:
        for a in ALGOS:
            for s in SEEDS:
                t.append({"mode": m, "algo": a, "arm": "score-only", "kind": "control", "seed": s})
                for rho in (0.5, 1.0):
                    t.append({"mode": m, "algo": a, "arm": "score-only", "kind": "online",
                              "direction": "bots", "rho": rho, "seed": s})
                t.append({"mode": m, "algo": a, "arm": "oracle", "kind": "online",
                          "direction": "bots", "rho": 1.0, "seed": s})
    # mixed old + covert bots after the shift: oracle at rho = 0.5 (seeds 0-2, appended)
    t += [{"mode": m, "algo": a, "arm": "oracle", "kind": "online", "direction": "bots", "rho": 0.5, "seed": s}
          for m in MODES for a in ALGOS for s in MIX_SEEDS]
    return t


def name(t):
    return "__".join(str(v) for v in t.values()).replace(" ", "")


def run_task(i):
    t = tasks()[i]
    sys.path.insert(0, str(HERE))
    import grounded_headline as G                  # sets up paths; trainer imported lazily below
    D = G.D
    out_dir = D.R.OUT / "emb_drift"
    out = out_dir / f"{name(t)}.csv"
    if out.exists():
        return
    G.no_power_throttling()
    import expkit.trainer as T
    assert (T.EMB_MODE or "none") == t["mode"], (T.EMB_MODE, t["mode"])
    q_tab = pickle.load(open(D.R.OUT / "emb" / "q_table.pkl", "rb"))
    real_pools, real_save = D.pools, D.C.save

    def pools(smoke, reason):
        p = real_pools(smoke, reason)
        p.cache._table.clear()
        p.cache._table.update(q_tab)                # the scorer-block scorer's q, everywhere
        return p

    def save(df, *_a, **_k):
        df["state_mode"] = t["mode"]
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    D.pools, D.C.save = pools, save
    task = {k: v for k, v in t.items() if k != "mode"}
    t0 = time.time()
    try:
        D.run_trained(task, False)
    finally:
        D.pools, D.C.save = real_pools, real_save
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs):
    T_ = tasks()
    sys.path.insert(0, str(HERE))
    import grounded_headline as G
    out_dir = G.D.R.OUT / "emb_drift"
    done = lambda i: (out_dir / f"{name(T_[i])}.csv").exists()
    todo = [i for i in range(len(T_)) if not done(i)]
    todo.sort(key=lambda i: T_[i]["algo"] != "dqn")            # long DQN runs first
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "RLC_EMB": "" if T_[i]["mode"] == "none" else T_[i]["mode"],
                   "RLC_EMB_DIR": str(G.D.R.OUT / "emb")}
            log = open(out_dir / "logs" / f"{i}.log", "w")
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
    d = G.D.R.OUT / "emb_drift"
    r = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*.csv"))], ignore_index=True)
    r.to_csv(d.parent / "emb_drift_runs.csv", index=False)
    s = r.groupby(["state_mode", "algo", "arm", "panel", "train_rho", "eval_direction", "eval_rho",
                   "train_seed"]).agg(DI=("DI", "mean"), fr=("friction_s", "mean"))
    g = s.groupby(level=list(range(7))).agg(DI=("DI", "mean"), sd=("DI", "std"), fr=("fr", "mean"),
                                            n=("DI", "size")).round(1)
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
