"""RL-state ablation under covert-bot drift (round 6, 2 seeds).

Each variant zeroes some state inputs at the network input (expkit.trainer RLC_MASK; stored states
and rewards are unchanged), on the 5-scalar state (`none`) or the q + embedding state (`both`):
  none:  full, -q, -qmean, -solved, -threat, -active
  both:  full, -emb_cur, -emb_avg
For each variant, DQN and PPO are trained (a) OFFLINE (round 4's run_offline: rl block, oracle
reward, then frozen and evaluated at rho = 0.1..1) and (b) ONLINE at rho = 1 (300 episodes, covert
bots from episode 100) with the score-only and the oracle reward. All runs use emb_drift's
scorer-block scorer. The online `full` variants are emb_drift's seeds 0-1 and are not rerun.

    python round4/state_ablation.py --list | --task N | --run-all J | --collect
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
VARIANTS = [("none", ""), ("none", "q"), ("none", "qmean"), ("none", "solved"), ("none", "threat"),
            ("none", "active"), ("both", ""), ("both", "emb_cur"), ("both", "emb_avg")]
ALGOS = ("dqn", "ppo")
SEEDS = (0, 1)


def tasks():
    t = []
    for mode, mask in VARIANTS:
        for a in ALGOS:
            for s in SEEDS:
                t.append({"mode": mode, "mask": mask, "kind": "offline", "family": a, "seed": s})
                if mask:                          # online full = emb_drift seeds 0-1
                    for arm in ("score-only", "oracle"):
                        t.append({"mode": mode, "mask": mask, "kind": "online", "algo": a, "arm": arm,
                                  "direction": "bots", "rho": 1.0, "seed": s})
    return t


def name(t):
    return "__".join(str(v) or "full" for v in t.values()).replace(" ", "")


def run_task(i):
    t = tasks()[i]
    sys.path.insert(0, str(HERE))
    import grounded_headline as G
    D = G.D
    out_dir = D.R.OUT / "state_ablation"
    out = out_dir / f"{name(t)}.csv"
    if out.exists():
        return
    G.no_power_throttling()
    import expkit.trainer as T
    assert (T.EMB_MODE or "none") == t["mode"] and ",".join(T.STATE_MASK) == t["mask"], t
    q_tab = pickle.load(open(D.R.OUT / "emb" / "q_table.pkl", "rb"))
    real_pools, real_save = D.pools, D.C.save

    def pools(smoke, reason):
        p = real_pools(smoke, reason)
        p.cache._table.clear()
        p.cache._table.update(q_tab)
        return p

    def save(df, *_a, **_k):
        df["state_mode"], df["mask"] = t["mode"], t["mask"] or "full"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    D.pools, D.C.save = pools, save
    task = {k: v for k, v in t.items() if k not in ("mode", "mask")}
    t0 = time.time()
    try:
        D.run_drift_task(task, False)
    finally:
        D.pools, D.C.save = real_pools, real_save
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs):
    T_ = tasks()
    sys.path.insert(0, str(HERE))
    import grounded_headline as G
    out_dir = G.D.R.OUT / "state_ablation"
    done = lambda i: (out_dir / f"{name(T_[i])}.csv").exists()
    todo = sorted((i for i in range(len(T_)) if not done(i)),
                  key=lambda i: (T_[i]["kind"] != "online", T_[i].get("algo", T_[i].get("family")) != "dqn"))
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "RLC_EMB": "" if T_[i]["mode"] == "none" else T_[i]["mode"],
                   "RLC_EMB_DIR": str(G.D.R.OUT / "emb"), "RLC_MASK": T_[i]["mask"]}
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
    d = G.D.R.OUT / "state_ablation"
    r = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*.csv"))], ignore_index=True)
    e = pd.read_csv(d.parent / "emb_drift_runs.csv")
    e = e[(e.panel == "online") & (e.train_rho == 1.0) & e.train_seed.isin(SEEDS) & e.state_mode.isin(["none", "both"])]
    r = pd.concat([r, e.assign(mask="full")], ignore_index=True)
    r.to_csv(d.parent / "state_ablation_runs.csv", index=False)
    r["algo"] = r.algo.fillna(r.family).str.upper()
    r["arm"] = r.arm.fillna("offline (oracle)")
    b = r[r.eval_direction == "bots"]
    c = ["DI", "human_survival", "bot_survival", "friction_s"]
    on = b[(b.panel == "online") & (b.eval_rho == 1.0)]
    g = on.groupby(["algo", "arm", "state_mode", "mask", "train_seed"])[c].mean().groupby(level=[0, 1, 2, 3]).mean()
    print("ONLINE, trained and evaluated at rho = 1\n", g.round(3).to_string())
    off = b[b.panel == "offline"]
    g = off.groupby(["algo", "state_mode", "mask", "eval_rho", "train_seed"]).DI.mean().groupby(level=[0, 1, 2, 3]).mean()
    print("\nOFFLINE (frozen), DI by eval rho\n", g.unstack("eval_rho").round(1).to_string())


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
