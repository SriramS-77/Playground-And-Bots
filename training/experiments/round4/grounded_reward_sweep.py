"""Reward-constant sensitivity in the grounded world: DQN retrained under each setting.

Same training as `grounded_headline.py` (rl block, GROUNDED world, cross-fit seed-1 scorer,
200 episodes, DQN 128-64), only the reward constants change; the world itself (solves, human
failures, leaving with tau = 45 s) is the same in every setting, so survival-based metrics are
comparable. Evaluated on the EVAL BLOCK, 100 humans x bots {20..1000}, 20 eval seeds. The
"published" setting is the headline DQN itself (`grounded_headline.py`), not rerun here.

    python round4/grounded_reward_sweep.py --list | --task N | --run-all J | --collect
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
OUT = G.OUT.parent / "reward_sweep"
B = C.GROUNDED_CFG
SETTINGS = {
    "cheap leak (R_leak -50)": dataclasses.replace(B, leakage_penalty=-50.0),
    "costly leak (R_leak -300)": dataclasses.replace(B, leakage_penalty=-300.0),
    "mild false positive (R_FP -50)": dataclasses.replace(B, false_positive_penalty=-50.0),
    "harsh false positive (R_FP -200)": dataclasses.replace(B, false_positive_penalty=-200.0),
    "gentle friction (c_f 1/s)": dataclasses.replace(B, friction_per_second=1.0),
    "harsh friction (c_f 4/s)": dataclasses.replace(B, friction_per_second=4.0),
    "low overkill (c_o 1)": dataclasses.replace(B, overkill=1.0),
    "low underestimation (c_u 2.5)": dataclasses.replace(B, underestimation=2.5),
    "high underestimation (c_u 10)": dataclasses.replace(B, underestimation=10.0),
}


def tasks():
    return [{"setting": s, "seed": i} for s in SETTINGS for i in range(C.SEEDS)]


def name(t):
    return f"{list(SETTINGS).index(t['setting'])}_s{t['seed']}"


def run_task(i):
    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = C.pools(final=True, reason=f"R6 grounded reward sweep {t}")
    spec = D.choice(False)["dqn"].split(":", 1)[1]
    variants, _ = R.train_candidate("dqn", spec, t["seed"], p.train_h, p.train_b, p.cache,
                                    R.EPISODES, solve=C.GROUNDED, cfg=SETTINGS[t["setting"]])
    (_, pol), = variants.items()
    pol.eval_mode()
    df = G.evaluate(pol, "DQN", p, 20, {"setting": t["setting"], "train_seed": t["seed"]},
                    fractions=())
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T = tasks()
    done = lambda i: (OUT / f"{name(T[i])}.csv").exists()
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
    import numpy as np
    from scipy import stats
    runs = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv"))], ignore_index=True)
    head = pd.read_csv(G.OUT.parent / "headline_runs.csv")
    head = head[(head.grid == "fixed") & (head.eval_seed < 20)]
    base = head[head.policy == "DQN"].assign(setting="as trained (Table)")
    statics = head[head.policy.str.startswith("Static")]
    runs = pd.concat([base, runs], ignore_index=True)
    rows = []
    for s, f in runs.groupby("setting", sort=False):
        ps = f.groupby("train_seed")[["DI", "human_survival", "bot_survival", "friction_s"]].mean()
        n = len(ps)
        h = stats.t.ppf(0.975, n - 1) * ps.DI.std(ddof=1) / np.sqrt(n)
        rows.append({"setting": s, "n": n, **ps.mean().round(4).to_dict(), "DI_ci95": round(h, 2),
                     "DI_seed_min": ps.DI.min()})
    t = pd.DataFrame(rows)
    best_static = statics.groupby("policy").DI.mean().max()
    t["beats_best_static"] = t.DI > best_static
    t.to_csv(OUT.parent / "reward_sweep_summary.csv", index=False)
    print(t.round(2).to_string(index=False), "\nbest static", round(best_static, 2))


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
