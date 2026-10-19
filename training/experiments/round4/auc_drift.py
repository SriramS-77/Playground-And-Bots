"""Calibration drift that ALSO destroys ranking (AUC falls toward 0.5) -- exploratory, round 6.

Round 4's shifts (advanced web bots, Balabit humans) move calibration while session AUC stays
>= 0.99. Here the shifted class is replaced by recordings of the OTHER class from the same block:

  bots    a fraction rho of bots are "replay bots": they replay real HUMAN recordings, so their
          scores come from the human distribution (a mouse-replay attack)
  humans  a fraction rho of humans move like BOTS (bot recordings), e.g. automation-assisted users

At rho = 1 the score carries no information about the shifted class (AUC -> 0.5). Training pools
use the RL block, evaluation pools the eval block, as in round 4. Everything else is round 4's:
grounded world, scorer seed 1, 100 humans x bots 20..1000, 20 eval seeds.

Tasks:  metrics  (x-axis: calibration error, session and step AUC per point)
        offline  saved headline agents (DQN, PPO, Thompson; 5 seeds), frozen, + raw-score and
                 posterior statics -- every point
        online   DQN and PPO, rewards oracle / score-only / posterior, 300 episodes with the
                 shift from episode 100, rho in {0.5, 1.0}, seeds 0-2 (no controls: the
                 frozen offline policies are the comparator)

    python round4/auc_drift.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
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
OUT = G.OUT.parent / "auc_drift"
LEVELS = (0.25, 0.5, 0.75, 1.0)
ONLINE_LEVELS = (0.5, 1.0)
ONLINE_SEEDS = (0, 1, 2)


def shifted_replay(p, direction, rho, held_out: bool):
    h, b = (p.ev_h, p.ev_b) if held_out else (p.train_h, p.train_b)
    if direction == "bots":
        b, real = D.mix(b, h, rho)          # bots replaying this block's human recordings
    elif direction == "humans":
        h, real = D.mix(h, b, rho)          # humans moving like this block's bots
    else:
        real = 0.0
    return h, b, real


D.shifted = shifted_replay                  # every round-4 helper now uses this shift


def _save(df, exp, name, smoke):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)


C.save = _save


def tasks():
    t = [{"kind": "metrics"}]
    t += [{"kind": "offline", "family": f, "seed": s} for f in ("dqn", "ppo", "thompson") for s in range(5)]
    t += [{"kind": "statics"}]
    t += [{"kind": "online", "algo": a, "arm": arm, "direction": d, "rho": r, "seed": s}
          for a in ("dqn", "ppo") for arm in ("oracle", "score-only", "posterior")
          for d in D.DIRECTIONS for r in ONLINE_LEVELS for s in ONLINE_SEEDS]
    return t


def name(t):
    return "__".join(str(v) for v in t.values()).replace(" ", "_")


def run_task(i):
    G.no_power_throttling()
    t = tasks()[i]
    if (OUT / f"{name(t)}.csv").exists():
        return
    t0 = time.time()
    pts = D.points(LEVELS)
    if t["kind"] == "metrics":
        p = D.pools(False, "R6 auc-drift metrics")
        _save(D.drift_metrics(p, LEVELS), None, name(t), False)
    elif t["kind"] == "offline":
        from exp_round3_fold2 import load_policy
        p = D.pools(False, f"R6 auc-drift offline {name(t)}")
        pol = load_policy(G.OUT / "agents" / f"{t['family']}_s{t['seed']}.pt")
        df = D._evaluate_points(pol, G.NAMES[t["family"]], p, pts, False, C.GROUNDED, C.GROUNDED_CFG,
                                {"panel": "offline", "train_seed": t["seed"]})
        _save(df, None, name(t), False)
    elif t["kind"] == "statics":
        import e7_proxy_reward as E
        import posterior_statics as PSt
        p = D.pools(False, "R6 auc-drift statics")
        calib = E.fit_calibration(p)
        pols = C.statics() + [PSt.PosteriorStatic(calib, calib.pi_cal, False, "Posterior single (fixed share)"),
                              PSt.PosteriorStatic(calib, calib.pi_cal, True, "Posterior multi (fixed share)")]
        frames = [D._evaluate_points(pol, pol.name, p, pts, False, C.GROUNDED, C.GROUNDED_CFG,
                                     {"panel": "offline", "train_seed": -1}) for pol in pols]
        _save(pd.concat(frames, ignore_index=True), None, name(t), False)
    else:
        D.run_trained({**t, "kind": "online"}, False)
        produced = OUT / f"{D._name({**t, 'kind': 'online'})}.csv"
        if produced.exists():
            produced.rename(OUT / f"{name(t)}.csv")
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs, first=0):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T = tasks()
    done = lambda i: (OUT / f"{name(T[i])}.csv").exists()
    todo = [i for i in range(first, len(T)) if not done(i)]
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
                print(f"task {i} {name(T[i])} exit {proc.returncode}{'' if done(i) else ' NO CSV'}", flush=True)
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.list:
        for i, t in enumerate(tasks()):
            print(i, t)
        print(len(tasks()), "tasks")
    elif a.run_all:
        run_all(a.run_all)
    elif a.task is not None:
        run_task(a.task)


if __name__ == "__main__":
    main()
