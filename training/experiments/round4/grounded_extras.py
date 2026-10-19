"""Sensitivity checks on the grounded-world headline agents (evaluation only, no training).

Loads the agents saved by `grounded_headline.py` (results/round4/fold2/drift/headline/agents) and
evaluates them on the EVAL BLOCK, 100 humans x bots {20, 100, 200, 500, 1000}, 20 eval seeds:

  alpha    bot-solve slope alpha in {0.5, 1, 1.5, 3, 6, 12} (training used 1.5)
  tau      leave-after-pass constant tau in {22, 45, 80} s, and no leaving (training used 45)
  freeze   DQN and PPO only: the active-user input frozen at {25, 100, 400, 1400} vs unfrozen,
           on humans {25, 100, 400} x bots {20, 200, 1000}, 10 eval seeds -- does security depend
           on reading the attack size off the population count? (round 3's E4 wrapper)

    python round4/grounded_extras.py --smoke | --task N | --run-all J | --collect
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
import grounded_headline as G                     # noqa: E402  (sets up r4_fold2, stage drift)

import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
sys.path.insert(0, str(HERE.parent / "round3"))
from e4_population import PopulationInput         # noqa: E402
from expkit.simx import evaluate_x, run_x         # noqa: E402
from expkit.stochastic import SolveModel          # noqa: E402

OUT = G.OUT.parent / "extras"
AGENTS = G.OUT / "agents"
ALPHAS = (0.5, 1.0, 1.5, 3.0, 6.0, 12.0)
TAUS = (22.0, 45.0, 80.0, None)
FREEZE = (25, 100, 400, 1400)
FREEZE_H, FREEZE_B = (25, 100, 400), (20, 200, 1000)


def tasks():
    t = [{"family": f, "seed": s} for f in ("dqn", "ppo", "thompson", "linucb", "ablation")
         for s in range(C.SEEDS)]
    return t + [{"family": "statics", "seed": -1}]


def _rows(pol, label, p, cells, eval_seeds, solve, cfg, tags):
    rows = []
    for nh, nb in cells:
        for s in range(eval_seeds):
            res, _ = run_x(pol, p.ev_h, p.ev_b, nb, cache=p.cache, n_humans=nh, seed=s,
                           solve=solve, cfg=cfg)
            m = evaluate_x(res)
            rows.append({"policy": label, "n_humans": nh, "bots": nb, "eval_seed": s, "DI": m.DI,
                         "human_survival": res.surviving_humans / nh,
                         "bot_survival": res.surviving_bots / max(nb, 1),
                         "friction_s": res.mean_human_friction_seconds,
                         "challenge_share": res.challenges_issued / max(sum(res.action_hist.values()), 1),
                         **tags})
    return rows


def run_task(i, smoke=False):
    G.no_power_throttling()
    from exp_round3_fold2 import load_policy

    task = tasks()[i]
    name = G.name(task)
    out = (OUT.parent.parent / "drift_smoke" / "extras" if smoke else OUT) / f"{name}.csv"
    if out.exists():
        print("exists:", out)
        return
    t0 = time.time()
    p = C.pools(final=not smoke, smoke=smoke, reason=f"R6 grounded extras {name}")
    eval_seeds = 2 if smoke else 20
    vols = (20, 100) if smoke else D.SCORED_VOLUMES
    fixed = [(100, v) for v in vols]
    if task["family"] == "statics":
        pols = [(pol, pol.name) for pol in C.statics()]
    else:
        ag = (G.SMOKE_OUT if smoke else G.OUT) / "agents" / f"{name}.pt"
        pols = [(load_policy(ag), G.NAMES[task["family"]])]
    rows = []
    tag = {"family": task["family"], "train_seed": task["seed"]}
    for pol, label in pols:
        for a in (ALPHAS[:2] if smoke else ALPHAS):
            rows += _rows(pol, label, p, fixed, eval_seeds, SolveModel(alpha=a), C.GROUNDED_CFG,
                          {**tag, "check": "alpha", "value": a})
        for tau in (TAUS[:2] if smoke else TAUS):
            cfg = (dataclasses.replace(C.GROUNDED_CFG, abandonment="none") if tau is None
                   else dataclasses.replace(C.GROUNDED_CFG, abandonment_tau=tau))
            rows += _rows(pol, label, p, fixed, eval_seeds, C.GROUNDED, cfg,
                          {**tag, "check": "tau", "value": "none" if tau is None else tau})
        if task["family"] in ("dqn", "ppo"):
            cells = [(h, b) for h in FREEZE_H for b in FREEZE_B]
            if smoke:
                cells = cells[:2]
            for v in ("unfrozen",) + tuple(map(str, FREEZE)):
                wp = pol if v == "unfrozen" else PopulationInput(pol, v, seed=task["seed"])
                rows += _rows(wp, label, p, cells, 2 if smoke else 10, C.GROUNDED, C.GROUNDED_CFG,
                              {**tag, "check": "freeze", "value": v})
    df = pd.DataFrame(rows)
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T = tasks()
    done = lambda i: (OUT / f"{G.name(T[i])}.csv").exists()
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
    runs = pd.concat([pd.read_csv(f, dtype={"value": str}) for f in sorted(OUT.glob("*.csv"))],
                     ignore_index=True)
    runs.to_csv(OUT.parent / "extras_runs.csv", index=False)
    for check in ("alpha", "tau"):
        f = runs[runs.check == check]
        t = f.groupby(["policy", "train_seed", "value"]).DI.mean().groupby(["policy", "value"]).mean().unstack()
        t.to_csv(OUT.parent / f"extras_{check}.csv")
        print(f"\n{check}: DI\n", t.round(1).to_string())
    f = runs[runs.check == "freeze"]
    cols = ["DI", "human_survival", "bot_survival", "challenge_share"]
    t = f.groupby(["policy", "value"])[cols].mean()
    t.to_csv(OUT.parent / "extras_freeze.csv")
    print("\nfreeze\n", t.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.collect:
        return collect()
    if a.smoke:
        log = R.OUT / "eval_access.log"
        before = log.read_text() if log.exists() else ""
        for i in (0, 5, 25):
            run_task(i, smoke=True)
        assert (log.read_text() if log.exists() else "") == before, "smoke opened the eval block"
        print("smoke: OK (eval block untouched)")
        return
    if a.run_all:
        return run_all(a.run_all)
    if a.task is not None:
        return run_task(a.task)
    ap.error("--smoke | --task N | --run-all J | --collect")


if __name__ == "__main__":
    main()
