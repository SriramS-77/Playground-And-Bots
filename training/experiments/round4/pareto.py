"""Pareto sweep: humans lost vs bots stopped, per method, as one trade-off knob varies (round 6).

Knob for RL: the false-positive penalty R_FP (cost of losing a human) in FP_GRID, with the rest of
GROUNDED_CFG unchanged; it sets how cautious the policy is.
  frozen   DQN / PPO trained offline (rl block, label-based reward, 200 episodes), seeds 0-2
  online   DQN / PPO trained online through the covert-bot shift (round 4's calibration shift,
           rho = 1 from episode 100 of 300) with the label-free score-only reward (whose outcome
           model uses the same R_FP), seeds 0-2
Static: NP threshold (per-user, from scorer-block humans) for alpha in NP_ALPHAS, and the raw single
threshold t in 0.1..0.9 (level 10 above t).
Evaluation: eval block, 100 humans x bots 20..1000, 20 eval seeds, at no drift and covert bots
rho in {0.5, 1}. Opens the eval block once per task (logged).

    python round4/pareto.py --list | --task N | --run-all J | --collect
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

import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
OUT = G.OUT.parent / "pareto"
FP_GRID = (-25.0, -50.0, -100.0, -200.0, -400.0)
SEEDS = (0, 1, 2)
NP_ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
RAW_T = tuple(round(0.1 * i, 1) for i in range(1, 10))
PTS = [("none", 0.0), ("bots", 0.5), ("bots", 1.0)]


def tasks():
    t = [{"kind": "static"}]
    for mode in ("frozen", "online"):
        t += [{"kind": mode, "algo": a, "fp": fp, "seed": s}
              for a in ("dqn", "ppo") for fp in FP_GRID for s in SEEDS]
    return t


def name(t):
    return "__".join(str(v) for v in t.values())


class Single:
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, t, label):
        self.t, self.name = t, label

    def select_action(self, user, n_active):
        return 10 if user.bot_score > self.t else 0


def run_task(i):
    import e7_proxy_reward as E
    import expkit.trainer as T
    import np_audit as NP
    from expkit.posterior_reward import OutcomeModel

    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = D.pools(False, f"R6 pareto {name(t)}")
    if t["kind"] == "static":
        frames = []
        for a in NP_ALPHAS:
            th = NP.np_threshold(p, a)
            frames.append(D._evaluate_points(Single(th, "NP threshold"), "NP threshold", p, PTS, False,
                                             C.GROUNDED, C.GROUNDED_CFG, {"knob": a, "t": th}))
        for th in RAW_T:
            frames.append(D._evaluate_points(Single(th, "Raw threshold"), "Raw threshold", p, PTS, False,
                                             C.GROUNDED, C.GROUNDED_CFG, {"knob": th, "t": th}))
        df = pd.concat(frames, ignore_index=True)
    else:
        algo, fp, seed = t["algo"], t["fp"], t["seed"]
        cfg = dataclasses.replace(C.GROUNDED_CFG, false_positive_penalty=fp)
        if t["kind"] == "frozen":
            ch = D.choice(False)
            spec = ch[algo].split(":", 1)[1]
            variants, _ = R.train_candidate(algo, spec, seed, p.train_h, p.train_b, p.cache, R.EPISODES,
                                            solve=C.GROUNDED, cfg=cfg)
            (_, agent), = variants.items()
        else:
            after_h, after_b, _ = D.shifted(p, "bots", 1.0, held_out=False)
            om = OutcomeModel(C.GROUNDED, cfg)
            spec = dict(E.CONFIGS["score-only"])
            spec.pop("hook")
            st = {"ep": 0}
            real_run_x, real_relabel = T.run_x, T.relabel

            def run_x_hooked(policy, humans, bots, n_bots, **kw):
                if kw.get("collect") and humans is p.train_h:
                    if st["ep"] >= D.INJECT_EP:
                        humans, bots = after_h, after_b
                    st["ep"] += 1
                return real_run_x(policy, humans, bots, n_bots, **kw)
            T.run_x, T.relabel = run_x_hooked, NP.make_relabel("score", om, 1.0, seed)
            try:
                agent, _ = D.train_online(algo, p.train_h, p.train_b, p.cache, seed, D.TOTAL, False,
                                          solve=C.GROUNDED, cfg=cfg, **spec)
            finally:
                T.run_x, T.relabel = real_run_x, real_relabel
            assert st["ep"] == D.TOTAL, st
        if hasattr(agent, "eval_mode"):
            agent.eval_mode()
        df = D._evaluate_points(agent, f"{algo.upper()} {t['kind']}", p, PTS, False, C.GROUNDED,
                                C.GROUNDED_CFG, {"knob": fp, "train_seed": seed})
    df["kind"] = t["kind"]
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
    runs.to_csv(OUT.parent / "pareto_runs.csv", index=False)
    g = runs.groupby(["kind", "policy", "knob", "eval_direction", "eval_rho"])[
        ["DI", "human_survival", "bot_survival", "friction_s"]].mean().reset_index()
    g.to_csv(OUT.parent / "pareto_points.csv", index=False)
    print(g.round(3).to_string(index=False))


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
