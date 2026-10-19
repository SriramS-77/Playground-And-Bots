"""Gated online training: frozen until an unsupervised shift detector fires, then online (round 6).

PPO, 400 episodes per run:
  0-199     offline training in simulation with the label-based reward (clean traffic)
  200-399   deployment, no labels. Detector burn-in on 200-249; covert bots (round 4's calibration
            shift, training-side pool) replace a fraction rho of bots from episode 250.
Arms:
  frozen        never updated in deployment
  online        score-only updates on every deployment episode
  gated_mid     frozen until CUSUM on the mid-band score share fires, then score-only updates
  gated_ks      the same with CUSUM on the KS distance to the burn-in scores
  online_audit  audit-recalibrated updates (2 % labels, np_audit "audit_recal") on every deployment
                episode: the ungated label-informed baseline (appended; earlier task ids unchanged)
Detector statistic is computed from the users' first-chunk scores of each deployment episode,
standardised by the burn-in; thresholds from detector_calib.py (5 % run-level false alarms).
Evaluation at the end, frozen, on the eval block: no shift and the trained rho, 20 eval seeds.

    python round4/gated.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import json
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
import detector_calib as DC                       # noqa: E402  (loads the same shift helpers)

OUT = G.OUT.parent / "gated"
PRE, DEPLOY, BURN_EPS, SHIFT_AT = 200, 200, 50, 250
ARMS = ("frozen", "online", "gated_mid", "gated_ks", "online_audit")
RHOS = (0.0, 0.5, 1.0)
SEEDS = tuple(range(int(os.environ.get("N_SEEDS", 3))))


def tasks():
    return [{"arm": a, "rho": r, "seed": s} for a in ARMS for r in RHOS for s in SEEDS]


def name(t):
    return f"{t['arm']}__{t['rho']}__{t['seed']}"


class Detector:
    def __init__(self, kind, threshold):
        self.kind, self.h = kind, threshold
        self.burn, self.ref, self.mu, self.sd = [], None, None, None
        self.s, self.alarm_ep = 0.0, None

    def stat(self, scores):
        if self.kind == "gated_mid":
            return float(np.mean((scores >= DC.MID[0]) & (scores <= DC.MID[1])))
        return DC.ks(scores, self.ref)

    def update(self, ep, scores):
        if len(self.burn) < BURN_EPS:
            self.burn.append(scores)
            if len(self.burn) == BURN_EPS:
                self.ref = np.concatenate(self.burn)
                x = np.array([self.stat(b) for b in self.burn])
                self.mu, self.sd = x.mean(), x.std() + 1e-9
            return False
        z = (self.stat(scores) - self.mu) / self.sd
        self.s = max(0.0, self.s + z - DC.K)
        if self.alarm_ep is None and self.s > self.h:
            self.alarm_ep = ep
        return self.alarm_ep is not None


def run_task(i):
    import e7_proxy_reward as E
    import expkit.ppo as PPO
    import expkit.trainer as T
    from expkit.posterior_reward import OutcomeModel, posterior_rewards

    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = D.pools(False, f"R6 gated {name(t)}")
    _, after_b, _ = D.shifted(p, "bots", t["rho"], held_out=False) if t["rho"] > 0 else (None, p.train_b, 0)
    om = OutcomeModel(C.GROUNDED, C.GROUNDED_CFG)
    thr = json.loads((R.OUT / "detector_thresholds.json").read_text())
    det = Detector(t["arm"], thr["cusum_mid" if t["arm"] == "gated_mid" else "cusum_ks"]) \
        if t["arm"].startswith("gated") else None
    st = {"ep": 0, "learn": True}
    audit_relabel = None
    if t["arm"] == "online_audit":
        import np_audit as NA
        audit_relabel = NA.make_relabel("audit_recal", om, None, t["seed"])
    real_run_x, real_relabel, real_update = T.run_x, T.relabel, PPO.TrainablePPO.update

    def run_x_hooked(policy, humans, bots, n_bots, **kw):
        if kw.get("collect") and humans is p.train_h and st["ep"] >= SHIFT_AT and t["rho"] > 0:
            bots = after_b
        return real_run_x(policy, humans, bots, n_bots, **kw)

    def relabel(transitions, cfg, rng):
        ep = st["ep"]
        st["ep"] += 1
        if ep < PRE:                                   # offline phase: label-based reward as collected
            st["learn"] = True
            return transitions, 0
        if t["arm"] == "frozen":
            st["learn"] = False
        elif t["arm"] == "online":
            st["learn"] = True
        elif t["arm"] == "online_audit":
            st["learn"] = True
            return audit_relabel(transitions, cfg, rng)
        else:
            first = {}
            for tr in transitions:
                first.setdefault(tr.user_id, tr.state[0])
            st["learn"] = det.update(ep, np.array(list(first.values()), dtype=float))
        if not st["learn"]:
            return transitions, 0
        from copy import copy
        r = posterior_rewards(transitions, om, None, mode="score_only")
        out_ = [copy(x) for x in transitions]
        for x, v in zip(out_, r):
            x.reward = float(v)
        return out_, 0

    def update(self, transitions):
        return real_update(self, transitions) if st["learn"] else {}

    spec = dict(E.CONFIGS["score-only"])
    spec.pop("hook")
    T.run_x, T.relabel, PPO.TrainablePPO.update = run_x_hooked, relabel, update
    try:
        agent, _ = D.train_online("ppo", p.train_h, p.train_b, p.cache, t["seed"], PRE + DEPLOY, False,
                                  solve=C.GROUNDED, cfg=C.GROUNDED_CFG, **spec)
    finally:
        T.run_x, T.relabel, PPO.TrainablePPO.update = real_run_x, real_relabel, real_update
    assert st["ep"] == PRE + DEPLOY, st
    agent.eval_mode()
    pts = [("none", 0.0)] + ([("bots", t["rho"])] if t["rho"] > 0 else [])
    df = D._evaluate_points(agent, f"PPO {t['arm']}", p, pts, False, C.GROUNDED, C.GROUNDED_CFG,
                            {"arm": t["arm"], "train_rho": t["rho"], "train_seed": t["seed"],
                             "alarm_ep": None if det is None else det.alarm_ep})
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min, alarm {None if det is None else det.alarm_ep}", flush=True)


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
    runs.to_csv(OUT.parent / "gated_runs.csv", index=False)
    g = runs.groupby(["arm", "train_rho", "eval_direction", "eval_rho"]).agg(
        DI=("DI", "mean"), humans=("human_survival", "mean"), bots=("bot_survival", "mean"),
        fr=("friction_s", "mean"), seeds=("train_seed", "nunique")).round(3)
    print(g.to_string())
    print(runs.groupby(["arm", "train_rho", "train_seed"]).alarm_ep.first().to_string())


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
