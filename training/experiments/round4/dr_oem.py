"""Adversarial-bot robustness for PPO without labels at deployment (exploratory, round 6).

A. DOMAIN-RANDOMISED TRAINING ("dr"). PPO trained OFFLINE (label-based reward, simulation) for
   300 episodes; each episode draws a drift regime IN SCORE SPACE from the RL block only -- no
   external bots, no evaluation data, nothing from the test attacks:
     clean        (p = 1/3) no change
     calibration  (p = 1/3) a fraction rho ~ U(0,1) of bots: logit(q) -> logit(q) - delta,
                  delta ~ U(0, 4)  (a generic miscalibration that keeps the ranking)
     ranking      (p = 1/3) a fraction rho ~ U(0,1) of bots receive the chunk-score trajectory
                  of a random RL-block HUMAN session (score evidence carries no information;
                  their challenge behaviour is still the bot's)
   Evaluated FROZEN on the unseen test drifts (bots direction only): round 4's calibration
   shift (Phase-2 advanced web bots, held-out half) and the ranking-destroying replay shift
   (evaluation-block human recordings as bots), rho in {0, .25, .5, .75, 1}. The headline PPO
   agents (no randomisation, 200 episodes) are evaluated in the same task as the comparator.

B. OUTCOME-ANCHORED EM ("oem" naive global share, "oem_bins" per-score-bin). Online PPO (round 4's protocol: 300 episodes,
   shift from episode 100). Each episode the bot share is estimated by EM on per-user likelihood
   ratios from CHALLENGE OUTCOMES only (OutcomeModel: pass / fail / leave rates the defender
   sets; no score), then the posterior reward uses that share, with the score's likelihood
   ratio ("oem") or without it ("oem_noscore"). Shifts: calibration and replay, bots, rho in
   {0.5, 1.0}, seeds 0-2; controls (no shift) per arm.

    python round4/dr_oem.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
CALIB_SHIFT = D.shifted
import auc_drift                                  # noqa: E402,F401  (patches D.shifted)
REPLAY_SHIFT = D.shifted
D.shifted = CALIB_SHIFT
SHIFTS = {"calibration": CALIB_SHIFT, "replay": REPLAY_SHIFT}

OUT = G.OUT.parent / "dr_oem"
EVAL_LEVELS = (0.25, 0.5, 0.75, 1.0)
ONLINE_LEVELS = (0.5, 1.0)
DR_SEEDS, ONLINE_SEEDS = range(5), (0, 1, 2)
DR_EPISODES = 300


# --------------------------------------------------------------------------- #
# A. score-space drift for training
# --------------------------------------------------------------------------- #

class DriftCache:
    """Wraps the score cache; for drifted BOT users returns a transformed score."""

    def __init__(self, inner, human_sessions, mode, rho, delta, rng):
        self.inner, self.humans = inner, human_sessions
        self.mode, self.rho, self.delta, self.rng = mode, rho, delta, rng
        self._who = {}
        self._table = inner._table

    def __getattr__(self, a):
        return getattr(self.inner, a)

    def score(self, user):
        v = self.inner.score(user)
        if not user.is_bot or self.mode == "clean":
            return v
        uid = id(user)
        if uid not in self._who:
            self._who[uid] = (self.rng.random() < self.rho, self.rng.choice(self.humans))
        drifted, hs = self._who[uid]
        if not drifted:
            return v
        if self.mode == "calibration":
            if v is None:
                return None
            q = min(max(v, 1e-6), 1 - 1e-6)
            return 1 / (1 + math.exp(-(math.log(q / (1 - q)) - self.delta)))
        idx = user.cache_key[1]                     # ranking: a human session's score at this step
        n = len(hs.chunks)
        return self.inner._table.get((hs.key, min(idx, n - 1)))


def train_dr(p, seed):
    import expkit.trainer as T
    rng = random.Random(50_000 + seed)
    real_run_x = T.run_x
    log = []

    def run_x_hooked(policy, humans, bots, n_bots, **kw):
        if kw.get("collect") and humans is p.train_h:
            mode = rng.choice(("clean", "calibration", "ranking"))
            rho, delta = rng.random(), rng.uniform(0, 4)
            kw["cache"] = DriftCache(kw["cache"], p.train_h, mode, rho, delta, rng)
            log.append((mode, rho, delta))
        return real_run_x(policy, humans, bots, n_bots, **kw)

    T.run_x = run_x_hooked
    try:
        agent, _ = D.train_online("ppo", p.train_h, p.train_b, p.cache, seed, DR_EPISODES, False,
                                  solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
    finally:
        T.run_x = real_run_x
    assert len(log) == DR_EPISODES, len(log)
    return agent.eval_mode()


# --------------------------------------------------------------------------- #
# B. outcome-anchored EM reward
# --------------------------------------------------------------------------- #

BINS, DECAY, PRIOR_N = 10, 0.97, 2.0


def _outcome_lr(transitions, om):
    """Per-user likelihood ratio P(outcomes | bot) / P(outcomes | human) from challenge outcomes
    only (posterior at share 0.5, no score), and the user's running-mean score at its last step."""
    from expkit.posterior_reward import posterior_rewards
    _, q = posterior_rewards(transitions, om, None, mode="posterior", share=0.5, return_q=True)
    last, score = {}, {}
    for t, qq in zip(transitions, q):
        last[t.user_id] = qq
        score[t.user_id] = t.state[1]
    u = list(last)
    qv = np.clip(np.array([last[k] for k in u]), 1e-6, 1 - 1e-6)
    return u, qv / (1 - qv), np.array([score[k] for k in u])


def make_oem_relabel(om, calib, mode, stats):
    """mode "oem":      one global share by EM over all users' outcome LRs (naive: outcomes exist
                        only for challenged users, so the share is the challenged users' share)
       mode "oem_bins": P(bot | score bin) by EM within each bin of the running-mean score, over
                        the challenged users of that bin (outcome LR != 1), decayed across
                        episodes, smoothed towards the bin's raw score; the score-only reward then
                        uses this recalibrated probability. No labels anywhere."""
    from expkit.posterior_reward import em_share, posterior_rewards
    centres = (np.arange(BINS) + 0.5) / BINS
    hist = {"lr": np.zeros(0), "bin": np.zeros(0, int), "w": np.zeros(0)}

    def relabel(transitions, cfg, rng):
        users, lr, sc = _outcome_lr(transitions, om)
        if mode == "oem":
            pi = em_share(np.log(lr), pi0=0.5)
            stats.append(pi)
            r = posterior_rewards(transitions, om, calib, mode="posterior", share=pi)
        else:
            inf = np.abs(np.log(lr)) > 1e-6
            b = np.minimum((sc * BINS).astype(int), BINS - 1)
            hist["w"] = np.r_[hist["w"] * DECAY, np.ones(inf.sum())]
            hist["lr"] = np.r_[hist["lr"], lr[inf]]
            hist["bin"] = np.r_[hist["bin"], b[inf]]
            keep = hist["w"] > 0.02
            for k in hist:
                hist[k] = hist[k][keep]
            p_bin = centres.copy()
            for k in range(BINS):
                m = hist["bin"] == k
                if not m.any():
                    continue
                w, lam, pi = hist["w"][m], hist["lr"][m], centres[k]
                for _ in range(200):                 # weighted EM, prior pseudo-counts at the centre
                    post = pi * lam / (pi * lam + 1 - pi)
                    new = (float(w @ post) + PRIOR_N * centres[k]) / (float(w.sum()) + PRIOR_N)
                    if abs(new - pi) < 1e-7:
                        break
                    pi = new
                p_bin[k] = pi
            stats.append(p_bin.tolist())
            masked = []
            for t in transitions:
                m_ = copy(t)
                m_.state = [float(p_bin[min(int(t.state[0] * BINS), BINS - 1)])] + list(t.state[1:])
                masked.append(m_)
            r = posterior_rewards(masked, om, None, mode="score_only")
        out = [copy(t) for t in transitions]
        for t, x in zip(out, r):
            t.reward = float(x)
        return out, 0
    return relabel


# --------------------------------------------------------------------------- #

def tasks():
    t = [{"kind": "dr", "seed": s} for s in DR_SEEDS]
    for arm in ("oem", "oem_bins"):
        t += [{"kind": "control", "arm": arm, "seed": s} for s in ONLINE_SEEDS]
        t += [{"kind": "online", "arm": arm, "shift": sh, "rho": r, "seed": s}
              for sh in SHIFTS for r in ONLINE_LEVELS for s in ONLINE_SEEDS]
    return t


def name(t):
    return "__".join(str(v) for v in t.values())


def evaluate_bots(pol, label, p, tags, shifts=SHIFTS, levels=EVAL_LEVELS):
    frames = []
    for sh, fn in shifts.items():
        D.shifted = fn
        try:
            pts = [("none", 0.0)] + [("bots", r) for r in levels]
            frames.append(D._evaluate_points(pol, label, p, pts, False, C.GROUNDED, C.GROUNDED_CFG,
                                             {**tags, "eval_shift": sh}))
        finally:
            D.shifted = CALIB_SHIFT
    return pd.concat(frames, ignore_index=True)


def run_task(i):
    import e7_proxy_reward as E
    import expkit.trainer as T
    from exp_round3_fold2 import load_policy
    from expkit.posterior_reward import OutcomeModel

    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = D.pools(False, f"R6 dr/oem {name(t)}")
    if t["kind"] == "dr":
        agent = train_dr(p, t["seed"])
        (OUT / "agents").mkdir(parents=True, exist_ok=True)
        agent.save(OUT / "agents" / f"ppo_dr_s{t['seed']}.pt")
        base = load_policy(G.OUT / "agents" / f"ppo_s{t['seed']}.pt")
        df = pd.concat([evaluate_bots(agent, "PPO domain-randomised", p, {"panel": "dr", "train_seed": t["seed"]}),
                        evaluate_bots(base, "PPO (no randomisation)", p, {"panel": "dr", "train_seed": t["seed"]})],
                       ignore_index=True)
    else:
        online = t["kind"] == "online"
        sh = t.get("shift", "calibration")
        rho = t.get("rho", 0.0) if online else 0.0
        D.shifted = SHIFTS[sh]
        after_h, after_b, real = D.shifted(p, "bots" if online else "none", rho, held_out=False)
        D.shifted = CALIB_SHIFT
        om = OutcomeModel(C.GROUNDED, C.GROUNDED_CFG)
        calib = E.fit_calibration(p) if t["arm"] == "oem" else None
        spec = dict(E.CONFIGS["score-only"])
        spec.pop("hook")
        st, shares = {"ep": 0}, []
        real_run_x, real_relabel = T.run_x, T.relabel

        def run_x_hooked(policy, humans, bots, n_bots, **kw):
            if kw.get("collect") and humans is p.train_h:
                if online and st["ep"] >= D.INJECT_EP:
                    humans, bots = after_h, after_b
                st["ep"] += 1
            return real_run_x(policy, humans, bots, n_bots, **kw)
        T.run_x = run_x_hooked
        T.relabel = make_oem_relabel(om, calib, t["arm"], shares)
        try:
            agent, _ = D.train_online("ppo", p.train_h, p.train_b, p.cache, t["seed"], D.TOTAL, False,
                                      solve=C.GROUNDED, cfg=C.GROUNDED_CFG, **spec)
        finally:
            T.run_x, T.relabel = real_run_x, real_relabel
        assert st["ep"] == D.TOTAL, st
        agent.eval_mode()
        tags = {"panel": t["kind"], "arm": t["arm"], "train_shift": sh, "train_rho": rho,
                "train_seed": t["seed"], "share_after_inject": str(np.round(np.mean(np.array(shares[D.INJECT_EP:], dtype=float), axis=0), 3).tolist()),
                "share_before_inject": str(np.round(np.mean(np.array(shares[:D.INJECT_EP], dtype=float), axis=0), 3).tolist())}
        if online:
            df = evaluate_bots(agent, f"PPO {t['arm']}", p, tags, {sh: SHIFTS[sh]}, levels=(rho,))
        else:
            df = evaluate_bots(agent, f"PPO {t['arm']}", p, tags, levels=ONLINE_LEVELS)
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
