"""Neyman-Pearson thresholds (per-user: at most alpha of humans ever cross t) and an audit-budget reward under calibration drift (exploratory).

NP threshold t_alpha: the (1 - alpha) quantile of HUMAN chunk scores in the scorer block (out of
fold, the posterior's own calibration rows) -- "challenge at most alpha of human decisions". It
uses no bot data and no bot prior, so bot-side drift cannot move it (Tong, Feng & Li 2018).

Policies / rewards:
  static NP single    level 10 if q > t_alpha else 0
  static NP multi     0 if q <= t_alpha else max(3, floor(10 q)) (a visible challenge, graded)
  NP reward           the score-only reward with q replaced by the NP decision 1[q > t_alpha]:
                      r = (1 - d) r_H + d E_b r_B  (DQN and PPO)
  audit reward        a random EPS of USERS is audited (true class revealed, e.g. a verified
                      hard challenge or manual review): their transitions keep the oracle reward;
                      every other transition gets the score-only reward (PPO)
Online protocol = round 4's (300 episodes, shift from episode 100, eval on the eval block at the
shifted point and at rho = 0); control = same arm, no shift. Shift: round 4's calibration-only
shift by default, the ranking-destroying replay shift with AUC_SHIFT=1.

    python round4/np_audit.py --list | --task N | --run-all J | --collect
"""
from __future__ import annotations

import argparse
import dataclasses
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
AUC = bool(os.environ.get("AUC_SHIFT"))
if AUC:
    import auc_drift                              # noqa: F401  (patches D.shifted to the replay shift)
OUT = G.OUT.parent / ("np_audit_auc" if AUC else "np_audit")
ALPHA = float(os.environ.get("NP_ALPHA", 0.05))
EPS = float(os.environ.get("AUDIT_EPS", 0.02))
LEVELS = tuple(float(x) for x in os.environ.get("AUDIT_LEVELS", "0.5,1.0").split(","))
SEEDS = tuple(range(int(os.environ.get("N_SEEDS", 3))))
ARMS = {"dqn": ("np", "audit_recal", "audit_ch"), "ppo": ("np", "audit", "combo", "audit_dr", "audit_recal", "audit_ch")}
AUDIT_LEVEL = int(os.environ.get("AUDIT_LEVEL", 9))
AUDIT = {"eps": EPS, "level": AUDIT_LEVEL, "log": []}   # shared by the run_x hook and the reward


def audit_fail_rates():
    """(P(fail | human), P(fail | bot)) at the audit level, from the simulator's own model: a human
    fails only after both attempts; a bot's capability is uniform on 0..9 (the training prior)."""
    g = C.GROUNDED
    ph = g.human_pass_probability(AUDIT_LEVEL)
    pb = np.mean([g.bot_pass_probability(b, AUDIT_LEVEL) for b in range(10)])
    return (1 - ph) ** 2, 1 - pb


def np_threshold(p, alpha=ALPHA):
    from expkit.posterior_reward import replay_scores
    from expkit.simx import sessions_from_refs
    sh, sb = sessions_from_refs(p.ctx.fold.scorer)
    rows = replay_scores(sh, sb, p.ctx.crossfit.cache(sh + sb))
    # per USER, not per decision: a human makes up to 12 decisions, so bound the share of humans
    # whose highest chunk score crosses t (at most alpha of humans ever challenged)
    mx = {}
    for step, latest, avg, is_bot, key in rows:
        if not is_bot:
            mx[key] = max(mx.get(key, 0.0), latest)
    return float(np.quantile(np.array(list(mx.values())), 1 - alpha))


class NPStatic:
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, t, multi):
        self.t, self.multi = t, multi
        self.name = f"NP {'multi' if multi else 'single'} (alpha {ALPHA})"

    def select_action(self, user, n_active):
        q = user.bot_score
        if q <= self.t:
            return 0
        return max(3, min(int(q * 10), 9)) if self.multi else 10


def tasks():
    t = [{"kind": "statics"}]
    for algo, arms in ARMS.items():
        for arm in arms:
            t += [{"kind": "control", "algo": algo, "arm": arm, "seed": s} for s in SEEDS]
            t += [{"kind": "online", "algo": algo, "arm": arm, "direction": d, "rho": r, "seed": s}
                  for d in D.DIRECTIONS for r in LEVELS for s in SEEDS]
    return t


def name(t):
    return "__".join(str(v) for v in t.values())


COMBO_LAMBDA = 0.5          # weight on the raw score in the combined reward
RECAL_BINS = 10             # score bins of the audit recalibration
RECAL_DECAY = 0.97          # per-episode decay of old audits (recent traffic counts more)
RECAL_PRIOR = 2.0           # pseudo-audits per bin, centred on the bin's raw score


EMB_PRIOR = 20.0           # pseudo-audits of shrinkage towards the raw score (= 10 bins x 2)
EMB_C = 1.0                # inverse L2 strength of the embedding recalibration


def emb_x(state):
    """(q, chunk embedding, session-mean embedding) from a stored embedding state."""
    s = np.asarray(state, dtype=np.float32)
    return np.concatenate([s[:1], s[5:]])


class EmbRecal:
    """P(y = 1 | q, embedding) from decayed audits: weighted L2 logistic regression, refitted
    every episode, shrunk towards a fallback by n_eff / (n_eff + EMB_PRIOR)."""

    def __init__(self):
        self.X, self.y, self.w = [], [], np.zeros(0)

    def add(self, X, y):
        self.w = np.concatenate([self.w * RECAL_DECAY, np.ones(len(y))])
        self.X += list(X)
        self.y += list(y)
        keep = self.w > 1e-3
        self.X = [x for x, k in zip(self.X, keep) if k]
        self.y = [v for v, k in zip(self.y, keep) if k]
        self.w = self.w[keep]

    def predict(self, X, fallback):
        from sklearn.linear_model import LogisticRegression
        y = np.array(self.y)
        if len(y) == 0 or y.min() == y.max():
            return np.asarray(fallback, dtype=float)
        m = LogisticRegression(C=EMB_C, max_iter=500).fit(np.array(self.X), y, sample_weight=self.w)
        n = self.w.sum()
        return (n * m.predict_proba(X)[:, 1] + EMB_PRIOR * np.asarray(fallback)) / (n + EMB_PRIOR)


def make_relabel(arm, om, t_np, seed):
    """Per-episode reward relabelling for the label-free / audited arms.

    np           q := 1[q > t_np]
    combo        q := lambda q + (1 - lambda) 1[q > t_np]
    audit        audited users keep the oracle reward; others score-only (round-6 first version)
    audit_dr     doubly robust: everyone gets r_score; an audited user's transitions get
                 r_score + (r_oracle - r_score) / EPS, so E[reward] = r_oracle for every user
    audit_recal  audits recalibrate the score on CURRENT traffic: P(bot | score bin) from the
                 audited users' (score, label) pairs (decayed counts, smoothed towards the raw
                 score); every non-audited transition gets the score-only reward with q replaced
                 by that recalibrated probability; audited users keep the oracle reward
    audit_recal_emb / audit_ch_emb   the same two audits, but P(bot) is a logistic regression on
                 (q, chunk embedding, session-mean embedding) instead of a function of the q bin,
                 so users with equal q but different embeddings get different rewards. Fitted
                 on decayed audits, shrunk towards the raw score (or the bin prior) by
                 n_eff / (n_eff + EMB_PRIOR). Needs the embedding state (RLC_EMB).
    """
    from expkit.posterior_reward import posterior_rewards
    audit_rng = random.Random(10_000 + seed)
    state = {"ep": 0, "n": np.zeros(RECAL_BINS), "b": np.zeros(RECAL_BINS)}
    centres = (np.arange(RECAL_BINS) + 0.5) / RECAL_BINS

    def with_q(transitions, fn):
        masked = []
        for t in transitions:
            m = copy(t)
            m.state = [float(fn(t.state[0]))] + list(t.state[1:])
            masked.append(m)
        return posterior_rewards(masked, om, None, mode="score_only")

    emb_fit = EmbRecal() if arm.endswith("_emb") else None

    def with_q_vec(transitions, qs):
        masked = []
        for t, q in zip(transitions, qs):
            m = copy(t)
            m.state = [float(q)] + list(t.state[1:])
            masked.append(m)
        return posterior_rewards(masked, om, None, mode="score_only")

    def relabel(transitions, cfg, rng):
        state["ep"] += 1
        users = sorted({t.user_id for t in transitions})
        audited = {u for u in users if audit_rng.random() < EPS}
        out = [copy(t) for t in transitions]
        if arm == "np":
            r = with_q(transitions, lambda q: 1.0 if q > t_np else 0.0)
        elif arm == "combo":
            r = with_q(transitions, lambda q: COMBO_LAMBDA * q + (1 - COMBO_LAMBDA) * (1.0 if q > t_np else 0.0))
        elif arm == "audit_recal":
            # update the recalibration with THIS episode's audits, then apply it to everyone
            state["n"] *= RECAL_DECAY
            state["b"] *= RECAL_DECAY
            for t in transitions:
                if t.user_id in audited:
                    k = min(int(t.state[0] * RECAL_BINS), RECAL_BINS - 1)
                    state["n"][k] += 1
                    state["b"][k] += float(t.is_bot)
            p_bin = (state["b"] + RECAL_PRIOR * centres) / (state["n"] + RECAL_PRIOR)
            r = with_q(transitions, lambda q: p_bin[min(int(q * RECAL_BINS), RECAL_BINS - 1)])
        elif arm == "audit_ch":
            # challenge-based audit: only pass/fail is observed; per bin the fail rate mixes the
            # classes, f = pi f_b + (1 - pi) f_h, so pi = (f - f_h) / (f_b - f_h)
            f_h, f_b = audit_fail_rates()
            state["n"] *= RECAL_DECAY
            state["b"] *= RECAL_DECAY
            for q0, failed in AUDIT["log"]:
                k = min(int(q0 * RECAL_BINS), RECAL_BINS - 1)
                state["n"][k] += 1
                state["b"][k] += float(failed)
            AUDIT["log"].clear()
            f = (state["b"] + RECAL_PRIOR * (centres * f_b + (1 - centres) * f_h)) / (state["n"] + RECAL_PRIOR)
            p_bin = np.clip((f - f_h) / (f_b - f_h), 0.0, 1.0)
            state["p_bin"] = p_bin
            r = with_q(transitions, lambda q: p_bin[min(int(q * RECAL_BINS), RECAL_BINS - 1)])
        elif arm == "audit_recal_emb":
            X = np.array([emb_x(t.state) for t in transitions])
            emb_fit.add([emb_x(t.state) for t in transitions if t.user_id in audited],
                        [float(t.is_bot) for t in transitions if t.user_id in audited])
            p = emb_fit.predict(X, X[:, 0])
            r = with_q_vec(transitions, p)
        elif arm == "audit_ch_emb":
            f_h, f_b = audit_fail_rates()
            log = list(AUDIT["log"])
            AUDIT["log"].clear()
            emb_fit.add([np.concatenate([[q0], f]) for q0, _, f in log], [float(fl) for _, fl, _ in log])
            X = np.array([emb_x(t.state) for t in transitions])
            f = emb_fit.predict(X, X[:, 0] * f_b + (1 - X[:, 0]) * f_h)
            r = with_q_vec(transitions, np.clip((f - f_h) / (f_b - f_h), 0.0, 1.0))
        else:
            r = posterior_rewards(transitions, om, None, mode="score_only")
        for t, x in zip(out, r):
            if arm in ("audit", "audit_recal", "audit_recal_emb") and t.user_id in audited:
                continue                        # the audited user's true (oracle) reward
            if arm == "audit_dr" and t.user_id in audited:
                t.reward = float(x) + (t.reward - float(x)) / EPS
                continue
            t.reward = float(x)
        return out, 0
    return relabel


def run_task(i):
    import e7_proxy_reward as E
    import expkit.trainer as T
    from expkit.posterior_reward import OutcomeModel

    G.no_power_throttling()
    t = tasks()[i]
    out = OUT / f"{name(t)}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = D.pools(False, f"R6 np/audit {'auc ' if AUC else ''}{name(t)}")
    t_np = np_threshold(p)
    if t["kind"] == "statics":
        pols = [NPStatic(t_np, False), NPStatic(t_np, True)]
        df = pd.concat([D._evaluate_points(pol, pol.name, p, D.points(LEVELS), False, C.GROUNDED,
                                           C.GROUNDED_CFG, {"panel": "static", "t_np": t_np})
                        for pol in pols], ignore_index=True)
    else:
        algo, arm, seed = t["algo"], t["arm"], t["seed"]
        online = t["kind"] == "online"
        direction, rho = (t["direction"], t["rho"]) if online else ("none", 0.0)
        after_h, after_b, real = D.shifted(p, direction, rho, held_out=False)
        om = OutcomeModel(C.GROUNDED, C.GROUNDED_CFG)
        spec = dict(E.CONFIGS["score-only"])
        spec.pop("hook")
        st = {"ep": 0}
        real_run_x, real_relabel = T.run_x, T.relabel

        def run_x_hooked(policy, humans, bots, n_bots, **kw):
            if kw.get("collect") and humans is p.train_h:
                if online and st["ep"] >= D.INJECT_EP:
                    humans, bots = after_h, after_b
                st["ep"] += 1
                if arm.startswith("audit_ch"):
                    kw["audit"] = AUDIT
                    if arm == "audit_ch_emb":
                        AUDIT["feat"] = T._emb.features
            return real_run_x(policy, humans, bots, n_bots, **kw)
        T.run_x, T.relabel = run_x_hooked, make_relabel(arm, om, t_np, seed)
        try:
            agent, _ = D.train_online(algo, p.train_h, p.train_b, p.cache, seed, D.TOTAL, False,
                                      solve=C.GROUNDED, cfg=C.GROUNDED_CFG, **spec)
        finally:
            T.run_x, T.relabel = real_run_x, real_relabel
        assert st["ep"] == D.TOTAL, st
        agent.eval_mode()
        pts = [("none", 0.0), (direction, rho)] if online else D.points(LEVELS)
        if arm.startswith("audit_ch"):              # the audit is part of the deployed system
            import functools
            real_eval_run_x = C.run_x
            C.run_x = functools.partial(real_eval_run_x, audit={"eps": EPS, "level": AUDIT_LEVEL, "log": []})
        df = D._evaluate_points(agent, f"{algo.upper()} {arm}", p, pts, False, C.GROUNDED,
                                C.GROUNDED_CFG, {"panel": t["kind"], "algo": algo, "arm": arm,
                                                 "train_direction": direction, "train_rho": rho,
                                                 "train_seed": seed, "t_np": t_np})
        if arm.startswith("audit_ch"):
            C.run_x = real_eval_run_x
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(t)}: {(time.time() - t0) / 60:.1f} min (t_np={t_np:.3f})", flush=True)


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
