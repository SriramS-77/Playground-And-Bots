"""E7 -- can a reward built only from what a deployment observes train a secure policy?
(Reviewer 1, point 1 -- the one that matters most; HANDOFF_ROUND3 §5 Priority 3,
HANDOFF_ROUND2 §6 Priority 2.)

Everything runs in the GROUNDED world (stochastic solves; humans can fail a challenge),
because only there is the immediate signal ambiguous: "failed, session ended" is a blocked
bot OR a false-positived human, and the proxy reward is a pure function of what the
operator sees -- level, observable outcome, seconds (`expkit.proxy.relabel`). The proxy
agent never reads the class. Class information reaches it only through a delayed, sparse
(p = 0.3), noisy (2 %) abuse label, and through what solve time genuinely reveals: since
round 3, bots spend measured solve times (`expkit.stochastic.BOT_SECONDS`) instead of 0 s,
which had made the time a perfect class label. Round 1-2 E7 numbers are not comparable.

Configurations -- every one the DQN winner, 5 seeds, trained on fold 2's rl block with
out-of-fold scores, evaluated on the eval block:

    oracle            the simulator's reward -- the matched-budget denominator
    proxy             the default proxy reward, abuse penalty -150
    proxy pen=P       P in {-600, -2400, -9600}: the break-even arithmetic says blocking
                      only pays once |P| > ~260
    proxy zfr=Z       solve 3, immediate gap: zero_friction_reward Z in {35, 25} (vs 50)
    pbrs pen=P        solve 1, potential-based shaping F = gamma*Phi(s') - Phi(s) with
                      Phi(s) = -|P| * label_probability * bot_score(s), at P in {-600,
                      -2400}. PBRS preserves the optimal policy BY THEOREM (Ng, Harada &
                      Russell 1999), so it can only fix credit assignment, never an
                      optimum that is already "never challenge" -- hence above break-even.
    learned rm        solve 2: a logistic model of "this session will be confirmed abusive"
                      fitted on logged behaviour -- state, action, observable outcome; NOT
                      seconds -- without class balancing, so its output is a probability.
                      Inverted for label coverage, P(leak) = (p_hat - 0.02) / (0.30 - 0.02)
                      (valid here because the simulator confirms leaks at random; Elkan &
                      Noto, KDD 2008), and P * P(leak) is placed where the label would land,
                      REPLACING the sparse label. Its immediate part is still the proxy's.
    normclip pen=P    solve 4: rewards scaled by 150/|P| and gradient norm clipped at 10,
                      at P in {-2400, -9600} -- is the large-penalty collapse optimisation?
    proxy 1000ep      solve 5: the default proxy reward at 1000 episodes
    posterior         the oracle reward in expectation over the operator's posterior on
                      (class, strength): calibrated humanity score + every challenge outcome
                      by Bayes, bot share by EM per episode, human friction at its EXPECTED
                      value (solve time is never evidence). `expkit.posterior_reward`.
                      Uses the simulator's exact outcome model and an offline-labelled
                      calibration set -- an upper bound on the idea, not a deployment.
    posterior+labels  the same, also conditioned on the delayed abuse label as evidence
                      (fires 0.30 for a bot not blocked by session end, 0.02 otherwise) --
                      never as an extra penalty, which would double count
    ... (floored)     SENSITIVITY: both posterior variants with the score's likelihood ratio
                      capped by per-class Laplace floors, eps_c = 1 / (n_c + 2) over the
                      calibration set's recordings of class c. Leave-one-recording-out CV
                      picks eps = 0 on fold 2 because no calibration recording is
                      confidently wrong -- CV cannot see a rate below ~1/n -- while the rl
                      block holds two confidently misscored humans. Declared after the
                      reward-bias diagnostic showed that; nothing from the rl block enters it.
    score-only        ABLATION: the raw latest chunk score as P(bot), no prior correction,
                      no outcome evidence -- the idea "as is", to show what the
                      corrections are for
    never challenge   level 0 for everyone -- the floor of the gap-closed ratio

Reward-bias diagnostic (task kind `reward_bias`): on logged random-level behaviour in the rl
block, mean(r_hat - oracle r) overall and by band of q(B), for each reward variant, plus the
posterior's Brier score / log-loss and the EM share error per episode. Two checks, fixed
before looking: the EXACT variant (no score evidence, true share) must be unbiased within
4 SE -- it tests the code; and in the uncertain band 0.2 <= q(B) < 0.8 a variant counts as
"not skewed" when |bias| <= 5 % of the mean |r_H - E r_B| there -- it tests calibration.

Not implemented: solve 6, off-policy evaluation. Every learned policy here is greedy, so
importance weights against it are degenerate; OPE needs a stochastic logging policy and
estimates value rather than fixing the reward. Stated, not hidden.

--collect reports DI, BOS, SI-F1, false-positive rate, humans, bots and friction per
configuration; the proxy/oracle ratio both raw and as
    (proxy - never_challenge) / (oracle - never_challenge)
excluding the zero-bot cell; and the observability table (what the immediate signal says
about class), measured on logged behaviour in the rl block.
"""

from __future__ import annotations

import dataclasses
import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch

import common as C
import expkit.trainer as T
from exp_policy_arch_search import BOT_VOLUMES
from expkit.proxy import OBSERVABLE, ProxyRewardConfig, immediate_reward, summarise_observability
from expkit.trainer import full_features

EXP = "e7"
D = ProxyRewardConfig()
CONFIGS = {
    "oracle": {"reward_source": "oracle"},
    "proxy": {"reward_source": "proxy", "proxy_cfg": D},
    **{f"proxy pen={p}": {"reward_source": "proxy",
                          "proxy_cfg": dataclasses.replace(D, abuse_penalty=float(p))}
       for p in (-600, -2400, -9600)},
    **{f"proxy zfr={z}": {"reward_source": "proxy",
                          "proxy_cfg": dataclasses.replace(D, zero_friction_reward=float(z))}
       for z in (35, 25)},
    **{f"pbrs pen={p}": {"reward_source": "proxy", "hook": "pbrs",
                         "proxy_cfg": dataclasses.replace(D, abuse_penalty=float(p))}
       for p in (-600, -2400)},
    "learned rm": {"reward_source": "proxy", "proxy_cfg": D, "hook": "learned_rm"},
    **{f"normclip pen={p}": {"reward_source": "proxy", "hook": "normclip",
                             "proxy_cfg": dataclasses.replace(D, abuse_penalty=float(p))}
       for p in (-2400, -9600)},
    "proxy 1000ep": {"reward_source": "proxy", "proxy_cfg": D, "episodes": 1000},
    # Rewards from the posterior. reward_source "proxy" routes them through `relabel` (hooked
    # below) and balances the replay buffer on the observable outcome, never the class.
    "posterior": {"reward_source": "proxy", "proxy_cfg": D, "hook": "posterior"},
    "posterior+labels": {"reward_source": "proxy", "proxy_cfg": D, "hook": "posterior_labels"},
    "score-only": {"reward_source": "proxy", "proxy_cfg": D, "hook": "score_only"},
    # Sensitivity: the score's likelihood ratio capped by per-class Laplace floors
    # (`ScoreCalibration.floored`) -- declared after the reward-bias diagnostic.
    "posterior (floored)": {"reward_source": "proxy", "proxy_cfg": D, "hook": "posterior_floored"},
    "posterior+labels (floored)": {"reward_source": "proxy", "proxy_cfg": D,
                                   "hook": "posterior_labels_floored"},
}
POSTERIOR_HOOKS = ("posterior", "posterior_labels", "score_only", "posterior_floored",
                   "posterior_labels_floored")
LOG_EPISODES = 40          # behaviour episodes for the observability table and the reward model
BIAS_EPISODES = 60         # behaviour episodes for the reward-bias diagnostic
SKEW_TOLERANCE = 0.05      # uncertain-band |bias| as a share of the class reward gap
UNCERTAIN = (0.2, 0.8)


def tasks():
    t = [{"kind": "train", "config": c, "seed": s} for c in CONFIGS for s in range(C.SEEDS)]
    return t + [{"kind": "never"}, {"kind": "observability"}, {"kind": "reward_bias"}]


def smoke_tasks():
    return [{"kind": "train", "config": c, "seed": 0}
            for c in ("oracle", "proxy", "pbrs pen=-600", "learned rm", "normclip pen=-2400",
                      "posterior", "posterior+labels", "score-only",
                      "posterior+labels (floored)")] + \
           [{"kind": "never"}, {"kind": "observability"}, {"kind": "reward_bias"}]


# --------------------------------------------------------------------------- #
# Behaviour data: a policy that tries every level, for the RM and the table
# --------------------------------------------------------------------------- #

class RandomLevel:
    name = "uniform random level"
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def features(self, user, n_active):
        return full_features(user, n_active)

    def select_action(self, user, n_active):
        return self.rng.randint(0, 10)


def logged_transitions(p, seed, episodes):
    rng, out = random.Random(seed), []
    for e in range(episodes):
        nh = rng.randint(*C.R.TRAIN_HUMAN_RANGE)
        _, tr = C.run_x(RandomLevel(seed * 1000 + e), p.train_h, p.train_b, rng.choice(BOT_VOLUMES),
                        cache=p.cache, n_humans=nh, seed=rng.randrange(10 ** 9), collect=True,
                        solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
        out.extend(tr)
    return out


def _labels(transitions, cfg, rng):
    """Per user: did the delayed abuse label fire? The same rule `relabel` applies."""
    last = {}
    for t in transitions:
        last[t.user_id] = t
    fired = {}
    for uid, t in last.items():
        leaked = t.is_bot and t.outcome != "bot_blocked"
        fired[uid] = rng.random() < (cfg.label_probability if leaked else cfg.label_noise)
    return fired


def _x(t):
    """State, action, observable outcome. Not seconds: a reward an adversary can move by
    choosing how long to take is not one to build on."""
    a = np.zeros(11, dtype=np.float32); a[t.action] = 1
    o = np.zeros(3, dtype=np.float32)
    o[["passed_continued", "passed_left", "failed_gone"].index(OBSERVABLE[t.outcome])] = 1
    return np.concatenate([np.asarray(t.state, dtype=np.float32), a, o])


def fit_reward_model(p, seed, episodes):
    """P(this session's abuse label fires | one transition's features). No class
    balancing: balancing inflates the odds by the negative:positive ratio, and the output
    is used as a probability."""
    from sklearn.linear_model import LogisticRegression

    tr = logged_transitions(p, seed, episodes)
    fired = _labels(tr, D, random.Random(seed))
    X = np.stack([_x(t) for t in tr])
    y = np.array([fired[t.user_id] for t in tr], dtype=int)
    if y.min() == y.max():
        raise RuntimeError("learned rm: logged data produced a single label class -- log more")
    return LogisticRegression(max_iter=2000).fit(X, y)


def p_leak(p_label, cfg):
    """P(leaked bot) from P(label fires): label = c * leak + noise * (1 - leak)."""
    return np.clip((np.asarray(p_label) - cfg.label_noise)
                   / (cfg.label_probability - cfg.label_noise), 0.0, 1.0)


def calibration_for(hook, calib):
    """The calibration a posterior hook uses: as fitted, floored, or none (score-only)."""
    if hook == "score_only" or calib is None:
        return None
    return calib.floored() if hook.endswith("_floored") else calib


def fit_calibration(p):
    """The humanity score's likelihood ratio, fitted on the SCORER block -- scored out of
    fold, disjoint from every session the policy trains or is evaluated on. The rl block's
    labels are never used for it."""
    from expkit.posterior_reward import ScoreCalibration, replay_scores
    from expkit.simx import sessions_from_refs

    sh, sb = sessions_from_refs(p.ctx.fold.scorer)
    cal = ScoreCalibration.fit(replay_scores(sh, sb, p.ctx.crossfit.cache(sh + sb)))
    print(f"  score calibration: {cal.n_sessions} scorer-block sessions ({cal.pi_cal:.0%} bots), "
          f"{cal.n_rows} session-steps; leave-one-recording-out CV chose C={cal.C}, "
          f"eps={cal.eps}; floored variants: eps_h={cal.floored().eps_h:.3f}, "
          f"eps_b={cal.floored().eps_b:.3f}, likelihood ratio within "
          f"[{cal.floored().bounds()[0]:.3g}, {cal.floored().bounds()[1]:.3g}]")
    print("  CV log-loss (rows C, columns eps):\n" +
          pd.DataFrame(cal.cv).pivot(index="C", columns="eps", values="log_loss").round(4).to_string())
    return cal


# --------------------------------------------------------------------------- #
# Hooks on the trainer, scoped to one training
# --------------------------------------------------------------------------- #

@contextmanager
def hooked(kind, proxy_cfg, rm=None, om=None, calib=None):
    orig_relabel, orig_dqn = T.relabel, T.TrainableDQN
    from copy import copy
    from collections import defaultdict

    def pbrs(transitions, cfg, rng):
        out, n = orig_relabel(transitions, cfg, rng)
        lam, g = abs(cfg.abuse_penalty) * cfg.label_probability, 0.95
        for t in out:
            phi_s = -lam * t.state[0]
            phi_n = 0.0 if t.done else -lam * t.next_state[0]
            t.reward += g * phi_n - phi_s
        return out, n

    def learned(transitions, cfg, rng):
        # The proxy's immediate reward, and the label's EXPECTATION -- corrected for
        # coverage -- where the label would have landed. No sparse label on top.
        out = [copy(t) for t in transitions]
        for t in out:
            t.reward = immediate_reward(t.action, t.outcome, t.seconds, cfg)
        by_user = defaultdict(list)
        for i, t in enumerate(out):
            by_user[t.user_id].append(i)
        last = [idxs[-1] for idxs in by_user.values()]
        leak = p_leak(rm.predict_proba(np.stack([_x(out[i]) for i in last]))[:, 1], cfg)
        for idxs, pl in zip(by_user.values(), leak):
            window = idxs[-cfg.label_delay_steps:]
            for i in window:
                out[i].reward += cfg.abuse_penalty * float(pl) / len(window)
        return out, 0

    def posterior(mode, use_labels):
        from expkit.posterior_reward import posterior_rewards

        def relabel_posterior(transitions, cfg, rng):
            labels = _labels(transitions, cfg, rng) if use_labels else None
            r = posterior_rewards(transitions, om, calib, mode=mode, labels=labels, proxy_cfg=cfg)
            out = [copy(t) for t in transitions]
            for t, x in zip(out, r):
                t.reward = float(x)
            return out, (int(sum(labels.values())) if labels else 0)
        return relabel_posterior

    def normalised(transitions, cfg, rng):
        out, n = orig_relabel(transitions, cfg, rng)
        k = 150.0 / abs(cfg.abuse_penalty)
        for t in out:
            t.reward *= k
        return out, n

    class ClippedDQN(orig_dqn):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            step = self.optimizer.step

            def clipped(*sa, **skw):
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                return step(*sa, **skw)
            self.optimizer.step = clipped

    try:
        if kind == "pbrs":
            T.relabel = pbrs
        elif kind == "learned_rm":
            T.relabel = learned
        elif kind == "posterior":
            T.relabel = posterior("posterior", use_labels=False)
        elif kind == "posterior_labels":
            T.relabel = posterior("posterior", use_labels=True)
        elif kind == "posterior_floored":
            T.relabel = posterior("posterior", use_labels=False)
        elif kind == "posterior_labels_floored":
            T.relabel = posterior("posterior", use_labels=True)
        elif kind == "score_only":
            T.relabel = posterior("score_only", use_labels=False)
        elif kind == "normclip":
            T.relabel, T.TrainableDQN = normalised, ClippedDQN
        yield
    finally:
        T.relabel, T.TrainableDQN = orig_relabel, orig_dqn


# --------------------------------------------------------------------------- #
# Reward-bias diagnostic: is r_hat the oracle reward in expectation, band by band?
# --------------------------------------------------------------------------- #

def reward_bias(p, calib, om, seed: int = 0, episodes: int = BIAS_EPISODES):
    """Logged random-level behaviour on the training pool. For each reward variant, r_hat
    minus the oracle reward of the same transition, by band of q(B); the posterior's Brier
    score and log-loss against the true class; and the EM share error per episode.

    Random levels are chosen independently of the class, so the EXACT variant -- no score
    evidence, the true share -- is the true posterior given outcomes alone and must be
    unbiased: a failure there is a code bug, not a calibration problem.
    """
    from expkit.posterior_reward import OBS, em_share, posterior_rewards

    rng = random.Random(seed)
    parts, shares = [], []
    for e in range(episodes):
        nh, nb = rng.randint(*C.R.TRAIN_HUMAN_RANGE), rng.choice(BOT_VOLUMES)
        _, tr = C.run_x(RandomLevel(seed * 1000 + e), p.train_h, p.train_b, nb, cache=p.cache,
                        n_humans=nh, seed=rng.randrange(10 ** 9), collect=True,
                        solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
        r = np.array([t.reward for t in tr])
        y = np.array([t.is_bot for t in tr], dtype=float)
        delta = np.array([abs(om.delta[t.action, OBS.index(OBSERVABLE[t.outcome])]) for t in tr])
        labels = _labels(tr, D, random.Random(seed * 1000 + e))
        variants = {
            "exact (no score, true share)": dict(calib=None, share=nb / (nb + nh)),
            "posterior": dict(calib=calib),
            "posterior, Platt only (eps = 0)": dict(calib=dataclasses.replace(calib, eps=0.0)),
            "posterior (floored)": dict(calib=calib.floored()),
            "posterior+labels (floored)": dict(calib=calib.floored(), labels=labels),
            "posterior+labels": dict(calib=calib, labels=labels),
            "score-only": dict(calib=None, mode="score_only"),
        }
        for name, kw in variants.items():
            rh, q = posterior_rewards(tr, om, kw.get("calib"), mode=kw.get("mode", "posterior"),
                                      labels=kw.get("labels"), proxy_cfg=D, share=kw.get("share"),
                                      return_q=True)
            parts.append(pd.DataFrame({"variant": name, "d": rh - r, "q": q, "y": y, "delta": delta}))
        proxy_tr, _ = T.relabel(tr, D, random.Random(seed * 1000 + e))
        parts.append(pd.DataFrame({"variant": "proxy (reference: a different objective)",
                                   "d": np.array([t.reward for t in proxy_tr]) - r, "q": np.nan,
                                   "y": y, "delta": delta}))
        first = {}
        for t in tr:
            first.setdefault(t.user_id, t)
        f = list(first.values())
        ll0 = calib.log_lr([t.state[0] for t in f], [t.state[1] for t in f], [t.step for t in f])
        p_cal = calib.prob([t.state[0] for t in f], [t.state[1] for t in f], [t.step for t in f])
        fl = calib.floored()
        ll0_f = fl.log_lr([t.state[0] for t in f], [t.state[1] for t in f], [t.step for t in f])
        shares.append({"episode": e, "humans": nh, "bots": nb, "true_share": nb / (nb + nh),
                       "em_share": em_share(ll0, pi0=calib.pi_cal),
                       "em_share_floored": em_share(ll0_f, pi0=calib.pi_cal),
                       "summed_probabilities": float(np.mean(p_cal)),
                       "raw_score_mean": float(np.mean([t.state[0] for t in f]))})
    df = pd.concat(parts, ignore_index=True)

    band_rows, summary_rows = [], []
    edges = [(0.0, 0.05), (0.05, UNCERTAIN[0]), UNCERTAIN, (UNCERTAIN[1], 0.95), (0.95, 1.0001)]
    for name, g in df.groupby("variant", sort=False):
        se = g.d.std() / np.sqrt(len(g))
        row = {"variant": name, "n": len(g), "bias": g.d.mean(), "se": se,
               "abs_error_mean": g.d.abs().mean()}
        if g.q.notna().any():
            qq = g.q.clip(1e-6, 1 - 1e-6)
            row["brier"] = float(((qq - g.y) ** 2).mean())
            row["log_loss"] = float(-(g.y * np.log(qq) + (1 - g.y) * np.log(1 - qq)).mean())
            for lo, hi in edges:
                b = g[(g.q >= lo) & (g.q < hi)]
                if not len(b):
                    continue
                band_rows.append({"variant": name, "band": f"[{lo:.2f}, {min(hi, 1):.2f})",
                                  "n": len(b), "bias": b.d.mean(), "se": b.d.std() / np.sqrt(len(b)),
                                  "class_gap": b.delta.mean(),
                                  "skew_ratio": abs(b.d.mean()) / b.delta.mean(),
                                  "true_bot_share": b.y.mean(), "mean_q": b.q.mean()})
        summary_rows.append(row)
    bands, summary = pd.DataFrame(band_rows), pd.DataFrame(summary_rows)
    un = f"[{UNCERTAIN[0]:.2f}, {UNCERTAIN[1]:.2f})"
    verdict = bands[bands.band == un].set_index("variant")["skew_ratio"] <= SKEW_TOLERANCE
    summary["uncertain_band_not_skewed"] = summary.variant.map(verdict)
    ex = summary.set_index("variant").loc["exact (no score, true share)"]
    assert abs(ex.bias) <= 4 * ex.se, (f"the exact posterior is biased ({ex.bias:.3f}, SE {ex.se:.3f}) "
                                       "-- a bug in expkit.posterior_reward, not calibration")
    return bands, summary, pd.DataFrame(shares)


def print_reward_bias(bands, summary, shares):
    print("reward bias, r_hat - oracle r (logged random-level behaviour):")
    print(summary.round(3).to_string(index=False))
    print(f"\nby band of q(B); skew_ratio = |bias| / class reward gap, tolerance {SKEW_TOLERANCE}:")
    print(bands.round(3).to_string(index=False))
    s = shares
    fl = (f"  EM floored {np.mean(np.abs(s.em_share_floored - s.true_share)):.3f}"
          if "em_share_floored" in s else "")
    print(f"\nbot share per episode, mean |error|: EM {np.mean(np.abs(s.em_share - s.true_share)):.3f}{fl}  "
          f"summed calibrated probabilities {np.mean(np.abs(s.summed_probabilities - s.true_share)):.3f}  "
          f"raw score mean {np.mean(np.abs(s.raw_score_mean - s.true_share)):.3f}")


def run_task(task, smoke):
    episodes, eval_seeds, volumes = C.budget(smoke)
    k = task["kind"]
    if k == "observability":
        p = C.pools(final=False, smoke=smoke)              # rl block only
        tr = logged_transitions(p, 0, 2 if smoke else LOG_EPISODES)
        obs = summarise_observability(tr)
        df = pd.DataFrame([{"observation": o, **v} for o, v in obs.items()]).fillna(0.0)
        C.save(df, EXP, "observability", smoke)
        return
    if k == "never":
        p = C.pools(final=True, smoke=smoke, reason="E7 never-challenge baseline")
        df = C.evaluate(C.NeverChallenge(), "never challenge", p.ev_h, p.ev_b, p.cache, eval_seeds,
                        volumes, solve=C.GROUNDED, cfg=C.GROUNDED_CFG, config="never challenge")
        C.save(df, EXP, "never", smoke)
        return
    if k == "reward_bias":
        from expkit.posterior_reward import OutcomeModel
        p = C.pools(final=False, smoke=smoke)              # rl block only
        bands, summary, shares = reward_bias(p, fit_calibration(p),
                                             OutcomeModel(C.GROUNDED, C.GROUNDED_CFG),
                                             episodes=4 if smoke else BIAS_EPISODES)
        d = C.out_dir(EXP, smoke)
        bands.to_csv(d / "e7_reward_bias_bands.csv", index=False)
        summary.to_csv(d / "e7_reward_bias_summary.csv", index=False)
        shares.to_csv(d / "e7_em_share.csv", index=False)
        print_reward_bias(bands, summary, shares)
        return
    spec = dict(CONFIGS[task["config"]])
    hook = spec.pop("hook", None)
    ep = spec.pop("episodes", episodes) if not smoke else episodes
    spec.pop("episodes", None)
    p = C.pools(final=True, smoke=smoke, reason=f"E7 {task['config']} seed {task['seed']}")
    rm = fit_reward_model(p, task["seed"], 2 if smoke else LOG_EPISODES) if hook == "learned_rm" else None
    om = calib = None
    if hook in POSTERIOR_HOOKS:
        from expkit.posterior_reward import OutcomeModel
        om = OutcomeModel(C.GROUNDED, C.GROUNDED_CFG)
        calib = calibration_for(hook, fit_calibration(p) if hook != "score_only" else None)
    with hooked(hook, spec.get("proxy_cfg"), rm, om=om, calib=calib):
        agent, log = C.train_winner(p.train_h, p.train_b, p.cache, seed=task["seed"], episodes=ep,
                                    smoke=smoke, solve=C.GROUNDED, cfg=C.GROUNDED_CFG, **spec)
    df = C.evaluate(agent.eval_mode(), task["config"], p.ev_h, p.ev_b, p.cache, eval_seeds, volumes,
                    solve=C.GROUNDED, cfg=C.GROUNDED_CFG, config=task["config"],
                    train_seed=task["seed"], episodes=ep,
                    labels_fired=int(np.nansum(log.labels_fired)) if log.labels_fired else 0)
    C.save(df, EXP, f"{task['config'].replace(' ', '_').replace('=', '')}_s{task['seed']}", smoke)


def collect(smoke):
    runs = C.load_tasks(EXP, smoke)
    d = C.out_dir(EXP, smoke)
    if "observation" in runs:
        obs = runs[runs.observation.notna()].dropna(axis=1, how="all")
        obs.to_csv(d / "e7_observability.csv", index=False)
        print(f"observability (what the immediate signal says about class):\n{obs.round(3).to_string(index=False)}\n")
    r = runs[runs.get("config").notna() & (runs.bots > 0)] if "config" in runs else pd.DataFrame()
    if not len(r):
        return
    cols = ["DI", "BOS", "SI_F1", "false_positive_rate", "human_survival", "bot_survival", "friction_s"]
    t = r.groupby("config")[cols].mean()
    if "train_seed" in r:
        t["DI_seed_sd"] = r.dropna(subset=["train_seed"]).groupby(["config", "train_seed"]).DI.mean() \
                           .groupby("config").std()
    if {"oracle", "never challenge"} <= set(t.index):
        o, n = t.loc["oracle", "DI"], t.loc["never challenge", "DI"]
        t["ratio_to_oracle"] = t["DI"] / o
        t["gap_closed_over_never"] = (t["DI"] - n) / (o - n) if o != n else np.nan
    per_seed = r.dropna(subset=["train_seed"]).groupby(["config", "train_seed"])[cols].mean() \
        if "train_seed" in r else pd.DataFrame()
    if len(per_seed):
        se = per_seed.groupby("config").sem()
        for c in ("DI", "human_survival", "bot_survival", "friction_s"):
            t[f"{c}_se"] = se[c]
    t = t.sort_values("DI", ascending=False)
    t.to_csv(d / "e7_configs.csv")
    print(t.round(3).to_string())
    if len(per_seed):
        h2h = head_to_head(per_seed)
        if len(h2h):
            h2h.to_csv(d / "e7_head_to_head.csv", index=False)
            print("\nhead to head (difference of seed means, Welch t-test over training seeds):")
            print(h2h.round(3).to_string(index=False))
    if (d / "e7_reward_bias_summary.csv").exists():
        print()
        print_reward_bias(*(pd.read_csv(d / f) for f in ("e7_reward_bias_bands.csv",
                                                         "e7_reward_bias_summary.csv", "e7_em_share.csv")))


HEAD_TO_HEAD = (("posterior+labels", "posterior"), ("posterior", "oracle"),
                ("posterior+labels", "oracle"), ("posterior", "score-only"),
                ("posterior", "proxy"), ("learned rm", "proxy"))


def head_to_head(per_seed, metrics=("DI", "human_survival", "bot_survival", "friction_s",
                                    "false_positive_rate")):
    from scipy.stats import ttest_ind

    have = set(per_seed.index.get_level_values("config"))
    rows = []
    for a, b in HEAD_TO_HEAD:
        if not {a, b} <= have:
            continue
        for m in metrics:
            xa, xb = per_seed.loc[a][m].to_numpy(), per_seed.loc[b][m].to_numpy()
            ok = len(xa) > 1 and len(xb) > 1
            rows.append({"a": a, "b": b, "metric": m, "a_mean": xa.mean(), "b_mean": xb.mean(),
                         "diff": xa.mean() - xb.mean(),
                         "p_value": ttest_ind(xa, xb, equal_var=False).pvalue if ok else np.nan,
                         "seeds": f"{len(xa)}/{len(xb)}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
