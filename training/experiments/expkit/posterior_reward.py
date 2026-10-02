"""Posterior-weighted expected reward -- the oracle reward in expectation over what an
operator can believe about each user (E7's posterior variants; Reviewer 1, point 1).

A deployment never sees the class or the bot strength, but it can compute

    r_hat_k = q_k(H) * r_H(T_k, o_k)  +  sum_b q_k(B, b) * r_B(b, T_k, o_k)

where q_k is the posterior over (class, strength) given everything observable up to and
including the outcome o_k of step k, and r_H, r_B are the oracle reward's components.
If q_k is the true posterior and the policy acts only on observables, then by the tower
property E[r_hat_k] = E[r_k] for every step and every such policy, so the optimal policy
is the oracle's and r_hat has lower variance than r (Rao-Blackwell).

The posterior (normalised over H and B x {0..9}):

    q_k(H)    oc (1 - pi)               * prod_{j<=k} P_H(o_j | T_j)            [* P(label | H)]
    q_k(B, b) oc   pi * Lambda_k * rho(b) * prod_{j<=k} P_B(o_j | T_j, b)       [* P(label | B, history)]

* pi        bot share of the window, estimated per episode by EM on the users' first-step
            score evidence (Saerens, Latinne & Decaestecker, Neural Computation 2002) -- the
            scorer's own training share is not the deployment's.
* Lambda_k  p(score evidence | B) / p(score evidence | H): a calibration WITH an intercept
            (Platt scaling; for two classes this is Alexandari et al.'s bias-corrected
            temperature scaling, ICML 2020) fitted on the scorer block's out-of-fold scores,
            divided by that block's prior odds. Its inputs are a superset of what the policy
            sees: the latest chunk score, the running average, and the step. ONE session-so-far
            likelihood ratio per step -- never a product over chunks, which are correlated.
            Then bounded by a recording-level contamination eps (`robust_log_lr`): a whole
            recording can look like the other class. eps and Platt's C are chosen by
            leave-one-recording-out CV on the calibration set. Added after the reward-bias
            diagnostic traced the uncapped posterior's skew to two confidently misscored
            human recordings (scores 0.756 and 0.987); the parameter is fitted only on the
            scorer block, never on the sessions it is applied to. On fold 2 the CV picks
            eps = 0 (no calibration recording is confidently wrong, and CV cannot see a rate
            below ~1/n), so `ScoreCalibration.floored` adds per-class Laplace floors
            eps_c = 1 / (n_c + 2) -- the "(floored)" variants, a declared sensitivity analysis.
* rho(b)    strength distribution, U{0..9} as the simulator draws it. A deployment would
            estimate it from logs (marginal ML by EM, Bock & Aitkin 1981).
* P_H, P_B  the simulator's solve and abandonment model (`OutcomeModel`). Exact here, so
            the variant is an UPPER BOUND on what the idea can do; a deployment estimates
            them from randomised challenges.
* label     the delayed, sparse, noisy abuse label, used as EVIDENCE -- it fires with
            `label_probability` for a bot whose session did not end in a failed challenge,
            `label_noise` otherwise -- never as an extra penalty, which would double count
            what r_hat already charges. A later label may condition an earlier step's reward:
            the policy never saw it, so unbiasedness is unaffected.

Solve time is not evidence. Human friction is charged at its EXPECTED value given level
and outcome, so an adversary cannot move the reward by controlling how long it takes.

The score-only ablation is the idea "as is": q(B) = the raw latest chunk score, no prior
correction, no outcome evidence, strength at its prior. It exists to show what the
corrections are for.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .proxy import OBSERVABLE
from .stochastic import N_LEVELS

STRENGTHS = np.arange(10)
OBS = ("passed_continued", "passed_left", "failed_gone")
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# Outcome likelihoods and the oracle's reward components
# --------------------------------------------------------------------------- #

class OutcomeModel:
    """P(observable outcome | class, strength, level) and the oracle's per-class rewards,
    derived from a `SolveModel` and a `StochasticRewardConfig` -- the same objects the
    simulator draws from, so nothing here is re-typed by hand."""

    def __init__(self, solve, cfg, strength_prior=None):
        if cfg.abandonment_applies_to_bots:
            raise NotImplementedError("bots that abandon need a bot leave model here")
        self.rho = np.full(len(STRENGTHS), 1.0 / len(STRENGTHS)) if strength_prior is None \
            else np.asarray(strength_prior, dtype=float)
        L = N_LEVELS
        self.p_h = np.zeros((L, 3))                 # P_H(obs | T)
        self.r_h = np.zeros((L, 3))                 # r_H(T, obs), expected friction
        self.p_b = np.zeros((len(STRENGTHS), L, 3))  # P_B(obs | b, T)
        self.r_b = np.zeros((len(STRENGTHS), L, 3))  # r_B(b, T, obs)
        for T in range(L):
            self._human(T, solve, cfg)
            for b in STRENGTHS:
                g = solve.bot_pass_probability(int(b), T)
                self.p_b[b, T] = (g, 0.0, 1.0 - g)
                leak = cfg.leakage_penalty - max(0, b - T) * cfg.underestimation
                block = cfg.max_reward - max(0, T - b) * cfg.overkill
                self.r_b[b, T] = (leak, leak, block)
        self.delta = self.r_h - np.einsum("b,bto->to", self.rho, self.r_b)   # r_H - E_rho r_B

    def _human(self, T, solve, cfg):
        """Mirror of `SolveModel.human_outcome` + the human branch of `stochastic_reward`."""
        s = solve.human_seconds[T]
        if T == 0:
            branches = [(1.0, True, 0.0)]                      # (prob, passed, seconds)
        else:
            p = solve.human_pass_probability(T)
            if p >= 1.0:
                branches = [(1.0, True, s)]
            else:
                A = solve.max_human_attempts
                branches = [((1 - p) ** (a - 1) * p, True, a * s) for a in range(1, A + 1)]
                branches.append(((1 - p) ** A, False, A * s))
        mass = np.zeros(3)
        fric = np.zeros(3)
        for prob, passed, sec in branches:
            f = cfg.friction(T, sec)
            if not passed:
                mass[2] += prob
                fric[2] += prob * f
                continue
            leave = cfg.leave_probability(T, sec)
            mass[0] += prob * (1 - leave)
            fric[0] += prob * (1 - leave) * f
            mass[1] += prob * leave
            fric[1] += prob * leave * f
        ef = np.divide(fric, mass, out=np.zeros(3), where=mass > 0)
        self.p_h[T] = mass
        pass_r = cfg.max_reward if T == 0 else cfg.max_reward / 2.0
        self.r_h[T] = (pass_r - (0.0 if T == 0 else ef[0]),
                       pass_r - (0.0 if T == 0 else ef[1]),
                       cfg.false_positive_penalty - ef[2])


# --------------------------------------------------------------------------- #
# Score evidence: one calibrated likelihood ratio per step
# --------------------------------------------------------------------------- #

def _features(latest, avg, step):
    la, lv = _logit(latest), _logit(avg)
    s = np.asarray(step, dtype=float) / 11.0
    return np.column_stack([la, lv, s, lv * s])


def robust_log_lr(log_lr, eps_h: float, eps_b: float | None = None):
    """Recording-level contamination: a human recording behaves like a bot's with
    probability `eps_h`, a bot recording like a human's with probability `eps_b` (default
    `eps_h`). Then p(x|B) = (1-eps_b) p_B + eps_b p_H and p(x|H) = (1-eps_h) p_H + eps_h p_B:

        Lambda_robust = ((1 - eps_b) Lambda + eps_b) / ((1 - eps_h) + eps_h Lambda),

    a proper likelihood ratio bounded in [eps_b / (1 - eps_h), (1 - eps_b) / eps_h]: the
    score can never be more certain than its out-of-sample record, so challenge outcomes
    and labels can still overturn it."""
    eps_b = eps_h if eps_b is None else eps_b
    ll = np.asarray(log_lr, dtype=float)
    if eps_h <= 0 and eps_b <= 0:
        return ll
    lam = np.exp(np.clip(ll, -50, 50))
    return np.log(((1 - eps_b) * lam + eps_b) / ((1 - eps_h) + eps_h * lam))


@dataclass
class ScoreCalibration:
    """P(bot | latest chunk score, running average, step), calibrated at `pi_cal`, with the
    recording-level contamination `eps` applied to the likelihood ratio."""

    coef: np.ndarray
    intercept: float
    pi_cal: float
    n_sessions: int
    n_rows: int
    n_humans: int = 0
    n_bots: int = 0
    eps: float = 0.0                # CV-chosen, symmetric
    C: float = 1.0
    cv: list | None = None          # [{"C", "eps", "log_loss"}] from leave-one-recording-out
    eps_h: float | None = None      # per-class contamination actually applied; None = eps
    eps_b: float | None = None

    def prob(self, latest, avg, step):
        """Platt-calibrated P(bot) at `pi_cal`, BEFORE the contamination term."""
        z = _features(np.atleast_1d(latest), np.atleast_1d(avg), np.atleast_1d(step)) @ self.coef
        return 1.0 / (1.0 + np.exp(-(z + self.intercept)))

    def log_lr(self, latest, avg, step):
        """log p(evidence | B) / p(evidence | H): the posterior odds minus the prior odds of
        the set the calibration was fitted on, then bounded by the contamination terms."""
        p = np.clip(self.prob(latest, avg, step), _EPS, 1 - _EPS)
        raw = np.log(p / (1 - p)) - math.log(self.pi_cal / (1 - self.pi_cal))
        eh = self.eps if self.eps_h is None else self.eps_h
        eb = self.eps if self.eps_b is None else self.eps_b
        return robust_log_lr(raw, eh, eb)

    def floored(self) -> "ScoreCalibration":
        """Per-class Laplace floors on the contamination: eps_c >= 1 / (n_c + 2), the
        posterior-mean rate of an event never seen in the n_c calibration recordings of
        class c. Leave-one-recording-out CV cannot estimate a rate below ~1/n when no
        calibration recording is confidently wrong, so it returns eps = 0 -- a property of
        the estimator, not evidence that the scorer never errs. Declared after the
        reward-bias diagnostic exposed tail errors CV cannot see; uses only the calibration
        set's class counts. Laplace (a point estimate), not the rule of three (an upper
        bound, which would cap the score far harder)."""
        import dataclasses
        return dataclasses.replace(self, eps_h=max(self.eps, 1.0 / (self.n_humans + 2)),
                                   eps_b=max(self.eps, 1.0 / (self.n_bots + 2)))

    def bounds(self):
        """(smallest, largest) likelihood ratio the score can contribute."""
        eh = self.eps if self.eps_h is None else self.eps_h
        eb = self.eps if self.eps_b is None else self.eps_b
        if eh <= 0 and eb <= 0:
            return 0.0, math.inf
        return eb / (1 - eh), ((1 - eb) / eh if eh > 0 else math.inf)

    @staticmethod
    def _platt(feats, y, w, C):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=C, max_iter=5000).fit(feats, y, sample_weight=w)

    @classmethod
    def fit(cls, rows, Cs=(0.03, 0.1, 0.3, 1.0), epss=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2)):
        """`rows`: (step, latest, avg, is_bot, session_key) from `replay_scores`.

        Platt's regularisation C and the contamination eps are chosen TOGETHER by
        leave-one-recording-out cross-validation: each recording's rows are predicted by a
        calibration that never saw that recording, and candidates are scored by session-
        weighted log-loss. Rows of one recording are correlated, so leaving out single
        rows would certify a confidence the scorer has not earned.
        """
        step, latest, avg, y = (np.array([r[i] for r in rows]) for i in range(4))
        y = y.astype(int)
        keys = np.array([r[4] for r in rows])
        per = defaultdict(int)
        for k in keys:
            per[k] += 1
        # One row per step per recording; weight recordings equally so the calibration
        # prior is the RECORDING share, not the share of recording-steps.
        w = np.array([1.0 / per[k] for k in keys])
        feats = _features(latest, avg, step)
        uniq = sorted(set(keys))

        cv = []
        for C in Cs:
            loss = {e: 0.0 for e in epss}
            for k in uniq:
                tr, te = keys != k, keys == k
                m = cls._platt(feats[tr], y[tr], w[tr], C)
                pi = float(np.sum(w[tr] * y[tr]) / np.sum(w[tr]))
                p = np.clip(m.predict_proba(feats[te])[:, 1], _EPS, 1 - _EPS)
                raw = np.log(p / (1 - p)) - math.log(pi / (1 - pi))
                for e in epss:
                    post = 1.0 / (1.0 + np.exp(-(robust_log_lr(raw, e) + math.log(pi / (1 - pi)))))
                    post = np.clip(post, _EPS, 1 - _EPS)
                    yy = y[te]
                    loss[e] += float(-np.mean(yy * np.log(post) + (1 - yy) * np.log(1 - post)))
            cv += [{"C": C, "eps": e, "log_loss": loss[e] / len(uniq)} for e in epss]
        best = min(cv, key=lambda r: r["log_loss"])
        m = cls._platt(feats, y, w, best["C"])
        sess_y = {k: yy for k, yy in zip(keys, y)}
        n_b = int(sum(sess_y.values()))
        return cls(coef=m.coef_[0].astype(float), intercept=float(m.intercept_[0]),
                   pi_cal=float(np.mean(list(sess_y.values()))), n_sessions=len(sess_y),
                   n_rows=len(rows), n_humans=len(sess_y) - n_b, n_bots=n_b,
                   eps=float(best["eps"]), C=float(best["C"]), cv=cv)


def replay_scores(humans, bots, cache, max_steps: int = 12):
    """(step, latest, avg, is_bot, key) exactly as the simulator exposes them: `User`'s own
    `next_chunk` / `observe`, including the empty-chunk fallback and the initial 0.5."""
    from rlcaptcha.data import User

    rows = []
    for sessions, is_bot in ((humans, False), (bots, True)):
        for s in sessions:
            u = User(session=s, is_bot=is_bot, bot_strength=0 if is_bot else None)
            for step in range(max_steps):
                chunk = u.next_chunk()
                if chunk is None:
                    break
                u.observe(cache.score(u))
                rows.append((step, u.bot_score, u.avg_bot_score, is_bot, s.key))
    return rows


def em_share(log_lr, pi0: float = 0.5, iters: int = 1000, tol: float = 1e-9) -> float:
    """Maximum-likelihood bot share of a window from per-user likelihood ratios (EM)."""
    lam = np.exp(np.clip(np.asarray(log_lr, dtype=float), -50, 50))
    pi = float(np.clip(pi0, 1e-4, 1 - 1e-4))
    for _ in range(iters):
        w = pi * lam / (pi * lam + 1 - pi)
        new = float(np.clip(w.mean(), 1e-4, 1 - 1e-4))
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


# --------------------------------------------------------------------------- #
# The reward
# --------------------------------------------------------------------------- #

def _lse(a):
    m = np.max(a)
    return m if not np.isfinite(m) else m + math.log(np.exp(a - m).sum())


def update_strength_prior(om: OutcomeModel, strength_out: dict, eta: float,
                          floor: float = 0.01, min_bot_mass: float = 1.0) -> bool:
    """One online-EM step on the strength distribution (Cappe & Moulines 2009 style):
    rho <- (1 - eta) rho + eta * rho_episode, where rho_episode is the bot-weighted mean of
    the users' final posteriors over strength. Causal: call it AFTER the episode's rewards
    were computed with the old rho. Floored and renormalised, so no strength ever gets zero
    mass (which would make outcomes impossible). Skipped when the episode holds less than
    `min_bot_mass` expected bots. Returns whether it updated."""
    if not strength_out:
        return False
    mass = np.sum(list(strength_out.values()), axis=0)
    if mass.sum() < min_bot_mass:
        return False
    rho = (1 - eta) * om.rho + eta * mass / mass.sum()
    rho = np.maximum(rho, floor)
    om.rho = rho / rho.sum()
    return True


def posterior_rewards(transitions, om: OutcomeModel, calib: ScoreCalibration | None,
                      mode: str = "posterior", labels: dict | None = None, proxy_cfg=None,
                      share: float | None = None, return_q: bool = False,
                      strength_out: dict | None = None):
    """Expected oracle reward per transition, aligned with `transitions`.

    mode      "posterior" (score + outcomes + EM share) or "score_only" (the ablation).
    calib     None switches the score evidence off (Lambda = 1) -- the exactness test.
    labels    {user_id: fired} to condition on the delayed abuse label, or None.
    share     overrides the EM share (the exactness test passes the true one).
    return_q  also return q(B) per transition.
    strength_out  if a dict, filled with each user's final posterior over (B, b) -- the
              E-step for `update_strength_prior`.
    """
    if mode not in ("posterior", "score_only"):
        raise ValueError(mode)
    by_user = defaultdict(list)
    for i, t in enumerate(transitions):
        by_user[t.user_id].append(i)
    for idxs in by_user.values():
        idxs.sort(key=lambda i: transitions[i].step)

    pi = None
    if mode == "posterior":
        if share is not None:
            pi = share
        elif calib is None:
            raise ValueError("no score evidence and no share: nothing to estimate the share from")
        else:
            first = [transitions[idxs[0]] for idxs in by_user.values()]
            ll0 = calib.log_lr([t.state[0] for t in first], [t.state[1] for t in first],
                               [t.step for t in first])
            pi = em_share(ll0, pi0=calib.pi_cal)
        pi = float(np.clip(pi, 1e-4, 1 - 1e-4))

    log_rho = np.log(om.rho)
    rewards = np.zeros(len(transitions))
    qb = np.zeros(len(transitions))
    for uid, idxs in by_user.items():
        ts = [transitions[i] for i in idxs]
        if mode == "score_only":
            for i, t in zip(idxs, ts):
                o = OBS.index(OBSERVABLE[t.outcome])
                q = float(np.clip(t.state[0], 0.0, 1.0))
                rewards[i] = (1 - q) * om.r_h[t.action, o] + q * float(om.rho @ om.r_b[:, t.action, o])
                qb[i] = q
            continue

        lab_h = lab_b = 0.0
        if labels is not None:
            fired = bool(labels[uid])
            ended_failed = OBSERVABLE[ts[-1].outcome] == "failed_gone"
            p_h = proxy_cfg.label_noise
            p_b = proxy_cfg.label_noise if ended_failed else proxy_cfg.label_probability
            lab_h = math.log(p_h if fired else 1 - p_h)
            lab_b = math.log(p_b if fired else 1 - p_b)

        hist_h, hist_b = 0.0, np.zeros(len(STRENGTHS))
        lam = (calib.log_lr([t.state[0] for t in ts], [t.state[1] for t in ts], [t.step for t in ts])
               if calib is not None else np.zeros(len(ts)))
        with np.errstate(divide="ignore"):
            for k, (i, t) in enumerate(zip(idxs, ts)):
                o = OBS.index(OBSERVABLE[t.outcome])
                hist_h += math.log(om.p_h[t.action, o]) if om.p_h[t.action, o] > 0 else -math.inf
                hist_b = hist_b + np.log(om.p_b[:, t.action, o])
                a = math.log(1 - pi) + hist_h + lab_h
                bvec = math.log(pi) + lam[k] + log_rho + hist_b + lab_b
                z = _lse(np.append(bvec, a))
                if not np.isfinite(z):              # impossible under the model: fall back
                    a, bvec, z = math.log(1 - pi), math.log(pi) + log_rho, 0.0
                    z = _lse(np.append(bvec, a))
                q_h = math.exp(a - z)
                q_b = np.exp(bvec - z)
                rewards[i] = q_h * om.r_h[t.action, o] + float(q_b @ om.r_b[:, t.action, o])
                qb[i] = float(q_b.sum())
        if strength_out is not None:
            strength_out[uid] = q_b                 # final posterior over (B, b), mass = q(B)
    return (rewards, qb) if return_q else rewards
