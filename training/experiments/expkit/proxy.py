"""A deployable reward: what an operator can actually observe (Reviewer 1, point 1).

The reviewer's sharpest objection has no textual answer. The simulator's reward reads
``user.is_bot`` and ``user.bot_strength``; a deployed controller knows neither at
decision time. So either the reward is an oracle the system cannot have, or the paper
must show that a reward built only from observable signals gets you to a comparable
policy.

WHAT A DEPLOYMENT ACTUALLY SEES
-------------------------------
Immediately, per challenge:

* which level was presented,
* whether the challenge was completed, failed, or timed out,
* how long it took,
* whether the session continued or ended.

Crucially, the immediate observation is *ambiguous about class*:

    challenge failed, session ends   <- a blocked bot AND a false-positived human
    challenge passed, session goes on <- a leaked bot AND a satisfied human

``relabel`` assigns reward as a pure function of the observable tuple (level, observable
outcome, seconds), so the two members of each pair receive identical immediate reward
whenever they spent the same time. Time is observable and does carry some class
information -- automated solvers answer text in about a second (humans ~7 s) and image
grids in about 15 s (humans ~10 s), `expkit.stochastic.BOT_SECONDS` -- as much as real
solve times carry, and no more. Before round 3 simulated bots spent 0 s, which made the time a perfect
class label and paid a leaked bot (+25) more than a satisfied human (25 - friction).

Later, and only sometimes:

* a confirmed-abuse label (chargeback, account-takeover review, spam report). It arrives
  for only a fraction of the sessions that truly warranted it (`label_probability`),
  arrives with a delay, and is occasionally wrong (`label_noise`). This is the only
  channel through which class information ever reaches the agent, which matches how
  fraud-detection systems are trained in practice -- sparse, delayed, noisy confirmation
  rather than ground truth at decision time.

Credit for a delayed label is spread over that session's last `label_delay_steps`
transitions, the standard fix for delayed-reward credit assignment in this setting.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

#: outcome -> what the operator observes. The mapping is many-to-one on purpose.
OBSERVABLE = {
    "human_passed": "passed_continued",
    "bot_leaked": "passed_continued",
    "human_abandoned": "passed_left",
    "human_false_positive": "failed_gone",
    "bot_blocked": "failed_gone",
}


@dataclass(frozen=True)
class ProxyRewardConfig:
    """Weights over observable signals. Nothing here reads a class label."""

    zero_friction_reward: float = 50.0     # no challenge shown, session continues
    engagement_reward: float = 25.0        # challenge passed, session continues
    friction_per_second: float = 2.0       # measured latency cost
    abandon_penalty: float = -60.0         # user left right after a challenge
    failed_gone_reward: float = 5.0        # challenge failed and the session ended:
                                           # usually a blocked bot, sometimes a lost
                                           # customer. Small and positive by default.
    # --- delayed, noisy confirmation -------------------------------------- #
    abuse_penalty: float = -150.0
    label_probability: float = 0.30        # P(a true abuse session is ever confirmed)
    label_noise: float = 0.02              # P(a legitimate session is wrongly flagged)
    label_delay_steps: int = 3             # credit window at the end of the session


def immediate_reward(action: int, outcome: str, seconds: float,
                     cfg: ProxyRewardConfig) -> float:
    """Reward from observables alone. `outcome` is mapped through OBSERVABLE first, so a
    blocked bot and a false-positived human who spent the same `seconds` are scored
    identically; class enters only through what the time itself reveals."""
    obs = OBSERVABLE[outcome]
    friction = cfg.friction_per_second * seconds

    if obs == "passed_continued":
        return cfg.zero_friction_reward if action == 0 else cfg.engagement_reward - friction
    if obs == "passed_left":
        return cfg.abandon_penalty - friction
    return cfg.failed_gone_reward - friction    # failed_gone


def relabel(transitions, cfg: ProxyRewardConfig, rng: random.Random):
    """Return a NEW transition list whose rewards use only observable information.

    `is_bot` is read in exactly one place -- to decide whether a *delayed abuse label*
    would have fired -- and the label is then subjected to `label_probability` and
    `label_noise` before the agent sees anything. The agent never reads `is_bot`.
    """
    from copy import copy

    out = [copy(t) for t in transitions]
    for t in out:
        t.reward = immediate_reward(t.action, t.outcome, t.seconds, cfg)

    by_user: dict[int, list] = defaultdict(list)
    for i, t in enumerate(out):
        by_user[t.user_id].append(i)

    n_labels = 0
    for uid, idxs in by_user.items():
        last = out[idxs[-1]]
        leaked = last.is_bot and last.outcome != "bot_blocked"
        fires = (rng.random() < cfg.label_probability) if leaked else (
            rng.random() < cfg.label_noise)
        if not fires:
            continue
        n_labels += 1
        window = idxs[-cfg.label_delay_steps:]
        share = cfg.abuse_penalty / len(window)
        for i in window:
            out[i].reward += share

    return out, n_labels


def summarise_observability(transitions) -> dict:
    """How much class information the immediate signal carries -- the number to quote.

    For each observable class, the share of the two true outcomes behind it. A value
    near 0.5 means the immediate signal is uninformative on its own.
    """
    from collections import Counter

    counts = defaultdict(Counter)
    for t in transitions:
        counts[OBSERVABLE[t.outcome]][t.outcome] += 1

    out = {}
    for obs, c in counts.items():
        total = sum(c.values())
        out[obs] = {"n": total, **{k: v / total for k, v in c.items()}}
    return out
