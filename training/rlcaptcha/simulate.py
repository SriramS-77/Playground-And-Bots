"""The evaluation loop.

One function replaces the five near-identical 130-170 line cells that were duplicated
across the original benchmark notebooks. The policy-specific behaviour that used to be
expressed by commenting lines in and out is now the Policy object.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import EVAL_BOT_COUNTS, EVAL_HUMANS, RewardConfig
from .data import Session, build_population
from .rewards import reward as compute_reward
from .scoring import ScoreCache


@dataclass
class SimResult:
    """Outcome of one simulation at one bot volume."""

    n_humans: int
    n_bots: int
    surviving_humans: int
    surviving_bots: int
    human_step_reward: float
    bot_step_reward: float
    human_session_reward: float
    bot_session_reward: float
    timesteps: list[int] = field(default_factory=list)
    human_counts: list[int] = field(default_factory=list)
    bot_counts: list[int] = field(default_factory=list)


def run_simulation(
    policy,
    humans: list[Session],
    bots: list[Session],
    n_bots: int,
    cache: ScoreCache | None = None,
    n_humans: int = EVAL_HUMANS,
    rewards: RewardConfig | None = None,
    seed: int | None = None,
) -> SimResult:
    """Run one episode: `n_humans` humans and `n_bots` bots until everyone leaves.

    Survivors are the users still active at the start of the final timestep -- the same
    definition the original notebooks printed as "Final remaining humans/bots".
    """
    rng = random.Random(seed)
    rewards = rewards or RewardConfig()
    if policy.needs_bot_score and cache is None:
        raise ValueError(f"{policy.name} needs a ScoreCache")

    active = build_population(humans, bots, n_humans, n_bots, rng)
    for user in active:
        user.last_threat_level = policy.initial_last_threat

    h_total = b_total = 0.0
    h_steps = b_steps = 0
    surviving_humans = n_humans
    surviving_bots = n_bots
    timesteps, human_counts, bot_counts = [], [], []
    t = 0

    while active:
        human_counts.append(sum(1 for u in active if not u.is_bot))
        bot_counts.append(sum(1 for u in active if u.is_bot))
        timesteps.append(t)
        t += 1

        n_active = len(active)
        survivors = []

        for user in active:
            # Advances the clock; returns None (and marks `exhausted`) at end of session.
            chunk = user.next_chunk()
            if policy.needs_bot_score:
                # No chunk means no observation, so the policy falls back to the running
                # average. Matters only for the bandits, which are the policies that act
                # on exhausted users.
                user.observe(None if chunk is None else cache.score(user))

            if user.exhausted and not policy.acts_when_exhausted:
                continue

            action = policy.select_action(user, n_active)
            r, terminated = compute_reward(user, action, rewards, rng)

            if user.is_bot:
                b_total += r
                b_steps += 1
            else:
                h_total += r
                h_steps += 1

            if not (user.exhausted or terminated):
                if policy.updates_last_threat:
                    user.last_threat_level = action
                survivors.append(user)

        if not survivors:
            surviving_humans = sum(1 for u in active if not u.is_bot)
            surviving_bots = sum(1 for u in active if u.is_bot)
        active = survivors

    return SimResult(
        n_humans=n_humans,
        n_bots=n_bots,
        surviving_humans=surviving_humans,
        surviving_bots=surviving_bots,
        human_step_reward=h_total / h_steps if h_steps else 0.0,
        bot_step_reward=b_total / b_steps if b_steps else 0.0,
        human_session_reward=h_total / n_humans if n_humans else 0.0,
        bot_session_reward=b_total / n_bots if n_bots else 0.0,
        timesteps=timesteps,
        human_counts=human_counts,
        bot_counts=bot_counts,
    )


def run_sweep(
    policy,
    humans: list[Session],
    bots: list[Session],
    cache: ScoreCache | None = None,
    bot_counts=EVAL_BOT_COUNTS,
    rewards: RewardConfig | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[int, SimResult]:
    """Run every simulation type for one policy -- i.e. one row of Table 4."""
    out = {}
    for i, n_bots in enumerate(bot_counts):
        out[n_bots] = run_simulation(
            policy, humans, bots, n_bots,
            cache=cache, rewards=rewards,
            seed=None if seed is None else seed + i,
        )
        if verbose:
            r = out[n_bots]
            print(
                f"  {policy.name:24s} {n_bots:>4} bots -> "
                f"humans {r.surviving_humans:3d} | bots {r.surviving_bots:4d}"
            )
    return out
