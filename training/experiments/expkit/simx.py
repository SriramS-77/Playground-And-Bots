"""Extended simulation loop.

`rlcaptcha.simulate.run_simulation` is faithful to the published code and is left alone.
This is a superset used by the new experiments: it accepts an arbitrary session pool, a
probabilistic action model, a proxy reward, and it records per-step transitions so the
same loop can both evaluate and generate training data.

With ``solve=DETERMINISTIC`` and ``cfg=PAPER_EQUIVALENT`` it produces the same survivor
counts as `rlcaptcha.simulate.run_simulation` -- checked in notebook 06.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rlcaptcha.config import MAX_CHUNKS, SESSION_CHUNK_MS
from rlcaptcha.data import Session, User, _split_into_chunks

from .stochastic import (
    DETERMINISTIC,
    PAPER_EQUIVALENT,
    SolveModel,
    StochasticRewardConfig,
    stochastic_reward,
)


# --------------------------------------------------------------------------- #
# Session pools from a partition
# --------------------------------------------------------------------------- #

#: Parsed-and-chunked sessions, memoised by file path.
#:
#: The bootstrap in `expkit.bootstrap` resamples the pool hundreds of times and calls
#: this on every draw. Without the memo that is tens of thousands of redundant JSON
#: reads and re-chunkings. A Session is never mutated -- `User` holds a reference and
#: tracks its own chunk index -- so one object can safely back many simulated users,
#: which is the same assumption the score cache already relies on.
_SESSION_MEMO: dict[str, Session] = {}


def _load_one(ref) -> Session | None:
    cached = _SESSION_MEMO.get(ref.path)
    if cached is not None:
        return cached
    raw = json.loads(Path(ref.path).read_text())
    if not raw.get("mouse_movements"):
        return None
    session = Session(
        key=ref.name,
        is_bot=ref.is_bot,
        chunks=_split_into_chunks(raw, SESSION_CHUNK_MS, MAX_CHUNKS),
    )
    _SESSION_MEMO[ref.path] = session
    return session


def sessions_from_refs(refs) -> tuple[list[Session], list[Session]]:
    """Load SessionRefs into rlcaptcha Session objects, split into (humans, bots).

    Duplicate refs -- which a bootstrap resample produces by design -- yield the same
    Session object repeated, so a recording drawn three times is genuinely three
    entries in the pool and is three times as likely to be sampled.
    """
    humans, bots = [], []
    for ref in refs:
        session = _load_one(ref)
        if session is None:
            continue
        (bots if ref.is_bot else humans).append(session)
    if not humans or not bots:
        raise RuntimeError("pool needs both human and bot sessions")
    return humans, bots


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class XSimResult:
    """Superset of rlcaptcha.simulate.SimResult, plus outcome accounting."""

    n_humans: int
    n_bots: int
    surviving_humans: int
    surviving_bots: int
    human_step_reward: float = 0.0
    bot_step_reward: float = 0.0
    human_session_reward: float = 0.0
    bot_session_reward: float = 0.0
    timesteps: list = field(default_factory=list)
    human_counts: list = field(default_factory=list)
    bot_counts: list = field(default_factory=list)
    outcomes: Counter = field(default_factory=Counter)
    human_seconds: float = 0.0
    challenges_issued: int = 0
    action_hist: Counter = field(default_factory=Counter)

    @property
    def false_positive_rate(self) -> float:
        """Legitimate users blocked because they failed a challenge."""
        return self.outcomes["human_false_positive"] / self.n_humans if self.n_humans else 0.0

    @property
    def abandonment_rate(self) -> float:
        return self.outcomes["human_abandoned"] / self.n_humans if self.n_humans else 0.0

    @property
    def mean_human_friction_seconds(self) -> float:
        return self.human_seconds / self.n_humans if self.n_humans else 0.0


@dataclass
class Transition:
    """One (s, a, r, s', done) plus the labels a *simulator* knows and a deployment
    does not. `is_bot` is used by the balanced replay buffer and by the oracle reward;
    the proxy-reward agent is never allowed to read it."""

    state: list
    action: int
    reward: float
    next_state: list
    done: bool
    is_bot: bool
    user_id: int
    step: int
    outcome: str
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

def run_x(
    policy,
    humans: list[Session],
    bots: list[Session],
    n_bots: int,
    cache=None,
    n_humans: int = 100,
    solve: SolveModel = DETERMINISTIC,
    cfg: StochasticRewardConfig = PAPER_EQUIVALENT,
    seed: int | None = None,
    collect: bool = False,
    human_skill_sd: float = 0.0,
    seed_global: bool = True,
) -> tuple[XSimResult, list[Transition]]:
    """One episode. Returns (result, transitions).

    `collect=True` records transitions for training; the state vector stored is whatever
    `policy.features(user, n_active)` returns, so a trainer can reuse it directly.
    """
    rng = random.Random(seed)
    if seed_global and seed is not None:
        # Two policies in the published code draw from GLOBAL generators that the `seed`
        # argument never reached: ThompsonPolicy via `torch.randn_like`, and DQNPolicy's
        # epsilon-greedy via `np.random.rand`. Runs of those two were therefore not
        # reproducible from their seed alone. Seeding both here closes that.
        import numpy as _np
        import torch as _torch
        _np.random.seed(seed % (2 ** 32 - 1))
        _torch.manual_seed(seed)
    if policy.needs_bot_score and cache is None:
        raise ValueError(f"{policy.name} needs a ScoreCache")

    users = [
        User(session=rng.choice(humans), is_bot=False, bot_strength=None,
             jitter=(rng.randint(-1, 1), rng.randint(-1, 1)))
        for _ in range(n_humans)
    ]
    users += [
        User(session=rng.choice(bots), is_bot=True, bot_strength=rng.randint(0, 9),
             jitter=(rng.randint(-1, 1), rng.randint(-1, 1)))
        for _ in range(n_bots)
    ]
    rng.shuffle(users)
    for i, u in enumerate(users):
        u.last_threat_level = policy.initial_last_threat
        u.skill = rng.gauss(0.0, human_skill_sd) if human_skill_sd else 0.0
        u.uid = i

    active = users
    result = XSimResult(n_humans=n_humans, n_bots=n_bots,
                        surviving_humans=n_humans, surviving_bots=n_bots)
    transitions: list[Transition] = []
    pending: dict[int, tuple] = {}
    h_total = b_total = 0.0
    h_steps = b_steps = 0
    t = 0

    while active:
        result.human_counts.append(sum(1 for u in active if not u.is_bot))
        result.bot_counts.append(sum(1 for u in active if u.is_bot))
        result.timesteps.append(t)
        t += 1

        n_active = len(active)
        survivors = []

        for user in active:
            chunk = user.next_chunk()
            if policy.needs_bot_score:
                user.observe(None if chunk is None else cache.score(user))

            if user.exhausted and not policy.acts_when_exhausted:
                continue

            state = policy.features(user, n_active).tolist() if collect else None
            if collect and user.uid in pending:
                ps, pa, pr, po, psec = pending.pop(user.uid)
                transitions.append(Transition(ps, pa, pr, state, False,
                                              user.is_bot, user.uid, t, po, psec))

            action = policy.select_action(user, n_active)
            r, terminated, info = stochastic_reward(user, action, solve, cfg, rng)

            result.outcomes[info["outcome"]] += 1
            result.action_hist[action] += 1
            result.challenges_issued += 1 if action > 0 else 0
            if not user.is_bot:
                result.human_seconds += info["seconds"]
                h_total += r
                h_steps += 1
            else:
                b_total += r
                b_steps += 1

            finished = user.exhausted or terminated
            if collect:
                if finished:
                    transitions.append(Transition(state, action, r, state, True,
                                                  user.is_bot, user.uid, t,
                                                  info["outcome"], info["seconds"]))
                    pending.pop(user.uid, None)
                else:
                    pending[user.uid] = (state, action, r, info["outcome"], info["seconds"])

            if not finished:
                if policy.updates_last_threat:
                    user.last_threat_level = action
                survivors.append(user)

        if not survivors:
            result.surviving_humans = sum(1 for u in active if not u.is_bot)
            result.surviving_bots = sum(1 for u in active if u.is_bot)
        active = survivors

    result.human_step_reward = h_total / h_steps if h_steps else 0.0
    result.bot_step_reward = b_total / b_steps if b_steps else 0.0
    result.human_session_reward = h_total / n_humans if n_humans else 0.0
    result.bot_session_reward = b_total / n_bots if n_bots else 0.0
    return result, transitions


def to_sim_result(x: XSimResult):
    """Adapt to rlcaptcha.simulate.SimResult so `rlcaptcha.metrics` can score it."""
    from rlcaptcha.simulate import SimResult

    return SimResult(
        n_humans=x.n_humans, n_bots=x.n_bots,
        surviving_humans=x.surviving_humans, surviving_bots=x.surviving_bots,
        human_step_reward=x.human_step_reward, bot_step_reward=x.bot_step_reward,
        human_session_reward=x.human_session_reward, bot_session_reward=x.bot_session_reward,
        timesteps=x.timesteps, human_counts=x.human_counts, bot_counts=x.bot_counts,
    )


def evaluate_x(x: XSimResult):
    """rlcaptcha metrics, computed on an XSimResult."""
    from rlcaptcha.metrics import evaluate

    return evaluate(to_sim_result(x))
