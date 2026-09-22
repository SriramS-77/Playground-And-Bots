"""The reward function from Section III-F-3 of the manuscript.

One important divergence between the published runs is captured here by
``RewardConfig.abandonment_applies_to_bots``.

In the DQN, ablation and static-threshold notebooks the "user gave up" check sits inside
the human branch, so only humans can abandon the site. In the LinUCB and Thompson
Sampling notebooks the same check is written one indent level out, at function scope, so
it also fires for BOTS: a bot shown threat level 8 is discarded with probability 0.75
regardless of its strength B_s, on top of being blocked when T > B_s.

That is not a cosmetic difference. It gives the two bandits an extra bot-removal channel
the other four policies never had, and it is the single largest driver of their bot-kill
rates at high load -- with it, LinUCB leaves 7 bots alive out of 500; without it, ~31.
Both settings are provided so every row of Table 4 reproduces, but the comparison in the
paper is not like-for-like. See README "Known inconsistencies".
"""

from __future__ import annotations

import math
import random

from .config import RewardConfig
from .data import User


def reward(user: User, threat: int, cfg: RewardConfig, rng: random.Random) -> tuple[float, bool]:
    """Return (reward, terminated) for presenting `threat` to `user`.

    Bots  : blocked when threat > B_s, with a linear penalty for overkill; otherwise a
            large leakage penalty scaled by how far the threat fell short.
    Humans: R_max at zero friction, else R_max/2 minus exponentially growing frustration.

    Abandonment: P_leave(T) is 0 for T<=6, 1-0.5/(T-6) for 6<T<10, and 1 at T=10.
    """
    terminated = False

    if user.is_bot:
        if threat > user.bot_strength:
            r = cfg.max_reward - (threat - user.bot_strength) * cfg.overkill
            terminated = True
            user.end_session()
        else:
            r = cfg.leakage_penalty - (user.bot_strength - threat) * cfg.underestimation
            user.captchas_solved += 1
    else:
        r = cfg.max_reward if threat == 0 else (
            cfg.max_reward / 2.0 - math.pow(cfg.friction_base, threat)
        )
        user.captchas_solved += 1

    if not user.is_bot or cfg.abandonment_applies_to_bots:
        if threat == 10 or rng.random() * (threat - 6) > 0.5:
            terminated = True

    return r, terminated
