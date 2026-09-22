"""Reproduction package for the adaptive CAPTCHA orchestration paper.

Typical use (see reproduce_table4.ipynb):

    from rlcaptcha import *

    humans, bots = load_sessions()
    cache = ScoreCache(BotScorer()).precompute(humans, bots)
    sweep = run_sweep(DQNPolicy(), humans, bots, cache, rewards=DQN_REWARDS)
    table({"DQN": sweep})
"""

from .config import (
    BANDIT_REWARDS,
    DQN_REWARDS,
    EVAL_BOT_COUNTS,
    EVAL_HUMANS,
    RewardConfig,
)
from .data import Session, User, build_population, load_sessions
from .metrics import Metrics, evaluate, summarise, table
from .policies.bandits import LinUCBPolicy, ThompsonPolicy
from .policies.dqn import DQNAblationPolicy, DQNPolicy, QNetwork
from .policies.static import MultiThresholdPolicy, SingleThresholdPolicy
from .rewards import reward
from .scoring import BotScorer, ScoreCache, verify_cache
from .simulate import SimResult, run_simulation, run_sweep

__all__ = [
    "BANDIT_REWARDS", "DQN_REWARDS", "EVAL_BOT_COUNTS", "EVAL_HUMANS", "RewardConfig",
    "Session", "User", "build_population", "load_sessions",
    "Metrics", "evaluate", "summarise", "table",
    "LinUCBPolicy", "ThompsonPolicy",
    "DQNAblationPolicy", "DQNPolicy", "QNetwork",
    "MultiThresholdPolicy", "SingleThresholdPolicy",
    "reward",
    "BotScorer", "ScoreCache", "verify_cache",
    "SimResult", "run_simulation", "run_sweep",
]
