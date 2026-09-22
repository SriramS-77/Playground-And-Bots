"""The two no-RL ablations: fixed thresholds on the behavioural score.

Remember the score is P(bot), so HIGH means bot-like and warrants a HIGHER threat level.
"""

from __future__ import annotations

from ..data import User


class SingleThresholdPolicy:
    """Binary: everything above the threshold gets denied outright."""

    name = "Static Single-Threshold"
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def select_action(self, user: User, n_active: int) -> int:
        return 10 if user.bot_score > self.threshold else 0


class MultiThresholdPolicy:
    """Tiered: thresholds at 0.1, 0.2, ... 0.9 map onto threat levels 1..9."""

    name = "Static Multi-Threshold"
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def select_action(self, user: User, n_active: int) -> int:
        return int(user.bot_score * 10)
