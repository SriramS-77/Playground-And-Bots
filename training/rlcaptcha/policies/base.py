"""The single interface every orchestration policy implements.

All six systems compared in Table 4 differ in exactly three ways, so those are the three
things a policy declares:

* ``needs_bot_score``       -- whether the LSTM has to run at all. The ablation variant
                               has no behavioural input, so it skips scoring entirely.
* ``acts_when_exhausted``   -- whether a user whose session data has run out still gets
                               one final action. The DQN and static baselines skip such
                               users; the two bandit notebooks did not. This is a real
                               behavioural difference in the published runs and it shifts
                               the per-step reward averages, so it is preserved.
* ``updates_last_threat``   -- whether the chosen action is fed back into the user's
                               ``last_threat_level``. The DQN and static loops did this;
                               the bandit loops READ the feature but never WROTE it, so
                               it never moved off its initial value for the whole of the
                               published LinUCB and Thompson Sampling runs. That is a bug
                               in the original code, but fixing it would change Table 4,
                               so it is reproduced here and flagged in the README.
* ``initial_last_threat``   -- the starting value of ``last_threat_level``. The DQN and
                               static loops used 0; the bandit loops used -1. Combined
                               with never updating it, that makes context index 3 a
                               CONSTANT -1 for the bandits -- which is not a no-op: it
                               shifts enough of their decisions from threat 9 to 10 to
                               change the published bot-survival numbers several-fold.
* ``select_action(user, n_active)`` -- the decision itself.
"""

from __future__ import annotations

from typing import Protocol

from ..data import User


class Policy(Protocol):
    name: str
    needs_bot_score: bool
    acts_when_exhausted: bool
    updates_last_threat: bool
    initial_last_threat: int

    def select_action(self, user: User, n_active: int) -> int:
        """Return a threat level in 0..10."""
        ...
