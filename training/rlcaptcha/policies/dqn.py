"""Deep Q-Network policy, and its no-Humanity-Score ablation.

Both share one network shape (128 -> 64 -> 32 -> 11, Table 1 of the manuscript); they
differ only in the state vector and therefore the input width. Keeping them in one class
makes the ablation legible as what it is: the same agent with two features deleted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DQN_ABLATION_CKPT, DQN_CKPT, N_ACTIONS
from ..data import User


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int = N_ACTIONS):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class DQNPolicy:
    """Full DQN. State: [bot_score, avg_bot_score, captchas_solved, last_threat, n_active]."""

    name = "DQN"
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0
    state_size = 5

    def __init__(self, checkpoint=DQN_CKPT, device: str | None = None, epsilon: float | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = QNetwork(self.state_size).to(self.device)
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["policy_net_state_dict"])
        self.net.eval()
        # The published runs kept the agent's final training epsilon (~0.01), so a small
        # fraction of actions are random. Pass epsilon=0.0 for a fully greedy policy.
        self.epsilon = ckpt.get("epsilon", 0.0) if epsilon is None else epsilon

    def features(self, user: User, n_active: int) -> np.ndarray:
        s = np.array(
            [
                user.bot_score,
                user.avg_bot_score,
                user.captchas_solved,
                user.last_threat_level,
                n_active,
            ],
            dtype=np.float32,
        )
        s[2] = np.tanh(s[2] / 5.0)
        s[4] = s[4] / 300.0
        return s

    def select_action(self, user: User, n_active: int) -> int:
        if self.epsilon and np.random.rand() <= self.epsilon:
            return int(np.random.randint(N_ACTIONS))
        with torch.no_grad():
            x = torch.tensor(self.features(user, n_active), device=self.device).unsqueeze(0)
            return int(torch.argmax(self.net(x)).item())


class DQNAblationPolicy(DQNPolicy):
    """DQN without the Humanity Score. State: [captchas_solved, last_threat, n_active].

    Because it never reads `bot_score`, evaluating this policy needs no TensorFlow at all.
    """

    name = "DQN without H-Score"
    needs_bot_score = False
    state_size = 3

    def __init__(self, checkpoint=DQN_ABLATION_CKPT, **kwargs):
        super().__init__(checkpoint=checkpoint, **kwargs)

    def features(self, user: User, n_active: int) -> np.ndarray:
        s = np.array(
            [user.captchas_solved, user.last_threat_level, n_active], dtype=np.float32
        )
        s[0] = np.tanh(s[0] / 5.0)
        s[1] = s[1] / 10.0
        s[2] = s[2] / 300.0
        return s
