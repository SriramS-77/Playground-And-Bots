"""Neural contextual bandits: LinUCB and Thompson Sampling.

Both use the same feature extractor (5 -> 128 -> ReLU -> 64, Table 1) and the same
per-arm ridge statistics A_a, b_a. They differ only in how an arm is chosen from those
statistics: an optimistic upper confidence bound, or a draw from the posterior.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import LINUCB_CKPT, N_ACTIONS, THOMPSON_CKPT
from ..data import User


class FeatureNet(nn.Module):
    def __init__(self, input_dim: int = 5, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class _NeuralBandit:
    """Shared context building and checkpoint loading."""

    needs_bot_score = True
    # The published bandit benchmarks acted on users whose session data had run out;
    # the DQN benchmark did not. Preserved so Table 4 reproduces.
    acts_when_exhausted = True
    # The bandit loops read context[3] (last threat level) but never wrote it back, so
    # it was identically 0 throughout the published runs. Preserved; see README.
    updates_last_threat = False
    # ...and it starts at -1, not 0. Since it is never written, context[3] is a constant
    # tanh(-0.1) throughout. Removing it changes bot survival by several fold.
    initial_last_threat = -1

    def __init__(self, checkpoint, weights_key: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.embed_dim = ckpt["embed_dim"]
        self.alpha = ckpt["alpha"]
        self.n_arms = ckpt.get("num_arms") or N_ACTIONS
        self.net = FeatureNet(5, self.embed_dim).to(self.device)
        self.net.load_state_dict(ckpt[weights_key])
        self.net.eval()
        self.A = [a.to(self.device) for a in ckpt["As"]]
        self.b = [b.to(self.device) for b in ckpt["bs"]]
        # A_a is fixed at evaluation time, so invert once instead of per decision.
        self._invA = [torch.linalg.inv(a) for a in self.A]
        self._theta = [inv @ b for inv, b in zip(self._invA, self.b)]

    def context(self, user: User, n_active: int) -> torch.Tensor:
        """Note the tanh normalisation differs from the DQN's; kept as published."""
        ctx = np.array(
            [
                user.bot_score,
                user.avg_bot_score,
                user.captchas_solved,
                user.last_threat_level,
                n_active,
            ],
            dtype=np.float32,
        )
        ctx[2] = np.tanh(ctx[2] / 10.0)
        ctx[3] = np.tanh(ctx[3] / 10.0)
        ctx[4] = np.tanh(ctx[4] / 200.0)
        return torch.tensor(ctx, device=self.device)


class LinUCBPolicy(_NeuralBandit):
    """a_t = argmax_a  phi(s)'theta_a + alpha * sqrt(phi(s)' A_a^-1 phi(s))."""

    name = "LinUCB"

    def __init__(self, checkpoint=LINUCB_CKPT, **kwargs):
        super().__init__(checkpoint, "feature_nn", **kwargs)

    def select_action(self, user: User, n_active: int) -> int:
        with torch.no_grad():
            z = self.net(self.context(user, n_active))
            scores = [
                (z @ self._theta[a] + self.alpha * torch.sqrt((z @ self._invA[a]) @ z)).item()
                for a in range(self.n_arms)
            ]
        return int(np.argmax(scores))


class ThompsonPolicy(_NeuralBandit):
    """a_t = argmax_a phi(s)'theta~_a,  theta~_a ~ N(mu_a, alpha^2 diag(A_a^-1))."""

    name = "Thompson Sampling"

    def __init__(self, checkpoint=THOMPSON_CKPT, **kwargs):
        super().__init__(checkpoint, "feature_net", **kwargs)
        self._sigma = [self.alpha * torch.sqrt(torch.diag(inv)) for inv in self._invA]

    def select_action(self, user: User, n_active: int) -> int:
        with torch.no_grad():
            z = self.net(self.context(user, n_active))
            scores = [
                (z @ (self._theta[a] + torch.randn_like(self._theta[a]) * self._sigma[a])).item()
                for a in range(self.n_arms)
            ]
        return int(np.argmax(scores))
