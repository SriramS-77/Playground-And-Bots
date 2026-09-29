"""Trainable neural bandits, aligned with the DQN, with a searchable architecture and
a mixture-of-Gaussians posterior for Thompson Sampling.

Why this module exists
----------------------

`rlcaptcha.policies.bandits` is a faithful port of the published LinUCB and Thompson
Sampling benchmarks and must not change. But those benchmarks were **not evaluated in
the same environment as the other four policies**, and the differences all favour the
bandits (`training/README.md`):

| # | mismatch | lives in |
|---|---|---|
| 1 | bots abandon the site regardless of strength | `cfg.abandonment_applies_to_bots` |
| 2 | overkill penalty 2.0, not 2.5 | `cfg.overkill` |
| 3 | `last_threat_level` initialised to -1 and never written back | policy attribute |
| 4 | acts on users whose session data has run out | policy attribute |
| 5 | different feature normalisation from the DQN | `features()` |

Mismatch 5 was not in the README list and is easy to miss:

    feature              published bandit      DQN
    population           tanh(n / 200)         n / 300, unbounded
    captchas solved      tanh(x / 10)          tanh(x / 5)
    last threat level    tanh(x / 10)          raw

The classes here fix all five: they take the DQN's flags, the DQN's `full_features`, and
whatever `StochasticRewardConfig` the caller passes to `run_x` -- so every policy in the
headline table sees one environment and one state encoding.

R1.6 -- the statistics accumulate in a moving feature space
-----------------------------------------------------------

`A_a = I + sum z z^T` and `b_a = sum r z` are accumulated in the embedding space `z`,
but `z` keeps changing while the feature network trains, so the two halves of `A_a` were
never measured in the same coordinates. The published code just lets this happen.

`train_bandit` stores the raw contexts alongside the rewards, then at the end **freezes
the network and recomputes every statistic from the whole buffer in the final z-space**
(`recompute_statistics`). Evaluation already freezes `A`, so after this the statistics
and the evaluation-time embedding agree exactly. Report this in the paper.

Two details of that refit, both of which change results:

* It uses **batch EM** (`MixturePosterior.fit_batch`), not one online sweep. With the
  whole buffer in hand, a single sweep would leave every mixture assignment made against
  a `theta` fitted on the first `warmup` samples -- an initialisation, not a fit. At K=1
  the two are identical (`A = I + ZᵀZ` is a sum; order does not matter), so the published
  equivalence is untouched.
* It **rebalances 50/50 on the true class** by default, matching the balanced batches the
  network and the published bandits were trained on. The raw buffer is bot-heavy at high
  volumes -- a 1000-bot episode contributes ten times the transitions of its humans -- and
  a head fitted on that skew tilts toward challenging everyone, which is the failure mode
  round 1 saw at 500 and 1000 bots. `balance=False` uses the raw buffer; whichever is
  chosen, state it.

A mixture posterior for Thompson Sampling
-----------------------------------------

The published sampler draws `theta~_a ~ N(mu_a, alpha^2 diag(A_a^-1))`, i.e. a single
Gaussian over a linear reward model. But the reward for a given arm is bimodal: the same
threat level pays very differently to a human and to a bot, and both are present in every
episode. One linear model cannot express that.

`MixturePosterior(K)` fits K components per arm, unsupervised, on `(z, r)` only -- never
on `is_bot`, which a deployed system does not have. **The gate is context-dependent**,
which is the part that matters:

    pi_k(z)  =  pi_k N(z; m_k, v_k) / sum_j pi_j N(z; m_j, v_j)
    score(a) =  sum_k pi_k(z) * (z . theta~_k),   theta~_k ~ N(mu_k, alpha^2 Sigma_k)

A context-free gate would be pointless: drawing one component per decision gives
`E[r|z,a] = z . sum_k pi_k theta_k`, still linear in `z`, so the mixture would add no
representational power and would just randomise over "the world is all human" versus
"all bot" -- noise, not a Thompson sample.

At K=1 the gate is identically 1 and the update reduces to the published ridge recursion,
so **K=1 reproduces Gaussian Thompson Sampling bit-exactly** under the same seed.
`exp_policy_arch_search.py` asserts it.

`cov="full"` is offered alongside so that "the mixture helped" cannot be confused with
"we dropped the published diagonal approximation".
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlcaptcha.config import N_ACTIONS

from .simx import run_x
from .stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from .trainer import TrainLog, full_features

MEMORY_SIZE = 10_000
_EPS = 1e-6

POSTERIORS = ("gaussian", "gaussian_full", "mog2", "mog3")


def posterior_spec(name: str) -> tuple[int, str]:
    """`name` -> (n_components, covariance)."""
    return {"gaussian": (1, "diag"), "gaussian_full": (1, "full"),
            "mog2": (2, "diag"), "mog3": (3, "diag")}[name]


# --------------------------------------------------------------------------- #
# Feature network
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BanditArch:
    """One searchable configuration.

    The default reproduces the published feature network exactly: 5 -> 128 -> ReLU -> 64,
    alpha 1.0, Adam at 1e-3.
    """

    name: str = "published"
    hidden: tuple[int, ...] = (128,)
    embed_dim: int = 64
    alpha: float = 1.0
    lr: float = 1e-3
    posterior: str = "gaussian"      # Thompson only; LinUCB ignores it

    @property
    def n_components(self) -> int:
        return posterior_spec(self.posterior)[0]

    @property
    def covariance(self) -> str:
        return posterior_spec(self.posterior)[1]


PUBLISHED_ARCH = BanditArch()


class FeatureNetX(nn.Module):
    """Configurable depth. At `hidden=(128,)` the module layout, parameter shapes and
    state_dict keys match `rlcaptcha.policies.bandits.FeatureNet`."""

    def __init__(self, input_dim: int = 5, hidden: tuple[int, ...] = (128,),
                 embed_dim: int = 64):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers += [nn.Linear(prev, embed_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Posterior
# --------------------------------------------------------------------------- #

class MixturePosterior:
    """K Bayesian linear regressions over one arm's reward, with a context gate.

    All components start from the ridge prior `A = I`, so `A` is invertible from the
    first step and K=1 is exactly the published recursion.
    """

    def __init__(self, dim: int, n_components: int = 1, alpha: float = 1.0,
                 covariance: str = "diag", warmup: int = 64, hard: bool = False,
                 device=None):
        if covariance not in ("diag", "full"):
            raise ValueError("covariance must be 'diag' or 'full'")
        self.d = dim
        self.K = n_components
        self.alpha = alpha
        self.covariance = covariance
        self.warmup = warmup
        self.hard = hard
        self.device = device or torch.device("cpu")

        z = lambda *s: torch.zeros(*s, device=self.device)                # noqa: E731
        self.A = [torch.eye(dim, device=self.device) for _ in range(self.K)]
        self.b = [z(dim) for _ in range(self.K)]
        self.n = [0.0 for _ in range(self.K)]
        self.m = [z(dim) for _ in range(self.K)]          # gate: mean of z
        self.v = [torch.ones(dim, device=self.device) for _ in range(self.K)]
        self.s2 = [1.0 for _ in range(self.K)]            # residual variance
        self.n[0] = 1.0                                   # all mass on comp 0 pre-warmup

        self._pending: list[tuple[torch.Tensor, float]] = []
        self._initialised = self.K == 1
        self._dirty = True
        self._invA: list[torch.Tensor] = []
        self._theta: list[torch.Tensor] = []
        self._scale: list[torch.Tensor] = []

    # -- cached linear algebra --------------------------------------------- #
    def _refresh(self):
        if not self._dirty:
            return
        self._invA = [torch.linalg.inv(A) for A in self.A]
        self._theta = [inv @ b for inv, b in zip(self._invA, self.b)]
        if self.covariance == "diag":
            self._scale = [self.alpha * torch.sqrt(torch.clamp(torch.diag(inv), min=0.0))
                           for inv in self._invA]
        else:
            self._scale = []
            for inv in self._invA:
                sym = 0.5 * (inv + inv.T)
                sym = sym + _EPS * torch.eye(self.d, device=self.device)
                self._scale.append(self.alpha * torch.linalg.cholesky(sym))
        self._dirty = False

    def freeze(self):
        """Invert once; call before evaluation."""
        self._refresh()
        return self

    # -- gate --------------------------------------------------------------- #
    def gate(self, z: torch.Tensor) -> torch.Tensor:
        """pi_k(z), shape (K,). Identically [1.0] at K=1."""
        if self.K == 1:
            return torch.ones(1, device=self.device)
        total = sum(self.n) or 1.0
        logp = torch.empty(self.K, device=self.device)
        for k in range(self.K):
            var = torch.clamp(self.v[k], min=_EPS)
            ll = -0.5 * (((z - self.m[k]) ** 2) / var + torch.log(2 * math.pi * var)).sum()
            logp[k] = math.log(max(self.n[k], _EPS) / total) + ll
        return torch.softmax(logp, dim=0)

    # -- sampling ----------------------------------------------------------- #
    def sample_theta(self) -> list[torch.Tensor]:
        self._refresh()
        out = []
        for k in range(self.K):
            eps = torch.randn_like(self._theta[k])
            if self.covariance == "diag":
                out.append(self._theta[k] + eps * self._scale[k])
            else:
                out.append(self._theta[k] + self._scale[k] @ eps)
        return out

    def thompson_score(self, z: torch.Tensor) -> float:
        """One Thompson draw. At K=1 this is `z @ (theta + randn * sigma)` -- the
        published expression, with the same single `randn` draw."""
        thetas = self.sample_theta()
        if self.K == 1:
            return float(z @ thetas[0])
        w = self.gate(z)
        return float(sum(w[k] * (z @ thetas[k]) for k in range(self.K)))

    def ucb_score(self, z: torch.Tensor) -> float:
        """Gate-weighted UCB. At K=1 this is the published LinUCB expression."""
        self._refresh()
        w = self.gate(z)
        out = 0.0
        for k in range(self.K):
            mean = z @ self._theta[k]
            unc = self.alpha * torch.sqrt(torch.clamp((z @ self._invA[k]) @ z, min=0.0))
            out += float(w[k]) * float(mean + unc)
        return out

    def mean_score(self, z: torch.Tensor) -> float:
        self._refresh()
        w = self.gate(z)
        return float(sum(w[k] * (z @ self._theta[k]) for k in range(self.K)))

    # -- fitting ------------------------------------------------------------ #
    def _init_components(self):
        """1-D k-means on the buffered rewards gives the initial hard assignment.

        Rewards, not labels -- the split has to be discoverable from what a deployed
        system observes.
        """
        rs = np.array([r for _, r in self._pending], dtype=np.float64)
        centres = np.quantile(rs, np.linspace(0.0, 1.0, self.K + 2)[1:-1])
        for _ in range(25):
            assign = np.abs(rs[:, None] - centres[None, :]).argmin(axis=1)
            moved = False
            for k in range(self.K):
                sel = rs[assign == k]
                if len(sel):
                    new = float(sel.mean())
                    moved |= abs(new - centres[k]) > 1e-9
                    centres[k] = new
            if not moved:
                break
        assign = np.abs(rs[:, None] - centres[None, :]).argmin(axis=1)

        self.A = [torch.eye(self.d, device=self.device) for _ in range(self.K)]
        self.b = [torch.zeros(self.d, device=self.device) for _ in range(self.K)]
        self.n = [0.0] * self.K
        self.m = [torch.zeros(self.d, device=self.device) for _ in range(self.K)]
        self.v = [torch.ones(self.d, device=self.device) for _ in range(self.K)]
        self.s2 = [1.0] * self.K
        for (z, r), k in zip(self._pending, assign):
            self._accumulate(int(k), 1.0, z, r)
        self._pending.clear()
        self._initialised = True
        self._dirty = True

    def _theta_stale(self, k: int) -> torch.Tensor | None:
        """The last inverted estimate, without forcing a re-inversion.

        `s2` is an exponentially-weighted mean of squared residuals and the
        responsibilities are a soft assignment, so a one-step-stale `theta` is a standard
        online-EM approximation. Forcing `_refresh()` here instead costs a 64x64 inversion
        per observation, which measured at 0.29 ms and dominated training -- about 184 s
        per 1100-user episode. It does not touch `A`, `b` or the sampled `theta`, so the
        K=1 equivalence with published Gaussian Thompson Sampling is unaffected
        (asserted in `exp_policy_arch_search.smoke_tests`).
        """
        return self._theta[k] if self._theta else None

    def _accumulate(self, k: int, w: float, z: torch.Tensor, r: float):
        if w < 1e-8:
            return
        self.A[k] = self.A[k] + w * torch.outer(z, z)
        self.b[k] = self.b[k] + w * r * z
        prev, self.n[k] = self.n[k], self.n[k] + w
        lr = w / self.n[k]
        delta = z - self.m[k]
        self.m[k] = self.m[k] + lr * delta
        self.v[k] = torch.clamp((1 - lr) * (self.v[k] + lr * delta * delta), min=_EPS)
        theta = self._theta_stale(k)
        resid = float(r - z @ theta) if (prev > 0 and theta is not None) else float(r)
        self.s2[k] = max((1 - lr) * self.s2[k] + lr * resid * resid, _EPS)
        self._dirty = True

    def responsibilities(self, z: torch.Tensor, r: float) -> torch.Tensor:
        if self.K == 1:
            return torch.ones(1, device=self.device)
        if not self._theta:
            self._refresh()
        total = sum(self.n) or 1.0
        logp = torch.empty(self.K, device=self.device)
        for k in range(self.K):
            var = torch.clamp(self.v[k], min=_EPS)
            gz = -0.5 * (((z - self.m[k]) ** 2) / var + torch.log(2 * math.pi * var)).sum()
            theta = self._theta_stale(k)
            resid = float(r - z @ theta) if theta is not None else float(r)
            gr = -0.5 * (resid * resid / self.s2[k] + math.log(2 * math.pi * self.s2[k]))
            logp[k] = math.log(max(self.n[k], _EPS) / total) + gz + gr
        w = torch.softmax(logp, dim=0)
        if self.hard:
            hard = torch.zeros_like(w)
            hard[int(torch.argmax(w))] = 1.0
            return hard
        return w

    def update(self, z: torch.Tensor, r: float):
        z = z.detach()
        if not self._initialised:
            self._pending.append((z.clone(), float(r)))
            self._accumulate(0, 1.0, z, float(r))
            if len(self._pending) >= self.warmup:
                self._init_components()
            return
        w = self.responsibilities(z, float(r))
        for k in range(self.K):
            self._accumulate(k, float(w[k]), z, float(r))

    # -- batch EM ----------------------------------------------------------- #
    def fit_batch(self, Z, R, iters: int = 20):
        """Proper EM over one arm's whole history, in one pass per iteration.

        `update` is a single online sweep: each observation is assigned once, using
        parameters estimated from everything before it. That is the right thing during
        training, where the data arrives a step at a time. But `recompute_statistics` has
        the whole buffer in hand, and a single online sweep there would leave every
        assignment made against a `theta` fitted on the first `warmup` samples -- an
        initialisation, not a fit.

        At K=1 this is `A = I + ZᵀZ`, `b = ZᵀR`, which is exactly what the sequential
        recursion accumulates (a sum does not care about order), so the published
        equivalence is untouched.
        """
        Z = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=self.device)
        R = torch.as_tensor(np.asarray(R, dtype=np.float32), device=self.device)
        n = int(R.numel())
        if n == 0:
            return self

        def m_step(W):                      # W: (n, K) responsibilities
            for k in range(self.K):
                w = W[:, k]
                nk = float(w.sum())
                self.n[k] = nk
                Zw = Z * w.unsqueeze(1)
                self.A[k] = torch.eye(self.d, device=self.device) + Zw.T @ Z
                self.b[k] = Zw.T @ R
                if nk > _EPS:
                    self.m[k] = (Zw.sum(0) / nk)
                    self.v[k] = torch.clamp(
                        ((Z - self.m[k]) ** 2 * w.unsqueeze(1)).sum(0) / nk, min=_EPS)
            self._dirty = True
            self._refresh()
            for k in range(self.K):
                resid = R - Z @ self._theta[k]
                nk = max(self.n[k], _EPS)
                self.s2[k] = max(float((resid ** 2 * W[:, k]).sum() / nk), _EPS)

        if self.K == 1 or n < self.K * 4:
            m_step(torch.ones(n, 1, device=self.device))
            self._initialised = True
            self._pending.clear()
            return self

        # k-means on the rewards for the initial hard assignment -- discoverable from
        # what a deployed system observes, never from `is_bot`.
        rs = R.cpu().numpy().astype(np.float64)
        centres = np.quantile(rs, np.linspace(0.0, 1.0, self.K + 2)[1:-1])
        for _ in range(25):
            assign = np.abs(rs[:, None] - centres[None, :]).argmin(axis=1)
            new = np.array([rs[assign == k].mean() if (assign == k).any() else centres[k]
                            for k in range(self.K)])
            if np.allclose(new, centres):
                break
            centres = new
        assign = np.abs(rs[:, None] - centres[None, :]).argmin(axis=1)
        W = torch.zeros(n, self.K, device=self.device)
        W[torch.arange(n), torch.tensor(assign, device=self.device)] = 1.0

        prev = None
        for _ in range(iters):
            m_step(W)
            logp = torch.empty(n, self.K, device=self.device)
            total = sum(self.n) or 1.0
            for k in range(self.K):
                var = torch.clamp(self.v[k], min=_EPS)
                gz = -0.5 * (((Z - self.m[k]) ** 2) / var
                             + torch.log(2 * math.pi * var)).sum(1)
                resid = R - Z @ self._theta[k]
                gr = -0.5 * (resid ** 2 / self.s2[k] + math.log(2 * math.pi * self.s2[k]))
                logp[:, k] = math.log(max(self.n[k], _EPS) / total) + gz + gr
            W = torch.softmax(logp, dim=1)
            ll = float(torch.logsumexp(logp, dim=1).mean())
            if prev is not None and abs(ll - prev) < 1e-4:
                break
            prev = ll
        m_step(W)
        self._initialised = True
        self._pending.clear()
        return self


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

class _NeuralBanditX:
    """Aligned with `expkit.trainer.TrainableDQN` on all five axes."""

    needs_bot_score = True
    acts_when_exhausted = False      # (4) fixed
    updates_last_threat = True       # (3) fixed
    initial_last_threat = 0          # (3) fixed

    def __init__(self, arch: BanditArch = PUBLISHED_ARCH, n_arms: int = N_ACTIONS,
                 device: str | None = None, seed: int = 0, name: str | None = None):
        self.arch = arch
        self.n_arms = n_arms
        self.name = name or f"{self.base_name} ({arch.name})"
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = FeatureNetX(5, arch.hidden, arch.embed_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=arch.lr)
        self.posteriors = [self._new_posterior() for _ in range(n_arms)]
        self.rng = random.Random(seed)
        self._frozen = False

    def _new_posterior(self) -> MixturePosterior:
        return MixturePosterior(self.arch.embed_dim, 1, self.arch.alpha, "diag",
                                device=self.device)

    # (5) fixed -- the DQN's state encoding, not the published bandits' tanh variant.
    def features(self, user, n_active) -> np.ndarray:
        return full_features(user, n_active)

    def embed(self, state) -> torch.Tensor:
        x = torch.tensor(np.asarray(state, dtype=np.float32), device=self.device)
        return self.net(x)

    def select_action(self, user, n_active) -> int:
        with torch.no_grad():
            z = self.embed(self.features(user, n_active))
            return int(np.argmax([self._score(a, z) for a in range(self.n_arms)]))

    def _score(self, arm: int, z: torch.Tensor) -> float:
        raise NotImplementedError

    def eval_mode(self, epsilon: float = 0.0):
        self.net.eval()
        for p in self.posteriors:
            p.freeze()
        self._frozen = True
        return self

    # -- learning ----------------------------------------------------------- #
    def update(self, state, action: int, reward: float):
        """Published recursion: statistics from the detached embedding, then one MSE
        step on the network against the arm's own linear prediction."""
        return self.update_batch([state], [action], [reward])

    def update_batch(self, states, actions, rewards) -> float:
        """One training batch: per-sample ridge updates, then **one** network step.

        The published loop takes a separate Adam step for each of the 64 items in a
        balanced batch. That measured at 2.62 ms per item -- about 184 s per 1100-user
        episode, i.e. 10 h for a single 200-episode training, which is not runnable at
        225 trainings. Here the ridge statistics still update per sample (they are rank-1
        accumulations and order matters), but the network takes **one** step on the mean
        squared error over the batch.

        This is a deliberate, stated deviation: same objective, better-conditioned
        gradient, ~50x less autograd overhead. Pass the batch one item at a time through
        `update()` to reproduce the published trajectory exactly.
        """
        X = torch.tensor(np.asarray(states, dtype=np.float32), device=self.device)
        R = torch.tensor(np.asarray(rewards, dtype=np.float32), device=self.device)

        # One forward. The statistics take the detached embedding, the loss keeps the
        # graph -- exactly the published split, just batched.
        Z = self.net(X)
        for z, a, r in zip(Z.detach(), actions, rewards):
            self.posteriors[a].update(z, float(r))

        preds = []
        for i, a in enumerate(actions):
            post = self.posteriors[a]
            post._refresh()
            zi = Z[i]
            if post.K == 1:
                preds.append(zi @ post._theta[0])
            else:
                w = post.gate(zi.detach())
                preds.append(sum(w[k] * (zi @ post._theta[k]) for k in range(post.K)))
        loss = ((torch.stack(preds) - R) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def recompute_statistics(self, buffer, balance: bool = True, seed: int = 0):
        """R1.6: freeze the network, then rebuild every statistic from the whole buffer
        in the final embedding space, so `A_a` and `b_a` are measured in one coordinate
        system rather than smeared across training.

        `buffer` holds `(state, action, reward)` or `(state, action, reward, is_bot)`.
        With the 4-tuple form and `balance=True` the replay is resampled 50/50 on the
        true class, matching the balanced batches the network and the published bandits
        were trained on. Without it the raw buffer is bot-heavy at high volumes -- a
        1000-bot episode contributes ten times the transitions of its humans -- and a
        head fitted on that skew would tilt toward challenging everyone, which is the
        failure mode round 1 saw at 500 and 1000 bots.
        """
        self.net.eval()
        # `_new_posterior` is the subclass's contract: LinUCB always K=1 (a UCB bonus has
        # no use for a mixture), Thompson honours `arch.posterior`.
        self.posteriors = [self._new_posterior() for _ in range(self.n_arms)]

        rows = list(buffer)
        if not rows:
            return self
        if balance and len(rows[0]) == 4:
            rng = random.Random(seed)
            bots = [r for r in rows if r[3]]
            humans = [r for r in rows if not r[3]]
            if bots and humans:
                n = max(len(bots), len(humans))
                rows = ([rng.choice(bots) for _ in range(n)]
                        + [rng.choice(humans) for _ in range(n)])
                rng.shuffle(rows)
        self.replay_size = len(rows)

        by_arm: dict[int, list] = {}
        with torch.no_grad():
            X = torch.tensor(np.asarray([r[0] for r in rows], dtype=np.float32),
                             device=self.device)
            Z = self.net(X)                     # one batched embedding pass
        for z, row in zip(Z, rows):
            by_arm.setdefault(int(row[1]), []).append((z, float(row[2])))

        for arm, items in by_arm.items():
            self.posteriors[arm].fit_batch([z for z, _ in items],
                                           [r for _, r in items])
        for p in self.posteriors:
            p.freeze()
        return self

    def save(self, path):
        torch.save({"arch": self.arch.__dict__, "net": self.net.state_dict(),
                    "As": [p.A for p in self.posteriors],
                    "bs": [p.b for p in self.posteriors],
                    "ns": [p.n for p in self.posteriors],
                    "ms": [p.m for p in self.posteriors],
                    "vs": [p.v for p in self.posteriors],
                    "s2s": [p.s2 for p in self.posteriors]}, str(path))


class LinUCBX(_NeuralBanditX):
    base_name = "LinUCB"

    def _score(self, arm: int, z: torch.Tensor) -> float:
        return self.posteriors[arm].ucb_score(z)


class ThompsonX(_NeuralBanditX):
    base_name = "Thompson Sampling"

    def _new_posterior(self) -> MixturePosterior:
        return MixturePosterior(self.arch.embed_dim, self.arch.n_components,
                                self.arch.alpha, self.arch.covariance,
                                device=self.device)

    def _score(self, arm: int, z: torch.Tensor) -> float:
        return self.posteriors[arm].thompson_score(z)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_bandit(
    kind: str,                                   # "linucb" | "thompson"
    humans, bots, cache,
    arch: BanditArch = PUBLISHED_ARCH,
    episodes: int = 200,
    solve=DETERMINISTIC,
    cfg=PAPER_EQUIVALENT,
    human_counts=(80, 100),
    bot_choices=(0, 20, 50, 100, 200, 500, 1000),
    batch_size: int = 64,
    train_every: int = 5,
    seed: int = 0,
    name: str | None = None,
    verbose: int = 20,
):
    """Train one bandit on `humans`/`bots`. Returns (policy, TrainLog).

    Deliberately mirrors `expkit.trainer.train_dqn`: same episode loop, same class-
    balanced 50/50 replay, same default episode count, same bot-volume sampler. An
    unequal budget between policies is what produced round 1's inversion.

    `bot_choices` reaches 1000 by default, unlike `train_dqn`'s published (0..500). At
    1000 bots the DQN's population feature is `1100/300 = 3.67` against a training
    maximum of 2.0 -- out of distribution, and a candidate explanation for its collapse
    in that cell. Give every learned policy this sampler for the headline table, and say
    so.
    """
    cls = {"linucb": LinUCBX, "thompson": ThompsonX}[kind]
    policy = cls(arch=arch, seed=seed, name=name)
    rng = random.Random(seed + 1)
    log = TrainLog()
    buffers = {"a": deque(maxlen=MEMORY_SIZE), "b": deque(maxlen=MEMORY_SIZE)}
    every: list[tuple] = []
    step = 0

    for ep in range(episodes):
        n_h = rng.randint(*human_counts)
        n_b = rng.choice(bot_choices)
        result, transitions = run_x(policy, humans, bots, n_b, cache=cache,
                                    n_humans=n_h, solve=solve, cfg=cfg,
                                    seed=rng.randrange(10 ** 9), collect=True)

        losses = []
        for t in transitions:
            item = (np.asarray(t.state, dtype=np.float32), t.action, t.reward)
            buffers["b" if t.is_bot else "a"].append(item)
            every.append(item + (t.is_bot,))      # is_bot only for the balanced refit
            step += 1
            if step % train_every == 0:
                half = batch_size // 2
                a, b = buffers["a"], buffers["b"]
                if len(a) >= half and len(b) >= half:
                    batch = rng.sample(list(a), half) + rng.sample(list(b), half)
                    rng.shuffle(batch)
                    ctxs, acts, rews = zip(*batch)
                    losses.append(policy.update_batch(ctxs, acts, rews))

        log.episodes.append(ep)
        log.loss.append(float(np.mean(losses)) if losses else float("nan"))
        log.surviving_humans.append(result.surviving_humans)
        log.surviving_bots.append(result.surviving_bots)
        log.n_bots.append(n_b)
        if verbose and (ep + 1) % verbose == 0:
            print(f"  [{policy.name}] ep {ep + 1:4d}/{episodes}  "
                  f"loss={log.loss[-1]:.2f}  {n_h}h/{n_b}b -> "
                  f"{result.surviving_humans}h/{result.surviving_bots}b")

    # R1.6: one coordinate system for the statistics.
    policy.recompute_statistics(every)
    # Kept so the posterior can be refit on the same weights and the same data --
    # see `fit_posteriors_from`.
    policy._buffer = every
    policy.eval_mode()
    return policy, log


def fit_posteriors_from(policy, buffer, posterior: str):
    """Refit a trained bandit's posterior without retraining the network.

    Gaussian, gaussian_full, mog2 and mog3 then differ in the posterior alone -- same
    weights, same buffer, same embedding -- so the comparison isolates the thing being
    varied, and the three extra variants cost almost nothing.
    """
    import copy

    held, policy._buffer = getattr(policy, "_buffer", None), None   # don't clone it
    try:
        clone = copy.deepcopy(policy)
    finally:
        policy._buffer = held
    clone.arch = type(policy.arch)(**{**policy.arch.__dict__, "posterior": posterior})
    clone.name = f"{policy.base_name} ({clone.arch.name}, {posterior})"
    clone.recompute_statistics(buffer)
    return clone.eval_mode()
