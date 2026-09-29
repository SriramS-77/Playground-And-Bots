"""DQN training, reproducing the published agent's hyperparameters.

Mirrors `offline/rl_service_offline_buffer_training.py` and `offline_training.z.py`:
128-64-32-11 network, Huber loss, gamma 0.95, Adam at 5e-4, epsilon 1.0 -> 0.01 with
decay 0.9995, target sync every 200 global steps, one gradient step every 5 environment
steps, batch 64 drawn 50/50 from two class-balanced replay buffers.

Two things are parameterised because the experiments need them:

* **Session pool.** The published trainer hard-codes ``DATA_DIR = "data"``, the same
  recordings it is later evaluated on. Here the pool is an argument.
* **Where the reward comes from.** ``"oracle"`` is the simulator's reward, which reads
  the true class. ``"proxy"`` is `expkit.proxy`, which reads only observable signals.
  A proxy agent also cannot balance its replay buffers by true class, so it balances on
  the observable outcome instead -- otherwise the buffer itself would leak the label.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlcaptcha.config import N_ACTIONS

from .proxy import OBSERVABLE, ProxyRewardConfig, relabel
from .simx import run_x
from .stochastic import DETERMINISTIC, PAPER_EQUIVALENT

MEMORY_SIZE = 10_000


PUBLISHED_HIDDEN = (128, 64, 32)


class QNet(nn.Module):
    """Published default is 128-64-32-11.

    `hidden` is configurable for the round-2 depth/width search. The layers are still
    named `layer1..layerN`, so at the default the state_dict keys and shapes are
    byte-compatible with the published checkpoints -- `exp_policy_arch_search.py`
    asserts that.
    """

    def __init__(self, state_size: int, action_size: int = N_ACTIONS,
                 hidden: tuple[int, ...] = PUBLISHED_HIDDEN):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer")
        dims = [state_size, *hidden]
        for i in range(len(hidden)):
            setattr(self, f"layer{i + 1}", nn.Linear(dims[i], dims[i + 1]))
        setattr(self, f"layer{len(hidden) + 1}", nn.Linear(hidden[-1], action_size))
        self.n_layers = len(hidden) + 1

    def forward(self, x):
        for i in range(1, self.n_layers):
            x = F.relu(getattr(self, f"layer{i}")(x))
        return getattr(self, f"layer{self.n_layers}")(x)


def full_features(user, n_active) -> np.ndarray:
    s = np.array([user.bot_score, user.avg_bot_score, user.captchas_solved,
                  user.last_threat_level, n_active], dtype=np.float32)
    s[2] = np.tanh(s[2] / 5.0)
    s[4] = s[4] / 300.0
    return s


def ablation_features(user, n_active) -> np.ndarray:
    s = np.array([user.captchas_solved, user.last_threat_level, n_active],
                 dtype=np.float32)
    s[0] = np.tanh(s[0] / 5.0)
    s[1] = s[1] / 10.0
    s[2] = s[2] / 300.0
    return s


class TrainableDQN:
    """A DQN that satisfies the `rlcaptcha.policies.base.Policy` protocol."""

    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, name="DQN (retrained)", use_score: bool = True,
                 device: str | None = None, seed: int = 0,
                 hidden: tuple[int, ...] = PUBLISHED_HIDDEN, lr: float = 5e-4):
        self.name = name
        self.needs_bot_score = use_score
        self.features_fn = full_features if use_score else ablation_features
        self.state_size = 5 if use_score else 3
        self.hidden = tuple(hidden)

        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNet(self.state_size, hidden=self.hidden).to(self.device)
        self.target_net = QNet(self.state_size, hidden=self.hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.buffers = {"a": deque(maxlen=MEMORY_SIZE), "b": deque(maxlen=MEMORY_SIZE)}
        self.rng = random.Random(seed)

    # -- Policy interface --------------------------------------------------- #
    def features(self, user, n_active) -> np.ndarray:
        return self.features_fn(user, n_active)

    def select_action(self, user, n_active) -> int:
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(N_ACTIONS))
        return self.greedy(self.features(user, n_active))

    def greedy(self, state) -> int:
        with torch.no_grad():
            x = torch.tensor(np.asarray(state, dtype=np.float32),
                             device=self.device).unsqueeze(0)
            return int(torch.argmax(self.policy_net(x)).item())

    def eval_mode(self, epsilon: float = 0.0) -> "TrainableDQN":
        self.epsilon = epsilon
        return self

    # -- Learning ----------------------------------------------------------- #
    def remember(self, t, bucket: str):
        self.buffers[bucket].append(
            (np.asarray(t.state, dtype=np.float32), t.action, t.reward,
             np.asarray(t.next_state, dtype=np.float32), float(t.done))
        )

    def train_step(self, batch_size: int = 64):
        half = batch_size // 2
        a, b = self.buffers["a"], self.buffers["b"]
        if len(a) < half or len(b) < half:
            return None
        batch = self.rng.sample(list(a), half) + self.rng.sample(list(b), half)
        self.rng.shuffle(batch)
        s, act, r, ns, d = zip(*batch)

        s = torch.tensor(np.array(s), device=self.device)
        act = torch.tensor(act, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(np.array(ns), device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.policy_net(s).gather(1, act)
        with torch.no_grad():
            nq = self.target_net(ns).max(1)[0].unsqueeze(1)
            target = r + self.gamma * nq * (1 - d)

        loss = self.criterion(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({"policy_net_state_dict": self.policy_net.state_dict(),
                    "epsilon": self.epsilon, "state_size": self.state_size,
                    "hidden": self.hidden}, str(path))

    def load(self, path):
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net_state_dict"])
        self.sync_target()
        self.epsilon = ckpt.get("epsilon", 0.0)
        return self


@dataclass
class TrainLog:
    episodes: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    surviving_humans: list = field(default_factory=list)
    surviving_bots: list = field(default_factory=list)
    n_bots: list = field(default_factory=list)
    labels_fired: list = field(default_factory=list)

    def frame(self):
        import pandas as pd
        return pd.DataFrame({k: v for k, v in self.__dict__.items() if v})


def train_dqn(
    humans, bots, cache,
    episodes: int = 120,
    use_score: bool = True,
    reward_source: str = "oracle",           # "oracle" | "proxy"
    solve=DETERMINISTIC,
    cfg=PAPER_EQUIVALENT,
    proxy_cfg: ProxyRewardConfig | None = None,
    human_counts=(80, 100),
    bot_choices=(0, 20, 50, 100, 200, 500),
    train_every: int = 5,
    target_every: int = 200,
    batch_size: int = 64,
    seed: int = 0,
    name: str | None = None,
    verbose: int = 20,
    hidden: tuple[int, ...] = PUBLISHED_HIDDEN,
    device: str | None = "cpu",
) -> tuple[TrainableDQN, TrainLog]:
    """Train one agent. Returns (agent, log)."""
    proxy_cfg = proxy_cfg or ProxyRewardConfig()
    name = name or f"DQN-{reward_source}{'' if use_score else '-noscore'}"
    # CPU by default: a 128-64-32-11 MLP with single-sample per-step inference is
    # slower on GPU than on CPU once transfers are counted. See bandits_x.DEFAULT_DEVICE.
    agent = TrainableDQN(name=name, use_score=use_score, seed=seed,
                         hidden=hidden, device=device)
    rng = random.Random(seed + 1)
    log = TrainLog()
    global_step = 0

    for ep in range(episodes):
        n_h = rng.randint(*human_counts)
        n_b = rng.choice(bot_choices)

        result, transitions = run_x(
            agent, humans, bots, n_b, cache=cache, n_humans=n_h,
            solve=solve, cfg=cfg, seed=rng.randrange(10**9), collect=True,
        )

        n_labels = 0
        if reward_source == "proxy":
            transitions, n_labels = relabel(transitions, proxy_cfg, rng)

        losses = []
        for t in transitions:
            if reward_source == "proxy":
                # Balance on what the operator can see, not on the true class.
                bucket = "a" if OBSERVABLE[t.outcome] == "failed_gone" else "b"
            else:
                bucket = "b" if t.is_bot else "a"
            agent.remember(t, bucket)

            global_step += 1
            if global_step % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)
            if global_step % target_every == 0:
                agent.sync_target()

        log.episodes.append(ep)
        log.epsilon.append(agent.epsilon)
        log.loss.append(float(np.mean(losses)) if losses else float("nan"))
        log.surviving_humans.append(result.surviving_humans)
        log.surviving_bots.append(result.surviving_bots)
        log.n_bots.append(n_b)
        log.labels_fired.append(n_labels)

        if verbose and (ep + 1) % verbose == 0:
            print(f"  [{name}] ep {ep+1:4d}/{episodes}  eps={agent.epsilon:.3f}  "
                  f"loss={log.loss[-1]:.2f}  {n_h}h/{n_b}b -> "
                  f"{result.surviving_humans}h/{result.surviving_bots}b")

    return agent, log
