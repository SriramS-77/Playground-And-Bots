"""PPO for the CAPTCHA threat-level policy -- the fourth learner beside DQN, LinUCB, Thompson.

A drop-in for `expkit.trainer.train_dqn`: same signature, same episode sampler, same state
vector (`full_features`: bot_score, avg_bot_score, captchas_solved, last_threat, n_active),
same network body as fold 2's DQN winner (`expkit.trainer.QNet`, one hidden layer of 64,
ReLU). Only the learning algorithm differs, so a comparison isolates it.

Design, each choice from published practice:
  * clipped surrogate objective, clip 0.2                      Schulman et al. 2017 (PPO)
  * GAE, lambda 0.95                                            Schulman et al. 2016 (GAE)
  * gamma 0.95 -- the DQN's, not PPO's usual 0.99, so both optimise the same return
  * 10 epochs, minibatch 64, Adam 3e-4: PPO's MuJoCo setting (Schulman et al. 2017, Table 3),
    the closest published regime: a low-dimensional state and ~2k steps per rollout (one
    simulator episode here is 1.5-3k decisions)
  * SEPARATE policy and value networks -- Andrychowicz et al. 2021 ("What matters in
    on-policy RL", ICLR) found separate networks perform better than a shared trunk
  * orthogonal init, gain sqrt(2) hidden / 0.01 policy head / 1.0 value head; Adam eps 1e-5;
    per-minibatch advantage normalisation; global grad-norm clip 0.5; entropy 0.01, value 0.5
    -- Huang et al. 2022, "The 37 implementation details of PPO" (ICLR blog track)
  * reward scaling by the running std of the discounted return (no mean shift): rewards here
    span about -200..+50 -- Engstrom et al. 2020 ("Implementation matters", ICLR)
  * no value clipping -- Andrychowicz et al. 2021 found it does not help
  * CONSTANT learning rate. The 37-details default anneals it to 0 over a fixed budget; that
    would remove adaptation by construction, and the drift / injection tests measure exactly
    adaptation after episode 100.

What the simulator dictates:
  * one PPO update per simulator episode (the rollout). The policy is fixed within an
    episode, so the behaviour log-probs are recomputed exactly from the pre-update network.
  * `run_x` interleaves users; each user is one trajectory. Transitions are grouped by
    `user_id` and ordered by `step` before GAE.
  * a user who reaches the end of their session gets no final transition from `run_x`
    (`exhausted` is set on the NEXT step, which is skipped), so their last recorded
    transition has done=False: it is bootstrapped with V(next_state). Terminations
    (blocked, abandoned, false positive) have done=True and bootstrap 0.
  * nothing in the update reads `is_bot`: no class balancing (on-policy data has no replay
    buffer to balance). The DQN balances its replay buffer -- by true class with the
    oracle reward, by observable outcome with deployable rewards. Stated, not hidden.
  * `epsilon` is kept as a switch for the harnesses that set it: 0 -> greedy (argmax of
    the policy), anything else -> sample from the policy. Evaluation is greedy, as the DQN's.

`T.run_x` and `T.relabel` are looked up on `expkit.trainer` at CALL time, so every hook the
DQN experiments install there (drift, injection, posterior rewards) reaches PPO unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import trainer as T
from .proxy import ProxyRewardConfig
from .stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from .trainer import QNet, TrainLog, ablation_features, full_features
from rlcaptcha.config import N_ACTIONS

DQN_WINNER_HIDDEN = (64,)          # fold 2's arch_choice.json: "dqn:64"


@dataclass
class PPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 64
    lr: float = 3e-4
    adam_eps: float = 1e-5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    scale_rewards: bool = True


class RunningVar:
    """Running variance (Chan et al. parallel update), for reward scaling."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        n_b, mean_b, var_b = x.size, float(x.mean()), float(x.var())
        delta, n = mean_b - self.mean, self.n + n_b
        self.mean += delta * n_b / n
        self.m2 += var_b * n_b + delta ** 2 * self.n * n_b / n
        self.n = n

    @property
    def std(self):
        return float(np.sqrt(self.m2 / self.n)) if self.n > 1 else 1.0


def _ortho(net: QNet, head_gain: float):
    layers = [getattr(net, f"layer{i}") for i in range(1, net.n_layers + 1)]
    for i, layer in enumerate(layers):
        nn.init.orthogonal_(layer.weight, gain=head_gain if i == len(layers) - 1 else np.sqrt(2))
        nn.init.zeros_(layer.bias)


class TrainablePPO:
    """A PPO actor-critic that satisfies the `rlcaptcha.policies.base.Policy` protocol."""

    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, name="PPO", use_score: bool = True, seed: int = 0,
                 hidden: tuple[int, ...] = DQN_WINNER_HIDDEN, cfg: PPOConfig | None = None,
                 device: str = "cpu"):
        self.name = name
        self.needs_bot_score = use_score
        self.features_fn = full_features if use_score else ablation_features
        self.state_size = 5 if use_score else 3
        self.hidden = tuple(hidden)
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.actor = QNet(self.state_size, N_ACTIONS, hidden=self.hidden).to(self.device)
        self.critic = QNet(self.state_size, 1, hidden=self.hidden).to(self.device)
        _ortho(self.actor, 0.01)
        _ortho(self.critic, 1.0)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.cfg.lr, eps=self.cfg.adam_eps)
        self.gen = torch.Generator().manual_seed(seed + 7)     # minibatch order, own stream
        self.ret_var = RunningVar()
        self._sample = True

    # -- the harnesses' exploration switch ----------------------------------- #
    @property
    def epsilon(self) -> float:
        return 1.0 if self._sample else 0.0

    @epsilon.setter
    def epsilon(self, value: float):
        self._sample = bool(value and value > 0)

    def eval_mode(self, epsilon: float = 0.0) -> "TrainablePPO":
        self.epsilon = epsilon
        return self

    # -- Policy interface ---------------------------------------------------- #
    def features(self, user, n_active) -> np.ndarray:
        return self.features_fn(user, n_active)

    def select_action(self, user, n_active) -> int:
        with torch.no_grad():
            x = torch.as_tensor(self.features(user, n_active)).unsqueeze(0)
            logits = self.actor(x)[0]
            if not self._sample:
                return int(torch.argmax(logits).item())
            p = torch.softmax(logits, -1).numpy().astype(np.float64)
        # np.random is re-seeded by run_x at the start of every episode.
        a = int(np.searchsorted(np.cumsum(p), np.random.rand() * p.sum(), side="right"))
        return min(a, N_ACTIONS - 1)

    def probs(self, states) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.actor(torch.as_tensor(np.asarray(states, np.float32))), -1).numpy()

    # -- Learning ------------------------------------------------------------ #
    def update(self, transitions) -> dict:
        """One PPO update on one episode's transitions (rewards as given: oracle or relabelled)."""
        c = self.cfg
        if not transitions:
            return {}
        order = sorted(range(len(transitions)),
                       key=lambda i: (transitions[i].user_id, transitions[i].step))
        tr = [transitions[i] for i in order]
        s = torch.as_tensor(np.array([t.state for t in tr], dtype=np.float32))
        ns = torch.as_tensor(np.array([t.next_state for t in tr], dtype=np.float32))
        a = torch.as_tensor([t.action for t in tr], dtype=torch.int64)
        r = np.array([t.reward for t in tr], dtype=np.float64)
        d = np.array([float(t.done) for t in tr])
        uid = np.array([t.user_id for t in tr])
        last = np.r_[uid[1:] != uid[:-1], True]           # last transition of its user

        if c.scale_rewards:
            # forward discounted sum per user, the quantity Engstrom et al. normalise by
            ret, run = np.empty_like(r), 0.0
            for i in range(len(r)):
                run = r[i] + (c.gamma * run if i and not last[i - 1] else 0.0)
                ret[i] = run
            self.ret_var.update(ret)
            r = r / (self.ret_var.std + 1e-8)

        with torch.no_grad():
            v = self.critic(s).squeeze(1).numpy().astype(np.float64)
            vn = self.critic(ns).squeeze(1).numpy().astype(np.float64)
            old_logp = torch.log_softmax(self.actor(s), -1).gather(1, a.unsqueeze(1)).squeeze(1)
        delta = r + c.gamma * vn * (1.0 - d) - v
        adv = np.zeros_like(r)
        acc = 0.0
        for i in range(len(r) - 1, -1, -1):
            if last[i]:
                acc = 0.0
            acc = delta[i] + c.gamma * c.gae_lambda * (1.0 - d[i]) * acc
            adv[i] = acc
        ret_t = torch.as_tensor(adv + v, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)

        n = len(tr)
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
        for _ in range(c.epochs):
            perm = torch.randperm(n, generator=self.gen)
            for start in range(0, n, c.minibatch):
                idx = perm[start:start + c.minibatch]
                if len(idx) < 2:
                    continue
                logits = self.actor(s[idx])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, a[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp - old_logp[idx])
                mb_adv = adv_t[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = -torch.min(ratio * mb_adv,
                                torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * mb_adv).mean()
                vf = F.mse_loss(self.critic(s[idx]).squeeze(1), ret_t[idx])
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg + c.value_coef * vf - c.entropy_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, c.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    lr_ = logp - old_logp[idx]
                    stats["kl"].append(float(((torch.exp(lr_) - 1) - lr_).mean()))
                    stats["clipfrac"].append(float(((ratio - 1).abs() > c.clip).float().mean()))
                stats["pg"].append(pg.item()); stats["vf"].append(vf.item()); stats["ent"].append(ent.item())
        return {k: float(np.mean(x)) if x else float("nan") for k, x in stats.items()}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "hidden": self.hidden, "state_size": self.state_size}, str(path))


@dataclass
class PPOLog(TrainLog):
    entropy: list = field(default_factory=list)
    value_loss: list = field(default_factory=list)
    approx_kl: list = field(default_factory=list)
    clipfrac: list = field(default_factory=list)
    challenge_share: list = field(default_factory=list)


def train_ppo(
    humans, bots, cache,
    episodes: int = 120,
    use_score: bool = True,
    reward_source: str = "oracle",           # "oracle" | "proxy" (relabelled via T.relabel)
    solve=DETERMINISTIC,
    cfg=PAPER_EQUIVALENT,
    proxy_cfg: ProxyRewardConfig | None = None,
    human_counts=(80, 100),
    bot_choices=(0, 20, 50, 100, 200, 500),
    seed: int = 0,
    name: str | None = None,
    verbose: int = 20,
    hidden: tuple[int, ...] = DQN_WINNER_HIDDEN,
    device: str | None = "cpu",
    ppo_cfg: PPOConfig | None = None,
    **_ignored,                              # train_every / target_every / batch_size: DQN-only
) -> tuple[TrainablePPO, PPOLog]:
    """Train one PPO agent with train_dqn's episode loop. Returns (agent, log)."""
    proxy_cfg = proxy_cfg or ProxyRewardConfig()
    name = name or f"PPO-{reward_source}"
    agent = TrainablePPO(name=name, use_score=use_score, seed=seed, hidden=hidden,
                         cfg=ppo_cfg, device=device or "cpu")
    rng = random.Random(seed + 1)
    log = PPOLog()
    for ep in range(episodes):
        n_h = rng.randint(*human_counts)
        n_b = rng.choice(bot_choices)
        result, transitions = T.run_x(agent, humans, bots, n_b, cache=cache, n_humans=n_h,
                                      solve=solve, cfg=cfg, seed=rng.randrange(10 ** 9),
                                      collect=True)
        n_labels = 0
        if reward_source == "proxy":
            transitions, n_labels = T.relabel(transitions, proxy_cfg, rng)
        st = agent.update(transitions)

        log.episodes.append(ep)
        log.epsilon.append(agent.epsilon)
        log.loss.append(st.get("pg", float("nan")))
        log.value_loss.append(st.get("vf", float("nan")))
        log.entropy.append(st.get("ent", float("nan")))
        log.approx_kl.append(st.get("kl", float("nan")))
        log.clipfrac.append(st.get("clipfrac", float("nan")))
        log.surviving_humans.append(result.surviving_humans)
        log.surviving_bots.append(result.surviving_bots)
        log.n_bots.append(n_b)
        log.labels_fired.append(n_labels)
        log.challenge_share.append(result.challenges_issued / max(sum(result.action_hist.values()), 1))
        if verbose and (ep + 1) % verbose == 0:
            print(f"  [{name}] ep {ep + 1:4d}/{episodes}  ent={log.entropy[-1]:.3f} "
                  f"kl={log.approx_kl[-1]:.4f} vf={log.value_loss[-1]:.3f} "
                  f"chal={log.challenge_share[-1]:.2f}  {n_h}h/{n_b}b -> "
                  f"{result.surviving_humans}h/{result.surviving_bots}b", flush=True)
    return agent, log


def load_ppo(path, device="cpu") -> TrainablePPO:
    """A greedy TrainablePPO from a checkpoint written by `TrainablePPO.save`."""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    agent = TrainablePPO(use_score=ck["state_size"] == 5, hidden=tuple(ck["hidden"]), device=device)
    agent.actor.load_state_dict(ck["actor"])
    agent.critic.load_state_dict(ck["critic"])
    return agent.eval_mode()
