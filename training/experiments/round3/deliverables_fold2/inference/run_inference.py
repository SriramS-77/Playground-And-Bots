"""Self-contained inference script for Adaptive CAPTCHA Orchestration (Round 3 -- Fold 2).

Rebuilt for Round 3:
1. Loads the calibrated kinematic Humanity Scorer (6,081-param LSTM with context=32,
   start_from_epoch=30, lr=3e-3, dropout=0.25, recurrent_dropout=0.2)
   trained on the fold 2 refit data (AUC 0.986).
2. Accepts mouse telemetry dicts supporting both 'timestamp' and 'time' keys,
   properly normalizing to 'timestamp' matching expkit.features.
3. Loads Round 3 trained DQN policy agents (Fold 2 winner 128-64),
   reading the network architecture dynamically from checkpoint metadata.
4. Standardises state vector [bot_score, avg_bot_score, captchas_solved, last_threat, n_active]
   matching the training protocol.
5. Predicts orchestrated threat level (0 to 10) in single-sample inference mode.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# expkit is the repository's own (training/experiments/expkit); the cluster bundle shipped a
# byte-identical copy, dropped here so the two cannot drift.
sys.path.insert(0, str(HERE.parents[2]))

from expkit.humanity_scorer import HumanityScorer

N_ACTIONS = 11

LEVEL_NAMES = [
    "Level 0: No challenge (allow)",
    "Level 1: Passive risk score (reCAPTCHA v3)",
    "Level 2: Honeypot / JS challenge",
    "Level 3: Checkbox challenge (reCAPTCHA v2 click)",
    "Level 4: Easy distorted text",
    "Level 5: Standard distorted text",
    "Level 6: Image grid selection",
    "Level 7: Hard distorted text",
    "Level 8: Very hard multi-round text",
    "Level 9: Interactive game challenge",
    "Level 10: Audio fallback / hard block",
]


class DynamicQNet(nn.Module):
    """Dynamic MLP Q-Network supporting configurable hidden layer architectures."""

    def __init__(self, state_size: int = 5, action_size: int = N_ACTIONS, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        dims = [state_size, *hidden]
        for i in range(len(hidden)):
            setattr(self, f"layer{i + 1}", nn.Linear(dims[i], dims[i + 1]))
        setattr(self, f"layer{len(hidden) + 1}", nn.Linear(hidden[-1], action_size))
        self.n_layers = len(hidden) + 1

    def forward(self, x):
        for i in range(1, self.n_layers):
            x = F.relu(getattr(self, f"layer{i}")(x))
        return getattr(self, f"layer{self.n_layers}")(x)


def load_scorer(directory: Path) -> HumanityScorer:
    """`HumanityScorer.load`, falling back to rebuilding the architecture from scorer.json and
    loading only the weights: the cluster saved model.keras with a newer Keras whose layer
    configs (`input_axes`, `quantization_config`) older Keras versions cannot parse."""
    try:
        return HumanityScorer.load(directory)
    except (TypeError, ValueError):
        from expkit.humanity_scorer import ScorerConfig, Standardiser
        meta = json.loads((directory / "scorer.json").read_text())
        cfg = ScorerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in meta["config"].items()})
        model = cfg.build()
        model.load_weights(str(directory / "model.keras"))
        return HumanityScorer(cfg, model, Standardiser.from_dict(meta["standardiser"]),
                              meta.get("temperature", 1.0))


def load_dqn_agent(weights_path: Path):
    """Loads a DQN model checkpoint, auto-detecting hidden layers and state size."""
    device = torch.device("cpu")  # CPU pinned for fast single-sample inference
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    state_size = ckpt.get("state_size", 5)
    hidden = ckpt.get("hidden", (128, 64))
    
    model = DynamicQNet(state_size=state_size, action_size=N_ACTIONS, hidden=hidden).to(device)
    state_dict = ckpt["policy_net_state_dict"] if "policy_net_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, hidden, state_size


def format_state(bot_score: float, avg_bot_score: float, captchas_solved: int,
                 last_threat_level: int, n_active_users: int) -> np.ndarray:
    """Standardises state features exactly matching training protocol."""
    s = np.array([
        bot_score,
        avg_bot_score,
        np.tanh(captchas_solved / 5.0),
        last_threat_level,
        n_active_users / 300.0,
    ], dtype=np.float32)
    return s


def normalize_movement_chunk(chunk: list[dict]) -> list[dict]:
    """Ensures each movement dict carries 'timestamp' (supporting 'time' alias)."""
    norm = []
    t_running = 0.0
    for i, m in enumerate(chunk):
        t = m.get("timestamp", m.get("time", t_running))
        t_running = float(t)
        norm.append({
            "x": float(m["x"]),
            "y": float(m["y"]),
            "timestamp": t_running,
        })
    return norm


def main():
    print("=" * 70)
    print("Adaptive CAPTCHA Orchestration -- Round 3 (Fold 2) Inference Pipeline")
    print("=" * 70)

    # 1. Load Humanity Scorer (Refit Scorer from Fold 2)
    scorer_dir = HERE / "scorer"
    print(f"\n[1] Loading Humanity Scorer from: {scorer_dir}")
    scorer = load_scorer(scorer_dir)
    print(f"    Scorer config: {scorer.cfg.representation} + {scorer.cfg.padding}, "
          f"context={scorer.cfg.context}, temperature={scorer.temperature:.3f}")

    # 2. Load Agent (Fold 2 Winning DQN 128-64 Architecture)
    agent_path = HERE / "agents" / "dqn_s0.pt"
    print(f"\n[2] Loading Round 3 Agent from: {agent_path}")
    dqn_model, device, hidden, state_size = load_dqn_agent(agent_path)
    print(f"    Loaded DQN architecture: hidden={hidden}, state_size={state_size} on {device}")

    # 3. Simulate Sample User Movement Telemetry
    print("\n[3] Generating Sample Mouse Movement Telemetry:")
    # Telemetry uses 'timestamp' (or 'time'), with realistic kinematic human-like cursor movements
    t_base = 1720000000.0
    sample_chunk = [
        {"x": 100.0 + i * 2.5 + np.sin(i / 3.0) * 4.0,
         "y": 200.0 + i * 1.8 + np.cos(i / 3.0) * 3.0,
         "timestamp": t_base + i * 0.016}
        for i in range(35)
    ]
    norm_chunk = normalize_movement_chunk(sample_chunk)
    print(f"    Collected {len(norm_chunk)} cursor telemetry events.")

    # 4. Behavioral Scoring
    print("\n[4] Scoring Trajectory with Kinematic Scorer:")
    # HumanityScorer.score_chunk returns P(bot) -- HIGH means bot-like, the same quantity the
    # policy was trained on as `bot_score`. The cluster bundle read it as a humanity score and
    # inverted it, so the DQN saw P(human). The manuscript's H is 1 - P(bot).
    bot_probability = scorer.score_chunk(norm_chunk)
    if bot_probability is None:          # empty chunk: the simulator's initial 0.5
        bot_probability = 0.5
    print(f"    Bot Probability: {bot_probability:.4f} (1.0 = bot)")
    print(f"    Humanity Score : {1.0 - bot_probability:.4f} (manuscript H = 1 - P(bot))")

    # 5. State Feature Formulation
    print("\n[5] Formulating Orchestrator State Vector:")
    avg_bot_prob = bot_probability
    captchas_solved = 0
    last_threat_level = 0
    n_active_users = 150  # 150 concurrent sessions on server

    state = format_state(bot_probability, avg_bot_prob, captchas_solved, last_threat_level, n_active_users)
    print(f"    State Vector: {np.round(state, 3)}")
    print(f"    Features: [bot_score={state[0]:.3f}, avg_bot={state[1]:.3f}, "
          f"solved_norm={state[2]:.3f}, last_threat={state[3]:.1f}, pop_norm={state[4]:.3f}]")

    # 6. Action Inference
    print("\n[6] Predicting Optimal Action:")
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = dqn_model(x).squeeze(0).numpy()
        action = int(np.argmax(q_values))

    print(f"    Q-Values by Threat Level (0..10): {np.round(q_values, 2)}")
    print(f"    --> Selected Action: Threat Level {action}")
    print(f"    --> Mechanism: {LEVEL_NAMES[action]}")
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
