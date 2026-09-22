"""Paths and the simulation / reward constants used for the paper results.

Everything the experiments depend on numerically lives here, so a reviewer can see
the whole configuration on one screen.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

HUMANITY_SCORER = CHECKPOINTS / "humanity_scorer" / "big_model.keras"
DQN_CKPT = CHECKPOINTS / "dqn" / "offline_rl_model_1_200.pt"
DQN_ABLATION_CKPT = CHECKPOINTS / "dqn_ablation" / "offline_rl_model_0_1373.pt"
LINUCB_CKPT = CHECKPOINTS / "linucb" / "best_model_1.pt"
THOMPSON_CKPT = CHECKPOINTS / "thompson_sampling" / "best_model.pt"

# --- Simulation ------------------------------------------------------------
N_ACTIONS = 11                  # threat levels 0..10
SIM_STEP_SECONDS = 10           # agent is called every 10s of session time
SESSION_CHUNK_MS = SIM_STEP_SECONDS * 1000
MAX_CHUNKS = 12                 # ~120s sessions -> 12 decision points
SCORER_CONTEXT = 100            # mouse movements per LSTM window

EVAL_HUMANS = 100
EVAL_BOT_COUNTS = (0, 20, 100, 200, 500, 1000)


@dataclass(frozen=True)
class RewardConfig:
    """Reward function parameters (Section III-F-3 of the manuscript).

    Two fields differ between the published runs and are NOT cosmetic:

    * `overkill` -- 2.5 in the DQN/ablation/static benchmarks (the value printed in the
      paper), 2.0 in the LinUCB and Thompson Sampling benchmarks.
    * `abandonment_applies_to_bots` -- False for DQN/ablation/static, True for the two
      bandits, where the "user gave up" check was written outside the human branch so
      bots abandon the site too. This is the single largest driver of the bandits' bot
      numbers at high load.

    Both presets are kept so every row of Table 4 reproduces; see README
    "Known inconsistencies" for what that means for the comparison.
    """

    max_reward: float = 50.0         # R_max
    leakage_penalty: float = -150.0  # R_leak
    friction_base: float = 2.0       # beta
    overkill: float = 2.5            # overkill penalty coefficient
    underestimation: float = 5.0     # underestimation penalty coefficient
    abandonment_applies_to_bots: bool = False


#: Used by DQN, the DQN ablation, and both static-threshold baselines.
DQN_REWARDS = RewardConfig(overkill=2.5, abandonment_applies_to_bots=False)

#: Used by LinUCB and Thompson Sampling.
BANDIT_REWARDS = RewardConfig(overkill=2.0, abandonment_applies_to_bots=True)
