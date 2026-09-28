"""Where everything lives.

`expkit` is additive: it imports `rlcaptcha` but never modifies it. Every artefact it
writes goes under `training/experiments/results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent.parent      # training/experiments
TRAINING_ROOT = EXP_ROOT.parent                         # training
PROJECT_ROOT = TRAINING_ROOT.parent                     # Captcha_RL_Project

# Make `import rlcaptcha` work from a notebook in experiments/.
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

# The two recording campaigns.
CAMPAIGN_A = PROJECT_ROOT / "offline" / "train_data"    # Aug 2025, 83 sessions
CAMPAIGN_B = TRAINING_ROOT / "data"                     # Nov 2025, 73 sessions

RESULTS = EXP_ROOT / "results"
PARTITION_JSON = RESULTS / "partition.json"
SCORERS = RESULTS / "scorers"
AGENTS = RESULTS / "agents"

for _d in (RESULTS, SCORERS, AGENTS):
    _d.mkdir(parents=True, exist_ok=True)
