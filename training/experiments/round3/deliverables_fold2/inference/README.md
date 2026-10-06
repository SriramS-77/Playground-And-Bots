# Standalone Inference Pipeline — Round 3 (Fold 2)

This directory provides a completely self-contained, out-of-the-box inference pipeline for the **Adaptive CAPTCHA Orchestration** agent trained during **Round 3**.

---

## 1. Components

1. **Calibrated Kinematic Humanity Scorer (`scorer/`):**
   - Fitted on Fold 2's `scorer + rl` block (the refit scorer, validation AUC `0.986`).
   - 6,081-parameter LSTM with kinematic cursor representations (dx, dy, dt, speed, acceleration, jerk, curvature) and masking.
   - Temperature calibrated (`temperature=1.452`).

2. **Winning Deep Q-Network Policy Checkpoints (`agents/`):**
   - Fold 2 winner selected in the pre-registered architecture search: `dqn:128-64` (9,739 parameters).
   - Checkpoints `dqn_s0.pt` through `dqn_s4.pt` (seeds 0 to 4).

3. **Feature extraction and scoring:** the repository's own `training/experiments/expkit/`. The cluster
   bundle shipped a copy, byte-identical apart from line endings; it was dropped so the two cannot drift.

---

## 2. Quickstart

Run inference directly using:

```bash
python run_inference.py
```

### Pipeline Flow:
1. Loads the calibrated Fold 2 Refit Humanity Scorer.
2. Loads the Fold 2 winning DQN `128-64` policy checkpoint (`agents/dqn_s0.pt`).
3. Ingests raw mouse movement telemetry (supporting either `'timestamp'` or `'time'`).
4. Extracts 32-step kinematic windows and predicts P(bot) (HIGH = bot-like; the manuscript's H is 1 - P(bot)).
5. Constructs the standard 5-dimensional environment state vector:
   `[bot_score, avg_bot_score, tanh(solved / 5), last_threat_level, n_active / 300]`
6. Performs forward Q-value evaluation across threat levels 0 to 10 and outputs the optimal challenge mechanism.

---

## Corrections made after delivery

* **Inverted score (fixed).** `HumanityScorer.score_chunk` returns P(bot). The delivered script read it
  as a humanity score and passed `1 - score` to the DQN, so the policy saw P(human) -- every decision
  ran backwards. It now passes P(bot), the quantity the policy was trained on as `bot_score`. Checked on
  real recordings: bot sessions score 0.64-0.996, human sessions 0.02-0.08.
* **Keras version.** `model.keras` was saved by a newer Keras (cluster: Python 3.14, Keras 3) whose layer
  configs older versions cannot parse. `load_scorer` falls back to rebuilding the architecture from
  `scorer.json` and loading only the weights.
* The sample "human-like" trajectory gets P(bot) = 0.034 and the DQN picks level 4: levels 0-4 have
  near-equal Q-values (978-982), consistent with the ~7 s of friction per human the headline reports.
