# Response to Reviewers

We thank the editors and reviewers. Every number below appears in the revised manuscript and comes from saved experiment outputs. Section, table and figure numbers refer to the revised manuscript.

## Associate Editor

- **Gap between Table 4 and the claims.** The main comparison is rebuilt on held-out recordings in a stochastic, literature-grounded simulator. Intervals are 95% CIs over 5 training seeds, and the zero-bot case is excluded (Table 4).
- **Claims narrowed.** Fig. 1 marks which tiers are implemented and which are simulated, and the Limitations (§V-B) state what was not evaluated. Usability is described as *modelled*, and the server-resource DoS claim is removed.
- **Clarity.** The methodology is now prose. We added a worked example (§III-D), a configuration table giving the source of every value (Table 2), and a scorer figure (Fig. 2).

## Reviewer 1

**1. Deterministic simulator; the deployed reward is unknown.**
- **Bots** now solve challenges with an item-response probability, σ(1.5(B_s + 0.5 − T)). **Human** pass rates and solve times come from Bursztein et al. (2010) and Searles et al. (2023) (§III-F, Table 1).
- **Results.** In this world PPO and DQN reach DI 99.8 and 97.8, against 95.5 for the best static policy, a Neyman–Pearson threshold (Table 4).
- **Deployable rewards** (§III-D, Table 5):
  - The label-based reward needs true labels, and the paper says so.
  - Two rewards that need no labels, score-only and an EM-based posterior, are defined and tested against it, along with two rewards that recalibrate the score from a 2% audit sample.
- **Online training under drift** (§IV-D):
  - When covert bots break the scorer's calibration, frozen RL policies degrade faster than static thresholds.
  - Online training with the label-free score-only reward recovers this. PPO keeps DI 82.7 at ρ = 1, against 3.8 frozen and 39.5 for the best static policy (Fig. 4).
  - The EM-based posterior reward fails here, because it inherits the scorer's blind spot.
  - Recalibrating the reward from a random 2% audit sample matches score-only under covert bots and, unlike it, holds up when bot-like humans shift the score (58.1 against 17.5 at ρ = 1; Table 5).
- **Gated online training.** Always-online training costs performance when nothing has changed (92.4 against 99.8 frozen). We therefore keep the policy frozen until an unsupervised CUSUM detector on the incoming scores raises an alarm. The detector needs no labels. It never fired in any no-drift run, and it detected covert bots within 1–5 episodes. The gated policy keeps the frozen policy’s 99.8 without drift and reaches 85.1 and 82.0 under covert bots, at least as good as always-online training (Table 6).
- **Trade-off between humans lost and bots stopped** (Fig. 5).
  - Without drift, PPO stops every bot while losing under 1% of humans; the best static threshold loses 4.4%.
  - Under covert bots, static thresholds stop all bots only at 27% of humans lost.
- **Stated limit.** Bots that replay human trajectories destroy the score's ranking, and then every policy falls to DI ≈ 0 (at most 22 even with full labels).

**2. Train/evaluation independence.**
- Recordings are split before windowing into scorer, RL and evaluation blocks, rotated across folds. The fold-2 evaluation block (14 humans, 36 bots) is never used for fitting or model selection.
- Scorers are cross-fitted, so every RL training session is scored by a model that did not see it.
- No data augmentation is used.
- The dataset has 156 recordings: 44 human, from 3 participants, and 112 bot (§III-B).

**3. Reward scale; usability claims.**
- **Friction.** The exponential friction term is replaced by a cost per measured second of challenge time.
- **Leaving after a pass** is labelled an assumption. With τ varied from 22 s to no leaving, PPO and DQN remain the best policies at every value (§IV-C).
- **Reward constants.** DQN retrained under 9 alternative reward settings beats the static threshold on average in every setting. The one seed that degraded is reported.
- **Wording.** Usability is described as modelled throughout.

**4. Table 4: zero-bot case, standard deviations, "significantly".**
- Zero-bot traffic is removed from every metric.
- ± is a 95% CI over 5 training seeds, and the worst seed is also reported.
- The word "significant" has been removed; differences are reported with 95% CIs over training seeds.

**5. DoS claim and the population confound.**
- The server-resource DoS claim is removed.
- Training draws humans (50–250) and bots (0–1000) independently.
- Table 4 adds a varying-traffic column, with 50–250 humans and bots making up 5–75% of users. Results are unchanged: PPO 99.96, DQN 99.4, best static 95.2.
- Freezing the active-user feature leaves bot survival at zero. The feature controls how cautious the policy is, not whether it stops bots (§IV-C).

**6. Reproducibility.**
- Table 2 gives episodes, seeds, architectures and all hyperparameters, with how each was selected.
- §III-E gives the update rules of all four learners.
- Bandit statistics are recomputed in the final embedding space.
- The Thompson Sampling posterior was chosen from four variants: diagonal Gaussian, full Gaussian, and mixtures of two and three Gaussians.

**7. Scorer description and calibration.**
- **Description.** It now matches the model used: one 32-unit LSTM, a 32-unit ReLU layer and a linear logit (6,081 parameters). Training uses a sigmoid with cross-entropy. At inference the logit is calibrated by temperature scaling, T = 1.14 (§III-C, Fig. 2).
- **Calibration.** On the evaluation block, the Brier score is 0.032 and ECE is 0.053. We report that temperature scaling does not reduce ECE there (§IV-A).

**8. Scope narrower than the motivation.** Fig. 1 marks tiers 1–3 as implemented and tier 4 as simulated. The Limitations (§V-B) list what was not evaluated: real CAPTCHA providers, server load (request rate, CPU, latency), and adaptive or VLM-based attackers; a user study is left to future work (§V-C).

**9. External-dataset protocol.**
- The original fine-tuning results are withdrawn. Balabit contains only humans, and the web-bot runs split windows at random.
- They are replaced by a reproducible benchmark comparison, described in §IV-A:
  - It uses the official session-level splits of Iliou et al.'s D1 and D2.
  - Early stopping uses validation sessions held out from training, never the test set.
  - Results are averaged over 5 seeds.
- The fine-tuned scorer classifies every test session correctly. This matches their mouse module on D1 and exceeds it on D2, the human-like bots (1.00 vs 0.93) (Table 3).
- The external populations otherwise serve only as unseen data in the drift study.

**Minor comments.**
- **Ethics statement and participant count** added.
- **References** corrected, and unverifiable or uncited ones removed.
- **Foundational citations added** for DQN, PPO, LinUCB, Thompson Sampling, LSTM, temperature scaling, CUSUM and CAPTCHA usability.
- **Figures.** All figures are referenced in the text.

## Reviewer 2

1. "Difficulty in design" and "research gaps" are now in the Introduction.
2. The methodology is written as prose, with motivation, references, a worked example (§III-D) and architecture figures.
3. The scorer figure and the reasons for its design are given (§III-C, Fig. 2).
4. Table 2 lists each parameter's source: literature, cross-validated search, or a stated assumption with its tested sensitivity range.
