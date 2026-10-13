# Response to Reviewers

We thank the editors and reviewers. Every number below appears in the revised manuscript and is taken from saved experiment outputs. Section, table and figure numbers refer to the revised manuscript.

## Associate Editor

- **Gap between Table 4 and the claims.** The main table has been rebuilt on held-out recordings. Intervals are now computed across 5 training seeds, and the zero-bot case is excluded (Table 4).
- **Claims narrowed.** A scope table separates what was evaluated, what was implemented and what is only proposed (Table 1). Usability is described as *modelled*. The server-resource DoS claim has been removed.
- **Clarity and examples.** Methodology bullet lists are now prose. We added:
  - a worked example (§III-D);
  - a parameter table with the source of each value (Table 2);
  - an LSTM figure (Fig. 2).

## Reviewer 1

**1. Deterministic simulator; the deployed reward is unknown.**
- **Stochastic bot model.** Bots now solve challenges with an item-response probability, σ(1.5(B_s + 0.5 − T)).
- **Human pass rates, solve times and abandonment** come from Bursztein et al. 2010 and Searles et al. 2023 (§III-F, Table 2).
- **Results in the stochastic world** are in Table 5. Thompson Sampling has the highest DI there (91.5 vs 88.6 for DQN), which qualifies the original DQN-first claim.
- **Deployable rewards are tested in §IV-D.**
  - All 12 rewards computed from observable outcomes collapse to "never challenge".
  - Scorer-based rewards work when the scorer is accurate, but inherit its errors under drift (§IV-E, Fig. 4).
  - The oracle reward needs true labels, and we state this explicitly.

**2. Train/evaluation independence.**
- **Split point.** The split is made at the recording level, before windowing.
- **Partition.** Scorer, RL and evaluation blocks are rotated across folds. In fold 2, the evaluation block has 14 humans and 36 bots and is never used for fitting or model selection.
- **Cross-fitting.** Scorers are cross-fitted so that every RL training session is scored by a model that did not see it.
- **Augmentation.** None is used: a search showed it does not help the correctly sized model.
- **Counts.** 156 recordings: 44 human (from 3 participants) and 112 bot.
- **Leakage test.** A role-reversal test measures a leakage gap of 4.5 DI (SE 2.1) for DQN, and exactly 0 for static policies. The policy ranking survives on unseen recordings.

**3. Reward scale; usability claims.**
- **Friction.** The exponential friction term (−999 at T = 10) is replaced by a cost per measured second.
- **After a leaving event.** The session ends.
- **Reward sensitivity.** Retraining under 9 reward settings (5 seeds each) beats every static baseline in all 9. 3 of 45 runs collapsed, and we disclose them.
- **Leave-after-pass.** This is now labelled an assumption. A sensitivity world that uses Searles' abandonment-at-presentation rates is reported as a pilot (Table 6).
- **Wording.** Usability is described as modelled in the abstract, the discussion and the conclusion.

**4. Table 4: zero-bot case, standard deviations, "significantly".**
- Zero-bot traffic is removed from every metric and average.
- ± is now a 95% CI over 5 training seeds, and we also report the worst seed.
- "Significant" is used only for Welch tests with Holm correction.

**5. DoS claim and the population confound.**
- The server-resource DoS claim has been removed.
- Training now draws humans (50–250) and bots (0–1000) independently.
- **Evaluation with both populations varied.** Table 4 has a new column for traffic in which human (64–239) and bot (3–717) counts are drawn independently for each run, on the same held-out recordings and 5 training seeds. The results barely change: DQN 99.7 ± 0.1, Thompson Sampling 88.9 ± 17.6, best static 78.1, DQN without the score 21.5. The counts are drawn per run, not crossed in a full factorial grid.
- Freezing the active-user feature changes human survival by at most 4.4 points and bot survival by at most 1.8. It does change the share of users challenged (0.204–0.357 vs 0.305), and we report this.

**6. Reproducibility.**
- **Training details.** Episodes, steps per episode, seeds, architecture searches and all hyperparameters are in Table 2.
- **Bandit statistics.** These are recomputed in the final embedding space (§III-E).
- **Per-class replay buffers.** These use the true class, as the oracle reward does. For label-free rewards they are balanced on the observable outcome instead. Both are stated in the paper.

**7. Scorer description and calibration.**
- **Description.** It now matches the model used: one 32-unit LSTM, a 32-unit dense layer, a sigmoid output and temperature scaling; 6,081 parameters.
- **Inputs.** Δx, Δy, Δt, speed and acceleration, plus a mask channel (Fig. 2).
- **Calibration.** ECE falls from 0.064 to 0.046 after temperature scaling; Brier score is 0.065.
- **Unseen users.** Leave-one-campaign-out AUC is 0.890.

**8. Scope narrower than the motivation.** Table 1 separates what was evaluated, implemented and proposed. VLM-based and adaptive attackers are listed as not evaluated.

**9. External-dataset protocol.**
- **Withdrawn results.** On re-examination, the original fine-tuning results did not support a bot-detection claim, and we have withdrawn them (§IV-A):
  - Balabit contains humans only.
  - The web-bot runs split windows at random and used the test set for early stopping.
- **New role for external data.** It is used only as unseen populations, with session-level and user-level splits, in the drift study.

**Minor comments.**
- **Ethics statement.** Participant count added.
- **References corrected:** Khan et al. (2024, ACM CSUR 56(6)), Iliou et al. (DOI), Searles et al. (authors, pages).
- **References removed:** the uncited [4], and [6], [14], [33] and [45], which are no longer cited. L-MACD could not be verified in any index and was dropped.
- **Foundational citations added:** LinUCB (Li et al. 2010), Thompson Sampling (Russo et al. 2018), DQN (Mnih et al. 2015), PPO, LSTM (Hochreiter & Schmidhuber 1997), temperature scaling (Guo et al. 2017), and CAPTCHA usability (Bursztein et al. 2010).
- **Figures and language.** All figures are referenced in the text, and the language has been revised.

## Reviewer 2

1. "Difficulty in design" and "research gaps" are now in the Introduction.
2. The methodology is written as prose, with motivation and references, a worked example (§III-D) and the architecture figure.
3. An LSTM figure and the reasons for the design are given (§III-C, Fig. 2).
4. Every parameter's source is listed: literature, cross-validated search, or a stated assumption with its sensitivity range (Table 2).
