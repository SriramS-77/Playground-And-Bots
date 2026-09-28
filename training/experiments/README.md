# Reviewer-response experiments

Everything run in answer to the OJCS reviews of *Balancing Security and Usability:
Adaptive CAPTCHA Orchestration via Reinforcement Learning* (OJCS-2026-04-0359).

**Nothing under `training/rlcaptcha/` is modified.** `expkit` imports it and extends it,
so `reproduce_table4.ipynb` and `reproduce_seeded_table_4.ipynb` still produce exactly
what they produced before. Every artefact lands in `experiments/results/`.

```bash
python build_notebooks.py      # nb_*.py  ->  *.ipynb
python run_all.py              # execute them in order, logs to logs_NN.txt
python run_all.py 03 06        # or just some
```

## The notebooks

| # | Answers | What it does |
|---|---|---|
| 01 | R1.2 | Audits the published split, builds the session-level three-way partition |
| 02 | R1.7 | Retrains the scorer with timestamps / kinematics / per-movement jitter; ECE, Brier, reliability, temperature scaling |
| 03 | R1.2 | Table 4 under progressively cleaner evaluation, with retrained agents |
| 04 | R1.3, R2.4 | Reward sensitivity — including retraining under each setting |
| 05 | R1.5 | Independent human/bot traffic, and ablating the population feature |
| 06 | R1.1 | Probabilistic action model grounded in published measurements |
| 07 | R1.1 | A reward built only from signals a deployment can observe |
| 08 | — | Whether 156 recordings can support the claims |

## `expkit`

| module | contents |
|---|---|
| `partition` | session-level three-way split, stratified by class / bot family / campaign |
| `features` | input representations (`xy`, `xy_dt`, `dxdy_dt`, `kinematic`) and perturbation models (`rigid`, `per_move`, `gaussian`) |
| `scorer_train` | retrains the LSTM under a clean protocol; architecture verified against the published checkpoint |
| `calibration` | ECE (uniform and equal-mass), MCE, Brier + Murphy decomposition, temperature scaling |
| `xscoring` | adapters letting a retrained scorer drive the simulation; `PerMoveScoreCache` for when the rigid-translation assumption is dropped |
| `stochastic` | the 2PL item-response action model and its reward config |
| `proxy` | observable-only reward with delayed, noisy abuse confirmation |
| `simx` | extended simulation loop — a verified superset of `rlcaptcha.simulate` |
| `trainer` | DQN training on an arbitrary pool and reward source |
| `bootstrap` | session-level uncertainty, variance decomposition, learning curves |

## The equivalence guarantee

`simx.run_x(policy, ..., solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT)` reproduces
`rlcaptcha.simulate.run_simulation` **exactly** — identical survivor counts, not merely
the same distribution — across all six policies, all six bot volumes and four seeds
(144/144, asserted in notebook 06). Getting there required matching the published code's
RNG *draw pattern*, not just its probabilities:

* a certain outcome draws no random number (the published code resolved `T > B_s` by
  comparison, and a human who always passes was never sampled);
* the abandonment check keeps its short circuit at level 10;
* in the bandit configuration the abandonment check is evaluated even for a bot that was
  already blocked, because the published bandit loops evaluate it unconditionally.

So the probabilistic model is a continuous deformation of the published one rather than a
different environment, and `alpha -> infinity` returns to Table 4.

## Two reproducibility defects found in the published evaluation

Both are in code faithfully ported into `rlcaptcha`, and neither is fixed there — they
are worked around on the caller side in `simx.run_x(seed_global=True)`:

1. **Thompson Sampling draws from the global torch RNG** (`torch.randn_like`), which the
   `seed` argument never reached. Published TS runs were not reproducible from their seed.
2. **DQN epsilon-greedy draws from the global numpy RNG** (`np.random.rand`). Same issue
   whenever epsilon > 0, which it is for the published checkpoint (~0.01).

This is worth a line in the revised paper's reproducibility statement, and it is part of
why Thompson Sampling was the high-variance outlier in `results_seeded/`.

## Known gaps in this work

Stated plainly so they do not get discovered by a reviewer instead.

1. **The two bandits were not retrained.** LinUCB and Thompson Sampling keep their
   published checkpoints throughout, and those were fitted on campaign B — which overlaps
   the `eval` pool. So in the "clean" condition of notebook 03 they are the one component
   whose training still touches the evaluation recordings, and their numbers there should
   not be read as leakage-free. Retraining a neural bandit (feature network plus per-arm
   ridge statistics) is a contained piece of work and is the first thing to add.
2. **Training budget is smaller than the published run.** The retrained agents get
   60-150 episodes against the published checkpoints' 200+, on a 31-session pool rather
   than 73. Notebook 03's `leaky control` condition exists to separate that from
   leakage; without it the drop from the published Table 4 is not interpretable.
3. **Participant count is unknown.** The recordings carry no participant field. If the
   44 human sessions come from a handful of people, the partition needs to be at
   participant level and the external-validity claim has to narrow accordingly. No
   amount of compute here can close this.
4. **The scorer carried into notebooks 03-08 is `kinematic`**, chosen by a rule fixed
   before results were seen. Its lead rests on one of three seeds, and no variant is
   statistically distinguishable from any other (notebook 02). A different defensible
   rule would have chosen `xy__per_move`. The absolute numbers downstream depend on this
   choice; the qualitative conclusions do not.
5. **`abandonment_tau` is a modelling choice.** The empirical curve fires per decision
   step while the source measurements are per CAPTCHA encounter, so a 12-step session
   compounds the hazard. `tau = 45s` is used by default and 22/45/80 are swept in
   notebook 04; the policy ranking is stable across that range but absolute survival is
   not.

## Literature the action model is built on

* Bursztein, Bethard, Fabry, Mitchell, Jurafsky. *How Good Are Humans at Solving
  CAPTCHAs? A Large Scale Evaluation.* IEEE S&P 2010. DOI 10.1109/SP.2010.31 —
  318k CAPTCHAs, 21 schemes; image accuracy 87% mean (authorize.net 0.98 / 6.8s,
  mail.ru 0.70 / 12.8s), audio 52%.
* Searles et al. *An Empirical Study & Evaluation of Modern CAPTCHAs.* USENIX Security
  2023 — 1,400 participants, 14k CAPTCHAs; reCAPTCHA checkbox 3.7s median, game-based
  18-42s, abandonment 120% higher in realistic contexts.
* Plesner, Vontobel et al. *Breaking reCAPTCHAv2.* 2024 — YOLO solver at 100% on image
  challenges, prior SOTA 68-71%.
* Teoh et al. *Are CAPTCHAs Still Bot-hard? Generalized Visual CAPTCHA Solving with
  Agentic Vision Language Model.* USENIX Security 2025 — 60.7% across 26 types, 70.6% on
  unseen challenges in the wild.
* Li, Chu, Langford, Schapire. *A Contextual-Bandit Approach to Personalized News Article
  Recommendation.* WWW 2010. DOI 10.1145/1772690.1772758.
* Russo, Van Roy, Kazerouni, Osband, Wen. *A Tutorial on Thompson Sampling.* FnT ML 2018.
  DOI 10.1561/2200000070.
* Mnih et al. *Human-Level Control Through Deep Reinforcement Learning.* Nature 2015.
  DOI 10.1038/nature14236.

The last three are the foundational citations Reviewer 1 asked for; the first four are
what the action model is calibrated against.
