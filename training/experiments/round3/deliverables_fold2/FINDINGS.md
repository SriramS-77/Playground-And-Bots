# Round 3 findings — fold 2 (verified against the raw outputs)

Every number below was re-read from the CSVs in `results/` (or, where marked *task files*, from
the per-task CSVs in the cluster zip `deliverables_round3.zip`, which is not committed). This
file **replaces** the cluster agent's `FINDINGS.md`: its protocol, search and headline sections
were correct, but its E7–E9 sections contained numbers that exist in no output, a failed
pre-declared check reported as a pass, and a mislabelled parameter. Those errors are listed in
§7 so nothing is carried forward from the old text.

Conventions: DI = humans kept % − bots let through %, bots > 0 only; "seed" = training seed;
SE over training seeds.

---

## 1. Provenance and integrity

| Check | Result | Source |
|---|---|---|
| Fold isolation | fold 2 only; 446 eval-block openings logged, none for folds 0/1 | `results/eval_access.log` |
| Scorer preparation | passed on attempt 0 (0 retries); val AUC 0.950–0.986, held-out part AUC 0.920–0.992; canary keeps 65 % of humans (floor 25 %) | `results/prepare_log.json` |
| Determinism | 144/144 exact | `prepare_log.json` |
| Task completeness | search 225, headline 21, E2 50, E3 69, E4 51, E6 70, E7 93, E8 26, E9 141 — all present; 748 Slurm logs, no tracebacks, no NaNs | cluster zip |
| Code | the bundled `expkit` is byte-identical to the repository's (CRLF line endings only) | — |
| Pushes | the cluster agent pushed nothing; `origin` has only `main` at `c160cc0` | — |

Compute: Slurm on `nvidiaserver`, CPU only (Python 3.14, PyTorch 2.14, TF 2.22, Keras 3).

---

## 2. Architecture search (`results/search_*.csv`, `arch_choice.json`)

Rule (D2): best mean DI over 3 rotations × 5 seeds; within 1 SE, least friction among candidates
with human survival ≥ best − 0.05 and bot survival ≤ best + 0.05; then fewest parameters.

| Family | Winner | DI ± SE | Humans | Bots | Friction |
|---|---|---|---|---|---|
| DQN | `dqn:128-64` | 83.60 ± 0.79 | 0.870 | 0.034 | 5.3 s |
| LinUCB | `linucb:h64_e32:gaussian` | 60.40 ± 2.47 | 0.678 | 0.074 | 20.3 s |
| Thompson | `thompson:h128_64_e64:mog3` | 72.26 ± 3.15 | 0.823 | 0.101 | 5.2 s |

- **The DQN choice is a tie.** `dqn:64` scores 82.81 and the 1-SE band ends at 82.811 — inside by
  0.001 DI. The friction tie-break (5.3 s vs 9.4 s) decided it.
- **Rotation 1 does not discriminate.** Every DQN architecture and seed scores exactly 72.7 there
  (humans 0.805, bots 0.078) — a ceiling set by that rotation's recordings, not by any learner.
  The choice was effectively made on rotations 0 and 2 (`dqn:128-64` 97.9 / 80.2; `dqn:64` 100.0 / 75.0).
- Depth hurts DQN: 4 hidden layers fall to 68.4 with seed SD 11.4.
- **Mixture-of-Gaussians Thompson posteriors beat the single Gaussian** in every feature
  architecture (e.g. h64_e32: mog2 76.3 vs gaussian 56.9). Robust.
- Local PPO and state-augmentation work (scratchpad, not committed) used `dqn:64`, the earlier
  local winner, not this one.

---

## 3. Headline — eval block, deterministic world (`results/headline_*.csv`, `headline_runs.csv`)

5 training seeds × 50 evaluation seeds.

| Policy | DI published grid | DI realistic grid | Humans (pub.) | Bots (pub.) | Friction (pub.) | DI by seed (pub.) |
|---|---|---|---|---|---|---|
| DQN 128-64 | **98.88** | **99.74** | 0.989 | 0.000 | 6.96 s | 98.6–99.3 |
| Thompson (mog3) | 90.94 | 88.90 | 0.945 | 0.036 | 1.73 s | 73.4–97.2 |
| Static Multi | 78.71 | 78.06 | 0.887 | 0.100 | 9.44 s | — |
| Static Single | 77.72 | 77.65 | 0.777 | 0.000 | 6.68 s | — |
| LinUCB | 51.53 | 49.62 | 0.549 | 0.033 | 33.03 s | 13.1–93.0 |
| DQN without H-score | 23.56 | 21.50 | 0.880 | 0.644 | 29.47 s | 13.9–36.1 |

- DQN beats Static Multi by ~20 DI on both grids, stably across seeds. The score is essential:
  removing it costs ~77 DI.
- **Near ceiling.** The eval block is 14 humans and 36 bots, and the refit scorer separates them
  perfectly (AUC 1.0, `e9/e9_scorer_auc.csv`). This is why validation (83.6) and the eval block
  (98.9) differ so much; they also use different scorers, so the two are not comparable.
- Under load DQN gives ground: humans kept 1.000 at ≤ 200 bots, 0.951 at 1000 bots.
- The bandits are seed-unstable: Thompson's worst seed is 73.4, LinUCB spans 13–93.

---

## 4. Downstream experiments

### E2 — policy-level leakage (`e2/e2_gap_summary.csv`, `e2_gaps_per_split.csv`)
- DQN scores 82.74 on recordings it trained on and 78.24 on recordings it did not: gap 4.49
  (SE 2.13; normal CI [0.32, 8.67]; a t-interval with n = 10 is [−0.32, 9.31] and includes 0).
- Two of ten splits carry it (19.4 and 13.3); median gap 1.73. Seven positive, two zero, one negative.
- Ablation gap −0.06, statics exactly 0 (the harness gate, passed) → what leaks flows through the score.
- Even on unseen recordings DQN (78.2) beats Static Multi (66.5) and Static Single (64.2).
- **Neutral to mildly harmful:** modest, borderline memorisation; the ranking survives it.

### E3 — reward sensitivity and abandonment (`e3/`)
- All 9 retrained reward settings beat every static on seed means (best static 77.98); one of
  the 9 is the published setting. `linear friction` (98.96, 1.68 s) and `overkill 2.0` are Pareto.
- **3 of 45 seeds collapse** (*task files*): harsh friction s0 44.9, overkill 2.0 s2 3.6,
  under 2.5 s2 61.1 — hence their seed SDs of 24.0, 42.8, 17.0.
- Fixed-policy landscape: DQN preferred in 18/18 reward settings (seed 0 policies only).
- Abandonment (seed 0 policies): at τ = 22 s DQN 91.7, Static Single 81.4, Thompson 75.9,
  Static Multi 64.8, LinUCB 1.8.
- **Mostly beneficial**, with the seed collapses disclosed.

### E4 — population confound (`e4/e4_variants.csv`, `e4_by_cell.csv`)
- Wrapper self-test: "true" = "unfrozen" on all 3,000 cells.
- Freezing `n_active` barely moves security outcomes (humans −0.044 and bots +0.018 at worst,
  both at the 1400 freeze) — **but it moves challenge volume**: challenge share 0.305 unfrozen,
  0.204–0.357 frozen. Per seed (*task files*) the effect is concentrated: seed 1 drops 0.591 → 0.180
  frozen at 400, seed 0 0.309 → 0.180; seeds 2–4 barely move.
- Freezing at 25–800 gives slightly *higher* DI (99.4–100 vs 99.19): the population input mostly
  buys unnecessary challenges.
- **Corrected conclusion:** security outcomes are robust to `n_active`; challenge volume is not.
  This partly supports Reviewer 1's population-confound concern.

### E6 — data sufficiency (`e6/`)
- Learning curve (5 seeds per fraction): 25 % → 100.0 (SD 0), 50 % → 98.8, **75 % → 94.1 (SD 10.4;
  one seed lets 24.5 % of bots through)**, 100 % → 98.8. Non-monotone.
- Bootstrap (200 pools): DQN 100.0 with zero variance; Thompson CI [38.7, 86.7] at 100 bots,
  LinUCB [5, 36], statics roughly [54, 95].
- **Uninformative**: a 50-recording block the scorer separates perfectly puts DQN at ceiling;
  "25 % of the data suffices" is not supported and should not be presented as a strength.

### E7 — deployable rewards, grounded world (`e7/`)

Trained agents, 5 seeds (`e7_configs.csv`, `e7_head_to_head.csv`):

| Reward | DI | Humans | Bots | Seed SD | Note |
|---|---|---|---|---|---|
| posterior | 92.79 | 0.970 | 0.043 | 5.9 | |
| posterior+labels | 91.28 | 0.917 | 0.004 | 6.9 | |
| posterior+labels (floored) | 89.18 | 0.897 | 0.005 | 14.2 | |
| oracle | 88.59 | 0.886 | 0.000 | 21.7 | one seed 49.8 |
| score-only | 85.34 | 0.898 | 0.044 | 8.3 | |
| posterior (floored) | 75.83 | 0.964 | 0.206 | 24.5 | one seed lets 61.5 % of bots through |
| **all 12 proxy-family rewards** | ≈ 0 | 1.000 | 1.000 | — | never challenge |

- The proxy family (proxy; penalties −600/−2400/−9600; zero-friction 25/35; PBRS −600/−2400;
  learned reward model; normalised-clipped −2400/−9600; proxy with 1000 episodes) **all collapse
  to "never challenge"**. Observable outcomes cannot separate the classes: 47.5 % of
  `passed_continued` are bots and 11.6 % of `failed_gone` are humans (`e7_observability.csv`).
- Head to head: posterior − oracle +4.2 DI (p = 0.70); posterior − score-only +7.4 (p = 0.15);
  posterior+labels − posterior −1.5 (p = 0.72). **No significant differences among the posterior family.**
- Reward-bias diagnostic (`e7_reward_bias_summary.csv`), r̂ − oracle r per transition:

| Variant | Bias (SE) | Log-loss | Uncertain band (0.2 ≤ q < 0.8) |
|---|---|---|---|
| exact (no score, true share) | −0.27 (0.22) | 0.366 | pass |
| posterior | −2.71 (0.14) | 0.179 | **fail** |
| posterior (floored) | −1.49 (0.13) | 0.141 | **fail** — bias −11.76, skew 0.064 > 0.05 |
| posterior+labels (floored) | −0.91 (0.13) | 0.135 | pass |
| posterior+labels | −2.38 (0.14) | 0.172 | fail |
| score-only | −8.82 (0.17) | 0.293 | fail |
| proxy (different objective) | +50.81 (0.38) | — | — |

  The exact variant (the code test) is unbiased, as it must be. Every posterior variant is biased
  low by 7–20 SE.
- Bot-share estimation (`e7_em_share.csv`, 60 episodes): mean absolute error EM 0.029,
  EM floored 0.017, summed probabilities 0.099, raw score mean 0.111.
- The posterior reward uses the simulator's exact outcome model, so it is an **upper bound** on
  the idea, not a deployable reward; the floored variants were declared after the bias
  diagnostic had been seen.
- **Harmful to "deployable reward":** nothing computable from observables trains a working policy;
  the posterior family ties the oracle (not significantly better) and its floored variant is unstable.

### E8 — stochastic world (`e8/`)
- α is the **bot solve model's IRT discrimination** — P(bot solves level T) = σ(α(θ − T)),
  `expkit/stochastic.py` — not human abandonment. Human pass rates do not depend on α, so a flat
  human false-positive rate (DQN 0.0020–0.0023) is expected by design.
- **Thompson beats DQN at every α**: 90.3–92.2 vs 86.1–86.9. Static Multi degrades 81.7 → 71.3 as
  α rises; Static Single is flat at ~82.7; LinUCB ~24–25.
- Grounded world, α = 1.5 (`e8_grounded.csv`): Thompson 91.48 (abandonment 0.032), DQN retrained
  in this world 88.59 (0.106), deterministic-trained DQN 86.37 (0.134), Static Single 82.63,
  Static Multi 80.59, LinUCB 24.29. The grounded-trained DQN is the same run as E7's oracle,
  including its collapsed seed.
- **Harmful to "DQN beats the bandits"** once the world is stochastic.

### E9 — online adversary (`e9/`)

Injection at episode 100 of 300; effect = injected − no-injection control of the same arm at
episode 300. The experiment's own K1/K2 checks (`round3/e9_adversary.py`) were **recomputed by us
from `e9_adaptation_effect.csv`**; the zip has no collection log. K1 = target condition improves
(bots −10 pts / humans +10 pts / DI +5); K2 = no harm on "seen" (humans ≥ −5 pts, bots ≤ +5 pts).

| Injection | Arm | K1 value | K1 | K2 | Final on target (humans / bots) |
|---|---|---|---|---|---|
| advanced bots | oracle | bots −0.957 | pass | fail | 0.810 / 0.000 |
| advanced bots | score-only | bots −0.313 | pass | pass | 0.933 / 0.393 |
| advanced bots | posterior | +0.047 | fail | fail | 0.994 / 0.998 |
| advanced bots | posterior (floored) | +0.121 | fail | fail | 0.994 / **0.999** |
| advanced bots | posterior+labels (±floored) | +0.034 / +0.103 | fail | pass | ~1.0 / 1.000 |
| Balabit humans | oracle | humans +0.166 | pass | pass | 0.461 / 0.008 |
| Balabit humans | all others | −0.094 … +0.001 | fail | pass | 0.21–0.28 kept |
| both | oracle | DI +99.9 | pass | fail | DI 32.1 |
| both | score-only | DI +35.8 (p < 0.001) | pass | pass | DI −8.1 |
| both | posterior | DI +5.5 (p = 0.24) | pass | pass | DI −58.8 |
| both | posterior (floored) | DI −7.8 | fail | pass | DI −63.5 |
| any | proxy | 0 | fail | pass | never challenges |

- **Scorer blind spots** (`e9_scorer_external.csv`, `e9_scorer_auc.csv`): advanced web-bots are
  never classified as bots (mean P(bot) 0.30); Balabit humans rank *below* advanced bots
  (AUC 0.13); only 28.7 % of Balabit humans are classified correctly. Moderate web-bots
  (AUC 0.05) were not injected.
- After injection the posterior reward's bot-share error rises from 0.02–0.03 to ~0.18–0.22 and
  its bias jumps to +44…+56 (bots) or −9…−15 (humans) (`e9_reward_diagnostics.csv`); controls unchanged.
- **Only the oracle adapts to advanced bots; score-only partly adapts; every posterior arm lets
  99–100 % of them through.** Every arm, oracle included, keeps only 21–56 % of Balabit humans.
- **Harmful to robustness:** a posterior reward inherits the scorer's blind spots and cannot learn
  around them.

---

## 5. Net assessment

**Supports the paper**
1. Deterministic world: DQN beats statics by ~20 DI and the bandits clearly, stably across seeds (§3).
2. The humanity score is essential (−77 DI without it).
3. Robust to reward settings (E3) and to abandonment.
4. Mixture-of-Gaussians Thompson is a real improvement over the single Gaussian.
5. Clean protocol and audit trail.

**Undercuts it, or is new**
1. In the stochastic world Thompson ≥ DQN (E8).
2. No reward computable from observables works; the posterior reward is an upper bound that ties the
   oracle, and its floored variant is unstable (E7).
3. No label-free arm adapts to bots the scorer misses (E9).
4. Challenge volume depends on `n_active` (E4).
5. The DQN architecture choice is a statistical tie, and rotation 1 is uninformative (§2).
6. The headline sits near ceiling on a small, perfectly separable eval block (§3, E6).
7. Modest, borderline memorisation of training recordings (E2).

---

## 6. Inference bundle (`inference/`)

- The delivered `run_inference.py` read `HumanityScorer.score_chunk` (which returns P(bot)) as a
  humanity score and fed the DQN `1 − P(bot)`, so the policy acted on P(human). **Fixed**; checked
  on real recordings (bots 0.64–0.996, humans 0.02–0.08).
- `model.keras` was saved by a newer Keras that older versions cannot parse; the script now falls
  back to rebuilding the architecture from `scorer.json` and loading the weights.
- It now imports the repository's `expkit` instead of a bundled copy.

---

## 7. Errors in the cluster agent's FINDINGS.md (not carried forward)

| Cluster claim | Outputs |
|---|---|
| E7 floored posterior uncertain band "+3.731, skew 0.021, passes" | −11.759, skew 0.064, **fails** (`uncertain_band_not_skewed = False`) |
| E7 bias: exact +2.514, floored +1.514, posterior −0.220, score-only −4.867, proxy +58.704 (abs 97.58) | −0.274, −1.486, −2.712, −8.821, +50.811 (abs 90.13); quoted values appear in no output |
| E7 log-loss floored 0.226, posterior 0.285, score-only 0.344 | 0.141, 0.179, 0.293 |
| E7 EM share error 0.007 vs 0.044 | EM 0.029, floored 0.017, raw score 0.111 |
| E8 α = "human abandonment steepness"; FPR "strictly constant" | bot-solve IRT discrimination; FPR 0.0020–0.0023 |
| E9 floored posterior "maintains bot control, Δ ≤ 0.016" | lets 99.9 % of advanced bots through; bots +0.121 |
| E9 "maintains consistent challenge share" | E9 does not record challenge share |
| E4 "proving the policy responds to kinematic risk rather than traffic density" | challenge volume does respond to `n_active` |
| E6 "25 % of training data already achieves 100.0 DI" (as a strength) | ceiling effect; 75 % row (94.1, SD 10.4) omitted |
| Cites `e2_summary.csv`, `e4_summary.csv`, `e9_summary.csv` | files are `e2_gap_summary.csv`, `e4_variants.csv`; no E9 summary exists |
