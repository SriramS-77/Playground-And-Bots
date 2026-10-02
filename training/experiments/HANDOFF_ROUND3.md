# Round 3 — fold 2 only

**This document supersedes `HANDOFF_ROUND2.md` wherever they differ.** Round 2's specs for
E2–E8 still hold where §5 below points back to them; everything about the scorer, the
partition roles and the selection rule is replaced here.

Everything runs from one script, `exp_round3_fold2.py`. E2–E9 import `context()` from it.
**Read §0–§3 before running anything.**

---

## 0. What this round is

1. **One fold: fold 2.** Finalise the protocol, the architectures and every metric on it.
   **Run nothing on folds 0 or 1** — no scorer, no training, no evaluation.
2. **Fold 2 settles the protocol, not the other folds' architectures.** Fold 2 uses every
   recording in some role: its `rl` block *is* fold 1's eval block and its `scorer` block
   *is* fold 0's eval block (verified against `rotation.json`). When the study later
   extends to three folds, **architecture selection is re-run inside each fold** (nested);
   fold 2's winners are not transplanted. What carries over is the method: scorer design,
   cross-fitting, posterior family, selection rule, gates.
3. **New scorer design** (§4.1): five cross-fitted sub-scorers produce the scores the
   policy trains and is selected on; one refit scorer scores the eval block. No scorer ever
   scores a recording it was fitted on.
4. **The eval block is opened only for final evaluations**, and every opening appends a
   line to `results/round3/fold2/eval_access.log`. That file is a deliverable.
5. **The pipeline, end to end, all on fold 2:** build the scorer (`--prepare`: 5 cross-fitted
   sub-scorers + refit) → search the RL architectures and posterior variants, selecting on
   the rl block's validation rotations → **offline evaluation**: the headline, on the eval
   block → **online training**: E7 (deployable rewards — oracle, proxy and its variants, the
   posterior variants — trained and evaluated in the grounded world) and E9 (the same rewards
   when unseen bots, unseen humans or both join mid-training, §4.8) → E2–E4, E6, E8.

---

## 1. The user's decisions — one line each, top of `exp_round3_fold2.py`

**Do not change these after launching anything.** They were fixed before the run, which is
what makes them legitimate.

| | switch | set to | alternative | evidence |
|---|---|---|---|---|
| D1 | `SCORER_CFG` | Phase 0's **fold-2 pick** — lr 3e-3, dropout 0.25, recurrent dropout 0.2 — plus `start_from_epoch=30` | round-2 baseline (lr 1e-3, no dropout) | Phase 0's own rule picked it on fold 2's `scorer ∪ rl` (AUC 0.968 vs baseline 0.951, baseline 19th of 52). Cross-fitted on fold 2 with `start_from_epoch=30`: held-out-part AUC **0.966** vs 0.950, humans misscored **6.7 %** vs 12.0 %, 0/10 collapses each |
| D2 | `SELECTION_TIEBREAK`, `HUMANS_MARGIN`, `BOTS_MARGIN` | `"friction"`, `0.05`, `0.05` — within 1 SE of the best mean DI, least friction on `rl_val` **among candidates no worse than the best-DI candidate on either side: human survival ≥ best − 5 points and bot survival ≤ best + 5 points**, then fewest params | `"params"` (round 2's rule) | The paper's claim is security *and* usability, and DI cannot see friction (one round-2 DQN seed had the highest DI, 84.1, and 42 s of friction). The guard has two sides because friction can be cheap two ways: driving humans away early (friction is per *enrolled* human), or waving bots through — the local single-seed check saw mixture variants at 0–3 s letting 17–37 % of bots through. Bandit SEs (~18 DI) put such candidates inside the band |
| D3 | `VAL_ROTATIONS` | `3` — three disjoint `rl_val` thirds of fold 2's rl block; every candidate trained on each | `1` — round 2's single split, 75 tasks | Round 2's fold-2 `rl_val` is 18 recordings, **5 humans, all campaign B**. Cost: 225 tasks, = round 2's whole search |
| D4 | `ON_GATE_FAILURE`, `MAX_GATE_RETRIES` | **`"retry"`, `2`** — the user's choice: if a scorer gate trips, `--prepare` refits the whole cross-fit with the next seed (attempt *i* uses seed *i*), at most 2 retries, and records every attempt | `"stop"` — halt at the first failure | Gates use training-side data only — never eval results — so a pre-registered retry is like restarting a crashed run, not selection on results. It is legitimate **only because it is fixed in advance and fully recorded** (§3, §7) |
| D5 | `TRAIN_HUMAN_RANGE` | `(50, 250)` humans per training episode (bots always 0–1000) | `(80, 100)` — round 2's sampler | The realistic evaluation grid draws humans from `U(50, 250)`; round 2's sampler leaves most of those counts outside the training distribution and makes population a near-perfect proxy for bot count during training. `(80, 100)` keeps continuity with round 2's fold-2 numbers — but then the realistic grid tests out-of-range human counts as well as composition |

---

## 2. Run order

From `training/experiments/`. Each step is a separate job; do not collapse them.

```bash
# 0. checks, minutes
python preflight.py
sbatch slurm/r3_prepare.sh                     # cross-fit 5 + refit, every gate, 144/144
#    -> its log must end "prepare: OK" and show 6 "preflight/scorer: val AUC" lines
#       and one "preflight/canary" line. A scorer "ready" in seconds was LOADED, not fit.
python exp_round3_fold2.py --smoke             # toy RL budget; evaluates on rl_val ONLY;
                                               # also asserts save -> load round trips

# 1. the search (Priority 1)
python exp_round3_fold2.py --list-search       # 225 tasks with VAL_ROTATIONS = 3
sbatch --array=0-224%25 slurm/r3_search.sh
python exp_round3_fold2.py --collect-search    # -> results/round3/fold2/arch_choice.json

# 2. the headline (Priority 2) -- the first time the eval block is opened
python exp_round3_fold2.py --list-headline     # 21 tasks
sbatch --array=0-20%21 slurm/r3_headline.sh
python exp_round3_fold2.py --collect-headline

# 3. experiments -- round3/, see §5: E2/E4/E7/E9 after step 1; E3/E6/E8 after step 2
#    E9 first: python -m expkit.external --check   (the external-data cache, §4.8)
```

Adjust `--partition`, `--time`, `--mem` and the conda path in `slurm/r3_*.sh` only.

---

## 3. Gates and stop conditions

| gate | fails when | runs in | caught in round 2 |
|---|---|---|---|
| `check_scorer_health` | validation AUC < 0.85 | every one of the 6 scorers, at fit **and** at every load | fold 0's scorer restored its epoch-1 weights (val AUC 0.62) and saved without error |
| held-out-part AUC | a sub-scorer scores < 0.85 on the part it never saw | every sub-scorer | a cross-fitted fit with best epoch 2, val AUC 0.616, held-out 0.667 |
| `check_threshold_canary` | Static Single keeps < 25 % of humans on the out-of-fold rl block | `--prepare` | the collapsed scorer made it keep 0 % |
| provenance | a cached scorer was fitted on other data or another config | every load | round 2 reloaded stale scorers in 4 s |
| disjoint scoring | an rl chunk not scored out-of-fold, or an eval chunk scored by a scorer that saw it | every cache build | — (new design) |
| `check_determinism` | the simulator stops reproducing the published one: not 144/144 | `--prepare` | held |
| `check_budget` | < 5 seeds, < 200 episodes, < 20 eval seeds | every task | round 1: 1 seed, 90 episodes |

**Stop and report — do not work around:**

* `--prepare` exits with "scorer gates failed on all 3 attempt(s)". Send
  `prepare_log.json` and the log. Do not lower a threshold, raise `MAX_GATE_RETRIES`, or
  reseed by hand — the retry budget is part of the pre-registered design. Every array task
  refuses to start until `prepare_log.json` records a passing attempt.

**Retries are allowed, but never silent.** If `--prepare` passes after retrying, its last
line says `after N RETRY(IES)` and `prepare_log.json` records, per attempt: the seed, start
time, duration, a `status` (`running` is written *before* fitting, so a crash or node kill
leaves a record; then `passed` or `failed`), **which gate failed** (`validation AUC`, `held-out-part AUC`, `canary` or
`best epoch`) and the full message; for the passing attempt, every scorer's diagnostics and
the canary value. `seed_used` in that file is the single source of truth — every task reads
it. **Report any retry in the run notes, and the findings' methods section must state it**
("the cross-fit passed its gates on attempt *k*; attempt *j* failed the *g* gate"). Two or
more retries on fold 2 would mean the collapse rate is far above the ~0–3 % measured
locally, and is a finding to report, not just a footnote.
* `--collect-search` warns that a candidate is short of (seed, rotation) cells. Re-run
  **those exact tasks** (`--search-task N`) — never new seeds.
* `eval_access.log` has a line whose reason does not name an experiment and a task index.
  Expect one line per final-evaluation task: 21 for the headline, and one per array task of
  each E-experiment's final evaluation. The code already refuses to open the eval block
  without a `reason`, or before `arch_choice.json` exists.
* A sanity floor is crossed: the round-3 DQN below ~60 DI, or Static Single keeping
  < 40 % of humans on the published grid. Round 2's fold 2 under a weaker scorer gave
  DQN 77.2 and Static Single 64.6 % humans — a large drop means something is broken.

**Never:** run anything on folds 0 or 1; load `results/fold*_scorer/` or
`results/final_scorer/`; run `nb_02` or `run_all.py`; change D1–D5 mid-round; look at
eval-block numbers before `arch_choice.json` is final.

---

## 4. What changed since round 2, and why

### 4.1 The scorer: cross-fitted, then refit

**Round 2's scorer-only fit was unstable.** Each fold's scorer trained on `fold.scorer`
alone, with early stopping on a ~13-recording validation split. On fold 0 validation loss
never beat epoch 1, the epoch-1 weights were restored, and every recording scored
0.41–0.64 (AUC 0.70). Static Single kept 0 % of humans; every fold-0 result and every
E2–E8 result (all of which ran on fold 0) measured that failure.

**Training the scorer on `scorer + rl` fixed stability but broke something else.** The
policy then trained on scores for recordings the scorer had fitted: fold 0 rl block
misscored **0.0 %** of human chunks against **6.7 %** on the unseen eval block. The policy
never met a misscored human in training, then met them at evaluation.

**Cross-fitting fixes both** — the standard remedy for a two-stage pipeline:

```
scorer + rl (~102 recordings) -> 5 parts, by recording, stratified by family
  sub-scorer k = fitted on the other 4 parts -> scores the rl recordings in part k
  refit        = fitted on all of scorer + rl -> scores the eval block, only
```

Out-of-fold rl scores misscored 7.5 % of human chunks, against 6.7 % on eval. Sources:
Wolpert (1992) and Breiman, *Stacked Regressions* (Machine Learning 24, 1996) — the
second stage trains on cross-validated first-stage predictions; Chernozhukov et al.
(Econometrics Journal, 2018) — "own-observation" bias and cross-fitting; scikit-learn's
`StackingClassifier` — base models refit on all data for prediction, final estimator on
cross-validated predictions, 5-fold by default; Kallus & Uehara (JMLR, 2020) — cross-fold
estimation of learned components inside RL off-policy evaluation.

**`start_from_epoch = 30`.** Across 30 cross-fitted fits the median best epoch was 35 and
9 of 30 peaked at ≥ 45: learning often starts late, and patience 15 from epoch 1 kills a
fit that stalls longer than that. 1/30 collapsed (best epoch 2, val AUC 0.616); with
`start_from_epoch = 30`, 0/30, and the same fit peaked at epoch 69 with val AUC 0.957.
Zero of 30 still allows a true rate up to ~10 %, which is why the gates stay.

Code: `expkit/crossfit.py` (`fit_crossfit`, `CrossFit.cache`), `ScorerConfig.start_from_epoch`.

### 4.2 Bandit posterior statistics are float64

11 of round 2's 225 search tasks died in the full-covariance Thompson posterior; 3 were
fold 2's (`h256_128_e128`). The cause was float32 accumulation of `A = I + ΣzzT`, which at
the conditioning of deep feature nets silently corrupted every posterior, not only the one
that crashed. Fixed in `expkit/bandits_x.py` (`_STAT_DTYPE`); regression-tested by smoke
test 3b. Round 2's deep-bandit rows are not evidence about depth.

### 4.3 The search records usability, and selects on it

Every search run now records `friction_s`, `human_survival` and `bot_survival` next to DI.
The selection rule (D2) breaks ties within 1 SE of the best DI on friction. Per-seed means
are taken over volumes, eval seeds **and** rotations before the spread across seeds.

### 4.4 The headline evaluates twice, and saves what it trained

* **Published grid** — 100 humans, bots `(0, 20, 100, 200, 500, 1000)`, zero-bot cell
  excluded from every mean. For continuity with Table 4.
* **Realistic grid** — bot fraction `(5, 15, 30, 43, 60, 75 %)` with humans drawn from
  `U(50, 250)` independently of it (Imperva 2025: 37 % bad bots; Akamai: 43 % of logins
  credential abuse). Population stops being a proxy for the attack level by construction.
* 50 evaluation seeds, 5 training seeds. **Every trained policy is saved** to
  `results/round3/fold2/agents/` — round 2 delivered no fold-2 checkpoints at all.

### 4.5 What round 2 already said about fold 2 (provisional, weaker scorer)

DQN 77.2 DI, Thompson 73.7, LinUCB 69.1, Static Multi 67.2; DQN `64` and `128-64` tied in
the search; the mixture posterior kept 75–84 % of humans against the Gaussian's 51–61 %.
Round 3 replaces these. They are here as a sanity reference (§3), not as results.

### 4.6 Simulator changes, round 3 — grounded world only

Two defects in `expkit/stochastic.py`, both fixed:

* **"No challenge" blocked bots.** In the stochastic model a bot at level 0 passed with
  `σ(1.5·(B_s + 0.5))` — 0.68 for the weakest bot — so showing nothing "blocked" 32 % of
  `B_s = 0` bots per step and about 20 % of all bots over 12 steps of never challenging,
  each paid the oracle's +50. Level 0 now passes every bot with certainty.
* **Bots spent 0 seconds on every challenge.** That made solve time a perfect class label
  and made the proxy reward pay a leaked bot (+25) more than a satisfied human
  (25 − friction). Bots now spend measured solve times — `BOT_SECONDS`, automated solvers,
  lognormal with log-sd 0.5; the evidence for each level is in the comments, and
  `SERVICE_BOT_SECONDS` is the human-solving-service alternative, unused. The oracle reward
  is unchanged: a bot's time is not a usability cost.

**What this affects:** E7 and E8 — the only round-3 scripts in the `GROUNDED` world. The
headline, the search and E2–E6 run in `DETERMINISTIC`, where level 0 already passed
every bot and bot time has no spread and enters no reward; the deterministic-limit check
is still 144/144. **E7 and E8 numbers are not comparable with rounds 1–2, nor with
`nb_06` / `nb_07`.** Expect E7 to move: the proxy now charges a blocked bot the time it
spent, so blocking at image levels (bots ≈ 15 s) scores below zero.

**Known remaining asymmetry.** Humans still spend exactly the median (`s_T`, or `2·s_T` after
a retry) while bots are continuous; an exact-value match on seconds would identify every
human. Nothing exploits that — the proxy reward is linear in seconds, the learned reward
model is a linear logistic, the DQN state carries no seconds — but do not add a component
that could.

### 4.7 E7's new reward variants: the posterior reward, and the learned model fixed

**Amends round 2's rule** that "the immediate reward is a pure function of the observable
tuple": the posterior variants also use the humanity score, deliberately. Solve time is
never evidence in them.

* **`posterior`** (`expkit/posterior_reward.py`) — the oracle reward in expectation over
  the operator's posterior on (class, bot strength):
  `r̂ = q(H)·r_H(T, o) + Σ_b q(B, b)·r_B(b, T, o)`, with the oracle's own weights. The
  posterior combines a calibrated humanity score (Platt on the latest chunk score, the
  running average and the step, fitted on the **scorer block's** out-of-fold scores), every
  challenge outcome so far (the simulator's exact solve and abandonment model, strength
  prior U{0..9}), and the bot share estimated **per episode by EM**. Human friction is
  charged at its expected value given level and outcome. If the posterior is right, the
  tower property makes every policy's value identical to the oracle's. It uses the exact
  outcome model and an offline-labelled calibration set: an **upper bound** on the idea.
* **`posterior+labels`** — the same, conditioned on the delayed abuse label as **evidence**
  (fires 0.30 for a bot not blocked by session end, 0.02 otherwise), never as an extra
  penalty.
* **`… (floored)`** — sensitivity: the score's likelihood ratio bounded by per-class Laplace
  floors ε_c = 1/(n_c + 2) over the calibration set (`ScoreCalibration.floored`). The
  leave-one-recording-out CV that picks C and ε chooses ε = 0 on fold 2: no calibration
  recording is confidently wrong, and CV cannot see a rate below ~1/n. Declared after the
  reward-bias diagnostic traced the skew to two confidently misscored human recordings in
  the rl block (0.987 and 0.756); nothing from the rl block enters it.
* **`score-only`** — ablation: the raw chunk score as P(bot), no prior correction, no
  outcome evidence.
* **`learned rm`, fixed** — no class balancing (its output is used as a probability); no
  `seconds` feature; P(leak) = (p̂ − 0.02)/(0.30 − 0.02), placed where the label would land and
  **replacing** the sparse label instead of adding to it. Its immediate part is still the
  proxy's, which charges observed seconds. **Expect it to stay at "never challenge"**: the
  proxy pays +50 for every step a bot is let through (~+600 over a session) against a
  corrected penalty of at most −150 per *session*, while the oracle charges −150 − 5b per
  leaked *step*. A session-level label cannot reach a per-step leak cost; the posterior
  variants work because they charge leaks per step.
* **`reward_bias` task** — on logged random-level behaviour: `mean(r̂ − oracle r)` by band of
  q(B) for every variant. Two checks fixed before looking: the exact variant (no score, true
  share) must be unbiased within 4 SE — asserted, it tests the code; and a variant is "not
  skewed" when, in the band 0.2 ≤ q(B) < 0.8, |bias| ≤ 5 % of the class reward gap.

**The eval block has been opened locally (user's decision, 1 Oct 2026).** For a drift test
(replay bots mid-training) and an unseen-data test (Balabit humans, web-bot moderate and
advanced bots), policies were trained on fold 2's full rl block and evaluated on the eval
block, with locally fitted cross-fit scorers; the access is logged in the local
`eval_access.log`. **E7's design is frozen from here** — no configuration, parameter or
criterion may change because of what those runs showed. The cluster's E7 numbers will
differ slightly (its scorers are fitted again), but they are no longer a blind first look at
the eval block for the posterior variants; say so when reporting them. Exploratory local runs
of two adaptation channels (online recalibration of the score from outcomes + labels, and a
5 % exploration budget) followed on the same data; they are not part of E7 and do not change it.

**Development reference (local, NOT results).** Rotation 0: trained on rl_fit, evaluated on
rl_val (18 recordings, 5 of them human); 5 seeds × 200 episodes, 20 evaluation seeds; the eval
block was not opened. Use it as a sanity check of the cluster run, the way §4.5 is used.

| config | humans kept | bots left | friction s | DI (± SE over seeds) |
|---|---|---|---|---|
| oracle | 99.9 % | 1.6 % | 0.05 | 98.3 ± 0.9 |
| posterior (floored) | 99.3 % | 0.3 % | 0.34 | 98.9 ± 0.2 |
| posterior+labels (floored) | 98.0 % | 0.4 % | 0.90 | 97.6 ± 0.8 |
| posterior | 90.9 % | 2.3 % | 4.3 | 88.5 ± 5.2 |
| posterior+labels | 86.5 % | 1.6 % | 6.4 | 85.0 ± 12.5 |
| score-only | 78.4 % | 1.8 % | 10.5 | 76.5 ± 11.5 |
| proxy, learned rm (fixed), never challenge | 100 % | 100 % | 0 | 0 |

The un-floored variants are unstable rather than uniformly worse: individual seeds kept
74 %, 35 % and 32 % of humans (posterior, posterior+labels, score-only), while every floored
seed kept ≥ 95 %. In the reward-bias diagnostic **every variant except the exact one failed
the uncertain-band check**; the floor removed the bias at the extremes (≥ 0.95 band: −9.1 →
−1.8) but not in the uncertain band (−46.5 → −31.3), where two confidently misscored human
recordings land. Expect the same on the cluster and report it; the oracle is near its
ceiling on rl_val, so the eval block is the comparison that can separate the variants.

### 4.8 E9 — online training against unseen users (adversary bots, adversary humans, both)

`round3/e9_adversary.py`. Every policy trains on fold 2's rl block; at episode 100 of 300
**unseen sessions join the playback** and training continues. Four injection conditions per
reward arm — the arms are E7's oracle, proxy and its five posterior variants:

| injection | humans after episode 100 | bots after episode 100 |
|---|---|---|
| no injection (**control**) | rl humans | rl bots |
| adversary bots | rl humans | rl bots × 2 + 60 web-bot **advanced** bots (~55/45) |
| adversary humans | rl humans × 10 + 150 **Balabit** sessions, users 1–5 (50/50) | rl bots |
| adversary both | both of the above | both of the above |

**Adaptation is the injected run minus the no-injection control at episode 300** — never
episode 300 minus episode 100, because the extra 200 episodes are also simply more
training (locally the floored posterior climbed from 82 to 98 DI with no change in data).
Every run is evaluated on five conditions: seen (eval block), adversary bots (eval humans +
held-out advanced bots), adversary humans (held-out Balabit users 6–10 + eval bots),
adversary both, and **control humans** (held-out web-bot humans + eval bots — the same
logger as the web-bot bots, so a scorer that separates by recording source shows up). Task
0 is the **scorer report**: the fold's refit, never retrained, on every external group,
with session-level AUCs and bootstrap intervals. The checks K1/K2 are in the docstring.

**Data — the one prerequisite.** Neither dataset is in git (both are git-ignored, ~550 MB raw).
`expkit/external.py` holds the fixed preprocessing rules and loads a cache of the cleaned
120 s movements, `data/external_sessions.json.gz` (8.6 MB), pinned by SHA-256
`cb287ad6510d70e9a400341271559cf8d2fd528705ee7d884dd7783c59d28248`; E9 refuses any other file.
Sessions are re-chunked with rlcaptcha's own code and **scored at run time by this fold's
refit** — no scores are shipped. Check it before E9: `python -m expkit.external --check`. If
the cache is not in the checkout, rebuild it from the raw datasets with
`EXTERNAL_RAW=<dir with balabit_dataset/ and web_bot_detection_dataset/> python -m expkit.external --build`
and confirm the hash matches.

**Disclose when reporting:**
* **E9 is not blind.** It is the local unseen-data test formalised, and its eval-block results
  were seen locally (1–2 Oct 2026, locally fitted scorers).
* **Moderate web-bots were dropped after the local run**, because the scorer is inverted on
  them (same-source AUC 0.05: they score 0.04, more human than humans at 0.13). They stay in
  the scorer report so the reason is visible.
* **The oracle collapsed locally after injection** — it let ~40 % of *our* bots through on the
  seen condition — and replay buckets by observable outcome did not fix it. Expect it, report
  it, and qualify every oracle-relative statement.
* Adversary bots draw strength U{0..9} like our own: strength is a simulator construct.
* Small human samples: 5 held-out Balabit users (150 sessions) and 10 web-bot humans.

**Local reference (fold 2, locally fitted refit; not results).** Scorer: Balabit humans mean
0.48, 46 % classified human (CI 41–52 %); web-bot humans 100 %; advanced bots mean 0.39, 2 %
classified bot, but ranked correctly against same-source humans (AUC 1.0); moderate bots
0 %. Online, with moderate and advanced bots and Balabit injected together, the floored
posterior went from 55 % to 38 % of advanced bots left and from 61 % to 29 % of Balabit humans
kept (over-challenged: their scores sit in the uncertain band and passing proves little).

**Not part of round 3** (local, exploratory, reported to the user only): a drift test (bots
switching to replayed human traces mid-training) and two adaptation channels for the
posterior reward (online recalibration of the score from outcomes + labels; a 5 % exploration
budget). Recalibration fixed the bot-share estimate but cost humans heavily and unstably;
exploration alone changed nothing.

---

## 5. The queue

**The experiment scripts exist — run them, do not rewrite them.** They are in `round3/`
(see `round3/README.md`): `e2_leakage.py`, `e3_reward_sensitivity.py`, `e4_population.py`,
`e6_data_sufficiency.py`, `e7_proxy_reward.py`, `e8_stochastic.py`, `e9_adversary.py`,
sharing `common.py`.
Round 2's hand-written experiment scripts are where its results went wrong, so there is
nothing left to improvise. If a script fails or a spec below looks wrong, **stop and
report** rather than patching around it.

All of them train **fold 2's DQN winner** from `arch_choice.json` (never the published
128-64-32), **5 seeds × 200 episodes** with the headline's training sampler, score through
the same cross-fitted scorers, open the eval block only through a logged, labelled call,
and report DI next to human survival, bot survival, friction seconds, false-positive and
abandonment rates. Each has the same interface:

```bash
python round3/e7_proxy_reward.py --smoke     # first, always: toy budget, rl_val only
python round3/e7_proxy_reward.py --list      # prints the sbatch line
sbatch --array=0-92%25 slurm/r3_e.sh e7
python round3/e7_proxy_reward.py --collect   # -> results/round3/fold2/e7/*.csv
```

**Order.** E2, E4, E7 and E9 need only `arch_choice.json` (after step 1; E9 also the external
cache, §4.8). E3, E6 and E8 also
evaluate the headline's saved checkpoints, so launch them after the headline (step 2).
**E4 has two stages:** its tasks 0–4 train and save checkpoints; launch 5–50 with
`--dependency=afterok:<jobid of 0-4>` — an evaluation task refuses to run without them.

| priority | script | tasks | what it answers — the docstring has the full design |
|---|---|---|---|
| 0 | `exp_round3_fold2.py --prepare`, `--smoke` | — | §2 |
| 1 | `exp_round3_fold2.py` search | 225 | §2. Report each family's selection table with friction, humans and bots beside DI, and mixture vs Gaussian per architecture |
| 2 | `exp_round3_fold2.py` headline | 21 | §2. Both grids, per-seed spread, each learned policy vs Static Multi seed by seed |
| 3 | `round3/e7_proxy_reward.py` | 93 | R1.1, the deployable reward, in the grounded world: oracle; proxy; penalty sweep −600/−2400/−9600; immediate gap (zero-friction reward 35, 25); PBRS at −600/−2400; learned reward model (fixed, §4.7); **posterior reward with and without the delayed labels, their floored sensitivity variants, and the score-only ablation (§4.7)**; reward-bias diagnostic; reward normalisation + gradient clipping at −2400/−9600; 1000 episodes; never-challenge floor; observability table. Ratio reported raw and as `(proxy − never) / (oracle − never)`. Off-policy evaluation is **not** implemented (greedy policies make importance weights degenerate) — say so |
| 3b | `round3/e9_adversary.py` | 141 | R1.1 online, against unseen users (§4.8): oracle, proxy and E7's five posterior variants × {no injection (control), adversary bots, adversary humans, both} × 5 seeds, 300 episodes with the switch at 100; scorer report on all external groups (task 0). Adaptation = injected − control. Run `python -m expkit.external --check` first, then `sbatch --array=0-140%25 slurm/r3_e.sh e9`. Its no-injection arms train 300 episodes, not E7's 200 — a control, not comparable with E7 |
| 4 | `round3/e4_population.py` | 51 | R1.5, the population confound: DQN retrained on humans `U(25, 400)` × bots 0–1000; evaluated on humans `{25…400}` × bots `{0…1000}` with `n_active` unfrozen, **true (the wrapper self-test, asserted)**, frozen at `{25…1400}`, random (OOD, not evidence). Reports the share of decisions that were challenges, so a policy that stops acting is visible |
| 5 | `round3/e3_reward_sensitivity.py` | 69 | R1.3: 9 one-at-a-time reward settings retrained × 5 seeds, Pareto frontier; fixed-policy landscape (18 settings, one task each, rl block); abandonment curves. "N of 9 beat every static" is computed, never written in advance |
| 6 | `round3/e8_stochastic.py` | 26 | R2, modelled usability: α sweep {0.5…12, deterministic} of every headline checkpoint; DQN retrained in the grounded world; false-positive rate and friction |
| 7 | `round3/e2_leakage.py` | 50 | R1.2, policy-level leakage: role reversal inside fold 2's rl block, 10 splits, DQN and the no-score ablation; statics' gap **exactly 0.000** (asserted). **Never opens the eval block** |
| 8 | `round3/e6_data_sufficiency.py` | 70 | R1.4: recording-level bootstrap (200 pools × 3 simulation seeds, bots 100 and 500) with the between-pool share **per policy**; learning curve over 25–100 % of the rl block |

`eval_access.log` should end up with one labelled line per final-evaluation task: 21
(headline) + 51 (E3) + 46 (E4) + 70 (E6) + 91 (E7) + 26 (E8) + 141 (E9). E2 adds none.

---

## 6. Compute

* `--prepare`: **measured 11.6 min on one CPU core** — 6 scorer fits at ~50 s each
  (recurrent dropout is slow on CPU) plus ~6.5 min for the 144/144 check.
* `--smoke`: **measured 10–21 min on one core** (3 search tasks, 4 headline tasks, the
  save→load round trips, toy RL budget, real scorers).
* Search: 225 tasks. Round 2 measured ~25 min per Thompson and ~55 min per LinUCB training
  at reduced volumes, longer at 1000 bots; budget ~a day at 25-way concurrency.
* Headline: 21 tasks; each evaluates 12 grid cells × 50 seeds. Bandit tasks are the long
  pole.
* E-experiments: 500 array tasks in all (E2 50, E3 69, E4 51, E6 70, E7 93, E8 26, E9 141),
  almost all a single 200-episode DQN training plus a 20-seed evaluation; E7's `proxy 1000ep`
  (5 tasks) and E4's evaluation grid (up to 400 humans + 1000 bots) are the long poles. An E9
  task is a 300-episode training with 10 light checkpoints on 5 conditions, then the full
  20-seed evaluation on all 5. The training part measured 5–20 min on one core locally; the
  final evaluation was never timed — estimate ~30–60 min per task, up to ~1.5 h for permissive
  arms (proxy keeps up to 1000 bots alive for 12 steps in every episode); well inside 8 h.
* Policies stay on CPU on purpose (`bandits_x.DEFAULT_DEVICE`); parallelism, not the GPU,
  is what makes this affordable.
* **A silent task is not a crashed task.** Search and headline tasks print nothing between
  loading the scorers and writing their CSV. A bandit training at 200 episodes ran **over
  an hour on one core** locally, during which its log did not change. A local dry run
  misreported exactly this as "crashed silently". Before calling a task failed, check that
  its process or Slurm job has actually ended (`sacct -j <id>`) and look for a traceback;
  never resubmit a task that is still running.

---

## 7. Deliverables

1. **The whole `results/round3/fold2/` tree:** `scorer/` (6 scorers, `parts.json`,
   `prepare_log.json`), `search/`, `search_runs.csv`, `search_{dqn,linucb,thompson}.csv`,
   `arch_choice.json`, `headline/`, `headline_runs.csv`, `headline_{published,realistic}.csv`,
   `agents/*.pt`, `e2/` … `e9/` (each with `tasks/` and the `--collect` tables; `e9/` also
   `curves/` and `diag/`), and
   **`eval_access.log`**.
2. **Any script you had to write, committed** — the provided ones should need no changes;
   if one did, commit the change and say why. Not attached — committed.
3. `logs/r3_*`.
4. Run notes: Python, TensorFlow and Keras versions, whether a GPU was used, array width,
   and **`retries_used` from `prepare_log.json` with the failed gate of every failed
   attempt** — "0 retries" stated explicitly if none.
5. A fold-2 findings note in which **every interpretive sentence cites the table cell it
   rests on**. Round 2's `FINDINGS.md` contained template prose that contradicted its own
   tables (E3, E4, E6); a sentence that cannot point to a number is not a finding.
6. Do not ship `inference/` as it stands: it carries fold 0's collapsed scorer. Rebuild it
   from `results/round3/fold2/scorer/refit/` and `agents/dqn_s*.pt` if it is shipped at all.

### The `.gitignore` trap

`*.json`, `*.csv`, `*.pt`, `*.keras`, `*.png` **and `*.log`** are all ignored — so a plain
`git add` silently drops every result, every checkpoint and `eval_access.log` itself.
Force-add, then check:

```bash
git add -f training/experiments/results/round3/
git ls-files training/experiments/results/round3 | wc -l   # non-zero, or nothing was added
```
