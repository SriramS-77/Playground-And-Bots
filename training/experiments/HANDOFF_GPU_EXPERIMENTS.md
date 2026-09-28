# Handoff: experiments to finalise the OJCS resubmission

**Audience:** an AI coding agent with a Blackwell GPU box and the shared repo.
**Goal:** produce robust, multi-seeded results that answer IEEE OJCS Reviewer 1 comments
on *Balancing Security and Usability: Adaptive CAPTCHA Orchestration via Reinforcement
Learning* (OJCS-2026-04-0359).

Everything here has been prototyped on CPU. The numbers quoted are real but
under-powered — they tell you what to expect, not what to report. Your job is to redo
them properly and, where noted, to settle questions the CPU budget could not.

---

## 0. Read this first

1. **The scorer design is decided. Phase 0 is now confirmation, not a blocker.**

   ```
   representation = kinematic      (dx, dy, dt, speed, accel)
   padding        = mask           (repeat_row + a binary real/pad channel)
   context        = 32             (NOT the inherited 100)
   lstm_units     = (32,)  dense_units = (32,)  lr = 1e-3
   -> 6,081 parameters
   ```

   Measured at the **chunk** operating point, deduplicated, grouped 5-fold CV over all
   156 recordings, 2 seeds: **AUC 0.975 ± 0.001, ECE 0.010**, `corr(score, padding)` 0.22.

   The same config trained on the **`lstm` pool only** (the RL-safe scorer, 73
   recordings) scores **AUC 0.875, ECE 0.065 after calibration, T = 1.775** on the
   unseen `eval` pool. **Report both numbers** — the gap is pool size (125 recordings per
   CV fold vs 59 after the validation split), and the RL work consumes the 0.875 model.

   Artefact: `results/final_scorer/` (model.keras + scorer.json with standardiser and
   temperature). Load with `HumanityScorer.load("results/final_scorer")`.

   Phase 0 should still run to add seeds and settle the refinements (§4.3), but the RL
   work can start now.
2. **The GPU will not speed up the bottleneck.** The simulation is a single-threaded
   Python loop and the networks are ~5k parameters. Throughput comes from running
   **one process per (config, seed)** with `OMP_NUM_THREADS=1`, not from bigger batches.
   Plan for 32–64 concurrent processes, not one big job.
3. **Verify TensorFlow works on Blackwell before anything else** (sm_100 needs a recent
   build). If it does not, either run the scorer on CPU — it is small — or port it to
   PyTorch and re-run the Phase 0 cross-validation to prove parity. Do not assume a port
   matches.
4. Campaign-holdout / leave-one-campaign-out evaluation is **dropped** at the user's
   direction. Use ordinary grouped cross-validation and the fixed partition. Keep the
   3-participant caveat in the write-up (§8) but do not gate on it.

---

## 0b. Superseded artefacts in the repo — ignore these

The repo contains a scorer selected at the **session** operating point. It is wrong for
RL use and must not be picked up:

| path | status |
|---|---|
| `results/scorers/final_rl_scorer.*` | **superseded** — `dxdy`, session-selected, T fitted on session windows |
| `results/scorer_arch_choice.json` | **superseded** — records `lstm32`/`dxdy` |
| `FINDINGS.md` §10–11 | **superseded** — states `dxdy` is chosen and that timing does not help |
| `nb_10*.py`, `nb_11_scorer_final.py` | session-level selection; kept as a record, not a recipe |
| `results/final_scorer_probe/` | a smoke-test artefact from module development, not a trained deliverable |

`expkit/humanity_scorer.py`, `partition.json`, and everything under §1 remain valid.

---

## 1. What is already established (do not redo)

| Finding | Evidence |
|---|---|
| Published LSTM train/test split is clean — two campaigns, zero session and content-hash overlap | nb01 |
| The RL agent trained and evaluated on the same recordings; leak quantified at **+7.1 DI [3.9, 10.3]**, and it flows *only* through the scorer's outputs (the no-H-Score ablation leaks exactly 0.0) | nb03b |
| Reward parameters cannot change Table 4 for a *fixed* policy — survival depends only on blocking and abandonment. They matter **only through training** | nb04 |
| 6/6 retrained reward settings still beat every static baseline | nb04 |
| Population-feature confound is real in the design but **not** driving the result — freezing the feature at a constant costs nothing (+2.8 DI) | nb05 |
| The probabilistic action model reduces to the published rule exactly (**144/144**), and conclusions survive the α sweep (DQN DI spread 2.3) | nb06 |
| Proxy reward yields a usable but not secure policy; not fixed by label coverage, penalty weight, or credit window | nb07 |
| 86–89% of variance is between recording pools; paper's error bars understate by 2–3× | nb08 |
| Published scorer is ~54× larger than the data supports (304,049 params on ~206 windows) | nb10 |
| **Thompson Sampling draws from the global torch RNG and DQN ε-greedy from global numpy** — published runs were not reproducible from their seeds | expkit/simx.py |

Two reproducibility defects and one metric trap were found the hard way; §7 makes them
into standing rules.

---

## 2. Environment

```bash
python -m venv env && ./env/bin/pip install -r training/requirements.txt
# requirements already include: tensorflow, torch, keras, scikit-learn, scipy,
# pandas, matplotlib, openpyxl, tabulate
```

**Determinism.** Every run must set, at process start:

```python
import os, random, numpy as np, torch, tensorflow as tf
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
tf.keras.utils.set_random_seed(seed)
```

The global seeding is not optional — `ThompsonPolicy` uses `torch.randn_like` and
`DQNPolicy` ε-greedy uses `np.random.rand`, neither of which the simulation's own `seed`
argument reaches.

**Parallelism recipe.** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process per
(config, seed), GNU parallel or a simple process pool. Scorer inference must go through
the batched path (§4.4) — a per-chunk `model.predict` call costs ~59 ms of graph-dispatch
overhead *regardless of model size*, versus 0.33 ms amortised when batched (180× measured).

---

## 3. Repo and data contract

```
training/
  rlcaptcha/            DO NOT MODIFY. Faithful port of the published code.
  experiments/
    expkit/             all new code lives here
    nb_*.py             cell-marked sources -> build_notebooks.py -> *.ipynb
    run_all.py          sequential executor
    results/            all artefacts
    results/partition.json   FIXED three-way split — do not regenerate
```

`partition.json`: `lstm` 73 sessions / `rl` 31 / `eval` 52, session-disjoint, stratified
by class, bot family and campaign. **Keep it.** Regenerating it invalidates every
cross-experiment comparison.

**The scorer used by RL trains on the `lstm` pool only.** nb03b showed leakage travels
through the scorer, so the `rl` and `eval` pools must stay unseen by it. Cross-validation
over all 156 recordings is for *model selection and reporting*, never for the RL scorer.

**Optional upgrade if compute allows (recommended):** a nested outer loop — repartition
into 3 outer folds, and for each fold retrain the scorer *and* the RL agents and evaluate
on that fold's held-out recordings. This triples the effective evaluation data and is the
single best use of spare GPU time. If you do this, report both the nested result and the
fixed-partition result.

---

## 4. PHASE 0 — finalise the humanity scorer (BLOCKING)

### 4.1 The problem that reopened the design

The RL simulation scores **10-second chunks**, not whole sessions. `_windows` pads a
chunk to 100 points by repeating the last point. Measured padding:

| family | chunks | median movements/chunk | median padding fraction |
|---|---|---|---|
| human | 440 | 65 | 0.35 |
| NaiveBot | 192 | 4 | **0.96** |
| HumanishBot | 293 | 12 | 0.88 |
| MimicBot | 399 | 44 | 0.56 |
| FallibleBot | 357 | 44 | 0.56 |

The architecture search selected on windows built from **whole sessions** (mostly real
points). Re-evaluating the same models at the **chunk** operating point inverts the
ranking:

| representation | chunk AUC (session-trained) | chunk AUC (chunk-trained) | session-level AUC used for selection | collapse |
|---|---|---|---|---|
| **kinematic** | **0.885** | **0.880** | 0.959 | 0.074 |
| xy (published) | 0.822 | 0.848 | 0.906 | 0.058 |
| dxdy | 0.802 | **0.594** | 0.977 | **0.175** |
| dxdy_dt | 0.798 | 0.700 | 0.973 | 0.175 |

> **Treat this table as a signal, not a result.** It is **one seed**, and reruns of
> identical code have moved AUC by 0.038 in this project. It also used the duplicated
> chunk set (see below). Phase 0 must redo it properly. Its job here is only to show
> that the operating point changes the answer.

**`dxdy` — the representation previously selected — is the worst at the real operating
point.** The mechanism is *not* what it first appears, and the distinction matters for
anyone reimplementing:

`expkit.features.windows` applies the feature transform **first** and then repeats the
last feature **row**. So for a difference representation the padded tail repeats the last
*velocity* — "the cursor kept moving at its final speed forever" — filling 35–96% of the
window with a large artificial constant. It does **not** become zeros. Verified:

```
30-point chunk, dxdy       last real row [-27, 15]   padded rows [-27, 15]   <- repeat_row
                            with repeat_point:        padded rows [  0,  0]
```

For `xy` the two orders coincide (a repeated position), which is physically sensible and
is why `xy` degrades least. `rlcaptcha._windows` pads *points* before any transform, so
the published pipeline was effectively `repeat_point` — but it only ever used `xy`, so
the distinction never surfaced.

**Padding order is therefore a first-class experimental variable, not an implementation
detail.** `expkit/humanity_scorer.py` implements all four modes explicitly and
self-tests them (`python -m expkit.humanity_scorer`).

**Duplicate chunks.** `_split_into_chunks` recursively replays earlier chunks when a
session is shorter than 12 windows, so every simulated user lasts a full episode. Across
the corpus that turns **760 distinct chunks into 1,681**, skewed hard toward short
sessions:

| family | chunks | distinct |
|---|---|---|
| NaiveBot | 192 | **19** |
| HumanishBot | 293 | 89 |
| human | 440 | 306 |

Training or computing metrics on the duplicated set silently up-weights short (bot)
sessions. **Deduplicate for scorer selection, calibration and reporting; keep duplicates
in the simulation**, because that is what the environment replays.
`build_chunk_dataset(dedup=True)` is the default.

A second concern falls out of the same data: `corr(score, padding fraction)` is **+0.28
to +0.38**. The scorer is partly reading *how much the cursor moved* rather than *how
humanlike the motion is*. That is a real signal but a degenerate one — an artifact of the
fixed 100-point window — and exactly the kind of shortcut Reviewer 1 would name.

**The honest operating-point AUC is far below the session-level figure.** A quick
end-to-end fit of `kinematic`+`mask` on the `lstm` pool alone, deduplicated, gave
**chunk AUC 0.687** on the eval pool. Expect the RL-safe scorer (73 recordings) to be
substantially weaker than the cross-validated estimate over all 156, and report both.

### 4.2 `expkit/humanity_scorer.py` — already written, use it

The module exists in the repo and is tested. **Do not reimplement it from this prose** —
the padding order and the dedup rule are exactly what a reimplementation gets wrong, and
both change the numbers. Run `python -m expkit.humanity_scorer` to see the self-test.

It provides:

```python
ScorerConfig(representation="kinematic",   # xy | xy_dt | dxdy | dxdy_dt | kinematic
             padding="mask",               # repeat_point | repeat_row | mask | zero_mask
             lstm_units=(32,), dense_units=(32,), dropout=0.0, recurrent_dropout=0.0,
             lr=1e-3, batch_size=64, context=100,
             class_balance="window",       # window | family  -> see P0.4
             augmentation="none", n_copies=1, magnitude=1, sigma=2.0)

HumanityScorer(cfg).fit(train_refs, val_refs, seed=0)
    .score_chunks(chunks, batch_size=4096)   # BATCHED -- the only path RL uses
    .score_chunk(movements)                  # single, for tests
    .save(dir) / HumanityScorer.load(dir)
    .temperature                             # fitted on held-out val AT CHUNK LEVEL

build_chunk_dataset(refs, cfg, dedup=True) -> ChunkSet(X, y, groups, families, pad_fraction)
```

The four padding modes, and why all four are in the sweep:

| mode | what it does | padded tail of a `dxdy` window |
|---|---|---|
| `repeat_point` | pad the raw points, then transform | `[0, 0]` — "cursor stopped" (matches `rlcaptcha`) |
| `repeat_row` | transform, then repeat the last row | `[-27, 15]` — "cursor flies on at final velocity" (what produced §4.1's table) |
| `mask` | `repeat_row` + a binary real/pad channel | `[-27, 15, 0]` |
| `zero_mask` | zeros + a binary real/pad channel | `[0, 0, 0]` |

Guarantees already implemented and asserted:

* `score_chunks` matches `score_chunk` to **1.19e-07** (verified).
* Temperature is fitted **at the chunk operating point** on held-out data. The previously
  reported T = 1.516 was fitted on session windows and is wrong for deployment; a chunk-level
  refit gave T ≈ 1.571 for `kinematic`+`mask`.
* `class_balance="family"` equalises the four bot generators, not just human-vs-bot.

### 4.2b What the context length did — the dominant effect

Sweeping `{dxdy, kinematic} × {repeat_point, mask} × context {100, 50, 32}` at the chunk
operating point, deduplicated, 5-fold CV, 2 seeds:

| representation | padding | ctx | AUC | ECE | corr(score, pad) | mean padding |
|---|---|---|---|---|---|---|
| **kinematic** | **mask** | **32** | **0.975 ± 0.001** | **0.010** | 0.222 | 0.09 |
| kinematic | mask | 50 | 0.976 ± 0.006 | 0.014 | 0.239 | 0.19 |
| kinematic | repeat_point | 50 | 0.964 | 0.028 | 0.280 | 0.19 |
| dxdy | mask | 32 | 0.955 | 0.018 | 0.187 | 0.09 |
| kinematic | repeat_point | 100 | 0.937 | 0.034 | 0.295 | 0.44 |
| kinematic | mask | 100 | 0.890 ± 0.018 | 0.032 | 0.411 | 0.44 |
| dxdy | mask | 100 | 0.750 ± 0.078 | 0.066 | 0.363 | 0.44 |

Three things fall out:

* **Context length dominated everything.** Going 100 → 32 moves `kinematic`+`mask` from
  0.890 to 0.975 and `dxdy`+`mask` from 0.750 to 0.955. The inherited 100-point window
  was the actual problem; the representation debate was largely an artifact of it.
  Mean padding falls 0.44 → 0.09.
* **`kinematic` beats `dxdy` at every context and padding combination.** The earlier
  session-level ranking, which put `dxdy` top, does not survive the operating point.
* **`mask` beats `repeat_point` once padding is low** (0.975 vs 0.961 at ctx 32) but is
  *worse* at ctx 100 (0.890 vs 0.937). A mask channel helps when there is a little
  padding to mark and cannot rescue a window that is 44% padding.

The padding shortcut is reduced but not eliminated: `corr(score, padding)` falls from
0.41 to 0.22. P0.6 still matters.

Selection applied the pre-declared rule: ctx 50 and 32 are within 1 SE on AUC
(0.976 vs 0.975, 1 SE ≈ 0.004), so the tie broke on ECE (0.010 vs 0.014), then on
`corr_pad` (0.222 vs 0.239). Both are defensible; the difference is at noise level.

### 4.3 Phase 0 experiments

**Status: P0.1 and P0.7 are substantially answered above at 2 seeds.** Re-run them at 5
seeds to confirm, and treat the rest as refinement worth ±0.02 AUC.

All at the **chunk operating point**, grouped 5-fold cross-validation by recording over
all 156 sessions, **5 seeds** each, out-of-fold pooling.

| id | sweep | why |
|---|---|---|
| **P0.1** | representation × padding — `{xy, xy_dt, dxdy, dxdy_dt, kinematic}` × `{repeat_point, repeat_row, mask, zero_mask}`, **at ctx 32** | **Partly answered (4.2b): `kinematic`+`mask` wins.** Re-run at 5 seeds with `xy`/`xy_dt` included, which 4.2b omitted. |
| **P0.2** | architecture — `{(16,), (16,8), (32,), (32,16), (64,32), (200,100)+dense(128,64)}` × best 2 representations | Re-confirm capacity at the right operating point. Include the published architecture as the reference row. |
| **P0.3** | lr `{3e-4, 1e-3, 3e-3}` × dropout `{0, 0.25}` × recurrent_dropout `{0, 0.2}` on the top 3 | Regularisation matters more when inputs are half padding. |
| **P0.4** | class balancing — `window` (current) vs `family` (equalise the four bot generators) | NaiveBot has only **19 distinct chunks** against MimicBot's and FallibleBot's hundreds, so those two dominate the loss. (At chunk level NaiveBot is ~11% of *rows* but those rows are 19 distinct chunks replayed — dedup first, then judge.) |
| **P0.5** | augmentation — `none`, `rigid ±1`, `per_move ±1/±2/±3`, `gaussian σ=1/2`, each ×4 copies | Redo at chunk level. On session windows `none` won; that may not hold once inputs are padded. |
| **P0.6** | **padding shortcut audit** — report `corr(score, padding_fraction)` and AUC restricted to chunks with padding < 0.4 | If AUC collapses on low-padding chunks, the scorer is reading activity volume, not dynamics. Must be reported either way. |
| **P0.7** | context length `{16, 32, 50, 64}` | **Largely answered — 32 chosen, see 4.2b.** Confirm at 5 seeds and probe below 32, since the trend had not clearly turned. |

**Selection rule — fix it in writing before running, and do not change it afterwards:**

0. All metrics at the **chunk** operating point, on **deduplicated** chunks.
1. Primary: **chunk-level AUC**, mean over 5 seeds.
2. Among configs within 1 SE of the best: lowest **chunk-level ECE after temperature
   scaling**.
3. Among those: lowest `|corr(score, padding_fraction)|`.
4. Among those: fewest parameters.

Report **per-family recall at threshold 0.5 with window counts**, never per-family AUC.
(AUC on a 5-window class is meaningless — it is what made NaiveBot look like a 0.975
failure when its recall was 16/16. See §7.)

### 4.4 Phase 0 deliverables

* `expkit/humanity_scorer.py`, tested.
* `results/phase0_*.csv` — full sweep tables.
* `results/final_scorer/` — the chosen model, standardiser, temperature, config JSON,
  trained on the **`lstm` pool only**.
* A one-paragraph statement of the chosen design and its chunk-level numbers, which
  becomes the paper's scorer section.
* Assertion: batched vs per-chunk scoring agree to < 1e-6.

---

## 5. PHASE 1 — re-run every positive result with the final scorer

Absolute values **will shift**. The prior numbers below are ballparks for sanity, not
targets. If a headline conclusion flips, that is a finding — report it, do not tune.

Standing requirements for all of Phase 1:

* **≥ 5 training seeds** per learned agent, **20–50 evaluation seeds** per cell.
* **≥ 200 training episodes**; report training curves.
* Headline averages **exclude the zero-bot cell** (see §7).
* Recording-level bootstrap CIs, **≥ 1000 resamples**.
* Every comparison includes an **untrained baseline** as a null.

### E1 — Table 4 on one equal-footing environment

The published bandit runs differ from the others in four ways (all flagged in
`training/README.md`): bots abandon the site, overkill 2.0 vs 2.5, a dead
`last_threat_level` feature initialised to −1, and acting on exhausted users.

Produce **two tables**:

* **Headline** — all six policies in one environment with **all four** mismatches fixed:
  `abandonment_applies_to_bots=False`, `overkill=2.5`, `initial_last_threat=0` with
  `updates_last_threat=True`, and `acts_when_exhausted=False`. This is the like-for-like
  comparison the paper currently lacks. **Depends on E9** — the bandit rows are only
  meaningful once the bandits are retrained on the `rl` pool; until then mark them
  provisional.
* **Reproduction** — the published configuration, to show Table 4 is recoverable.

Prior: DQN avg DI ≈ 74.8, Multi 54.6, Single 46.6, LinUCB 52.7, TS 57.1, ablation 40.5.

### E2 — leakage cross-fit (nb03b), **10 splits**

Role-swapped design: train A on P1, B on P2; `seen = {A|P1, B|P2}`,
`unseen = {A|P2, B|P1}`. Pool difficulty cancels algebraically. **Assert the untrained
baselines' gap is exactly 0.000** — that assertion is what proves the design works.
Prior: DQN +7.1 DI [3.9, 10.3] on 3 splits; ablation +0.0.

### E3 — reward sensitivity (nb04), **5 seeds per setting**

The weakest existing table — it had **one** seed per setting. Grid: β `{1.5, 2.0, 2.5}` ×
R_leak `{−50, −150, −300}` × overkill `{2.0, 2.5}` × underestimation `{2.5, 5.0, 10}`,
plus a linear (seconds-priced) friction variant. Retrain for each; re-scoring a fixed
policy is provably a no-op.

Also **report a Pareto frontier** (bots blocked vs human friction seconds) rather than
collapsing to DI — DI hides which settings buy security with usability.

Abandonment curve sweep: `paper`, `empirical τ ∈ {22, 45, 80}`, `none`. This is the only
reward-side parameter that moves the metrics directly.

### E4 — independent traffic (nb05)

Grid humans `{25, 50, 100, 200, 400}` × bots `{0, 20, 100, 200, 500, 1000}`.
**Retrain on the varied grid** — the prototype only evaluated on it.
Population-feature ablation: frozen-at-constant is the valid test; randomised is
out-of-distribution and must be labelled as not a test.

### E5 — stochastic action model (nb06)

Assert the deterministic limit first (**144/144 exact**). Sweep α `{0.5, 1, 1.5, 3, 6, 12, ∞}`,
and `solver_era ∈ {0, 1}`. Report false-positive rate and friction seconds — quantities the
deterministic model cannot express and which answer "usability is modeled, not measured".

### E6 — data sufficiency (nb08)

Session-level bootstrap (≥1000), variance decomposition, learning curve over pool
fraction. Prior: 86–89% of variance is between recordings; the learning curve had **not**
flattened at the full pool.

---

## 6. PHASE 2 — the negative results, and the best solves for them

### E7 — proxy reward for deployment (the most valuable experiment here)

**Status: unsolved.** A reward built only from observable signals produced a *usable* but
not *secure* policy — it kept all 100 humans (zero false positives, zero friction) and
left ~255 bots alive against the oracle's ~11.

Three fixes were tried and all failed:

| tried | result |
|---|---|
| label coverage 0.02 → 1.00 | flat; ~252–263 bots left throughout |
| abuse penalty −150 → −9600 | non-monotonic; only −2400 defends (146 bots), −9600 collapses |
| credit window 3 → 12 steps | **worse** (255.8 vs 145.3) — dilutes the per-step signal |

**Diagnosis.** The immediate reward table is:

```
level   pass+stay   pass+leave   fail+gone
0          +50.0        -60.0        +5.0
2          +24.0        -61.0        +4.0
7           +1.2        -83.8       -18.8
```

Doing nothing pays +50 with certainty every step; challenging pays at most +24. So
challenging costs 26/step immediately, while the abuse penalty is delayed, discounted by
γ=0.95, fires with probability 0.3, and is spread over 3 steps. Undiscounted break-even
needs a penalty above 26 × 3 / 0.3 = 260; empirically it takes ~2400, and by then training
destabilises.

**Solves to implement, in priority order:**

1. **Potential-based reward shaping** (Ng, Harada & Russell 1999). Add
   `F(s,s') = γΦ(s') − Φ(s)` with Φ a function of the behavioural score. The theorem
   guarantees the optimal policy is **unchanged**, so this adds a dense, immediate,
   *observable* signal pointing toward blocking bots without biasing the solution. This is
   the principled version of what was attempted by hand and is the most likely to work.
2. **Learned reward model.** Train a classifier to predict "will this session be confirmed
   abusive" from observable features, and use its continuous output as a dense per-step
   reward. Standard practice in fraud RL; turns one sparse label per session into a signal
   at every step.
3. **Sweep the immediate gap directly** — `zero_friction_reward` ∈ {50, 35, 25} against
   `engagement_reward` 25. The 50-vs-24 gap is the lever the break-even analysis
   identified, and it is far cheaper than a −2400 penalty that destabilises training.
4. **Reward normalisation + gradient clipping** (clip to ±10, normalise returns), to test
   whether the −9600 collapse is optimisation rather than reward design.
5. **1000-episode runs.** The proxy signal is much noisier than the oracle's; 90 episodes
   may simply be below threshold.
6. **Off-policy evaluation** (importance sampling / doubly-robust) to estimate a policy's
   value from logged data without deploying it — the natural companion result.

If (1) or (2) works, the message changes from "deployable rewards have a structural
limitation" to "here is how to build one", which is materially stronger.

Keep the **observability constraint** intact whatever you try: the immediate reward must
be a pure function of the observable tuple, so a blocked bot and a false-positived human
receive *identical* immediate reward. Measured ambiguity: `failed_gone` is 95.1% blocked
bot / 4.9% false-positived human; `passed_continued` is 17.4% leaked bot / 82.6% satisfied
human. And the proxy agent must not balance its replay buffer by true class — balance on
the observable outcome.

### E8 — stochastic-model retraining at matched budget

An agent retrained inside the grounded environment scored 33.7 against the published
checkpoint's 45.2 — but it got 90 episodes on 31 recordings while the published checkpoint
got 200+ on 73 *and* carries partial leakage into that comparison. **Redo at matched
budget with cross-fitting.** As it stands this comparison is uninterpretable.

### E9 — retrain both bandits on the `rl` pool

The one real hole in the leakage answer. LinUCB and Thompson Sampling keep published
checkpoints fitted on campaign B, which overlaps the eval pool. Start from
`offline/offline_training_linucb.ipynb` and `offline_training_thompson_sampling.ipynb`.

Implementation note the reviewer explicitly asks about (R1.6): the per-arm statistics
`A_a = I + Σ z zᵀ` and `b_a = Σ r z` accumulate in feature space `z`, but `z` changes
meaning as the feature network trains. State and implement the handling — periodic reset
of A and b, or freeze the network after a warm-up — and report which.

---

## 7. Standing rules (each of these caught a wrong number)

1. **Put an untrained baseline through every comparison.** It caught three of five wrong
   headline numbers. A static-threshold policy cannot memorise, cannot learn, and cannot
   leak — whatever gap *it* shows is your null.
2. **Exclude the zero-bot cell from headline averages.** With no bots, `S_B` is undefined
   and DI = 100 for any policy that lets everyone through. Averaging it in rewards
   permissiveness — it is what made a proxy agent that challenges *nobody* appear to beat
   the oracle by 14%. Report the all-six average only for continuity with the paper.
3. **State the BOS convention and recompute.** Table 4 reports BOS = 0.890 at zero bots
   for DQN, which is just `s_h`; Eq. (17) with "no bots ⇒ perfect rejection" gives
   F1(0.89, 1.0) = **0.942**. Every Average row inherits the error.
4. **Per-family: report recall with window counts, never AUC.** NaiveBot's "0.975" was AUC
   over 5 windows; its cross-validated recall is **16/16 = 1.000**. AUC is a ranking metric
   and one inversion on a 5-sample class costs 0.025.
5. **Seed the global generators**, not just the simulation's `seed` argument (§2).
6. **Never trust `nbconvert --inplace` after editing.** It writes the executed notebook
   back, silently reverting source edits. Rebuild from source and *verify the built
   artefact* before launching. This cost three wasted runs.
7. **Recording-level bootstrap, not seed-level.** Seed-level intervals will be tight
   regardless of how small the pool is.

### Automated assertions to run in CI

```
assert simx.run_x(..., DETERMINISTIC, PAPER_EQUIVALENT) == rlcaptcha.run_simulation(...)
       for all 6 policies x 6 bot volumes x 4 seeds            # 144/144
assert crossfit_gap(static_baseline) == 0.0                     # role swap cancels
assert max|score_chunks - score_chunk| < 1e-6                   # batched path
assert scorer trained only on partition['lstm']                 # no RL-pool leakage
```

---

## 8. Seed and budget floors

| quantity | floor |
|---|---|
| scorer CV | 5-fold grouped, **5 seeds**, repeated k-fold if budget allows |
| RL training seeds | **5** per learned agent |
| RL evaluation seeds | **20** (50 preferred) per cell |
| training episodes | **200+**, curves reported |
| bootstrap resamples | **1000+** |
| cross-fit splits | **10** |
| reward-sensitivity seeds | **5 per setting** (currently 1) |

---

## 9. Known limitations to state, not fix

* **3 participants**, 44 human recordings, across two sittings. The effective number of
  independent human samples is 3, not 44. Supportable: "distinguishes these three
  participants from these four bot generators". Not supportable: "generalises to human
  users".
* **4 scripted bot generators.** Bot-side conclusions characterise four programs.
* **NaiveBot recordings carry a median of 4 mouse movements** — that family is detected by
  absence of movement, not by movement dynamics. Say so.
* The learning curve has not flattened at the full pool: more recordings would still
  tighten every result.

---

## 10. Optional, high value if compute allows

* **E10 — adaptive attacker.** Bots whose strength responds to the policy's recent
  actions. This is the single most valuable addition: it moves the paper from
  "orchestration against fixed traffic" to "orchestration against an adversary", which is
  what the introduction promises and what R1.8 says is missing.
* **E11 — a queueing layer for the DoS claim (R1.5).** Model request arrival rates, a
  finite-capacity server (M/M/c), latency and drop rate. Without it, "DoS mitigation"
  should narrow to "robustness to traffic composition".
* **E12 — nested outer-fold pipeline** (§3), tripling evaluation data.
* **E13 — a real VLM solver or a public automation framework** recorded as a fifth bot
  family, to widen bot diversity beyond code the authors wrote.

---

## 11. Deliverables

1. `expkit/humanity_scorer.py` + Phase 0 sweep tables + the chosen scorer artefact.
2. One results CSV per experiment, plus a regenerated `FINDINGS.md`
   (`python summarise_results.py` builds it from the CSVs — keep that property, nothing
   hand-entered).
3. Paper-format Table 4 (headline + reproduction) as XLSX and PDF.
4. Figures at 300 dpi.
5. A short delta note: which prior conclusions held, which moved, which flipped.
