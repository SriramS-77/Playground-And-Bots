# Round 2 — addendum to `HANDOFF_GPU_EXPERIMENTS.md`

Read this **together with** `HANDOFF_GPU_EXPERIMENTS.md`. That document's science is
unchanged. This one records what round 1 (Slurm job 1263, 2026-09-28, 1095 min) actually
executed, the defects it exposed, the design changes since, and the exact queue for
round 2.

**Where the two documents disagree, this one wins.** Specifically:

| `HANDOFF_GPU_EXPERIMENTS.md` says | superseded by |
|---|---|
| §3 — one fixed `partition.json` (73/31/52) | §3.1 — a 3-fold **rotation**; 31 RL recordings is what starved the agent |
| §4.2 — load `results/final_scorer/` | §3.2 — carry the **config**, refit weights and `T` **per fold** |
| §4.3 — Phase 0 cross-validated over all 156 | §6 Priority 0 — per fold on `scorer ∪ rl`; the eval block never picks the config |
| §4.4 — scorer trained on the `lstm` pool | §3.2 — on **this fold's** scorer block |
| §7 — `preflight` shown as illustrative asserts | §4 — `preflight.py`, real and importable |
| §9 — leave-one-campaign-out ≈ an unseen participant | §3.1 — **same three participants**; it is a temporal holdout, nothing more |
| E9 as a separate experiment | §6 Priority 1 — subsumed; the bandits are retrained by construction |
| E1's four environment mismatches | §2 — there are **five**; the state encoding also differs |

Everything else in that document — the reviewer mapping, the metric definitions, the
standing rules, the seed and budget floors, the limitations to state rather than fix —
still holds.

**New code shipped with this document** (all under `training/experiments/`, nothing in
`rlcaptcha/` was touched):

| file | what it adds |
|---|---|
| `expkit/partition.py` | `stratify_by="family"`, `EQUAL_THIRDS`, `split_refs`, **`make_rotation`** (3-fold role rotation), `save/load_rotation` |
| `expkit/bandits_x.py` | **`LinUCBX`, `ThompsonX`** — trainable bandits aligned with the DQN on all five axes; `BanditArch`; **`MixturePosterior`** (MoG Thompson Sampling); `train_bandit`; `fit_posteriors_from`; `recompute_statistics` (the R1.6 fix) |
| `expkit/trainer.py` | `QNet(hidden=...)` and `TrainableDQN(hidden=..., lr=...)` for the depth/width search |
| `expkit/simx.py` | `verify_against_published()` — the 144/144 assertion, lifted out of `nb_06` so it can run in preflight |
| **`exp_scorer_phase0.py`** | **Phase 0 runner** — per fold, chunk operating point, 5 seeds, the fixed selection rule, the P0.6 audit |
| **`exp_policy_arch_search.py`** | the policy + posterior search, its selection rule, and eight smoke tests |
| **`preflight.py`** | the gates, importable: `import preflight; preflight.check()` |
| **`slurm/`** | `phase0.sh`, `prepare_scorers.sh`, `arch_search.sh` + a README with the run order and the measured budget |
| `summarise_results.py` | **fixed**: the zero-bot cell no longer leaks into headline means (see §1.1) |

All defaults reproduce round-1 behaviour exactly — verified: `make_partition()` with no
arguments still yields the identical 73/31/52 split, and the default `QNet` still loads
the published DQN checkpoint.

**Start here:** `slurm/README.md` has the run order end to end. The three `--smoke` modes
(`preflight.py`, `exp_scorer_phase0.py --smoke`, `exp_policy_arch_search.py --smoke`) take
minutes and validate the whole chain before anything expensive launches.

---

## 0. What round 1 ran, and what it did not

Round 1 executed the **pre-handoff notebook chain** (`nb_01`–`nb_09` plus the four
probes), not the handoff. Evidence, all from the shipped logs:

| observation | where |
|---|---|
| `retrained scorer: xy_dt__s2` | `log_07_clean_table4.txt`, line 1 |
| shipped `scorer/model.keras` md5 == the repo's pre-existing `results/final_scorer/model.keras` | byte-identical, i.e. copied, not retrained |
| `FINDINGS.md` §10 and §11 render empty | `nb_10`, `nb_10b`, `nb_11` never ran |
| `Could not find cuda drivers ... GPU will not be used` on every step | 27 occurrences in `slurm_experiments_1263.out` |
| 150 episodes (nb03) / 90 episodes (nb06, nb07) | handoff floor is **200+** |
| one training run per configuration | the `seed` column in every `*_runs.csv` is an **evaluation** seed, not a training seed |
| `expkit/*.py` identical to the repo copy | no code was written |

**Consequence: no RL result in round 1 used the final humanity scorer.** `nb_03` reads
`results/scorer_choice.json` (written by `nb_02`) and calls `XScorer.from_name(...)`.
Everything downstream of it inherits that scorer.

Genuinely more comprehensive than the prior local run, and worth keeping:

* **E2 at 10 splits** (was 3) — the leakage number is now solid.
* **20 evaluation seeds** in `nb_03` (was 12) and **4 independent unseen draws**.
* Independent cross-platform reproduction of the 144/144 deterministic-limit assertion
  and of the scorer probes (`kinematic`+`mask`+ctx 32 → chunk AUC 0.970, ECE 0.014).

---

## 1. Defects to fix **before** running anything

### 1.1 `summarise_results.py` — zero-bot cell leaked into headline means — **FIXED**

It grouped `proxy_vs_oracle_runs.csv` by policy and took an unfiltered mean, so the
zero-bot cell was included. That printed **81 %** into `FINDINGS.md` §7 and `README.md`
claim 5, while `nb_07`'s own log said:

```
including the zero-bot cell : proxy / oracle = 81%  <- inflated, do not quote
excluding it                : proxy / oracle = 65%  (35.7 vs 55.1)
```

**Already fixed in this push.** A shared `scored(df)` helper now drops `bots == 0`, and it
is applied to the proxy table, the stochastic-environment table, the label-coverage sweep
and the abuse-penalty sweep. Use `scored()` (or `preflight.headline()`) for every new
headline mean you add. DI is undefined at zero bots and equals 100 for any policy that
lets everyone through, so averaging that cell in rewards permissiveness — it is what made
a proxy agent that challenges *nobody* look like it beat the oracle.

Re-running `python summarise_results.py` on the round-1 CSVs will now print 65 %, not
81 %. **`README.md` claim 5 and `FINDINGS.md` §7 still carry the old number** until it is
regenerated.

### 1.2 `nb_05` — the population ablation is over-claimed

The notebook freezes `n_active` at the **single** constant 100 and reads the verdict off
DI alone. At that constant:

| variant | DI | human survival | bot survival |
|---|---|---|---|
| DQN | 69.0 | 0.795 | 0.105 |
| DQN (population frozen at 100) | 82.3 | 1.000 | 0.177 |

Freezing keeps *every* human and leaks **69 % more bots**. It does not "cost essentially
nothing" — it makes the policy markedly more permissive, and DI happens to reward that.
100 is also near the bottom of the grid's 25–1400 population range, so the constant
injects a "low traffic" signal rather than removing information.

Compounding this, the notebook's own regression says the advantage tracks population
(−0.983) more than bot fraction (+0.665), and its printed `VERDICT` text is hardcoded
prose that does not read the numbers. **`README.md` claim 3 is not supported.**

Fix: see E4 below.

### 1.3 Not a defect — checked and cleared

The `91h/0b`, `82h/0b`, `84h/0b` lines in `log_11`/`log_12` are a logging coincidence, not
a broken sampler. Replaying `random.Random(seed+1)` through `train_dqn`'s loop reproduces
exactly those three draws; over 90 episodes the bot volumes are 13/10/15/20/15/17 across
`(0, 20, 50, 100, 200, 500)`. No action needed.

---

## 2. The finding that now needs deciding first

Recomputed from `clean_partition_runs.csv`, `condition == "clean"`, **zero-bot cell
excluded**:

| policy | avg DI (bots > 0) | avg DI (all six, as shipped) |
|---|---|---|
| Thompson Sampling | **46.5** | 55.4 |
| LinUCB | **41.7** | 51.5 |
| DQN (clean) | **37.8** | 43.9 |
| Static Single-Threshold | 26.7 | 29.1 |
| Static Multi-Threshold | 26.1 | 30.6 |
| DQN without H-Score | 12.3 | 26.9 |

On the leaky published condition the same file gives DQN 72.8 — first by a wide margin.
**Once the overlap is removed, the DQN is third.** That is the paper's central claim.

### But the averages hide the mechanism. Break it down by bot volume

| policy (clean condition) | 0 | 20 | 100 | 200 | **500** | **1000** |
|---|---|---|---|---|---|---|
| Thompson Sampling | 100 | 85.5 | 88.6 | **22.4** | 22.9 | 23.1 |
| LinUCB | 100 | 77.8 | 42.6 | 42.5 | 22.4 | 23.5 |
| DQN (clean, retrained) | 74.2 | 60.2 | 65.2 | 61.1 | **3.4** | **−0.8** |
| DQN (published, same eval pool) | 100 | 81.7 | 84.0 | 82.6 | **81.5** | **80.6** |

Three facts fall out:

1. **The bandits do not scale — they floor at ~23.** Both fall off a cliff above 100 bots
   and sit at a constant thereafter. At 500 bots Thompson keeps **24.6 of 100 humans**
   while killing 99 % of bots: that is a near-always-challenge policy, and DI ≈ 23 is
   simply what "block everyone" scores. Their average is carried entirely by the low-bot
   cells.
2. **The clean DQN collapses at exactly those two volumes**, keeping 7 of 100 humans at
   500 bots. Below 200 bots it is healthy.
3. **That collapse is not the algorithm.** Same evaluation pool, the published agent
   scores 81.5 and 80.6 in those cells. The cross-fit puts leakage at **+8.7 DI**, which
   cannot explain a 78-point drop.

### Six reasons the comparison is stacked against the DQN

1. The bandits still load **campaign-B checkpoints that overlap the eval pool** — leaky,
   where the DQN is not.
2. **Bots abandon the site** in the bandit environment, regardless of strength, on top of
   being blocked. Free bot removal.
3. **Overkill 2.0 vs 2.5.**
4. **`last_threat_level` dead at −1.** At 500 bots that constant alone shifts many
   decisions from threat 9 to 10 — it makes the bandits maximally aggressive for free,
   which is precisely the 99 %-blocked / 75 %-humans-killed signature.
5. **The bandits act on exhausted users.**
6. **Different feature normalisation** (not in the README's list of four, easy to miss):

   | feature | published bandit | DQN |
   |---|---|---|
   | population | `tanh(n / 200)` | `n / 300`, unbounded |
   | captchas solved | `tanh(x / 10)` | `tanh(x / 5)` |
   | last threat level | tanh-scaled | raw |

   A smoke test asserts both halves of this: that `LinUCBX`/`ThompsonX` produce the DQN's
   state vector exactly, **and** that the published class still produces a different one —
   so if someone later "fixes" `rlcaptcha`, the claim fails loudly instead of going stale.

Plus one that hurts every learned policy equally and may explain the 1000-bot cell:
`train_dqn` samples bot volumes up to **500**, but evaluation goes to **1000**. At 1000
bots the DQN's population feature is `1100/300 = 3.67` against a training maximum of 2.0
— out of distribution. It does **not** explain the collapse at 500, where the input is
in range, so undertraining still has to be tested separately.

`expkit/bandits_x.py` fixes 2–6 by construction and `train_bandit` defaults to
`bot_choices=(0, 20, 50, 100, 200, 500, 1000)`. Reason 1 is fixed by retraining, which is
now the default path rather than a separate experiment.

**If the DQN is still third on genuinely equal footing, the honest move is to reframe the
contribution around *adaptive orchestration versus static policies*** — where the margin
is large and holds at every bot volume — rather than around DQN-beats-bandits. Decide
this from the numbers, not before them.

---

## 3. Design changes since the original handoff

### 3.1 Partition: a 3-fold role rotation, no campaign holdout

The two campaigns are two sittings by the **same three participants** three months apart.
Campaign is therefore a nuisance variable, not a population boundary. Two consequences:

* Stratify on `family`, not `campaign:family`. Campaign is still reported, never held out.
* **Stop describing leave-one-campaign-out as a proxy for an unseen participant or
  setup.** It is a 3-month temporal holdout with the same people. `FINDINGS.md` §9 and the
  old handoff both overstate this; fix the wording in the paper.

The old 45/20/35 allocation gave the RL agent **31 recordings, 8 of them human** — the
leading suspect for §2's collapse. `make_rotation` replaces it:

```python
from expkit.partition import index_sessions, make_rotation
folds = make_rotation(index_sessions())      # 3 folds
# fold i: scorer = block i, policy = block i+1, eval = block i+2
```

Verified output:

```
fold 0: scorer 52  rl 50 (fit 32 / val 18)  eval 54
fold 1: scorer 50  rl 54 (fit 36 / val 18)  eval 52
fold 2: scorer 54  rl 52 (fit 34 / val 18)  eval 50
```

Every recording is evaluated **exactly once** across the three folds, so results pool over
all 156 instead of one arbitrary 52-recording slice. The three folds are independent —
run them as three array tasks and the wall-clock is unchanged.

### 3.2 The scorer is refit per fold — do **not** load `final_scorer/`

`results/final_scorer/` was fitted on the **old** `lstm` pool. Under the rotation those
recordings land in the new policy and evaluation blocks, which is exactly the leak E2
measured at +8.7 DI. Carry forward the finalised **config**, not the weights:

```python
ScorerConfig(representation="kinematic", padding="mask", context=32,
             lstm_units=(32,), dense_units=(32,), class_balance="window",
             augmentation="none")
```

Refit weights **and temperature** on each fold's scorer block
(`exp_policy_arch_search.fold_scorer` does this, and asserts the scorer never saw the
policy or evaluation blocks). `T = 1.376` was a property of the old fit; do not assert it.

### 3.3 Architecture and posterior selection happens on `rl_val`, never on `eval`

Choosing an architecture by evaluation-pool DI is test-set selection — Reviewer 1's point
2 all over again. The policy block is split 2:1 by recording; candidates rank on the
held-out third; the winner is retrained on the full policy block and evaluated once.

**Selection rule, fixed in writing before running and not to be changed afterwards:**

1. Primary: mean DI over evaluation seeds, **zero-bot cells excluded**.
2. Among candidates within 1 SE of the best: **fewest parameters**.
3. Ties broken by name, for determinism.

### 3.4 Thompson Sampling: a mixture-of-Gaussians posterior

The published sampler draws `θ̃_a ~ N(μ_a, α² diag(A_a⁻¹))` — a single Gaussian over a
linear reward model. But an arm's reward is bimodal: the same threat level pays very
differently to a human and to a bot, and both are present in every episode. One linear
model cannot express that.

`MixturePosterior(K)` fits K components per arm, **unsupervised on `(z, r)` only, never on
`is_bot`** — a deployed system does not have the label. The part that matters is that
**the gate depends on the context**:

```
π_k(z)  =  π_k N(z; m_k, v_k) / Σ_j π_j N(z; m_j, v_j)
score(a) =  Σ_k π_k(z) · (z · θ̃_k),    θ̃_k ~ N(μ_k, α² Σ_k)
```

A context-*free* gate would be pointless: drawing one component per decision gives
`E[r|z,a] = z · Σ_k π_k θ_k`, still linear in `z`, so the mixture would add no
representational power — it would just randomise over "the world is all human" versus
"all bot", which is noise, not a Thompson sample.

Four variants, swept: `gaussian` (published), `gaussian_full` (full covariance),
`mog2`, `mog3`. **`gaussian_full` is in the grid deliberately** — without it, "the mixture
helped" cannot be distinguished from "we dropped the published diagonal approximation".

`fit_posteriors_from` refits the posterior on the **same trained weights and the same
buffer**, so the posterior is the only variable and the three extra variants cost almost
nothing.

**Asserted in `smoke_tests`, and passing:** at K=1 the mixture reproduces the published
Gaussian Thompson draw **bit-exactly** under the same seed (`|Δ| < 1e-9`); the gate is
verifiably context-dependent at K=2; and batch EM fits a two-regime reward (K=1 MSE 13.8 →
K=2 0.08).

#### The gate has to be tempered, or the mixture is degenerate by construction

Found while testing, and it would have wrecked this arm of the search. The gate multiplies
a diagonal-Gaussian density over all 64 embedding coordinates, so the log-likelihood
**gap** between components scales with the embedding width and the softmax saturates.
Measured on embedding-like data:

| | gate entropy (max) | sampled-score sd |
|---|---|---|
| K=1 | — | 3.07 |
| K=2, untempered | **0.0022** (0.693) | 4.71 |
| K=3, untempered | **0.0073** (1.099) | 5.69 |
| K=2, tempered `1/d` | **0.6702** (0.693) | 2.56 |
| K=3, tempered `1/d` | **1.0700** (1.099) | 3.03 |

Untempered, the "soft" gate is a hard one-hot assignment that flips discontinuously with
small changes in `z`. Combine that with each component seeing only ~1/K of the data — so
`A_k` is smaller, `A_k⁻¹` larger, the Thompson draw noisier — and the policy degenerates
toward random actions. On a real 4-episode task that showed up as **mog2 DI −0.5 and mog3
−0.9 against Gaussian's 26.2**, which is squarely the random-action regime (§3.7 measures
random at −3.5).

`MixturePosterior` therefore scales the context log-likelihood by `gate_scale`, default
`1/dim` — a per-coordinate geometric mean, so the gate stays soft and comparable to the
reward term whatever the embedding width. Ordinary likelihood tempering, and a **declared
sweep axis** rather than a hidden constant: if you have budget, sweep
`gate_scale ∈ {1/d, 2/d, 4/d}` and report it.

Re-running that same 4-episode Thompson task with the gate tempered:

| posterior | untempered | tempered `1/d` |
|---|---|---|
| `gaussian` | 26.2 | 26.2 |
| `gaussian_full` | 28.0 | 28.0 |
| `mog2` | **−0.5** | **38.2** |
| `mog3` | **−0.9** | **41.7** |

The K=1 rows are identical, as they must be — at K=1 the gate is identically 1, so
`gate_scale` cannot touch them. That is the control: the change moves only the mixtures,
and it moves them from *below random* to *well above the published Gaussian*.

**Read this as a caution and a signal, not a result.** Four episodes, one seed, two
evaluation seeds, two bot volumes, an 18-recording validation pool — it is nowhere near
evidence that mixtures help. It is firm evidence that the untempered version was broken,
and it suggests the hypothesis behind the mixture (an arm's reward is bimodal because the
same threat level pays very differently to a human and to a bot) has something in it.
Whether `mog2`/`mog3` beat `gaussian_full` on a properly trained agent, at 5 seeds and 200
episodes, is exactly what the search is for.

**Operational tell:** if a mixture posterior ever scores near 0 or negative DI, suspect the
gate before suspecting the hypothesis. Check `MixturePosterior.gate` entropy against
`ln K` — saturation means the tempering is wrong for that embedding width.

### 3.5 R1.6 — statistics accumulated in a moving feature space

`A_a = I + Σ z zᵀ` and `b_a = Σ r z` accumulate in the embedding `z`, but `z` keeps
changing while the feature network trains, so the two halves of `A_a` were never measured
in the same coordinates. `train_bandit` stores the raw contexts, then at the end **freezes
the network and recomputes every statistic from the whole buffer in the final z-space**
(`recompute_statistics`). Evaluation already freezes `A`, so statistics and
evaluation-time embedding then agree exactly. **State this handling in the paper** — the
reviewer asks about it directly.

### 3.6 One stated deviation from the published bandit loop, for tractability

The published trainer takes a **separate Adam step for each of the 64 items** in a
balanced batch. Profiled on this machine that is **2.62 ms per item**, i.e. ~184 s for a
single 1100-user episode and roughly 10 h for one 200-episode training — not runnable at
225 trainings.

`update_batch` keeps the per-sample ridge accumulation (rank-1 updates, order matters) but
takes **one** network step on the batch mean. Same objective, better-conditioned gradient,
measured **0.36 ms per sample — 7.3x faster**. A second change, using the last inverted
`theta` for the residual instead of forcing a 64x64 inversion per observation, took the
posterior update from 0.29 ms to 0.036 ms. Neither touches `A`, `b` or the sampled
`theta`, so the K=1 bit-exact equivalence still holds — the smoke test asserts it after
both changes.

**State this deviation in the paper.** `policy.update(state, action, reward)` still exists
and reproduces the published trajectory exactly, item by item, if a reviewer asks.

If the search is still too slow, raise `train_every` (currently 5, a 12.8x replay ratio)
before touching anything else — that is a budget knob, not a design change.

### 3.7 Null baselines for the new bandits, measured

Standing rule 1 says put an untrained baseline through every comparison. Measured on
fold 0's `rl_val` at bot volumes {20, 200}, 3 seeds:

| policy | DI | humans left | bots left (of 200) |
|---|---|---|---|
| Never challenge | **0.0** | 100 | 200 |
| LinUCBX, untrained | **0.0** | 100 | 200 |
| ThompsonX, untrained | −0.2 | 3.7 | 6.0 |
| Random actions | **−3.5** | 2.0 | 6.7 |

Three things this establishes, all worth keeping:

* **An untrained LinUCB is exactly "never challenge."** With `A = I` and `b = 0` the
  estimate is zero and the UCB bonus `√(zᵀz)` is identical for every arm, so `argmax`
  ties to arm 0. DI is exactly 0.0 — a free, exact null for the bandit side, the
  counterpart of the static baselines' exact 0.000 in the cross-fit.
* **An untrained Thompson is a random-action policy**, because `θ̃ ~ N(0, α²I)`. It scores
  −0.2, next to random's −3.5.
* **Indiscriminate challenging is worse than doing nothing** (−3.5 vs 0.0): it leaves 2 of
  100 humans alive. So DI is not degenerate at these volumes — it genuinely penalises
  friction, and any policy scoring well above 0 has learned something real.

These four rows are clean — untrained policies cannot leak. **Do not** read anything from
the trained numbers in that wiring run: it loaded the old `results/final_scorer/`, whose
training pool overlaps fold 0's `rl_val`, so every trained figure there is leaky. It
showed only that the pipeline runs end to end.

### 3.8 What the search trains on, and what to do if the folds disagree

Each fold's `rl_fit` is **32–36 recordings, roughly 9–10 of them human** — close to
round 1's starved 31-recording pool. That is deliberate: `rl_fit` selects the
architecture, and the winner is then retrained on the **full** 50–54-recording policy
block before evaluation. State the sizes in the paper.

If the three folds pick different architectures, that is not a failure. Each fold uses its
own winner — that is proper nested cross-validation — and `policy_arch_choice.json`
records `cross_fold_agreement`. Report the agreement: unanimous across folds is a strong
claim about the architecture; disagreement means the choice does not matter much, which is
also worth saying.

---

## 4. Hard gates — assert these in code, fail the job if they trip

These are **real code**, in `preflight.py`. Do not reimplement them:

```python
import preflight
preflight.check()                                   # every script, first line of work

preflight.check_scorer(scorer, fold, train_refs, val_refs)
preflight.check_batched_scoring(scorer, chunks)
preflight.check_rotation(folds)
preflight.check_determinism()                       # the 144/144 assertion
preflight.check_budget(n_train_seeds=5, n_episodes=200, n_eval_seeds=20)

df = preflight.headline(runs)                       # drops bots == 0
```

| gate | what it caught in round 1 |
|---|---|
| `check_no_round1_scorer_choice` | the whole RL chain ran against the wrong scorer |
| `check_scorer` | config identity, and that the scorer never saw this fold's rl/eval blocks |
| `check_batched_scoring` | the RL loop only uses `score_chunks`; if it disagrees with `score_chunk` every cached score is wrong and nothing downstream reveals it |
| `check_rotation` | disjoint roles, every recording evaluated exactly once |
| `check_determinism` | 144/144 — licenses "continuous deformation of the published environment" rather than "a different environment" |
| `report_hardware` | 18.3 h on CPU on a node called `nvidiaserver`, discovered only afterwards |
| `headline` | the 81 % proxy figure |
| `check_budget` | 1 training seed and 90–150 episodes against floors of 5 and 200 |

`check_scorer` deliberately does **not** assert on the temperature. `T = 1.376` belonged
to the old 73-recording `lstm` pool; weights and temperature are refit per fold.

Also enforced by the driver rather than in Python:

* **≥ 5 training seeds** per learned agent — count the saved checkpoints, don't trust a flag.
* **Selection metrics come from `rl_val` / `scorer ∪ rl`, never from `eval`.**

**Do not run `run_all.py`. Do not run `nb_02`.**

One clarification, because the obvious reading is wrong: **`results/scorer_choice.json` is
not in git.** It is gitignored and was never pushed, so a fresh clone does not have it —
`nb_02` *creates* it, and `nb_03` onwards then reads it. That is exactly how round 1 went
wrong, and it means the protection is not "delete the file" but "never run `nb_02` or
`run_all.py`". `preflight.check()` fails the job if the file ever appears, which is the
signal that something ran `nb_02`.

### 4.1 Porting the existing notebooks (E2–E8) to the rotation

Every one of `nb_03`, `nb_03b`, `nb_04`, `nb_05`, `nb_06`, `nb_07`, `nb_08` currently
opens with `json.load(open(OUT / "scorer_choice.json"))` and `XScorer.from_name(...)`.
Replace that header with the same five lines everywhere:

```python
from expkit.partition import load_rotation
from exp_policy_arch_search import batched_cache, fold_scorer_cached

results = []
for fold in load_rotation(RESULTS / "rotation.json"):
    scorer = fold_scorer_cached(fold)                  # cached by --prepare
    humans, bots = sessions_from_refs(fold.eval)       # or fold.rl, per experiment
    cache = batched_cache(scorer, humans, bots)
    results.append(<the notebook's existing body, unchanged>)
# pool across folds -- every recording is evaluated exactly once
```

**E2 (`nb_03b`) is the one exception.** Its role-swap splits a pool into halves P1/P2 and
trains one agent on each, which is a mechanism *inside* a fold, not a replacement for one.
Run the 10-split cross-fit within each fold's `scorer ∪ rl` block and pool the 30 resulting
gaps. Keep the assertion that the untrained baselines' gap is exactly 0.000 — it is what
proves the design works.

---

## 5. Compute: the bottleneck is parallelism, not the GPU

Round 1 spent 18.3 hours wall-clock, **entirely on CPU**, on a node named `nvidiaserver`.
Python 3.14 has no TensorFlow GPU wheels, which is why TF never found a device.

1. **Get a device, or declare CPU-only.** Pin Python **3.11 or 3.12** and install
   `tensorflow[and-cuda]` plus a CUDA-12 PyTorch build. `preflight.check()` logs
   `nvidia-smi` and `torch.cuda.is_available()` at job start. If Blackwell wheels are
   unavailable, say so in the run notes and proceed on CPU — the science does not depend
   on it.

   **The policies are pinned to CPU on purpose** (`bandits_x.DEFAULT_DEVICE`,
   `train_dqn(device="cpu")`). The networks are tiny and `select_action` is single-sample
   inference once per user per step, where host/device transfer dominates any kernel
   saving — a GPU makes it *slower*, and it removes a class of device-mismatch bugs. The
   scorer's Keras training is the only part that would benefit, and it is the cheap part.
   Override with `device=` if you want to measure it.
2. **Parallelise, which matters far more.** Every workload is small and single-threaded: a
   6,081-parameter LSTM, a ≤256-wide MLP, and a pure-Python simulation loop. A GPU buys
   almost nothing per run. Throughput does. `nb_03b`'s **10 independent splits ran
   serially for 357 minutes**; as an array they finish in ~40.

**Search size**, so the array can be sized: 3 folds × (5 DQN archs + 5 LinUCB archs +
5 Thompson archs) × 5 seeds = **225 policy trainings** at 200 episodes, plus 3 scorer
fits.

**Measured on one CPU core here**, at reduced bot volumes `(0, 20, 100, 200)` — treat as
indicative and re-measure on your hardware before sizing the array:

| | per episode | per 200-episode training |
|---|---|---|
| Thompson | ~7 s | ~25 min |
| LinUCB | ~16 s | ~55 min |
| posterior refit, `gaussian` / `gaussian_full` | — | 4–5 s |
| posterior refit, `mog2` / `mog3` | — | ~30 s |

LinUCB is the slower of the two because its UCB bonus `√(zᵀA⁻¹z)` is evaluated for all 11
arms at every decision, against Thompson's single sampled dot product. The full sampler
reaching 1000 bots will be several times slower again, since episode cost scales with the
user count.

So budget on the order of **150–250 core-hours** for the search. At 25-way concurrency
that is most of a day, not two hours. If that is too much, in order of preference: raise
`train_every` from 5; drop to 3 seeds for the first pass and re-run 5 seeds only on the
finalists; trim `BANDIT_ARCHS` to 3.

**The workflow, in order.** Each step is a separate job; don't collapse them. The scripts
are in `slurm/`, and `slurm/README.md` is the operational copy of this list.

```bash
# 0. validate everything, minutes
python preflight.py
python exp_scorer_phase0.py --smoke
python exp_policy_arch_search.py --smoke

# 1. Phase 0 (no dependencies -- launch first)
python exp_scorer_phase0.py --list
sbatch --array=0-779%25 slurm/phase0.sh
python exp_scorer_phase0.py --audit && python exp_scorer_phase0.py --collect

# 2. fit the three per-fold scorers ONCE
sbatch slurm/prepare_scorers.sh

# 3. the policy + posterior search
python exp_policy_arch_search.py --list --seeds 5   # prints the exact --array line
sbatch --array=0-224%25 slurm/arch_search.sh
python exp_policy_arch_search.py --collect
```

Adjust `--partition`, `--time`, `--mem` and the conda path in `slurm/*.sh` for your
cluster — those are the only lines that should need editing.

Step 2 is not optional. Each task loads the cached scorer from
`results/fold{i}_scorer/`; if tasks refit their own, TF/oneDNN nondeterminism gives
slightly different weights and candidates end up ranked against different yardsticks.

`--task N` runs exactly one `(fold, family, architecture, seed)` unit and writes its own
CSV under `results/arch_search/`, so array tasks never collide. `--collect` is what
applies the selection rule; no individual task does.

---

## 6. Round-2 queue, in priority order

### Priority 0 — Phase 0, config confirmation only (cheap, launch first)

**Run `exp_scorer_phase0.py`.** Do **not** use `nb_10_scorer_search.py` — it is superseded.
It searches at the *session* operating point through `expkit.scorer_search`/`XScorer`, and
the ranking inverts at the chunk operating point the RL loop actually uses; that inversion
is what produced the current design. `probe_decide.py` had the right operating point but
cross-validated over all 156 recordings, so the evaluation block helped choose the config.

```bash
python exp_scorer_phase0.py --validate    # builds all 52 configs, no training
python exp_scorer_phase0.py --list        # 780 tasks at 5 seeds -> the --array line
sbatch --array=0-779%25 slurm/phase0.sh
python exp_scorer_phase0.py --audit       # P0.6, no training, seconds
python exp_scorer_phase0.py --collect
```

It implements `P0.1`–`P0.7` as named grids (`p01`…`p07`, selectable with `--sweeps`), each
varying **one** axis away from the round-1 answer:

| sweep | axis | configs |
|---|---|---|
| `p01` | representation × padding, at ctx 32 | 20 |
| `p02` | capacity, on the two strongest representations (last row = the published 304,049-parameter architecture, as reference) | 12 |
| `p03` | lr × dropout × recurrent dropout | 12 |
| `p04` | class balancing, `window` vs `family` | 2 |
| `p05` | augmentation — none / rigid / per-move ±1,±2,±3 / gaussian σ=1,2 | 7 |
| `p07` | context length {16, 24, 32, 50, 64} | 5 |

58 grid entries, **52 distinct** — every sweep varies one axis away from the same
baseline, so the baseline appears six times. `task_list` deduplicates by config identity;
without that the baseline would collect 6x the rows of any competitor, shrinking its
standard error by ~√6, narrowing the 1-SE band around it and pushing competitors out of
the ECE tiebreak. "Confirmation" would then be partly an artefact of the bookkeeping.

**Two properties it enforces that the old scripts did not:**

* **Per fold, on `scorer ∪ rl` only** — the evaluation block never participates in
  choosing a scorer config (asserted). If all three folds agree, the design is clean and
  the paper can say so; `phase0_choice.json` reports `unanimous` either way.
* **Per-family recall at 0.5 with window counts, never per-family AUC.** AUC over a
  5-window class is meaningless — it is what made NaiveBot look like a 0.975 failure when
  its recall was 16/16.

The selection rule is fixed in the module docstring: chunk AUC (SE over seeds) → within
1 SE, lowest ECE after temperature scaling → lowest `|corr(score, padding_fraction)|` →
fewest parameters.

Expect confirmation, not change. Round 1's independent probes already put
`kinematic`+`mask`+ctx 32 on top at chunk AUC 0.970 / ECE 0.014, and a smoke run of this
script on fold 0 reselected it (`window` 0.954 vs `family` 0.949).

**Phase 0 confirms; it does not silently re-pin the design.** Downstream code reads
`preflight.FINAL_SCORER_CFG`, which is hardcoded, and `phase0_choice.json` is a *report*.
If Phase 0 picks something else — plausible, with 52 configs and an ECE tiebreak —
**stop and tell the user.** Changing the scorer invalidates every Phase 1 and Phase 2
result computed before it, so that is their call, not an automatic re-run. Note also that
`p02` includes the published 304,049-parameter architecture as a reference row: if it wins,
that is a genuinely interesting negative result about the whole "right-sized scorer"
section, not a config update.

Validate the grid first — it takes seconds and catches a typo before 780 array tasks fail
hours later:

```bash
python exp_scorer_phase0.py --validate     # builds all 52 distinct configs, no training
```

Only `p04` has ever actually trained. `repeat_row`, `zero_mask`, `xy_dt`, the rigid and
gaussian augmentations, `recurrent_dropout` and the (200,100)+dense(128,64) architecture
have not, though all 58 grid entries are confirmed to build and produce well-shaped
windows.

### Priority 1 — the policy and posterior search, then the headline table

`exp_policy_arch_search.py`, all three folds. This **subsumes E9**: the bandits are
retrained on the policy block by construction, so there is no separate "retrain the
bandits" step. Keep the published bandit classes only for the reproduction table.

Then **E1**, the headline table. This is the paper's central table and the step most
likely to be improvised — which is how round 1 went wrong — so the spec is exact. Write
`exp_headline_table.py` to do precisely this:

1. **Per fold**, read `policy_arch_choice.json` and take the winner for **each family
   separately** — `dqn`, `linucb`, `thompson`. The Thompson winner includes its
   **posterior** (`gaussian` / `gaussian_full` / `mog2` / `mog3`), not just its
   architecture.
2. **The ablation** (`DQN without H-Score`) uses the **DQN winner's architecture** with
   `use_score=False`. It is an ablation of the winner, not of the published net.
3. **Retrain every learned policy on the full `fold.rl`** — not `rl_fit`, which was only
   for selection — at **5 seeds × 200+ episodes**, all with the same `cfg`, the same
   `bot_choices`, and the same budget. An unequal budget is what produced §2's inversion.
4. **Evaluate on `fold.eval`**, all six bot volumes, **50 evaluation seeds**. The statics
   need no training and go through the identical loop.
5. **Pool the three folds** — every recording is then evaluated exactly once across all
   156 — and bootstrap **at the recording level**, 1000+ resamples. Seed-level intervals
   are tight no matter how small the pool is and are not the uncertainty a reader cares
   about.
6. **Exclude the zero-bot cell** from every headline mean (`preflight.headline`). Print
   the all-six average alongside, explicitly labelled, for continuity with the published
   Table 4 and nothing else.

Produce it **twice**: **headline** (equal footing, selected architectures, retrained
bandits) and **reproduction** (published configuration and published checkpoints, to show
Table 4 is recoverable). The two differ in every way listed in §2 — say so in the caption.

This is what decides §2.

### Priority 2 — E7, the deployment-reward question (the reviewer point that matters most)

Round 1 re-ran the two sweeps that had already failed and attempted none of the six
solves. Status unchanged: the proxy agent keeps all 100 humans at zero friction and leaves
~219 bots alive against the oracle's ~16.

**A correction to the original handoff.** Potential-based reward shaping preserves the
optimal policy *by theorem* (Ng, Harada & Russell 1999). The break-even analysis says that
at penalty −150 the proxy MDP's optimum **is** "never challenge" — challenging costs 26
reward per step immediately and with certainty, while the penalty is delayed, discounted
by γ = 0.95, fires with probability 0.3 and spreads over 3 steps, so the penalty must
exceed ≈ 260 before blocking pays. PBRS therefore **cannot** fix the default setting; it
can only help where the optimum is already right and credit assignment is the obstacle.
Run each solve where it can actually work:

| # | solve | run at | what it tests |
|---|---|---|---|
| 1 | PBRS, `F(s,s') = γΦ(s') − Φ(s)`, Φ a function of the behavioural score | penalty **−600 and −2400**, above break-even | credit assignment, optimum already correct |
| 2 | learned reward model — classifier predicting "will be confirmed abusive" from observable features, its output as a dense per-step reward | default −150 | changes the optimum; strongest candidate at the default |
| 3 | immediate-gap sweep — `zero_friction_reward ∈ {50, 35, 25}` against `engagement_reward` 25 | default −150 | changes the optimum; cheapest lever the arithmetic identifies |
| 4 | reward normalisation + gradient clipping to ±10 | −2400, −9600 | whether the −9600 collapse is optimisation, not reward design |
| 5 | 1000-episode runs | all of the above | whether 90 episodes was simply below threshold |
| 6 | off-policy evaluation (importance sampling / doubly-robust) | logged data | estimating a policy's value without deploying it |

Non-negotiable constraints:

* The immediate reward stays a **pure function of the observable tuple** — a blocked bot
  and a false-positived human get identical immediate reward. Measured ambiguity from
  round 1: `failed_gone` = 97.8 % blocked bot / 2.2 % false-positived human;
  `passed_continued` = 24.6 % leaked bot / 75.4 % satisfied human.
* Balance the replay buffer on the **observable outcome**, never the true class.
* **Matched-budget oracle denominator.** Round 1's oracle scored 57.0 with FP 0.114 after
  90 episodes — *below* the published DQN's 65.6 in the same environment. It is
  undertrained, which flatters every ratio computed against it. Train the oracle at ≥ 200
  episodes and 5 seeds.
* Report the ratio **excluding the zero-bot cell**, and additionally report
  `(proxy − never_challenge) / (oracle − never_challenge)` — how much of the *achievable*
  gap the proxy closes over doing nothing.

If solve 2 or 3 works, the message changes from "deployable rewards have a structural
limitation" to "here is how to build one". If none works, the negative result is
publishable **with the break-even arithmetic attached**.

### Priority 3 — E4, the DoS confound, done properly

The single-constant ablation does not support its verdict (§1.2). Redo:

* **Retrain on the varied grid**, humans `{25, 50, 100, 200, 400}` × bots
  `{0, 20, 100, 200, 500, 1000}` — round 1 only *evaluated* on it.
* Freeze `n_active` at **several constants spanning the training range**
  (`{25, 100, 200, 400, 800, 1400}`), not one.
* Report **human survival, bot survival and friction seconds separately**, not DI alone.
  DI conflates "ignores the feature" with "became more permissive".
* Keep the randomised variant, labelled out-of-distribution and **not** a test.
* Delete the hardcoded `VERDICT` prose; derive the conclusion from the numbers.

Honest framing either way: the reviewer's design criticism is correct — the published
design had `corr(total population, bots) = 1.0000`. The grid fixes the design; the ablation
establishes whether the *result* depended on it.

**Note the scope limit.** This addresses traffic *composition*. It does not address server
resources: there is no arrival-rate model, no finite-capacity server, no queue, no latency
and no drop rate anywhere in the simulator. Either build E11 (an M/M/c layer) or narrow
the paper's claim from "DoS mitigation" to "robustness to traffic composition". The second
is well evidenced; the first currently is not.

### Priority 4 — E3, reward sensitivity at 5 seeds

Round 1 still had **one training seed per setting** (six trainings, 2.5 min each). The
between-setting spread (44.3 to 64.7 DI) is of the same order as the seed noise, so
"6/6 beat every static baseline" rests on six single draws.

Full grid: β `{1.5, 2.0, 2.5}` × R_leak `{−50, −150, −300}` × overkill `{2.0, 2.5}` ×
underestimation `{2.5, 5.0, 10}`, plus the linear seconds-priced friction variant.
**5 training seeds each**, retrained (re-scoring a fixed policy is provably a no-op).

Add a **Pareto frontier** (bots blocked vs human friction seconds) — DI hides which
settings buy security with usability.

Report, rather than bury, the fixed-policy landscape result: Static Single-Threshold wins
**4 of the 18 cells** (all at `friction_base = 1.5`). "The reward prefers the same policy
in 78 % of settings" is the honest sentence, and a fair answer to R1.3.

Abandonment sweep `paper`, `empirical τ ∈ {22, 45, 80}`, `none` — already clean, just
re-run under the per-fold scorer.

### Priority 5 — E8, stochastic retraining at matched budget

Round 1's stochastic-trained agent got **1.8 minutes and 90 episodes** and produced a
bimodal policy (79.5 % level 0, 15.0 % level 10) scoring 58.9 against the published
checkpoint's 65.6. Uninterpretable. Redo at matched budget with cross-fitting, 5 seeds,
200+ episodes.

The α sweep itself is sound and reproduced (144/144 exact at the deterministic limit);
keep it. The false-positive-rate and friction-seconds columns are the ones that answer
"usability is modeled, not measured" — 2.8 % FP and 12.9 s friction for the DQN versus
17.8 % and 33.0 s for Static Single — and the deterministic simulator cannot express them.

### Priority 6 — E2 and E6 under the per-fold scorer

The **mechanism** is settled and needs no re-deriving: the leak flows only through the
scorer's outputs, and the no-H-Score ablation leaks +0.1 DI [−0.1, +0.3] against the full
agent's +8.7 [+4.1, +13.2] over 10 splits, with the untrained baselines asserting exactly
0.000. The **magnitude** flows through the scorer, so re-measure it. Keep 10 splits;
per-split gaps ranged −3.1 to +19.0, so fewer would not be safe.

Same for E6. The DQN's variance share moved from 89 % (local, `kinematic` scorer) to 65 %
(round 1, `xy_dt` scorer) while the statics barely moved, so **do not quote a single
"78 %" figure** until it is measured once under the final scorer. The qualitative claim —
most variance is between recordings, not between simulation seeds — held in both runs and
is safe.

---

## 7. Do not re-run

* `nb_02` / `scorer_variants` — superseded, and it is what selected the wrong scorer. Its
  one durable finding stands and belongs in the paper: across six representations every
  paired-bootstrap interval against `xy` straddles zero, so **no representation is
  distinguishable on this data**. Round 1 is now independent evidence — it ranked
  `kinematic` last (0.863) where the local run ranked it first (0.941), same data,
  different seeds. Report it as a limitation, not a result.
* The data audit (`nb_01`) and participant structure (`nb_09`) — stable and correct: 156
  recordings, 44 human from **3 participants**, zero filename and zero content-hash overlap
  between campaigns. But fix the §9 wording per §3.1: leave-one-campaign-out is a temporal
  holdout with the same participants, not a proxy for new users.

---

## 8. Deliverables for round 2

1. `results/phase0/` (per-task CSVs), `phase0_runs.csv`, `phase0_table_fold*.csv`,
   `phase0_choice.json`, `phase0/p06_activity_audit.csv`, and `results/fold*_scorer/`.
   State whether the config choice was **unanimous across the three folds**.
2. `results/rotation.json`, `results/arch_search/` (per-task CSVs), `policy_arch_runs.csv`,
   `policy_arch_{dqn,bandit}_fold*.csv`, `policy_arch_choice.json` — including its
   `cross_fold_agreement` field.
3. One CSV per experiment; a regenerated `FINDINGS.md` from a **fixed**
   `summarise_results.py` — keep the property that nothing is hand-entered.
4. **Two** Table 4s — headline (equal footing) and reproduction — XLSX + PDF, each
   reported with and without the zero-bot cell, clearly labelled.
5. Figures at 300 dpi.
6. A delta note: which round-1 conclusions held, moved, flipped — with the clean-partition
   ordering (§2) called out explicitly whichever way it lands, and a statement of whether
   the MoG posterior beat the published Gaussian and by how much.
7. Run notes: Python version, CUDA availability, array width.
8. **Do not ship `run_inference.py` as is.** Its agents were trained on states built from
   the `xy_dt` scorer but it loads the kinematic `HumanityScorer` — the inference state
   distribution will not match training. Rebuild from the round-2 agents, and check the
   movement-dict key (`time` vs `timestamp`) against `expkit.features`.

### A `.gitignore` trap, if you push results back

`.gitignore` line 216 is a blanket `*.json`, so **every result JSON is silently ignored** —
`phase0_choice.json`, `policy_arch_choice.json`, and anything else. This has already cost
this project once: the training data appeared to be committed and was not. Use

```bash
git add -f training/experiments/results/*.json
git status --short          # confirm they are actually staged
```

and check `git ls-files` afterwards rather than trusting a clean `git status`. Git applies
the **last** matching rule, so adding a `!` exception above a later blanket rule does
nothing.

`results/rotation.json` is already committed (force-added), so the three folds are fixed
and identical on every machine — do not regenerate it. The scripts load it if present and
only fall back to `make_rotation` when it is missing.
