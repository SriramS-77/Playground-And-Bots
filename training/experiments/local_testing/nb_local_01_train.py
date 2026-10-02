# %% [markdown]
# # Local 01 — train every policy fresh, then evaluate on realistic traffic
#
# A single-seed local preview of what the GPU round will produce. Nothing here is
# evidence — one training seed is exactly what round 1 got wrong — but it tells us
# whether the conclusions are likely to move before a day of cluster time is spent.
#
# **Everything is trained fresh.** No published checkpoints, and not
# `results/final_scorer/` either, whose training pool overlaps this fold's evaluation
# block. The scorer is refit on fold 0's `scorer + rl` blocks; every learner is trained on
# fold 0's policy block; everything is evaluated on fold 0's evaluation block, which
# **neither has seen**. That last property is the whole point — it is the invariant E2
# showed the published work violated.
#
# The scorer deliberately uses `scorer + rl`, not `scorer` alone. On `scorer` alone the
# fit is unstable: early stopping runs on a ~13-recording validation split, and on this
# fold at seed 0 it restored the epoch-1 weights — eval AUC ~0.65–0.70, human and bot
# means 0.506 vs 0.525, useless to any fixed threshold — while seed 1 reached 0.93. With
# `scorer + rl` it reached **AUC 0.95–0.98** on every fold and seed tried. This is not the
# leak E2 measured; that was the agent training on the recordings it was *evaluated* on.
#
# Each policy is trained and evaluated in turn and its rows appended to disk, so the run
# is restartable and partial results are usable.
#
# ## How the population is set, and why
#
# The published design fixes humans at 100 and sweeps bots over `(0, 20, 100, 200, 500,
# 1000)` — bot fractions of 0 %, 17 %, 50 %, 67 %, 83 %, 91 %. Four of those six cells sit
# at or above 50 % bots and two above 80 %: a sustained-attack regime, not ordinary
# traffic. It also makes total population a perfect proxy for the attack level
# (`corr = 1.0`), which is Reviewer 1's confound.
#
# Published measurements of real traffic composition:
#
# | source | figure |
# |---|---|
# | Imperva Bad Bot Report 2025 | bad bots **37 %** of all web traffic; automated 51 % vs human 49 % |
# | Imperva 2025, travel sector | **48 %** bad bots |
# | Akamai | **43 %** of login requests are credential abuse |
#
# For a CAPTCHA-protected endpoint the realistic band is therefore roughly **30–45 %
# bots**, with excursions upward under attack. So this notebook draws the human population
# **from a range** (50–250) instead of fixing it, and sets bot volume by **fraction**, from
# 5 % to 75 %.
#
# Three things follow.
#
# 1. **The DoS confound is broken by construction, not by ablation.** Humans are drawn
#    independently of the bot fraction, so `corr(total population, bot fraction)` falls
#    from the published design's **+1.000** to **+0.76**, and — the property that actually
#    matters — *all six* bot fractions occur at the same middle band of total population.
#    Population is therefore no longer a usable proxy for the attack level. The residual
#    correlation is physically real: a genuine surge does raise total traffic.
# 2. **There is no zero-bot cell**, so the `DI = 100 for any permissive policy` pathology
#    that inflated round 1's proxy-reward headline cannot arise. No exclusion rule needed.
# 3. **The realistic operating point is now represented.** Four of the published design's
#    six cells were at or above 50 % bots; here four of six are at or below 43 %.

# %%
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd

import expkit  # noqa: F401
import preflight
from expkit.bandits_x import POSTERIORS, BanditArch, fit_posteriors_from, train_bandit
from expkit.humanity_scorer import HumanityScorer
from expkit.partition import load_rotation, split_refs
from expkit.paths import RESULTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.trainer import train_dqn
from exp_policy_arch_search import batched_cache, fold_scorer

from rlcaptcha.policies.static import MultiThresholdPolicy, SingleThresholdPolicy

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
CKPT = HERE / "agents"; CKPT.mkdir(exist_ok=True)
RUNS = OUT / "local_runs.csv"

SEED = 0
EPISODES = 200                       # the handoff's floor for "stable training"
HUMAN_RANGE = (50, 250)              # drawn per episode / per eval seed, not fixed
# Absolute counts chosen so that at the midpoint of HUMAN_RANGE (150 humans) the bot
# FRACTIONS are ~5, 15, 30, 43, 60, 75 %. Because the human count varies independently,
# the realised fraction varies around each of these.
BOT_CHOICES = (8, 26, 64, 113, 225, 450)

# Evaluation grid, by FRACTION. Labels are the justification, carried into the figures.
BOT_FRACTIONS = {0.05: "quiet", 0.15: "light", 0.30: "Imperva baseline (37% band)",
                 0.43: "Akamai login abuse", 0.60: "elevated", 0.75: "surge"}
EVAL_SEEDS = 12

preflight.check(hardware=True)

# %% [markdown]
# ## 1.1 The fold, and a scorer trained only on its scorer block

# %%
fold = load_rotation(RESULTS / "rotation.json")[0]
print(f"fold 0: scorer {len(fold.scorer)} | policy {len(fold.rl)} | eval {len(fold.eval)}")

scorer_dir = CKPT / "scorer"
if (scorer_dir / "model.keras").exists():
    scorer = HumanityScorer.load(scorer_dir); print("loaded cached fold-0 scorer")
else:
    t = time.time(); scorer = fold_scorer(fold, seed=SEED); scorer.save(scorer_dir)
    print(f"refit fold-0 scorer in {time.time() - t:.0f}s")

tr_refs, va_refs = split_refs(list(fold.scorer) + list(fold.rl), 0.25, seed=SEED)
preflight.check_scorer(scorer, fold, tr_refs, va_refs)
print(f"scorer: {scorer.cfg.representation}/{scorer.cfg.padding}/ctx{scorer.cfg.context}"
      f"  T={scorer.temperature:.3f}  params={scorer.model.count_params():,}")

rl_h, rl_b = sessions_from_refs(fold.rl)
ev_h, ev_b = sessions_from_refs(fold.eval)
cache = batched_cache(scorer, rl_h, rl_b, ev_h, ev_b)
preflight.check_batched_scoring(scorer, [c for s in (rl_h + rl_b) for c in s.chunks][:64])
print(f"policy pool {len(rl_h)}h/{len(rl_b)}b   eval pool {len(ev_h)}h/{len(ev_b)}b")

# %% [markdown]
# ## 1.2 The evaluation harness
#
# One human count per (fraction, seed), drawn from the range with a fixed RNG so every
# policy sees an identical population sequence. Bots follow from the fraction.

# %%
def population(fraction: float, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(10_000 + seed)
    n_h = int(rng.integers(HUMAN_RANGE[0], HUMAN_RANGE[1] + 1))
    return n_h, int(round(n_h * fraction / (1 - fraction)))


def evaluate(policy, name: str) -> pd.DataFrame:
    rows = []
    for frac, label in BOT_FRACTIONS.items():
        for s in range(EVAL_SEEDS):
            n_h, n_b = population(frac, s)
            res, _ = run_x(policy, ev_h, ev_b, n_b, cache=cache, n_humans=n_h, seed=s)
            m = evaluate_x(res)
            rows.append({
                "policy": name, "bot_fraction": frac, "regime": label, "seed": s,
                "n_humans": n_h, "n_bots": n_b, "total_population": n_h + n_b,
                "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                "human_survival": res.surviving_humans / n_h,
                "bot_survival": res.surviving_bots / n_b,
                "humans_left": res.surviving_humans, "bots_left": res.surviving_bots,
                "friction_s_per_human": res.human_seconds / max(n_h, 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
    g = df.groupby("bot_fraction")[["DI", "human_survival", "bot_survival"]].mean()
    print(f"  {name:32s} meanDI={df.DI.mean():6.2f}  "
          f"humans={df.human_survival.mean():.2f}  bots={df.bot_survival.mean():.3f}")
    return df


def done(name: str) -> bool:
    if not RUNS.exists():
        return False
    return name in set(pd.read_csv(RUNS).policy.unique())

# %% [markdown]
# ## 1.3 Static baselines — no training, so they go first

# %%
for pol in (SingleThresholdPolicy(), MultiThresholdPolicy()):
    if not done(pol.name):
        evaluate(pol, pol.name)

# %% [markdown]
# ## 1.4 Train and evaluate each learner
#
# Every learner gets the **same** episode count, population sampler, seed and `cfg`. An
# unequal budget is what produced round 1's inversion, where the DQN appeared to lose to
# bandits that were handed four environment advantages and a leaky checkpoint.

# %%
COMMON = dict(episodes=EPISODES, seed=SEED, human_counts=HUMAN_RANGE,
              bot_choices=BOT_CHOICES, verbose=50)
timings, curves = {}, {}

for name, use_score, tag in (("DQN", True, "dqn"),
                             ("DQN without H-Score", False, "dqn_ablation")):
    if done(name):
        print(f"  {name}: already done"); continue
    t = time.time()
    agent, log = train_dqn(rl_h, rl_b, cache, use_score=use_score, name=name, **COMMON)
    agent.save(CKPT / f"{tag}.pt")
    timings[name] = time.time() - t; curves[name] = log.frame()
    print(f"  trained {name} in {timings[name] / 60:.1f} min")
    evaluate(agent.eval_mode(), name)

# %%
ARCH = BanditArch()          # published feature net: 5 -> 128 -> ReLU -> 64, alpha 1.0
for kind in ("linucb", "thompson"):
    variants = [f"{kind}:gaussian"] if kind == "linucb" else \
               [f"{kind}:{p}" for p in POSTERIORS]
    if all(done(v) for v in variants):
        print(f"  {kind}: already done"); continue
    t = time.time()
    policy, log = train_bandit(kind, rl_h, rl_b, cache, arch=ARCH, **COMMON)
    policy.save(CKPT / f"{kind}.pt")
    timings[kind] = time.time() - t; curves[kind] = log.frame()
    print(f"  trained {kind} in {timings[kind] / 60:.1f} min, buffer {len(policy._buffer):,}")
    evaluate(policy, f"{kind}:gaussian")
    if kind == "thompson":
        # Same weights, same buffer -- the posterior is the only thing that varies.
        for post in POSTERIORS:
            if post == "gaussian":
                continue
            v = fit_posteriors_from(policy, policy._buffer, post)
            print(f"    posterior {post}: K={v.posteriors[0].K} "
                  f"cov={v.posteriors[0].covariance} gate_scale={v.posteriors[0].gate_scale:.4f}")
            evaluate(v, f"{kind}:{post}")

# %%
if curves:
    pd.concat([c.assign(policy=k) for k, c in curves.items()], ignore_index=True) \
      .to_csv(OUT / "training_curves.csv", index=False)
json.dump({"seed": SEED, "episodes": EPISODES, "human_range": HUMAN_RANGE,
           "bot_choices": BOT_CHOICES, "eval_seeds": EVAL_SEEDS,
           "bot_fractions": {str(k): v for k, v in BOT_FRACTIONS.items()},
           "fold": 0, "scorer_temperature": float(scorer.temperature),
           "minutes": {k: round(v / 60, 2) for k, v in timings.items()}},
          open(OUT / "train_config.json", "w"), indent=2)
print("\nminutes:", {k: round(v / 60, 1) for k, v in timings.items()})
print("Notebook local-01 complete.")
