# %% [markdown]
# # 8 — Is the dataset big enough to support the claims?
#
# No reviewer asks this directly, but it sits underneath several of their comments, and
# it is the question most likely to decide a resubmission. The honest answer has to come
# before the paper is rewritten, not after.
#
# ## The distinction that matters
#
# `results_seeded/` reports ± over 50 simulation seeds. That measures how much the
# *simulation* wobbles when the recording pool is held fixed. It is not the uncertainty
# a reader cares about, which is uncertainty over the **recordings** — 44 human and 112
# bot sessions, four scripted bot generators, and a participant count the manuscript
# never states.
#
# Resampling seeds with a fixed pool will always produce tight intervals, however small
# the pool is. Resampling the pool is what tells you whether the result generalises.
#
# This notebook: bootstrap over sessions, a variance decomposition separating the two
# sources, a learning curve over pool size, and the effect of the perturbation model on
# effective diversity. It ends with limitation text that can go into the paper as-is.

# %%
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import expkit
from expkit.bootstrap import (bootstrap_pool, ci, learning_curve, required_sessions,
                              variance_decomposition)
from expkit.partition import index_sessions, load_partition
from expkit.paths import RESULTS
from expkit.simx import sessions_from_refs
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from expkit.xscoring import XScorer
from rlcaptcha.scoring import ScoreCache
from rlcaptcha.policies import DQNPolicy, MultiThresholdPolicy, SingleThresholdPolicy

OUT = RESULTS
partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
scorer = XScorer.from_name(chosen)

all_refs = index_sessions()
ev_refs = partition["eval"]
ev_h, ev_b = sessions_from_refs(ev_refs)

# One cache over every session; scores are keyed by session name, so it stays valid for
# any resample of the pool.
all_h, all_b = sessions_from_refs(all_refs)
cache_all = ScoreCache(scorer).precompute(all_h, all_b, verbose=False)
CACHE = lambda h, b: cache_all
RUN_KW = dict(solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT)

# %% [markdown]
# ## 8.1 What we actually have

# %%
inv = pd.read_csv(OUT / "session_inventory.csv")
print(f"sessions: {len(inv)}   humans: {(~inv.is_bot).sum()}   bots: {inv.is_bot.sum()}")
print(f"mouse movements: {inv.movements.sum():,}")
print(f"\nwindows of 100 movements available in total: ~{int(inv.movements.sum() / 100):,}")
print(inv.groupby("family").agg(sessions=("name", "size"),
                                movements=("movements", "sum"),
                                median_movements=("movements", "median")).to_string())
print("""
Three structural limits, none of which more simulation seeds can fix:

1. Only 44 human recordings exist, from two sittings. If they came from a handful of
   participants -- which the timestamps suggest -- the effective number of independent
   human samples is nearer the participant count than 44.
2. The bot side is four scripted generators. Bot diversity is bounded by four programs,
   not by 112 recordings, so 'generalises to bots' means 'generalises to these four'.
3. NaiveBot recordings carry a median of ~4-10 movements: a single padded window. That
   family is separated by absence of movement, not by movement dynamics.""")

# %% [markdown]
# ## 8.2 Session-level bootstrap
#
# Resample the recording pool with replacement and re-run. The spread is the confidence
# interval that belongs in the paper.

# %%
N_BOOT = 150
policies = [DQNPolicy(epsilon=0.0), MultiThresholdPolicy(), SingleThresholdPolicy()]
rows = []
t0 = time.time()
for pol in policies:
    for nb in (100, 500):
        vals = bootstrap_pool(pol, ev_refs, nb, CACHE, key="DI", n_boot=N_BOOT,
                              seed=0, **RUN_KW)
        lo, hi = ci(vals)
        rows.append({"policy": pol.name, "bots": nb, "DI_mean": vals.mean(),
                     "DI_sd": vals.std(ddof=1), "ci_lo": lo, "ci_hi": hi,
                     "ci_width": hi - lo})
        print(f"  {pol.name:24s} {nb:4d} bots: DI {vals.mean():5.1f} "
              f"[{lo:5.1f}, {hi:5.1f}]  ({(time.time()-t0)/60:.1f} min)")
boot = pd.DataFrame(rows)
boot.to_csv(OUT / "session_bootstrap.csv", index=False)

# %% [markdown]
# ### Compare with the seed-level spread already reported

# %%
seeded_path = Path("../results_seeded/raw_runs.csv")
if seeded_path.exists():
    seeded = pd.read_csv(seeded_path)          # columns: Policy, Bots, seed, DI, ...
    s = (seeded[seeded.Policy.isin(["DQN", "Static Multi-Threshold",
                                    "Static Single-Threshold"])]
         .groupby(["Policy", "Bots"]).DI.std().reset_index(name="DI_sd_seeds"))
    merged = boot.merge(s, left_on=["policy", "bots"],
                        right_on=["Policy", "Bots"], how="left")
    merged["inflation"] = (merged.DI_sd / merged.DI_sd_seeds).round(2)
    print(merged[["policy", "bots", "DI_sd_seeds", "DI_sd", "inflation"]]
          .to_string(index=False))
    merged.to_csv(OUT / "sd_seed_vs_session.csv", index=False)
    print("""
'inflation' is how many times wider the session-level sd is than the seed-level sd the
paper currently reports. Note the two are measured on different pools -- the seeded run
used all 73 campaign-B recordings, this bootstrap uses the 52-session held-out pool -- so
read the ratio as an order of magnitude, not a precise factor.""")
else:
    print("results_seeded/raw_runs.csv not found; skipping the comparison")

# %% [markdown]
# ## 8.3 Variance decomposition
#
# A nested design — bootstrap pools crossed with simulation seeds — splits total variance
# into a between-pool (session) component and a within-pool (seed) component.

# %%
rows = []
for pol in policies:
    df_v, comp = variance_decomposition(pol, ev_refs, 200, cache_all, key="DI",
                                        n_pools=12, n_seeds=12, seed=0, **RUN_KW)
    rows.append({"policy": pol.name, **comp})
    print(f"  {pol.name:24s} session sd={comp['sd_between_pools']:5.1f}  "
          f"seed sd={comp['sd_within_pool']:5.1f}  "
          f"session share={comp['share_session_level']:.0%}")
var = pd.DataFrame(rows)
var.to_csv(OUT / "variance_decomposition.csv", index=False)

# %% [markdown]
# ## 8.4 Learning curve over pool size
#
# If the curve has flattened by the full pool, more recordings of the same kind would not
# change the answer. If it has not, the dataset is the binding constraint.

# %%
SIZES = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
curve = learning_curve(DQNPolicy(epsilon=0.0), ev_refs, 200, cache_all, SIZES,
                       key="DI", n_repeats=10, seed=0, **RUN_KW)
curve.to_csv(OUT / "learning_curve.csv", index=False)
suff = required_sessions(curve, key="DI")
print(suff["table"].round(2).to_string())
print(f"\nfull-pool mean DI: {suff['full_pool_mean']:.1f}")
print(f"sd at full pool  : {suff['sd_at_full_pool']:.1f}")
print(f"sd at half pool  : {suff['sd_at_half_pool']:.1f}")

# %% [markdown]
# ## 8.5 What the perturbation model does to effective diversity
#
# With the published rigid perturbation, two users drawn from one recording differ by a
# one-pixel translation and receive essentially the same behavioural score — which is why
# the cache works at all. So a 1,100-user episode contains far fewer than 1,100
# independent behavioural samples. Per-movement jitter changes that.

# %%
import random as _r
from expkit.features import PERTURBATIONS

sample = [s for s in ev_h[:6]] + [s for s in ev_b[:6]]
rows = []
for mode, kw in (("rigid", {"magnitude": 1}), ("per_move", {"magnitude": 1}),
                 ("per_move (mag 3)", {"magnitude": 3}), ("gaussian", {"sigma": 2.0})):
    fn = PERTURBATIONS["per_move" if mode.startswith("per_move") else mode]
    spreads = []
    for sess in sample:
        for chunk in sess.chunks:
            if len(chunk) < 20:
                continue
            rng = _r.Random(0)
            scores = [scorer.score_chunk(fn(chunk, rng, **kw)) for _ in range(8)]
            scores = [s for s in scores if s is not None]
            if len(scores) > 1:
                spreads.append(np.std(scores))
    rows.append({"perturbation": mode, "mean_score_sd_across_copies": np.mean(spreads),
                 "max": np.max(spreads), "n_chunks": len(spreads)})
div = pd.DataFrame(rows)
div.to_csv(OUT / "perturbation_diversity.csv", index=False)
print(div.round(5).to_string(index=False))
print("""
The rigid perturbation does not move the score at all (sd < 1e-5 across eight copies):
two 'different' users drawn from one recording are literally the same sample, which is
why the score cache is exact under it. Per-movement jitter produces a genuinely
different score for each copy. The
consequence for the paper: under the published augmentation the effective sample size of
an episode is the number of RECORDINGS drawn, not the number of simulated users, and
that is what the confidence intervals must reflect.""")

# %% [markdown]
# ## 8.6 Figures

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))

ax = axes[0]
g = curve.groupby("fraction").DI.agg(["mean", "std"])
n_sess = curve.groupby("fraction")[["n_human_sessions", "n_bot_sessions"]].first().sum(axis=1)
ax.errorbar(n_sess.values, g["mean"], yerr=g["std"], marker="o", capsize=3)
ax.set_xlabel("distinct recordings available"); ax.set_ylabel("DI at 200 bots")
ax.set_title("Learning curve over pool size"); ax.grid(alpha=.3)

ax = axes[1]
x = np.arange(len(var))
ax.bar(x - .2, var.sd_between_pools, .4, label="session-level sd")
ax.bar(x + .2, var.sd_within_pool, .4, label="seed-level sd (what the paper reports)")
ax.set_xticks(x); ax.set_xticklabels([p.replace(" ", "\n") for p in var.policy], fontsize=7)
ax.set_ylabel("sd of DI"); ax.set_title("Where the uncertainty lives")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")

ax = axes[2]
ax.bar(range(len(div)), div.mean_score_sd_across_copies)
ax.set_yscale("log")
ax.set_xticks(range(len(div))); ax.set_xticklabels(div.perturbation, rotation=20,
                                                   ha="right", fontsize=7)
ax.set_ylabel("sd of score across 8 copies of one chunk")
ax.set_title("How different are two 'users' from one recording?"); ax.grid(alpha=.3, axis="y")

plt.tight_layout(); plt.savefig(OUT / "data_sufficiency.pdf", dpi=300)
plt.savefig(OUT / "data_sufficiency.png", dpi=110); plt.close()

# %% [markdown]
# ## 8.7 Limitation text for the paper
#
# Drafted so it can be adapted directly. Fill in the participant count.

# %%
print(f"""
LIMITATIONS (draft)

Our evaluation draws on {len(inv)} recorded sessions ({(~inv.is_bot).sum()} human,
{inv.is_bot.sum()} automated) collected in two sittings, with automated traffic produced
by four scripted generators. Simulated populations are constructed by resampling these
recordings, so although an episode contains up to 1,100 users, the number of independent
behavioural samples is bounded by the number of distinct recordings drawn. We therefore
report confidence intervals obtained by bootstrapping over recordings rather than over
simulation seeds; the latter understates uncertainty by roughly the factor reported in
Section 8.2 above.

Three consequences follow. First, our results characterise these four bot generators and
should not be read as a claim about automated traffic in general, and in particular not
about adaptive or VLM-driven adversaries, which we do not evaluate. Second, with N human
participants the human side of the evaluation has limited statistical power, and
per-scheme or per-demographic claims are not supportable at this sample size. Third, the
behavioural scorer is trained and evaluated on data from the same two collection
sittings; robustness to a change of site layout, input device or population remains
untested.

MITIGATIONS, in decreasing order of value

1. Recruit more participants. A participant-level split, not a session-level one, is what
   the external-validity claim requires. This is the binding constraint.
2. Add bot diversity: record adversaries that were not written by the authors -- public
   automation frameworks, a commercial solving service, a VLM-driven agent.
3. Report recording-level intervals throughout, and drop point comparisons that those
   intervals do not support.
4. Keep the per-movement perturbation. Rigid translation does not create new samples and
   makes the population look more diverse than it is.
5. If recruitment is impossible before resubmission, narrow the claim explicitly to
   'under a simulator parameterised by these recordings' and move the generalisation
   argument to future work.""")
print("\nNotebook 08 complete.")
