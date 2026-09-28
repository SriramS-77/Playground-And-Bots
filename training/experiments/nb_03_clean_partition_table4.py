# %% [markdown]
# # 3 — Table 4 on a clean partition
#
# **Reviewer 1, point 2:** *"The 283 samples described as the LSTM test set are also said
# to be used in the web-traffic simulation, and it is unclear whether they were
# subsequently used for RL training, final evaluation, or both."*
#
# They were used for both. This notebook separates the two things that could explain a
# change in Table 4, by moving one at a time:
#
# | condition | scorer | RL checkpoint | evaluation pool |
# |---|---|---|---|
# | **paper** | published | published | all of campaign B (what Table 4 used) |
# | **held-out data** | published | published | `eval` pool only |
# | **clean** | retrained on `lstm` | retrained on `rl` | `eval` pool — disjoint from training |
# | **leaky control** | retrained on `lstm` | retrained on `rl` | the `rl` pool — the agent's **own training recordings** |
#
# If *paper* and *held-out data* agree, the evaluation pool was not the problem.
#
# The last row is the control that makes the result interpretable. *clean* and *leaky
# control* use the **same agent, same training budget, same scorer** and differ in one
# thing only: whether the evaluation recordings were seen during training. So
# *clean* vs *leaky control* is the price of the leak with everything else held fixed,
# while *clean* vs *paper* also mixes in a much shorter training run and a smaller pool.
# Without this control a drop from *paper* to *clean* could be read as leakage when it is
# really training budget — a mistake worth avoiding before it reaches a reviewer.
#
# The two bandits are a known gap: their published checkpoints were fitted on campaign B,
# and retraining a neural bandit is out of scope here. They are evaluated on the held-out
# pool with their published weights and flagged in the results.

# %%
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import expkit
from expkit.partition import load_partition
from expkit.paths import RESULTS, AGENTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from expkit.trainer import train_dqn
from expkit.xscoring import XScorer

from rlcaptcha.data import load_sessions
from rlcaptcha.scoring import BotScorer, ScoreCache
from rlcaptcha.policies import (DQNPolicy, DQNAblationPolicy, LinUCBPolicy,
                                MultiThresholdPolicy, SingleThresholdPolicy,
                                ThompsonPolicy)
import dataclasses

OUT = RESULTS
N_SEEDS = 20
BOT_COUNTS = (0, 20, 100, 200, 500, 1000)
BANDIT_CFG = dataclasses.replace(PAPER_EQUIVALENT, overkill=2.0,
                                 abandonment_applies_to_bots=True)

partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
print("retrained scorer:", chosen)

# %% [markdown]
# ## 3.1 Pools and score caches
#
# Three caches: the published scorer over the full campaign-B pool (what Table 4 used),
# the published scorer over the `eval` pool, and the retrained scorer over the `eval`
# pool.

# %%
pub_humans, pub_bots = load_sessions()                       # campaign B, all 73
ev_humans, ev_bots = sessions_from_refs(partition["eval"])   # 52 held-out
rl_humans, rl_bots = sessions_from_refs(partition["rl"])     # 31 for RL training

published_scorer = BotScorer()
cache_paper = ScoreCache(published_scorer).precompute(pub_humans, pub_bots, verbose=False)
cache_pub_eval = ScoreCache(published_scorer).precompute(ev_humans, ev_bots, verbose=False)

new_scorer = XScorer.from_name(chosen)
cache_new_eval = ScoreCache(new_scorer).precompute(ev_humans, ev_bots, verbose=False)
cache_new_rl = ScoreCache(new_scorer).precompute(rl_humans, rl_bots, verbose=False)

print(f"pools: paper {len(pub_humans)}h/{len(pub_bots)}b | "
      f"eval {len(ev_humans)}h/{len(ev_bots)}b | rl {len(rl_humans)}h/{len(rl_bots)}b")

# %% [markdown]
# ## 3.2 Retrain the RL agents on the `rl` pool
#
# Same hyperparameters as `offline/rl_service_offline_buffer_training.py`: 128-64-32-11,
# Huber, gamma 0.95, Adam 5e-4, epsilon 1.0 -> 0.01 at decay 0.9995, target sync every
# 200 steps, one gradient step per five environment steps, batch 64 drawn 50/50 from two
# class-balanced buffers. Only the session pool changes.

# %%
EPISODES = 150
t0 = time.time()
dqn_new, log_dqn = train_dqn(
    rl_humans, rl_bots, cache_new_rl, episodes=EPISODES, use_score=True,
    reward_source="oracle", solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT,
    seed=0, name="DQN (clean)", verbose=25)
dqn_new.save(AGENTS / "dqn_clean.pt")

abl_new, log_abl = train_dqn(
    rl_humans, rl_bots, None, episodes=EPISODES, use_score=False,
    reward_source="oracle", solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT,
    seed=0, name="DQN without H-Score (clean)", verbose=25)
abl_new.save(AGENTS / "dqn_ablation_clean.pt")
print(f"training took {(time.time()-t0)/60:.1f} min")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for a, log, t in zip(ax, (log_dqn, log_abl), ("DQN (clean)", "Ablation (clean)")):
    d = log.frame()
    a.plot(d.episodes, d.loss, lw=1)
    a.set_title(f"{t} — training loss"); a.set_xlabel("episode"); a.grid(alpha=.3)
    a.set_yscale("log")
plt.tight_layout(); plt.savefig(OUT / "clean_training_curves.pdf", dpi=300)
plt.savefig(OUT / "clean_training_curves.png", dpi=110); plt.close()

# %% [markdown]
# Training curves are one of the reproducibility items Reviewer 1 asks for (point 6).
# They are saved as `clean_training_curves.pdf`.

# %% [markdown]
# ## 3.3 Evaluate the three conditions

# %%
def sweep(policies, humans, bots, cache, condition, n_seeds=N_SEEDS):
    rows = []
    for pol, cfg in policies:
        for nb in BOT_COUNTS:
            for s in range(n_seeds):
                seed = 1000 + s
                torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1))
                r, _ = run_x(pol, humans, bots, nb, cache=cache,
                             solve=DETERMINISTIC, cfg=cfg, seed=seed)
                m = evaluate_x(r)
                rows.append({"condition": condition, "policy": pol.name, "bots": nb,
                             "seed": s, "humans_left": m.surviving_humans,
                             "bots_left": m.surviving_bots, "DI": m.DI, "BOS": m.BOS,
                             "SP_F1": m.SP_F1, "SI_F1": m.SI_F1})
    return pd.DataFrame(rows)


pub_policies = [
    (DQNPolicy(epsilon=0.0), PAPER_EQUIVALENT),
    (DQNAblationPolicy(epsilon=0.0), PAPER_EQUIVALENT),
    (SingleThresholdPolicy(), PAPER_EQUIVALENT),
    (MultiThresholdPolicy(), PAPER_EQUIVALENT),
    (LinUCBPolicy(), BANDIT_CFG),
    (ThompsonPolicy(), BANDIT_CFG),
]
clean_policies = [
    (dqn_new.eval_mode(0.0), PAPER_EQUIVALENT),
    (abl_new.eval_mode(0.0), PAPER_EQUIVALENT),
    (SingleThresholdPolicy(), PAPER_EQUIVALENT),
    (MultiThresholdPolicy(), PAPER_EQUIVALENT),
    (LinUCBPolicy(), BANDIT_CFG),
    (ThompsonPolicy(), BANDIT_CFG),
]

# The leaky control re-uses the very same retrained agents, evaluated on the recordings
# they trained on. No extra training, so the budget is identical by construction.
leaky_policies = [(dqn_new, PAPER_EQUIVALENT), (abl_new, PAPER_EQUIVALENT),
                  (SingleThresholdPolicy(), PAPER_EQUIVALENT),
                  (MultiThresholdPolicy(), PAPER_EQUIVALENT)]

# --- size-matched unseen pools ------------------------------------------- #
# `leaky control` (31 rl sessions) and `clean` (52 eval sessions) differ in POOL as well
# as in seen/unseen, and pools of different size and composition are not equally hard.
# So the honest comparison for the leaky control is against unseen pools drawn to the
# same size and class balance. Averaging over several such draws also gives the
# sampling noise, which the single 52-session `clean` figure hides.
import random as _rnd

_rng_sub = _rnd.Random(4242)
_rl_h = [r for r in partition["rl"] if not r.is_bot]
_rl_b = [r for r in partition["rl"] if r.is_bot]
_ev_h_refs = [r for r in partition["eval"] if not r.is_bot]
_ev_b_refs = [r for r in partition["eval"] if r.is_bot]
N_MATCHED = 4
print(f"size-matched unseen pools: {len(_rl_h)} human + {len(_rl_b)} bot sessions, "
      f"{N_MATCHED} independent draws from the eval pool")

t0 = time.time()
frames = [
    sweep(pub_policies, pub_humans, pub_bots, cache_paper, "paper"),
    sweep(pub_policies, ev_humans, ev_bots, cache_pub_eval, "held-out data"),
    sweep(clean_policies, ev_humans, ev_bots, cache_new_eval, "clean"),
    sweep(leaky_policies, rl_humans, rl_bots, cache_new_rl, "leaky control"),
]
for k in range(N_MATCHED):
    subset = (_rng_sub.sample(_ev_h_refs, len(_rl_h))
              + _rng_sub.sample(_ev_b_refs, len(_rl_b)))
    sh, sb = sessions_from_refs(subset)
    f = sweep(leaky_policies, sh, sb, cache_new_eval, "unseen matched", n_seeds=8)
    f["draw"] = k
    frames.append(f)

df = pd.concat(frames, ignore_index=True)
df.to_csv(OUT / "clean_partition_runs.csv", index=False)
print(f"{len(df)} runs in {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## 3.4 Does it align?
#
# Average over bot volumes, per condition. The `paper` column should match the published
# Table 4 within the seed noise established in `results_seeded/`.

# %%
def canon(n):
    return (n.replace(" (clean)", "")
             .replace("DQN without H-Score", "DQN without H-Score"))

df["policy"] = df.policy.map(canon)
avg = (df.groupby(["condition", "policy"])[["DI", "BOS", "SP_F1", "SI_F1"]]
         .agg(["mean", "std"]).round(3))

# Kept deliberately flat -- one single-level frame per statistic. `pivot_table` with a
# list aggfunc inserts the value name as a third column level, which silently breaks
# ("mean", condition) lookups.
CONDS = ["paper", "held-out data", "clean", "leaky control", "unseen matched"]
di_mean = df.groupby(["policy", "condition"]).DI.mean().unstack("condition").reindex(columns=CONDS)
di_std = df.groupby(["policy", "condition"]).DI.std().unstack("condition").reindex(columns=CONDS)
print("Average Discrimination Index (over all bot volumes and seeds)\n")
print(pd.concat({"mean": di_mean.round(1), "sd": di_std.round(1)}, axis=1).to_string())

PAPER_DI = {"LinUCB": 52.7, "Thompson Sampling": 57.1, "DQN": 74.8,
            "DQN without H-Score": 40.5, "Static Single-Threshold": 46.6,
            "Static Multi-Threshold": 54.6}
cmp = pd.DataFrame({
    "seeded reproduction": pd.Series(PAPER_DI),
    "paper (this run)": di_mean["paper"],
    "held-out data": di_mean["held-out data"],
    "clean": di_mean["clean"],
    "seen (leaky control)": di_mean["leaky control"],
    "unseen matched": di_mean["unseen matched"],
}).round(1)
cmp["clean - paper"] = (cmp["clean"] - cmp["paper (this run)"]).round(1)
cmp["seen - unseen"] = (cmp["seen (leaky control)"] - cmp["unseen matched"]).round(1)
print("\n", cmp.to_string())
cmp.to_csv(OUT / "clean_partition_DI_comparison.csv")

# %% [markdown]
# ### Reading the differences apart
#
# `clean − paper` mixes three changes at once — different scorer, different agent,
# different pool — so it is **not** a measure of leakage. The static baselines prove the
# point: they never train, yet they drop by a comparable amount, which can only be the
# scorer and pool.
#
# The isolated quantity is **`seen − unseen`**: the same agent, same training budget,
# same scorer, evaluated on pools of the *same size and class balance*, differing only in
# whether the recordings were seen during training.
#
# The untrained baselines give the null. They cannot memorise anything, so whatever
# `seen − unseen` they show is sampling noise plus residual pool difficulty, and a
# learned policy has to clear that bar before the gap can be called memorisation.

# %%
static = [p for p in cmp.index if p.startswith("Static")]
learned = [p for p in ("DQN", "DQN without H-Score") if p in cmp.index]

null = df[(df.condition.isin(["leaky control", "unseen matched"]))
          & (df.policy.isin(static))]
null_gap = (null[null.condition == "leaky control"].groupby("policy").DI.mean()
            - null[null.condition == "unseen matched"].groupby("policy").DI.mean())
print("Untrained baselines -- the null distribution for seen minus unseen:")
for p, v in null_gap.items():
    print(f"  {p:26s} {v:+6.1f} DI")
print(f"  {'mean':26s} {null_gap.mean():+6.1f} DI   (sd {null_gap.std():.1f})\n")

# Spread across the independent unseen draws, for a sense of the sampling noise.
draws = (df[df.condition == "unseen matched"].groupby(["policy", "draw"]).DI.mean()
         .unstack("draw"))
print("Average DI on each independent unseen draw:")
print(draws.round(1).to_string())
print()

for p in learned:
    print(f"  {p:22s} seen - unseen = {cmp.loc[p, 'seen - unseen']:+6.1f} DI")

print(f"""
DO NOT read these as leakage. The untrained baselines disagree by
{null_gap.max() - null_gap.min():.0f} DI on a quantity that must be zero for them, and the four
independent unseen draws differ by as much as 34 DI for a single policy. Pool-to-pool
difficulty dominates the seen/unseen effect, so this design cannot attribute the gap.

The flaw is structural: `leaky control` and `unseen matched` use DIFFERENT pools, and a
31-session pool is not interchangeable with another 31-session pool. Matching size and
class balance was not enough.

Notebook 03b fixes it by swapping roles -- two agents trained on two halves, each half
serving once as seen and once as unseen -- so pool difficulty cancels in the difference
by construction rather than by hope.""")

# %% [markdown]
# ## 3.5 Per-cell table, paper format

# %%
def paper_table(sub):
    blocks = []
    for nb in BOT_COUNTS:
        s = sub[sub.bots == nb]
        for metric, label in [("humans_left", "Remaining Humans"),
                              ("bots_left", "Remaining Bots"),
                              ("DI", "DI"), ("BOS", "BOS"),
                              ("SP_F1", "SP-F1"), ("SI_F1", "SI-F1")]:
            g = s.groupby("policy")[metric].agg(["mean", "std"])
            fmt = (lambda r: f"{r['mean']:.1f} +/- {r['std']:.1f}") if metric in (
                "humans_left", "bots_left", "DI") else (
                lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f}")
            blocks.append({"Simulation Type": f"{nb} Bots", "Metric": label,
                           **{p: fmt(g.loc[p]) for p in g.index}})
    return pd.DataFrame(blocks)

tables = {c: paper_table(df[df.condition == c]) for c in df.condition.unique()}
with pd.ExcelWriter(OUT / "clean_partition_table4.xlsx", engine="openpyxl") as xl:
    for c, t in tables.items():
        t.to_excel(xl, sheet_name=c[:28], index=False)
    cmp.to_excel(xl, sheet_name="DI comparison")
    df.to_excel(xl, sheet_name="Raw runs", index=False)
print(tables["clean"].head(12).to_string(index=False))

# %% [markdown]
# ## 3.6 Figure

# %%
fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharex=True)
order = ["DQN", "Thompson Sampling", "LinUCB", "Static Multi-Threshold",
         "Static Single-Threshold", "DQN without H-Score"]
conds = ["paper", "held-out data", "clean", "leaky control", "unseen matched"]
w = 0.16
for ax, metric in zip(axes, ["DI", "BOS", "SP_F1", "SI_F1"]):
    x = np.arange(len(order))
    for i, c in enumerate(conds):
        g = df[df.condition == c].groupby("policy")[metric].agg(["mean", "std"])
        g = g.reindex(order)
        ax.bar(x + (i - 2) * w, g["mean"], w, yerr=g["std"], capsize=2, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=6.5)
    ax.set_title(metric); ax.grid(alpha=.3, axis="y")
axes[0].legend(fontsize=8)
plt.suptitle("Table 4 metrics under progressively cleaner evaluation "
             f"(mean +/- sd over {N_SEEDS} seeds)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "clean_partition.pdf", dpi=300)
plt.savefig(OUT / "clean_partition.png", dpi=110); plt.close()
print("Notebook 03 complete.")
