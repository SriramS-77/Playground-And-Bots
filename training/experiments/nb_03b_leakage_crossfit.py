# %% [markdown]
# # 3b — Measuring the leak, with pool difficulty cancelled
#
# Notebook 03 tried to isolate the train/eval overlap by comparing an agent's score on
# its own training recordings against size-matched unseen pools. That failed, and the
# evidence it failed is in the notebook: **Static Multi-Threshold showed a seen−unseen
# gap of +23.2 DI**, and it never trains on anything. Four independent unseen draws of
# the same size differed by up to 34 DI for one policy. Pool difficulty swamped the
# effect.
#
# ## The design that cancels it
#
# Split the recordings into two halves, `P1` and `P2`. Train **two** agents:
#
# ```
# agent A  ->  trained on P1
# agent B  ->  trained on P2
#
# seen    = { A on P1 ,  B on P2 }
# unseen  = { A on P2 ,  B on P1 }
# ```
#
# Every pool appears exactly once as seen and once as unseen, so whatever makes `P1`
# intrinsically easier than `P2` enters both sides of the difference and cancels
# **by construction**, not by matching.
#
# The untrained baselines prove it: for them "A on P1" and "B on P1" are the same run, so
# their seen and unseen sets are identical and the gap is *exactly* zero. Any non-zero
# value for a learned agent is therefore attributable to having trained on those
# recordings — which is the quantity Reviewer 1's point 2 is really asking about.
#
# Repeated over several random halvings to get an interval.

# %%
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random as _rnd

import expkit
from expkit.partition import load_partition
from expkit.paths import RESULTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from expkit.trainer import train_dqn
from expkit.xscoring import XScorer
from rlcaptcha.scoring import ScoreCache
from rlcaptcha.policies import MultiThresholdPolicy, SingleThresholdPolicy

OUT = RESULTS
partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
scorer = XScorer.from_name(chosen)

# The scorer was trained on the `lstm` pool, which is disjoint from everything used
# here, so it is a fixed component across all conditions.
POOL = partition["rl"] + partition["eval"]
all_h, all_b = sessions_from_refs(POOL)
CACHE = ScoreCache(scorer).precompute(all_h, all_b, verbose=False)
print(f"pool for cross-fitting: {len(POOL)} sessions "
      f"({sum(1 for r in POOL if not r.is_bot)} human, {sum(1 for r in POOL if r.is_bot)} bot)")

N_SPLITS = 3
EPISODES = 100
N_SEEDS = 10
BOT_COUNTS = (20, 100, 200, 500, 1000)   # the zero-bot cell is degenerate; see nb 07


def halve(refs, seed):
    """Split into two halves, stratified by (class, bot family)."""
    rng = _rnd.Random(seed)
    strata = defaultdict(list)
    for r in refs:
        strata[r.family].append(r)
    p1, p2 = [], []
    for fam in sorted(strata):
        members = sorted(strata[fam], key=lambda r: r.name)
        rng.shuffle(members)
        p1.extend(members[: len(members) // 2])
        p2.extend(members[len(members) // 2:])
    return p1, p2


def mean_DI(policy, refs, seed0):
    h, b = sessions_from_refs(refs)
    vals = []
    for nb in BOT_COUNTS:
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            r, _ = run_x(policy, h, b, nb, cache=CACHE, solve=DETERMINISTIC,
                         cfg=PAPER_EQUIVALENT, seed=seed0 + s)
            vals.append(evaluate_x(r).DI)
    return float(np.mean(vals))

# %% [markdown]
# ## 3b.1 Cross-fit

# %%
rows = []
t0 = time.time()
for split in range(N_SPLITS):
    P1, P2 = halve(POOL, seed=100 + split)
    h1, b1 = sessions_from_refs(P1)
    h2, b2 = sessions_from_refs(P2)
    print(f"\nsplit {split}: P1 = {len(P1)} sessions, P2 = {len(P2)} sessions")

    agents = {}
    for use_score, tag in ((True, "DQN"), (False, "DQN without H-Score")):
        for name, (hh, bb) in (("A", (h1, b1)), ("B", (h2, b2))):
            a, _ = train_dqn(hh, bb, CACHE if use_score else None, episodes=EPISODES,
                             use_score=use_score, reward_source="oracle",
                             solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT,
                             seed=split, name=f"{tag}-{name}", verbose=0)
            agents[(tag, name)] = a.eval_mode(0.0)
        print(f"  trained {tag} on both halves  ({(time.time()-t0)/60:.1f} min)")

    for tag in ("DQN", "DQN without H-Score"):
        A, B = agents[(tag, "A")], agents[(tag, "B")]
        seen = np.mean([mean_DI(A, P1, 3000), mean_DI(B, P2, 3000)])
        unseen = np.mean([mean_DI(A, P2, 3000), mean_DI(B, P1, 3000)])
        rows.append({"split": split, "policy": tag, "seen": seen, "unseen": unseen,
                     "gap": seen - unseen})
        print(f"    {tag:22s} seen {seen:5.1f}  unseen {unseen:5.1f}  gap {seen-unseen:+5.1f}")

    # Untrained baselines: seen and unseen sets are the same two runs, so the gap is
    # exactly zero. Computed rather than assumed, as a check on the implementation.
    for pol in (MultiThresholdPolicy(), SingleThresholdPolicy()):
        d1, d2 = mean_DI(pol, P1, 3000), mean_DI(pol, P2, 3000)
        seen = np.mean([d1, d2]); unseen = np.mean([d2, d1])
        rows.append({"split": split, "policy": pol.name, "seen": seen,
                     "unseen": unseen, "gap": seen - unseen})

cross = pd.DataFrame(rows)
cross.to_csv(OUT / "leakage_crossfit.csv", index=False)
print(f"\ntotal {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## 3b.2 The answer

# %%
g = cross.groupby("policy").agg(
    seen=("seen", "mean"), unseen=("unseen", "mean"),
    gap=("gap", "mean"), gap_sd=("gap", "std"), n=("gap", "size")).round(2)
g["se"] = (g.gap_sd / np.sqrt(g.n)).round(2)
g["ci_lo"] = (g.gap - 1.96 * g.se).round(1)
g["ci_hi"] = (g.gap + 1.96 * g.se).round(1)
print(g.to_string())

zero_check = g.loc[[p for p in g.index if p.startswith("Static")], "gap"].abs().max()
print(f"\nimplementation check -- untrained baselines' gap: {zero_check:.3f} DI "
      f"(must be 0 by construction)")
assert zero_check < 1e-6, "role swap did not cancel; check the design"

print("\nLeakage attributable to having trained on the evaluation recordings:\n")
for p in ("DQN", "DQN without H-Score"):
    if p not in g.index:
        continue
    row = g.loc[p]
    sig = "excludes zero" if (row.ci_lo > 0 or row.ci_hi < 0) else "includes zero"
    print(f"  {p:22s} {row.gap:+6.1f} DI   95% CI [{row.ci_lo:+.1f}, {row.ci_hi:+.1f}]  ({sig})")

# %% [markdown]
# ## 3b.3 What to write
#
# This number — not the drop from the published Table 4 — is the cost of the original
# train/eval overlap. The Table 4 drop in notebook 03 mixes in a different scorer, a
# different pool and a shorter training run, and the untrained baselines fall by a
# comparable amount there, which proves it cannot be leakage alone.
#
# Report the interval. With three splits it is wide, and that is the truthful state of
# the evidence on 83 recordings.

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

sub = cross[cross.policy.isin(["DQN", "DQN without H-Score"])]
for i, p in enumerate(["DQN", "DQN without H-Score"]):
    s = sub[sub.policy == p]
    ax[0].scatter(s.unseen, s.seen, label=p, s=45)
lims = [min(sub.unseen.min(), sub.seen.min()) - 3, max(sub.unseen.max(), sub.seen.max()) + 3]
ax[0].plot(lims, lims, "k--", lw=1, label="no leakage")
ax[0].set_xlabel("DI on unseen recordings"); ax[0].set_ylabel("DI on training recordings")
ax[0].set_title("Every point above the line is memorisation")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

order = [p for p in g.index]
ax[1].barh(range(len(order)), g.loc[order, "gap"],
           xerr=1.96 * g.loc[order, "se"], capsize=3,
           color=["tab:blue" if not p.startswith("Static") else "tab:grey" for p in order])
ax[1].axvline(0, color="k", lw=1)
ax[1].set_yticks(range(len(order)))
ax[1].set_yticklabels([p.replace(" ", "\n") for p in order], fontsize=7)
ax[1].set_xlabel("seen - unseen DI  (95% CI)")
ax[1].set_title("Leakage, pool difficulty cancelled")
ax[1].grid(alpha=.3, axis="x")

plt.tight_layout(); plt.savefig(OUT / "leakage_crossfit.pdf", dpi=300)
plt.savefig(OUT / "leakage_crossfit.png", dpi=110); plt.close()
print("Notebook 03b complete.")
