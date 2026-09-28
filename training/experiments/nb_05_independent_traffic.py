# %% [markdown]
# # 5 — Independent human and bot traffic
#
# **Reviewer 1, point 5:** *"There's also a confound: legitimate traffic is fixed at 100
# users while bot traffic varies up to 1,000, and total active users is part of the
# controller's state, so population size alone may reveal much of the attack condition."*
#
# This is a sharp and testable objection. In the published design `n_humans = 100`
# always, so `server_total_users` is a near-deterministic function of the bot count — the
# agent can read the attack level straight off its own state vector without any
# behavioural signal at all.
#
# Two experiments:
#
# * **5.2** — vary humans and bots independently over a grid. If the policy's advantage
#   was an artefact of the confound, it should degrade when population size stops
#   identifying the attack.
# * **5.3** — ablate the population feature directly: freeze `n_active` at a constant
#   before it reaches the network. If performance collapses, the agent was leaning on it.
#
# The retreat to "traffic-volume robustness" that the review offers as an alternative is
# not needed if these come out well.

# %%
import dataclasses, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import expkit
from expkit.partition import load_partition
from expkit.paths import RESULTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from expkit.xscoring import XScorer
from rlcaptcha.scoring import ScoreCache
from rlcaptcha.policies import DQNPolicy, MultiThresholdPolicy, SingleThresholdPolicy

OUT = RESULTS
partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
ev_h, ev_b = sessions_from_refs(partition["eval"])
cache = ScoreCache(XScorer.from_name(chosen)).precompute(ev_h, ev_b, verbose=False)

HUMANS = (25, 50, 100, 200, 400)
BOTS = (0, 20, 100, 200, 500, 1000)
N_SEEDS = 8

# %% [markdown]
# ## 5.1 How strong is the confound?
#
# Under the published design, the correlation between total population and bot count is
# essentially 1. Across the grid below it is far weaker, because a 400-human/0-bot
# episode and a 100-human/300-bot episode look the same from the population feature.

# %%
pub = pd.DataFrame({"humans": 100, "bots": BOTS})
pub["total"] = pub.humans + pub.bots
grid = pd.DataFrame([{"humans": h, "bots": b} for h in HUMANS for b in BOTS])
grid["total"] = grid.humans + grid.bots
print(f"published design : corr(total population, bots) = {pub.total.corr(pub.bots):.4f}")
print(f"this grid        : corr(total population, bots) = {grid.total.corr(grid.bots):.4f}")
print(f"                   corr(total population, bot FRACTION) = "
      f"{grid.total.corr(grid.bots / grid.total.clip(lower=1)):.4f}")

# %% [markdown]
# ## 5.2 The grid

# %%
policies = [DQNPolicy(epsilon=0.0), MultiThresholdPolicy(), SingleThresholdPolicy()]
rows = []
t0 = time.time()
for pol in policies:
    for nh in HUMANS:
        for nb in BOTS:
            for s in range(N_SEEDS):
                torch.manual_seed(s); np.random.seed(s)
                r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache, n_humans=nh,
                             solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT, seed=4000 + s)
                m = evaluate_x(r)
                rows.append({"policy": pol.name, "n_humans": nh, "n_bots": nb, "seed": s,
                             "humans_left": m.surviving_humans, "bots_left": m.surviving_bots,
                             "human_survival": m.surviving_humans / nh,
                             "bot_survival": m.surviving_bots / nb if nb else 0.0,
                             "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                             "bot_fraction": nb / (nh + nb)})
traffic = pd.DataFrame(rows)
traffic.to_csv(OUT / "independent_traffic_runs.csv", index=False)
print(f"{len(traffic)} runs in {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ### Does the ranking hold across the grid?

# %%
cell = traffic.groupby(["policy", "n_humans", "n_bots"]).DI.mean().unstack(0)
winner = cell.idxmax(axis=1)
print("Winner by cell (average DI):\n")
print(winner.unstack().to_string())
print("\nShare of the %d cells won by each policy:" % len(winner))
print((winner.value_counts() / len(winner)).round(3).to_string())

# %% [markdown]
# ### Is the DQN's advantage a function of the bot fraction, or of raw population?
#
# If the agent were exploiting the confound, its advantage would track total population.
# If it is genuinely orchestrating, it should track the bot *fraction* — the thing a
# security policy should respond to.

# %%
adv = (traffic[traffic.policy == "DQN"].groupby(["n_humans", "n_bots"]).DI.mean()
       - traffic[traffic.policy == "Static Multi-Threshold"]
         .groupby(["n_humans", "n_bots"]).DI.mean()).reset_index(name="DI_advantage")
adv["total"] = adv.n_humans + adv.n_bots
adv["bot_fraction"] = adv.n_bots / adv.total
print(f"corr(DQN advantage, total population) = {adv.DI_advantage.corr(adv.total):+.3f}")
print(f"corr(DQN advantage, bot fraction)     = {adv.DI_advantage.corr(adv.bot_fraction):+.3f}")
print(f"corr(total population, bot fraction)  = {adv.total.corr(adv.bot_fraction):+.3f}")

# The two predictors are themselves correlated, so marginal correlations cannot say which
# one carries the effect. Regress on both, standardised, and read the coefficients.
Z = adv[["total", "bot_fraction"]].astype(float)
Z = (Z - Z.mean()) / Z.std(ddof=0)
y = (adv.DI_advantage - adv.DI_advantage.mean()) / adv.DI_advantage.std(ddof=0)
X = np.column_stack([np.ones(len(Z)), Z.values])
beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
print(f"\nstandardised regression of DQN advantage on both predictors:")
print(f"  total population : {beta[1]:+.3f}")
print(f"  bot fraction     : {beta[2]:+.3f}")
print("""
Read the larger absolute coefficient as the variable the advantage actually tracks. If it
is total population rather than bot fraction, the reviewer's confound concern has real
content and the next section is the decisive test.""")

# %% [markdown]
# ## 5.3 Ablating the population feature
#
# A wrapper that keeps everything about the DQN identical but freezes the fifth state
# feature (`server_total_users`) at a constant, so the attack level is invisible to it.
# Nothing in `rlcaptcha` changes — the wrapper only intercepts `features`.

# %%
class FrozenPopulationDQN:
    """DQN with `n_active` pinned to a constant before it reaches the network."""
    needs_bot_score = True
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def __init__(self, inner, constant=100, name=None):
        self.inner, self.constant = inner, constant
        self.name = name or f"DQN (population frozen at {constant})"

    def features(self, user, n_active):
        return self.inner.features(user, self.constant)

    def select_action(self, user, n_active):
        return self.inner.select_action(user, self.constant)


class ShuffledPopulationDQN(FrozenPopulationDQN):
    """DQN fed a population value drawn at random from the grid's range -- destroys the
    feature's information without changing its marginal distribution."""

    def __init__(self, inner, rng, name="DQN (population randomised)"):
        super().__init__(inner, name=name)
        self.rng = rng

    def features(self, user, n_active):
        return self.inner.features(user, self.rng.randint(25, 1400))

    def select_action(self, user, n_active):
        return self.inner.select_action(user, self.rng.randint(25, 1400))


import random
variants = [
    DQNPolicy(epsilon=0.0),
    FrozenPopulationDQN(DQNPolicy(epsilon=0.0), 100),
    ShuffledPopulationDQN(DQNPolicy(epsilon=0.0), random.Random(0)),
]
rows = []
for pol in variants:
    for nh in (50, 100, 200):
        for nb in BOTS:
            for s in range(N_SEEDS):
                torch.manual_seed(s); np.random.seed(s)
                r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache, n_humans=nh,
                             solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT, seed=5000 + s)
                m = evaluate_x(r)
                rows.append({"variant": pol.name, "n_humans": nh, "n_bots": nb,
                             "DI": m.DI, "BOS": m.BOS, "SI_F1": m.SI_F1,
                             "human_survival": m.surviving_humans / nh,
                             "bot_survival": m.surviving_bots / nb if nb else 0.0})
abl = pd.DataFrame(rows)
abl.to_csv(OUT / "population_feature_ablation.csv", index=False)
summ = abl.groupby("variant")[["DI", "BOS", "SI_F1", "human_survival", "bot_survival"]]\
          .agg(["mean", "std"]).round(3)
print(summ.to_string())

base = summ.loc["DQN", ("DI", "mean")]
print("\nChange in average DI when the population feature is destroyed:")
for v in summ.index:
    if v != "DQN":
        print(f"  {v:38s} {summ.loc[v, ('DI', 'mean')] - base:+6.1f} DI")

# %% [markdown]
# ### Which of the two ablations is the valid test
#
# They are not equivalent, and only one of them answers the reviewer.
#
# * **Frozen at a constant** keeps the input inside the range the network was trained on
#   and simply removes its *information*. If performance survives, the feature was not
#   carrying anything the policy needed — the confound was not being exploited.
# * **Randomised** removes the information too, but also feeds values far outside the
#   training distribution. A collapse here is the ordinary failure of a network on
#   out-of-distribution input and is **not** evidence that the feature was being relied
#   on. It is reported for completeness, not as the test.
#
# So the frozen row is the one to quote.

# %%
frozen = summ.loc["DQN (population frozen at 100)", ("DI", "mean")] - base
rand = summ.loc["DQN (population randomised)", ("DI", "mean")] - base
print(f"frozen at a constant : {frozen:+.1f} DI   <- the valid test")
print(f"randomised           : {rand:+.1f} DI   <- out-of-distribution input, not a test\n")
if frozen > -2.0:
    print("""VERDICT: replacing the population feature with a constant costs essentially
nothing. The policy is therefore NOT reading the attack level off total population, and
Reviewer 1's confound -- real as a design flaw -- is not what produces the result. The
paper can keep its claim, report this ablation, and additionally fix the design by
varying legitimate traffic as in Section 5.2.""")
else:
    print("""VERDICT: performance drops materially without the population feature, so the
policy IS using it. Given humans were fixed at 100 in the published design, part of the
reported advantage may come from reading the attack level off population size. The claim
should be narrowed, or the agent retrained on the varied-traffic grid.""")

# %% [markdown]
# ## 5.4 Figures

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))

pv = traffic[traffic.policy == "DQN"].pivot_table(index="n_humans", columns="n_bots",
                                                  values="DI")
im = axes[0].imshow(pv.values, cmap="viridis", aspect="auto", origin="lower")
axes[0].set_xticks(range(len(pv.columns))); axes[0].set_xticklabels(pv.columns)
axes[0].set_yticks(range(len(pv.index))); axes[0].set_yticklabels(pv.index)
axes[0].set_xlabel("bots"); axes[0].set_ylabel("legitimate users")
axes[0].set_title("DQN — Discrimination Index across the traffic grid")
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        axes[0].text(j, i, f"{pv.values[i, j]:.0f}", ha="center", va="center",
                     color="w", fontsize=7)
plt.colorbar(im, ax=axes[0])

for pol, mk in zip(["DQN", "Static Multi-Threshold", "Static Single-Threshold"], "os^"):
    g = traffic[traffic.policy == pol].groupby("bot_fraction").DI.mean()
    axes[1].plot(g.index, g.values, mk + "-", ms=4, label=pol)
axes[1].set_xlabel("bot fraction of total traffic"); axes[1].set_ylabel("average DI")
axes[1].set_title("DI vs. attack intensity, humans varied too")
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

g = abl.groupby(["variant", "n_bots"]).DI.mean().unstack(0)
g.plot(ax=axes[2], marker="o", ms=4)
axes[2].set_xlabel("bots"); axes[2].set_ylabel("average DI")
axes[2].set_title("Ablating the population feature")
axes[2].legend(fontsize=7); axes[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig(OUT / "independent_traffic.pdf", dpi=300)
plt.savefig(OUT / "independent_traffic.png", dpi=110); plt.close()

with pd.ExcelWriter(OUT / "independent_traffic.xlsx", engine="openpyxl") as xl:
    cell.to_excel(xl, sheet_name="DI by cell")
    winner.unstack().to_excel(xl, sheet_name="Winner by cell")
    summ.to_excel(xl, sheet_name="Population ablation")
    traffic.to_excel(xl, sheet_name="Raw runs", index=False)
print("Notebook 05 complete.")
