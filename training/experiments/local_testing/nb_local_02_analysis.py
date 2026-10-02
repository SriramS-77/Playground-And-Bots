# %% [markdown]
# # Local 02 — how the policies actually compare
#
# Reads `results/local_runs.csv` from notebook 01. Nothing is recomputed here, so this is
# cheap to re-run while looking at the numbers.
#
# **One training seed.** Every number below is a single draw. Round 1's headline inversion
# came from exactly this, so read the *pattern across bot fractions*, not the decimals —
# a policy that leads at every fraction is saying something; a 3-point gap in one cell is
# not.

# %%
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = HERE / "results"
runs = pd.read_csv(OUT / "local_runs.csv")
cfg = json.load(open(OUT / "train_config.json"))
print(f"{len(runs):,} runs | {runs.policy.nunique()} policies | "
      f"{runs.bot_fraction.nunique()} bot fractions x {runs.seed.nunique()} eval seeds")
print(f"trained: {cfg['episodes']} episodes, seed {cfg['seed']}, humans ~U{tuple(cfg['human_range'])}")

ORDER = ["DQN", "thompson:mog3", "thompson:mog2", "thompson:gaussian_full",
         "thompson:gaussian", "linucb:gaussian", "Static Multi-Threshold",
         "Static Single-Threshold", "DQN without H-Score"]
order = [p for p in ORDER if p in set(runs.policy)] + \
        [p for p in runs.policy.unique() if p not in ORDER]

# %% [markdown]
# ## 2.1 Headline — mean over the whole grid
#
# `DI = (human survival - bot survival) x 100`. Because no cell has zero bots, DI is
# well-defined everywhere and no exclusion rule is needed.

# %%
head = (runs.groupby("policy")
            .agg(DI=("DI", "mean"), DI_sd=("DI", "std"),
                 humans=("human_survival", "mean"), bots=("bot_survival", "mean"),
                 BOS=("BOS", "mean"), SI_F1=("SI_F1", "mean"),
                 friction_s=("friction_s_per_human", "mean"))
            .reindex(order).round(3))
head.to_csv(OUT / "headline.csv")
print(head.to_string())

# %% [markdown]
# ## 2.2 The pattern that matters — DI by bot fraction
#
# An average can be carried entirely by a couple of easy cells. Round 1's bandits looked
# competitive on the average while flooring at ~23 DI above 100 bots, because they were
# killing three quarters of the humans.

# %%
by_frac = runs.pivot_table(index="policy", columns="bot_fraction", values="DI").reindex(order)
by_frac.columns = [f"{c:.0%}" for c in by_frac.columns]
by_frac["mean"] = by_frac.mean(axis=1)
by_frac.round(1).to_csv(OUT / "di_by_fraction.csv")
print(by_frac.round(1).to_string())

print("\nWins per bot fraction (excluding the ablation, which is a diagnostic):")
compet = runs[runs.policy != "DQN without H-Score"]
wins = (compet.pivot_table(index="bot_fraction", columns="policy", values="DI")
              .idxmax(axis=1))
for f, w in wins.items():
    print(f"  {f:.0%} bots -> {w}")

# %% [markdown]
# ## 2.3 Security and usability separately
#
# DI collapses two quantities that trade off. A policy can score well by keeping humans
# and leaking bots, or by blocking bots and driving humans away. The paper's claim is
# about doing both, so report both.

# %%
su = (runs.groupby("policy")
          .agg(human_survival=("human_survival", "mean"),
               bot_survival=("bot_survival", "mean"),
               friction_s=("friction_s_per_human", "mean"))
          .reindex(order).round(3))
su["bots_blocked"] = (1 - su.bot_survival).round(3)
su.to_csv(OUT / "security_usability.csv")
print(su[["human_survival", "bots_blocked", "friction_s"]].to_string())

# %% [markdown]
# ## 2.4 Does any policy read the attack level off total population?
#
# Humans were drawn independently of the bot fraction, so total population is no longer a
# proxy for the attack level. If a policy's behaviour still tracks population rather than
# bot fraction, that is the R1.5 confound showing up on its own.

# %%
rows = []
for p, g in runs.groupby("policy"):
    rows.append({"policy": p,
                 "corr(DI, bot_fraction)": g.DI.corr(g.bot_fraction),
                 "corr(DI, total_population)": g.DI.corr(g.total_population),
                 "corr(friction, total_population)": g.friction_s_per_human.corr(g.total_population)})
conf = pd.DataFrame(rows).set_index("policy").reindex(order).round(3)
conf.to_csv(OUT / "population_confound.csv")
print(conf.to_string())
print(f"\ncorr(total population, bot fraction) in this grid = "
      f"{runs.total_population.corr(runs.bot_fraction):+.3f}   (published design: +1.000)")

# %% [markdown]
# ## 2.5 Figures

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
fracs = sorted(runs.bot_fraction.unique())
for p in order:
    g = runs[runs.policy == p].groupby("bot_fraction")
    m, sd = g.DI.mean(), g.DI.std()
    style = dict(lw=2.4, marker="o") if p.startswith(("DQN", "thompson:mog")) else \
            dict(lw=1.2, marker="s", alpha=.75)
    ax[0].plot(fracs, m.values, label=p, **style)
    ax[0].fill_between(fracs, (m - sd).values, (m + sd).values, alpha=.08)
ax[0].set_xlabel("bot fraction of traffic"); ax[0].set_ylabel("Discrimination Index")
ax[0].axvspan(0.30, 0.45, color="k", alpha=.05)
ax[0].annotate("published\nreal-traffic band", (0.375, ax[0].get_ylim()[0] + 3),
               ha="center", fontsize=7, color="0.35")
ax[0].set_title("DI across traffic composition"); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=6.5, ncol=1)

for p in order:
    g = runs[runs.policy == p]
    ax[1].scatter(g.friction_s_per_human.mean(), 1 - g.bot_survival.mean(), s=70)
    ax[1].annotate(p, (g.friction_s_per_human.mean(), 1 - g.bot_survival.mean()),
                   fontsize=6.5, xytext=(5, 3), textcoords="offset points")
ax[1].set_xlabel("friction, seconds per human"); ax[1].set_ylabel("fraction of bots blocked")
ax[1].set_title("Security vs usability (up and left is better)"); ax[1].grid(alpha=.3)

for p in order:
    g = runs[runs.policy == p]
    ax[2].scatter(g.bot_survival.mean(), g.human_survival.mean(), s=70)
    ax[2].annotate(p, (g.bot_survival.mean(), g.human_survival.mean()),
                   fontsize=6.5, xytext=(5, 3), textcoords="offset points")
ax[2].set_xlabel("bot survival"); ax[2].set_ylabel("human survival")
ax[2].set_title("Who survives (up and left is better)"); ax[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig(OUT / "local_comparison.png", dpi=120)
plt.savefig(OUT / "local_comparison.pdf", dpi=300); plt.close()
print("wrote local_comparison.{png,pdf}")

# %% [markdown]
# ## 2.6 Training stability
#
# One seed says nothing about variance between seeds, but a training curve that is still
# moving at the end says the budget was too small — which is what round 1's 90-episode
# runs looked like.

# %%
cur = OUT / "training_curves.csv"
if cur.exists():
    c = pd.read_csv(cur)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for p, g in c.groupby("policy"):
        g = g.sort_values("episodes")
        w = max(5, len(g) // 20)
        ax[0].plot(g.episodes, g.surviving_humans.rolling(w, min_periods=1).mean(), label=p)
        if "loss" in g:
            ax[1].plot(g.episodes, g.loss.rolling(w, min_periods=1).mean(), label=p)
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("surviving humans"); ax[0].grid(alpha=.3)
    ax[0].set_title("Training: humans kept"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("episode"); ax[1].set_ylabel("loss"); ax[1].set_yscale("log")
    ax[1].set_title("Training: loss"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=120); plt.close()

    last = c.groupby("policy").apply(
        lambda g: pd.Series({
            "final_third_mean_humans": g.sort_values("episodes").surviving_humans.tail(len(g) // 3).mean(),
            "middle_third_mean_humans": g.sort_values("episodes").surviving_humans
                                         .iloc[len(g) // 3: 2 * len(g) // 3].mean()}),
        include_groups=False).round(1)
    last["still_drifting"] = (last.final_third_mean_humans -
                              last.middle_third_mean_humans).abs().round(1)
    print(last.to_string())
print("\nNotebook local-02 complete.")
