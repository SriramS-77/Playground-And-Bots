# %% [markdown]
# # 4 — Reward sensitivity
#
# **Reviewer 1, point 3:** *"With the stated parameters, the human reward is 50 at T=0,
# 23 at T=1, and −999 at T=10 — a much steeper scale than the other reward terms, which
# may strongly influence which policy appears best."*
#
# **Reviewer 2, point 4:** *"How to determine the value of each parameter? By using the
# experience of the authors?"*
#
# The reviewer is right about the arithmetic: `R_max/2 - 2^T` gives
# 50, 23, 21, 17, 9, −7, −39, −103, −231, −487, **−999**, so from T=8 the friction term
# alone exceeds the entire leakage penalty of −150.
#
# ## One thing to get straight first
#
# In this simulator the reward parameters do **not** change the reported metrics for a
# fixed policy. Survival is decided by two things only: whether the bot was blocked
# (`T > B_s`) and whether the user abandoned (`P_leave(T)`). `beta`, `R_leak`, `overkill`
# and the underestimation coefficient enter the *reward value* and nothing else — so
# re-scoring a fixed policy under a different `beta` returns the identical Table 4.
#
# That has a consequence worth stating in the paper: **the reward parameters can only
# matter through training.** So this notebook does three separate things:
#
# * **4.2** — how the reward function *ranks fixed policies*, which is what it is for;
# * **4.3** — **retrain** the agent under each reward setting and ask whether the same
#   policy still wins. This is the sensitivity analysis the reviewer actually wants;
# * **4.4** — the abandonment curve, which is the one parameter that *does* move the
#   metrics directly, for every policy at once.

# %%
import dataclasses, json, os, sys, time
from itertools import product
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
from expkit.trainer import train_dqn
from expkit.xscoring import XScorer
from rlcaptcha.scoring import ScoreCache
from rlcaptcha.policies import (DQNPolicy, MultiThresholdPolicy, SingleThresholdPolicy)

OUT = RESULTS
partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
scorer = XScorer.from_name(chosen)

ev_h, ev_b = sessions_from_refs(partition["eval"])
rl_h, rl_b = sessions_from_refs(partition["rl"])
cache_ev = ScoreCache(scorer).precompute(ev_h, ev_b, verbose=False)
cache_rl = ScoreCache(scorer).precompute(rl_h, rl_b, verbose=False)
BOT_COUNTS = (0, 20, 100, 200, 500, 1000)
N_SEEDS = 12

# %% [markdown]
# ## 4.1 The reward landscape as published

# %%
levels = np.arange(11)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for beta in (1.5, 2.0, 2.5):
    hr = [50.0 if t == 0 else 25.0 - beta ** t for t in levels]
    ax[0].plot(levels, hr, "o-", label=f"beta={beta}")
ax[0].axhline(-150, color="crimson", ls="--", lw=1, label="R_leak = -150")
ax[0].set_xlabel("threat level T"); ax[0].set_ylabel("reward to a legitimate user")
ax[0].set_title("Human reward, $R_{max}/2 - \\beta^T$"); ax[0].legend(fontsize=8)
ax[0].grid(alpha=.3); ax[0].set_ylim(-1100, 100)
ax[1].plot(levels, [50.0 if t == 0 else 25.0 - 2.0 ** t for t in levels], "o-")
ax[1].set_yscale("symlog"); ax[1].axhline(-150, color="crimson", ls="--", lw=1)
ax[1].set_title("Same curve, symlog — note the crossing at T=8")
ax[1].set_xlabel("threat level T"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "reward_landscape.pdf", dpi=300)
plt.savefig(OUT / "reward_landscape.png", dpi=110); plt.close()

tbl = pd.DataFrame({"T": levels,
                    "human reward (beta=2)": [50.0 if t == 0 else 25.0 - 2.0 ** t for t in levels],
                    "P_leave (paper)": [0.0 if t <= 6 else (1.0 if t == 10 else 1 - 0.5 / (t - 6))
                                        for t in levels]}).round(3)
print(tbl.to_string(index=False))

# %% [markdown]
# ## 4.2 How the reward ranks fixed policies
#
# Mean reward per user-step for three fixed policies, under each reward setting. This is
# the reward function doing its job: expressing a preference. It says nothing about which
# policy *survives* better — see the note at the top.

# %%
GRID = [dict(friction_base=b, leakage_penalty=l, overkill=o)
        for b, l, o in product((1.5, 2.0, 2.5), (-50.0, -150.0, -300.0), (2.0, 2.5))]

fixed = [SingleThresholdPolicy(), MultiThresholdPolicy(), DQNPolicy(epsilon=0.0)]
rows = []
for g in GRID:
    cfg = dataclasses.replace(PAPER_EQUIVALENT, abandonment="paper", **g)
    for pol in fixed:
        for nb in (100, 500):
            for s in range(4):
                torch.manual_seed(s); np.random.seed(s)
                r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev,
                             solve=DETERMINISTIC, cfg=cfg, seed=s)
                rows.append({**g, "policy": pol.name, "bots": nb, "seed": s,
                             "human_step_reward": r.human_step_reward,
                             "bot_step_reward": r.bot_step_reward,
                             "total_step_reward": r.human_step_reward + r.bot_step_reward})
rank = pd.DataFrame(rows)
rank.to_csv(OUT / "reward_ranking_fixed.csv", index=False)

piv = rank.groupby(["friction_base", "leakage_penalty", "overkill", "policy"])\
          .total_step_reward.mean().unstack().round(1)
piv["best"] = piv.idxmax(axis=1)
print(piv.to_string())
print(f"\nThe reward function prefers the same policy in "
      f"{(piv['best'] == piv['best'].mode()[0]).mean():.0%} of the {len(piv)} settings.")

# %% [markdown]
# ## 4.3 Retraining under each reward setting
#
# The real test. A reduced grid (6 settings) retrained from scratch, then evaluated on
# the held-out pool with the *same* metric definitions throughout, so the only thing that
# varies is what the agent was taught to want.

# %%
RETRAIN_GRID = [
    dict(name="published", friction_base=2.0, leakage_penalty=-150.0, overkill=2.5),
    dict(name="gentle friction", friction_base=1.5, leakage_penalty=-150.0, overkill=2.5),
    dict(name="harsh friction", friction_base=2.5, leakage_penalty=-150.0, overkill=2.5),
    dict(name="cheap leak", friction_base=2.0, leakage_penalty=-50.0, overkill=2.5),
    dict(name="costly leak", friction_base=2.0, leakage_penalty=-300.0, overkill=2.5),
    dict(name="linear friction", friction_base=None, leakage_penalty=-150.0, overkill=2.5),
]

EPISODES = 60
agents, train_rows = {}, []
t0 = time.time()
for g in RETRAIN_GRID:
    label = g.pop("name")
    cfg = dataclasses.replace(PAPER_EQUIVALENT, abandonment="paper", **g)
    agent, log = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                           reward_source="oracle", solve=DETERMINISTIC, cfg=cfg,
                           seed=0, name=f"DQN[{label}]", verbose=0)
    agents[label] = agent.eval_mode(0.0)
    g["name"] = label
    print(f"  trained {label:18s} ({(time.time()-t0)/60:.1f} min elapsed)")

rows = []
for label, agent in agents.items():
    for nb in BOT_COUNTS:
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            r, _ = run_x(agent, ev_h, ev_b, nb, cache=cache_ev,
                         solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT, seed=2000 + s)
            m = evaluate_x(r)
            rows.append({"reward_setting": label, "bots": nb, "seed": s,
                         "humans_left": m.surviving_humans, "bots_left": m.surviving_bots,
                         "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1})
# Baselines that do not train, for reference under the same evaluation.
for pol in (SingleThresholdPolicy(), MultiThresholdPolicy()):
    for nb in BOT_COUNTS:
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev,
                         solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT, seed=2000 + s)
            m = evaluate_x(r)
            rows.append({"reward_setting": pol.name, "bots": nb, "seed": s,
                         "humans_left": m.surviving_humans, "bots_left": m.surviving_bots,
                         "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1})

sens = pd.DataFrame(rows)
sens.to_csv(OUT / "reward_sensitivity_runs.csv", index=False)
summary = sens.groupby("reward_setting")[["DI", "BOS", "SP_F1", "SI_F1"]]\
              .agg(["mean", "std"]).round(3)
print("\n", summary.to_string())

# %% [markdown]
# ### The question that matters
#
# Does a DQN trained under *any* of these reward settings still beat the static
# baselines? If yes, the headline conclusion does not depend on the hand-chosen
# constants, and that is a stronger claim than the paper currently makes.

# %%
dqn_rows = [g["name"] for g in RETRAIN_GRID]
base_best = summary.loc[["Static Single-Threshold", "Static Multi-Threshold"],
                        ("DI", "mean")].max()
beats = summary.loc[dqn_rows, ("DI", "mean")] > base_best
print(f"best static baseline, average DI: {base_best:.1f}\n")
for k in dqn_rows:
    v = summary.loc[k, ("DI", "mean")]
    print(f"  DQN[{k:16s}] DI={v:6.1f}  {'beats' if beats[k] else 'DOES NOT BEAT'} the best static")
print(f"\n=> {int(beats.sum())}/{len(beats)} reward settings still produce a DQN that "
      f"beats every static baseline.")

# %% [markdown]
# ## 4.4 The abandonment curve
#
# `P_leave` is the only reward-side parameter that changes survival directly, so it
# changes Table 4 for every policy at once — including the ones that never train. The
# published curve is 0 below T=7 then 0.5 / 0.75 / 0.83 / 1.0, chosen without supporting
# data. The `empirical` alternative derives it from measured solving time,
# `P_leave = 1 - exp(-seconds / tau)`, with the per-level seconds taken from Bursztein
# et al. (2010) and Searles et al. (2023).

# %%
from expkit.stochastic import HUMAN_SECONDS
curves = {"paper": "paper", "empirical (tau=22s)": "empirical",
          "empirical (tau=45s)": "empirical", "empirical (tau=80s)": "empirical",
          "none": "none"}
rows = []
for label, mode in curves.items():
    tau = 80.0 if "80" in label else (45.0 if "45" in label else 22.0)
    cfg = dataclasses.replace(PAPER_EQUIVALENT, abandonment=mode, abandonment_tau=tau)
    for pol in (SingleThresholdPolicy(), MultiThresholdPolicy(), DQNPolicy(epsilon=0.0)):
        for nb in BOT_COUNTS:
            for s in range(N_SEEDS):
                torch.manual_seed(s); np.random.seed(s)
                r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev,
                             solve=DETERMINISTIC, cfg=cfg, seed=3000 + s)
                m = evaluate_x(r)
                rows.append({"curve": label, "policy": pol.name, "bots": nb,
                             "DI": m.DI, "humans_left": m.surviving_humans,
                             "bots_left": m.surviving_bots})
ab = pd.DataFrame(rows)
ab.to_csv(OUT / "abandonment_sensitivity.csv", index=False)
print(ab.groupby(["curve", "policy"])[["DI", "humans_left", "bots_left"]]
        .mean().round(1).to_string())

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
lv = np.arange(11)
ax[0].plot(lv, [0 if t <= 6 else (1 if t == 10 else 1 - .5/(t-6)) for t in lv], "o-",
           label="paper (hand-chosen)")
for tau in (22.0, 45.0, 80.0):
    ax[0].plot(lv, [1 - np.exp(-HUMAN_SECONDS[t]/tau) for t in lv], "s--",
               label=f"empirical, tau={tau:.0f}s")
ax[0].set_xlabel("threat level"); ax[0].set_ylabel("P(user abandons)")
ax[0].set_title("Abandonment curves"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

g = ab.groupby(["curve", "policy"]).DI.mean().unstack()
g.plot.bar(ax=ax[1], rot=20); ax[1].set_ylabel("average DI")
ax[1].set_title("DI by abandonment curve"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
ax[1].tick_params(labelsize=7)

s = summary.loc[dqn_rows + ["Static Single-Threshold", "Static Multi-Threshold"], ("DI", "mean")]
e = summary.loc[s.index, ("DI", "std")]
ax[2].barh(range(len(s)), s.values, xerr=e.values,
           color=["tab:blue"] * len(dqn_rows) + ["tab:grey"] * 2)
ax[2].set_yticks(range(len(s))); ax[2].set_yticklabels(s.index, fontsize=7)
ax[2].axvline(base_best, color="crimson", ls="--", lw=1, label="best static")
ax[2].set_xlabel("average DI"); ax[2].set_title("Retrained under each reward setting")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(OUT / "reward_sensitivity.pdf", dpi=300)
plt.savefig(OUT / "reward_sensitivity.png", dpi=110); plt.close()

with pd.ExcelWriter(OUT / "reward_sensitivity.xlsx", engine="openpyxl") as xl:
    summary.to_excel(xl, sheet_name="Retrained summary")
    piv.to_excel(xl, sheet_name="Fixed-policy ranking")
    ab.groupby(["curve", "policy"])[["DI", "humans_left", "bots_left"]].mean()\
      .to_excel(xl, sheet_name="Abandonment")
    sens.to_excel(xl, sheet_name="Raw runs", index=False)
print("Notebook 04 complete.")
