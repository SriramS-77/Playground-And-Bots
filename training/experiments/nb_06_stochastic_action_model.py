# %% [markdown]
# # 6 — A probabilistic action model
#
# **Reviewer 1, point 1:** *"Each bot is assigned a strength (Bs), and it is blocked
# whenever T>Bs. Stronger actions succeed because the simulator is explicitly defined
# that way, and level 10 blocks every bot by construction. This shows an RL agent can
# learn the simulator's rule, but it does not yet demonstrate effective orchestration of
# real CAPTCHA services, where outcomes are probabilistic."*
#
# This is the comment the Associate Editor led with, and it is correct.
#
# ## The design
#
# Bot outcomes come from a two-parameter logistic item-response model,
# `P(solve) = sigma(alpha * (theta(B_s) - d(T)))` with `theta(B_s) = B_s + 0.5` and
# `d(T) = T`. Bot strength is latent ability; challenge level is item difficulty; `alpha`
# is discrimination.
#
# The published rule is the `alpha -> infinity` limit of exactly this model, because
# `theta > d` iff `B_s + 0.5 > T` iff `B_s >= T` for integers. So the deterministic
# simulator is not a different environment — it is one end of a continuum, and `alpha`
# walks away from it. Section 6.1 verifies that limit is reproduced to the survivor.
#
# Human outcomes come from measurement rather than assumption:
#
# * Bursztein et al., *How Good Are Humans at Solving CAPTCHAs? A Large Scale
#   Evaluation*, IEEE S&P 2010 — 318k CAPTCHAs across 21 schemes. Image schemes average
#   87% solving accuracy (authorize.net 98%, mail.ru 70%); audio 52%. Mean solving times
#   6.8s to 13.0s for image, 19-35s for audio.
# * Searles et al., *An Empirical Study & Evaluation of Modern CAPTCHAs*, USENIX Security
#   2023 — 1,400 participants, 14k CAPTCHAs. reCAPTCHA checkbox median 3.7s; distorted
#   text 9-15s; game-based 18-42s; abandonment 120% higher in realistic task contexts.
#
# For the bot side under contemporary solvers we also offer a `solver_era` shift, since
# YOLO-based solvers reach 100% on reCAPTCHAv2 image challenges (Plesner et al., 2024)
# while a generalised agentic VLM solver manages 60.7% across 26 CAPTCHA types and 70.6%
# on unseen challenges in the wild (Teoh et al., USENIX Security 2025) — i.e. legacy
# visual challenges are the broken ones, not the interactive ones.

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
from expkit.paths import RESULTS, AGENTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import (DETERMINISTIC, DETERMINISTIC_FALLIBLE_HUMANS, GROUNDED,
                               GROUNDED_MODERN, LEVEL_MECHANISM, PAPER_EQUIVALENT,
                               SolveModel, StochasticRewardConfig)
from expkit.trainer import train_dqn
from expkit.xscoring import XScorer
from rlcaptcha.data import load_sessions
from rlcaptcha.scoring import BotScorer, ScoreCache
from rlcaptcha.simulate import run_simulation
from rlcaptcha.config import DQN_REWARDS, BANDIT_REWARDS, EVAL_BOT_COUNTS
from rlcaptcha.policies import (DQNPolicy, DQNAblationPolicy, LinUCBPolicy,
                                MultiThresholdPolicy, SingleThresholdPolicy,
                                ThompsonPolicy)

OUT = RESULTS
partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
ev_h, ev_b = sessions_from_refs(partition["eval"])
rl_h, rl_b = sessions_from_refs(partition["rl"])
scorer = XScorer.from_name(chosen)
cache_ev = ScoreCache(scorer).precompute(ev_h, ev_b, verbose=False)
cache_rl = ScoreCache(scorer).precompute(rl_h, rl_b, verbose=False)
BOT_COUNTS = (0, 20, 100, 200, 500, 1000)
N_SEEDS = 12
BANDIT_EQ = dataclasses.replace(PAPER_EQUIVALENT, overkill=2.0,
                                abandonment_applies_to_bots=True)

# %% [markdown]
# ## 6.1 The deterministic limit is reproduced exactly
#
# Before changing anything, check that the new environment reduces to the old one. All
# six policies, all six bot volumes, four seeds each: survivor counts must be identical
# to `rlcaptcha.simulate.run_simulation`, not merely equal in distribution.

# %%
pub_h, pub_b = load_sessions()
cache_pub = ScoreCache(BotScorer()).precompute(pub_h, pub_b, verbose=False)
checks = [(DQNPolicy(epsilon=0.0), DQN_REWARDS, PAPER_EQUIVALENT),
          (DQNAblationPolicy(epsilon=0.0), DQN_REWARDS, PAPER_EQUIVALENT),
          (SingleThresholdPolicy(), DQN_REWARDS, PAPER_EQUIVALENT),
          (MultiThresholdPolicy(), DQN_REWARDS, PAPER_EQUIVALENT),
          (LinUCBPolicy(), BANDIT_REWARDS, BANDIT_EQ),
          (ThompsonPolicy(), BANDIT_REWARDS, BANDIT_EQ)]
bad = tot = 0
for pol, rc, xc in checks:
    for nb in EVAL_BOT_COUNTS:
        for s in (1, 2, 3, 4):
            torch.manual_seed(s); np.random.seed(s)
            a = run_simulation(pol, pub_h, pub_b, nb, cache=cache_pub, rewards=rc, seed=s)
            b, _ = run_x(pol, pub_h, pub_b, nb, cache=cache_pub,
                         solve=DETERMINISTIC, cfg=xc, seed=s)
            tot += 1
            bad += ((a.surviving_humans, a.surviving_bots)
                    != (b.surviving_humans, b.surviving_bots))
print(f"exact matches: {tot - bad}/{tot}")
assert bad == 0, "the stochastic environment no longer reduces to the published one"

# %% [markdown]
# ## 6.2 The level ladder
#
# What each threat level is assumed to be, and the measured numbers behind it. This table
# belongs in the revised manuscript — it is the thing that converts "eleven abstract
# levels" into "eleven mechanisms with published completion rates".

# %%
ladder = GROUNDED.describe()
ladder["Source (human side)"] = [
    "-", "reCAPTCHA v3 (passive)", "-", "Searles 2023: 3.7s median",
    "Bursztein 2010: authorize.net 0.98 / 6.8s", "Bursztein 2010: eBay 0.93 / 7.3s",
    "Bursztein 2010: Google 0.86 / 9.7s", "Bursztein 2010: reCAPTCHA 0.75 / 11.9s",
    "Bursztein 2010: Microsoft 0.80, mail.ru 0.70 / 13s",
    "Searles 2023: game-based 18-42s", "Bursztein 2010: audio 0.52",
]
ladder.to_csv(OUT / "action_model_ladder.csv", index=False)
print(ladder.to_string(index=False))

# %% [markdown]
# ## 6.3 Sweeping alpha from stochastic to deterministic
#
# Policies are held fixed (published checkpoints) so the only thing moving is the
# environment's determinism.
#
# **A note on `abandonment_tau`.** The empirical curve is
# `P_leave = 1 - exp(-seconds / tau)` and it fires at *every* decision step, whereas
# Searles et al. measure abandonment for a single CAPTCHA encounter. A 12-step session
# that shows a challenge each step therefore compounds the hazard, and a short tau makes
# any always-challenge policy look catastrophic for reasons that are an artefact of the
# step count rather than a property of the policy. We use `tau = 45s` as the default and
# sweep 22 / 45 / 80 against the published curve in notebook 04; the policy *ranking*
# does not move across that range.

# %%
ALPHAS = [0.5, 1.0, 1.5, 3.0, 6.0, 12.0]
policies = [(DQNPolicy(epsilon=0.0), PAPER_EQUIVALENT),
            (MultiThresholdPolicy(), PAPER_EQUIVALENT),
            (SingleThresholdPolicy(), PAPER_EQUIVALENT),
            (LinUCBPolicy(), BANDIT_EQ), (ThompsonPolicy(), BANDIT_EQ)]

GROUNDED_CFG = StochasticRewardConfig(abandonment="empirical", abandonment_tau=45.0,
                                      false_positive_penalty=-100.0)
rows = []
t0 = time.time()
for alpha in ALPHAS + ["deterministic"]:
    solve = DETERMINISTIC_FALLIBLE_HUMANS if alpha == "deterministic" else SolveModel(alpha=alpha)
    for pol, base_cfg in policies:
        cfg = dataclasses.replace(GROUNDED_CFG,
                                  overkill=base_cfg.overkill,
                                  abandonment_applies_to_bots=base_cfg.abandonment_applies_to_bots)
        for nb in BOT_COUNTS:
            for s in range(N_SEEDS):
                torch.manual_seed(s); np.random.seed(s)
                r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev, solve=solve,
                             cfg=cfg, seed=6000 + s)
                m = evaluate_x(r)
                rows.append({"alpha": str(alpha), "policy": pol.name, "bots": nb, "seed": s,
                             "DI": m.DI, "BOS": m.BOS, "SI_F1": m.SI_F1,
                             "humans_left": m.surviving_humans, "bots_left": m.surviving_bots,
                             "false_positive_rate": r.false_positive_rate,
                             "abandonment_rate": r.abandonment_rate,
                             "mean_friction_s": r.mean_human_friction_seconds})
alpha_df = pd.DataFrame(rows)
alpha_df.to_csv(OUT / "alpha_sweep.csv", index=False)
print(f"{len(alpha_df)} runs in {(time.time()-t0)/60:.1f} min\n")
print(alpha_df.groupby(["alpha", "policy"]).DI.mean().unstack().round(1).to_string())

# %% [markdown]
# ### What the fixed policies lose
#
# The published checkpoints were trained against the deterministic rule. Under a
# probabilistic one they face two things they never saw: a strong bot can beat a hard
# challenge, and a legitimate user can fail an easy one.

# %%
g = alpha_df.groupby("alpha")[["DI", "false_positive_rate", "abandonment_rate",
                               "mean_friction_s"]].mean().round(3)
g.index = pd.CategoricalIndex(g.index, [str(a) for a in ALPHAS] + ["deterministic"],
                              ordered=True)
print(g.sort_index().to_string())

# %% [markdown]
# ## 6.4 Retraining inside the probabilistic environment
#
# The fair comparison: an agent that was taught in the world it is tested in.

# %%
EPISODES = 90
t0 = time.time()
dqn_sto, log_sto = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                             reward_source="oracle", solve=GROUNDED, cfg=GROUNDED_CFG,
                             seed=0, name="DQN (stochastic-trained)", verbose=30)
dqn_sto.save(AGENTS / "dqn_stochastic.pt")

dqn_modern, _ = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                          reward_source="oracle", solve=GROUNDED_MODERN, cfg=GROUNDED_CFG,
                          seed=0, name="DQN (modern-solver-trained)", verbose=0)
dqn_modern.save(AGENTS / "dqn_stochastic_modern.pt")
print(f"training took {(time.time()-t0)/60:.1f} min")

# %%
compare = [(dqn_sto.eval_mode(0.0), GROUNDED, "grounded"),
           (DQNPolicy(epsilon=0.0), GROUNDED, "grounded"),
           (MultiThresholdPolicy(), GROUNDED, "grounded"),
           (SingleThresholdPolicy(), GROUNDED, "grounded"),
           (dqn_modern.eval_mode(0.0), GROUNDED_MODERN, "modern solvers"),
           (DQNPolicy(epsilon=0.0), GROUNDED_MODERN, "modern solvers"),
           (MultiThresholdPolicy(), GROUNDED_MODERN, "modern solvers")]
rows = []
for pol, solve, env in compare:
    for nb in BOT_COUNTS:
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev, solve=solve,
                         cfg=GROUNDED_CFG, seed=7000 + s)
            m = evaluate_x(r)
            rows.append({"environment": env, "policy": pol.name, "bots": nb, "seed": s,
                         "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                         "humans_left": m.surviving_humans, "bots_left": m.surviving_bots,
                         "false_positive_rate": r.false_positive_rate,
                         "mean_friction_s": r.mean_human_friction_seconds})
sto = pd.DataFrame(rows)
sto.to_csv(OUT / "stochastic_runs.csv", index=False)
summ = sto.groupby(["environment", "policy"])[
    ["DI", "BOS", "SI_F1", "false_positive_rate", "mean_friction_s"]]\
    .agg(["mean", "std"]).round(3)
print(summ.to_string())

# %% [markdown]
# ## 6.5 Action distributions
#
# A deterministic world rewards "pick the smallest T that exceeds B_s". A probabilistic
# one does not: the cost of a false positive on a legitimate user makes very high levels
# expensive even when they would stop every bot.

# %%
def action_profile(pol, solve, cfg, nb=200, seeds=6):
    from collections import Counter
    c = Counter()
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev, solve=solve, cfg=cfg, seed=8000 + s)
        c.update(r.action_hist)
    tot = sum(c.values())
    return [c.get(a, 0) / tot for a in range(11)]

profiles = {
    "DQN (published, det. env)": action_profile(DQNPolicy(epsilon=0.0), DETERMINISTIC, PAPER_EQUIVALENT),
    "DQN (published, stoch. env)": action_profile(DQNPolicy(epsilon=0.0), GROUNDED, GROUNDED_CFG),
    "DQN (stochastic-trained)": action_profile(dqn_sto, GROUNDED, GROUNDED_CFG),
    "Static Multi-Threshold": action_profile(MultiThresholdPolicy(), GROUNDED, GROUNDED_CFG),
}
prof = pd.DataFrame(profiles, index=[f"L{a}" for a in range(11)]).round(3)
prof.to_csv(OUT / "action_profiles.csv")
print(prof.to_string())

# %% [markdown]
# ## 6.6 Figures

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

ax = axes[0, 0]
lv = np.arange(11)
for bs in (2, 5, 9):
    for a, ls in ((1.5, "-"), (6.0, "--")):
        ax.plot(lv, [SolveModel(alpha=a).bot_pass_probability(bs, t) for t in lv],
                ls, label=f"B_s={bs}, alpha={a}")
ax.plot(lv, GROUNDED.human_pass, "ko-", lw=2, label="P(human passes)")
ax.set_xlabel("threat level T"); ax.set_ylabel("P(pass)")
ax.set_title("The action model"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)

ax = axes[0, 1]
g = alpha_df.groupby(["alpha", "policy"]).DI.mean().unstack()
g = g.reindex([str(a) for a in ALPHAS] + ["deterministic"])
g.plot(ax=ax, marker="o", ms=4)
ax.set_xlabel("alpha (rightmost = published rule)"); ax.set_ylabel("average DI")
ax.set_title("Published policies as the world becomes stochastic")
ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = axes[1, 0]
sub = summ.xs("grounded", level="environment")[("DI", "mean")]
err = summ.xs("grounded", level="environment")[("DI", "std")]
ax.barh(range(len(sub)), sub.values, xerr=err.values)
ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=8)
ax.set_xlabel("average DI"); ax.set_title("Grounded environment: retrained vs. published")
ax.grid(alpha=.3, axis="x")

ax = axes[1, 1]
prof.plot.bar(ax=ax, width=.8)
ax.set_ylabel("share of actions"); ax.set_xlabel("threat level")
ax.set_title("Action distribution at 200 bots"); ax.legend(fontsize=6.5); ax.grid(alpha=.3, axis="y")

plt.tight_layout(); plt.savefig(OUT / "stochastic_action_model.pdf", dpi=300)
plt.savefig(OUT / "stochastic_action_model.png", dpi=110); plt.close()

with pd.ExcelWriter(OUT / "stochastic_action_model.xlsx", engine="openpyxl") as xl:
    ladder.to_excel(xl, sheet_name="Level ladder", index=False)
    g.to_excel(xl, sheet_name="Alpha sweep")
    summ.to_excel(xl, sheet_name="Retrained comparison")
    prof.to_excel(xl, sheet_name="Action profiles")
    sto.to_excel(xl, sheet_name="Raw runs", index=False)
print("Notebook 06 complete.")
