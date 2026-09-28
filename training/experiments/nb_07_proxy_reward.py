# %% [markdown]
# # 7 — A reward the system could actually have
#
# **Reviewer 1, point 1 (second half):** *"The paper should also explain what generates
# the reward in deployment: the simulator knows the true user class and bot strength, but
# a deployed controller normally knows neither at decision time."*
#
# There is no textual answer to this. Either the reward is an oracle the system cannot
# have, or the paper shows that a reward built from observable signals reaches a
# comparable policy. This notebook does the latter.
#
# ## What a deployment observes
#
# Immediately, per challenge: which level was shown, whether it was completed, how long
# it took, whether the session continued. Crucially the immediate signal is **ambiguous
# about class**:
#
# | observation | could be |
# |---|---|
# | challenge failed, session ends | a blocked bot **or** a false-positived human |
# | challenge passed, session continues | a leaked bot **or** a satisfied human |
#
# Later, and only sometimes: a confirmed-abuse label (chargeback, account-takeover
# review, spam report). It covers a fraction of the sessions that deserved it, arrives
# with a delay, and is occasionally wrong. That is the only channel through which class
# information ever reaches the agent — which is how fraud systems are actually trained:
# sparse, delayed, noisy confirmation rather than ground truth at decision time.
#
# `expkit.proxy` enforces the ambiguity by construction: reward is a pure function of the
# observable tuple, so the two outcomes behind each observation receive identical
# immediate reward. The proxy agent also cannot balance its replay buffers by true class,
# so it balances on the observable outcome instead.

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
from expkit.proxy import OBSERVABLE, ProxyRewardConfig, summarise_observability
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import GROUNDED, StochasticRewardConfig
from expkit.trainer import train_dqn
from expkit.xscoring import XScorer
from rlcaptcha.scoring import ScoreCache
from rlcaptcha.policies import DQNPolicy, MultiThresholdPolicy, SingleThresholdPolicy

OUT = RESULTS
partition = load_partition()
chosen = json.load(open(OUT / "scorer_choice.json"))["chosen"]
ev_h, ev_b = sessions_from_refs(partition["eval"])
rl_h, rl_b = sessions_from_refs(partition["rl"])
scorer = XScorer.from_name(chosen)
cache_ev = ScoreCache(scorer).precompute(ev_h, ev_b, verbose=False)
cache_rl = ScoreCache(scorer).precompute(rl_h, rl_b, verbose=False)

GROUNDED_CFG = StochasticRewardConfig(abandonment="empirical", abandonment_tau=45.0,
                                      false_positive_penalty=-100.0)
BOT_COUNTS = (0, 20, 100, 200, 500, 1000)
N_SEEDS = 12
EPISODES = 90

# %% [markdown]
# ## 7.1 How ambiguous is the immediate signal?
#
# Run the published policy once and count what sits behind each observable class. A value
# near 0.5 would mean the immediate signal carries no class information at all.

# %%
_, tr = run_x(DQNPolicy(epsilon=0.0), ev_h, ev_b, 200, cache=cache_ev,
              solve=GROUNDED, cfg=GROUNDED_CFG, seed=11, collect=True)
obs = summarise_observability(tr)
rows = []
for k, v in obs.items():
    row = {"observation": k, "n": v["n"]}
    row.update({kk: round(vv, 3) for kk, vv in v.items() if kk != "n"})
    rows.append(row)
amb = pd.DataFrame(rows).fillna(0)
amb.to_csv(OUT / "observability.csv", index=False)
print(amb.to_string(index=False))
print("""
'failed_gone' mixes blocked bots with false-positived humans; 'passed_continued' mixes
leaked bots with satisfied humans. An operator seeing only the immediate signal cannot
separate them, which is exactly the constraint the proxy reward has to work under.""")

# %% [markdown]
# ## 7.2 Train an oracle agent and a proxy agent in the same world
#
# Identical network, hyperparameters, session pool, environment and seed. The only
# difference is where the reward comes from.

# %%
t0 = time.time()
oracle, log_o = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                          reward_source="oracle", solve=GROUNDED, cfg=GROUNDED_CFG,
                          seed=0, name="DQN (oracle reward)", verbose=30)
oracle.save(AGENTS / "dqn_oracle_grounded.pt")

proxy_cfg = ProxyRewardConfig()
proxied, log_p = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                           reward_source="proxy", solve=GROUNDED, cfg=GROUNDED_CFG,
                           proxy_cfg=proxy_cfg, seed=0, name="DQN (proxy reward)",
                           verbose=30)
proxied.save(AGENTS / "dqn_proxy_grounded.pt")
print(f"training took {(time.time()-t0)/60:.1f} min")
lp = log_p.frame()
print(f"delayed abuse labels fired: {lp.labels_fired.sum():,} across {EPISODES} episodes "
      f"({lp.labels_fired.mean():.1f} per episode)")

# %% [markdown]
# ## 7.3 Evaluate both against the TRUE objective
#
# Both agents are scored with the same ground-truth metrics on the held-out pool. The
# proxy agent never saw those labels during training; it is only judged by them.

# %%
def sweep(pols, tag_field="policy"):
    rows = []
    for pol in pols:
        for nb in BOT_COUNTS:
            for s in range(N_SEEDS):
                torch.manual_seed(s); np.random.seed(s)
                r, _ = run_x(pol, ev_h, ev_b, nb, cache=cache_ev, solve=GROUNDED,
                             cfg=GROUNDED_CFG, seed=9000 + s)
                m = evaluate_x(r)
                rows.append({tag_field: pol.name, "bots": nb, "seed": s,
                             "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                             "humans_left": m.surviving_humans,
                             "bots_left": m.surviving_bots,
                             "false_positive_rate": r.false_positive_rate,
                             "mean_friction_s": r.mean_human_friction_seconds})
    return pd.DataFrame(rows)

comp = sweep([oracle.eval_mode(0.0), proxied.eval_mode(0.0), DQNPolicy(epsilon=0.0),
              MultiThresholdPolicy(), SingleThresholdPolicy()])
comp.to_csv(OUT / "proxy_vs_oracle_runs.csv", index=False)
summ = comp.groupby("policy")[["DI", "BOS", "SP_F1", "SI_F1",
                               "false_positive_rate", "mean_friction_s"]]\
           .agg(["mean", "std"]).round(3)
print(summ.to_string())

# %% [markdown]
# ### The zero-bot cell has to come out of the average
#
# Reviewer 1, point 4 objects that `S_B = B_surv / B_total` is undefined with no bots,
# yet DI, BOS and SI-F1 are still reported for that condition. This experiment shows
# exactly why that matters: at zero bots **any policy that simply lets everyone through
# scores DI = 100**, because there are no bots to fail to block. Averaging that cell in
# rewards permissiveness.
#
# So the headline number is the average over the five volumes that actually contain
# bots. The all-six average is kept alongside it only because it is the paper's current
# convention.

# %%
all_six = comp.groupby("policy").DI.mean()
with_bots = comp[comp.bots > 0].groupby("policy").DI.mean()
surv = comp[comp.bots > 0].groupby("policy")[["humans_left", "bots_left"]].mean()
head = pd.DataFrame({
    "avg DI (all 6 volumes)": all_six.round(1),
    "avg DI (5 with bots)": with_bots.round(1),
    "humans left": surv.humans_left.round(1),
    "bots left": surv.bots_left.round(1),
    "false positive rate": comp[comp.bots > 0].groupby("policy")
                               .false_positive_rate.mean().round(3),
    "friction (s/user)": comp[comp.bots > 0].groupby("policy")
                             .mean_friction_s.mean().round(1),
})
head.to_csv(OUT / "proxy_headline.csv")
print(head.to_string())

o_all, p_all = all_six["DQN (oracle reward)"], all_six["DQN (proxy reward)"]
o, p = with_bots["DQN (oracle reward)"], with_bots["DQN (proxy reward)"]
print(f"\nincluding the zero-bot cell : proxy / oracle = {p_all / o_all:.0%}  "
      f"<- inflated, do not quote")
print(f"excluding it                : proxy / oracle = {p / o:.0%}  ({p:.1f} vs {o:.1f})")

# %% [markdown]
# ### What the proxy agent actually learned
#
# Read the survivor columns, not just DI. The proxy agent keeps **every** legitimate user
# — zero false positives, zero friction seconds — and pays for it by leaving far more
# bots alive. That is not a bug in the reward; it is the structural bias the reward has:
#
# * friction and abandonment are observed **immediately and exactly**, on every session;
# * abuse is observed **rarely, late and noisily**, through a label that fires for only
#   `label_probability` of the sessions that deserved it.
#
# An agent optimising that signal will systematically under-weight security relative to
# usability. That is a real and reportable property of deployable reward design, and it
# is what Section 7.4 measures: how much label confirmation is needed before the balance
# is restored.

# %%
share_zero = (comp.bots == 0).mean()
print(f"zero-bot cells are {share_zero:.0%} of the runs behind the all-six average\n")
for pol in ("DQN (oracle reward)", "DQN (proxy reward)"):
    sub = comp[comp.policy == pol].groupby("bots")[["humans_left", "bots_left"]].mean()
    print(f"{pol}:")
    print(sub.round(1).to_string(), "\n")

# %% [markdown]
# ## 7.4 How much confirmation does it need?
#
# `label_probability` is the fraction of genuinely abusive sessions that ever get
# confirmed. Production values are low. This sweep says how low the paper can honestly
# claim to go.

# %%
LABEL_P = [0.02, 0.05, 0.10, 0.30, 0.60, 1.00]
rows = []
t0 = time.time()
for lp_ in LABEL_P:
    cfgp = dataclasses.replace(proxy_cfg, label_probability=lp_)
    agent, _ = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                         reward_source="proxy", solve=GROUNDED, cfg=GROUNDED_CFG,
                         proxy_cfg=cfgp, seed=0, name=f"proxy p={lp_}", verbose=0)
    agent.eval_mode(0.0)
    for nb in BOT_COUNTS:
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            r, _ = run_x(agent, ev_h, ev_b, nb, cache=cache_ev, solve=GROUNDED,
                         cfg=GROUNDED_CFG, seed=9500 + s)
            m = evaluate_x(r)
            rows.append({"label_probability": lp_, "bots": nb, "seed": s, "DI": m.DI,
                         "BOS": m.BOS, "SI_F1": m.SI_F1,
                         "humans_left": m.surviving_humans, "bots_left": m.surviving_bots,
                         "false_positive_rate": r.false_positive_rate})
    print(f"  p={lp_:.2f} done ({(time.time()-t0)/60:.1f} min)")
lab = pd.DataFrame(rows)
lab.to_csv(OUT / "label_probability_sweep.csv", index=False)
lab_b = lab[lab.bots > 0]          # same exclusion as the headline above
g = lab_b.groupby("label_probability")[["DI", "BOS", "SI_F1", "humans_left",
                                        "bots_left", "false_positive_rate"]].mean().round(3)
print("\n", g.to_string())
print(f"\noracle reference: DI={o:.1f}")

# %% [markdown]
# ## 7.5 The label sweep is the wrong axis — and here is why
#
# Section 7.4 varies how often abuse is confirmed, and it barely moves the agent: even
# with **every** abusive session confirmed, it still leaves ~250 bots alive and
# challenges nobody. That is not a failure of label coverage. It is arithmetic in the
# reward weights.
#
# Print the immediate reward the agent can see, per action:

# %%
from expkit.proxy import immediate_reward
from expkit.stochastic import HUMAN_SECONDS

print("%-6s %10s %12s %11s" % ("level", "pass+stay", "pass+leave", "fail+gone"))
for t in (0, 2, 3, 5, 7, 10):
    s = HUMAN_SECONDS[t]
    print("%-6d %10.1f %12.1f %11.1f" % (
        t, immediate_reward(t, "human_passed", s, proxy_cfg),
        immediate_reward(t, "human_abandoned", s, proxy_cfg),
        immediate_reward(t, "bot_blocked", s, proxy_cfg)))

gap = (immediate_reward(0, "human_passed", 0.0, proxy_cfg)
       - immediate_reward(2, "human_passed", HUMAN_SECONDS[2], proxy_cfg))
print(f"\nchallenging costs {gap:.0f} reward per step, immediately and with certainty.")
print(f"the abuse penalty is {proxy_cfg.abuse_penalty:.0f} spread over "
      f"{proxy_cfg.label_delay_steps} steps, fires with probability "
      f"{proxy_cfg.label_probability}, and arrives only at session end.")
need = gap * proxy_cfg.label_delay_steps / proxy_cfg.label_probability
print(f"\nbreak-even (undiscounted): the penalty must exceed {need:.0f} in magnitude "
      f"before blocking is worth more than doing nothing.")
print(f"at the default it is {abs(proxy_cfg.abuse_penalty):.0f} -- roughly "
      f"{need / abs(proxy_cfg.abuse_penalty):.0f}x too small.")

# %% [markdown]
# So the prediction is that the agent should start defending once the abuse penalty
# passes roughly that threshold, at *fixed* label coverage. Testing it:

# %%
PENALTIES = [-150.0, -600.0, -2400.0, -9600.0]
rows = []
t0 = time.time()
for pen in PENALTIES:
    cfgp = dataclasses.replace(proxy_cfg, abuse_penalty=pen, label_probability=0.30)
    agent, _ = train_dqn(rl_h, rl_b, cache_rl, episodes=EPISODES, use_score=True,
                         reward_source="proxy", solve=GROUNDED, cfg=GROUNDED_CFG,
                         proxy_cfg=cfgp, seed=0, name=f"proxy R={pen}", verbose=0)
    agent.eval_mode(0.0)
    for nb in BOT_COUNTS:
        if nb == 0:
            continue
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            r, _ = run_x(agent, ev_h, ev_b, nb, cache=cache_ev, solve=GROUNDED,
                         cfg=GROUNDED_CFG, seed=9800 + s)
            m = evaluate_x(r)
            rows.append({"abuse_penalty": pen, "bots": nb, "DI": m.DI,
                         "humans_left": m.surviving_humans,
                         "bots_left": m.surviving_bots,
                         "false_positive_rate": r.false_positive_rate,
                         "friction_s": r.mean_human_friction_seconds})
    print(f"  R={pen:>8.0f} done ({(time.time()-t0)/60:.1f} min)")

pen_df = pd.DataFrame(rows)
pen_df.to_csv(OUT / "abuse_penalty_sweep.csv", index=False)
gp = pen_df.groupby("abuse_penalty")[["DI", "humans_left", "bots_left",
                                      "false_positive_rate", "friction_s"]].mean().round(2)
print("\n", gp.to_string())
print(f"\noracle reference: DI={o:.1f}, humans {surv.loc['DQN (oracle reward)','humans_left']:.0f}, "
      f"bots {surv.loc['DQN (oracle reward)','bots_left']:.0f}")

# %% [markdown]
# ### What this means for deployable reward design
#
# The hard part is **not** getting enough confirmed-abuse labels. It is weighting a
# signal that is observed immediately and exactly (friction, abandonment) against one
# that is sparse, delayed and probabilistic (abuse). Get that ratio wrong in the obvious
# direction and the agent quietly converges on never challenging anybody — which looks
# excellent on every usability metric and is worthless as a defence.
#
# That is a concrete, checkable design rule for the paper, and it is more useful than the
# result we expected to report.

# %% [markdown]
# ## 7.6 Figures

# %%
fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))

ax = axes[0]
for log, lbl in ((log_o, "oracle reward"), (log_p, "proxy reward")):
    d = log.frame()
    ax.plot(d.episodes, d.loss.rolling(5, min_periods=1).mean(), lw=1.3, label=lbl)
ax.set_yscale("log"); ax.set_xlabel("episode"); ax.set_ylabel("Huber loss (5-ep mean)")
ax.set_title("Training"); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[1]
order = ["DQN (oracle reward)", "DQN (proxy reward)", "DQN",
         "Static Multi-Threshold", "Static Single-Threshold"]
m = with_bots.reindex(order); e = comp[comp.bots > 0].groupby("policy").DI.std().reindex(order)
ax.barh(range(len(order)), m.values, xerr=e.values,
        color=["tab:green", "tab:orange", "tab:blue", "tab:grey", "tab:grey"])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o_.replace(" (", "\n(") for o_ in order], fontsize=7)
ax.set_xlabel("average DI (bot-bearing volumes only)")
ax.set_title("Judged on the true objective")
ax.grid(alpha=.3, axis="x")

ax = axes[2]
ax.plot(g.index, g.DI, "o-", label="proxy-trained DQN")
ax.axhline(o, color="tab:green", ls="--", lw=1.2, label="oracle-trained DQN")
ax.axhline(with_bots[["Static Multi-Threshold", "Static Single-Threshold"]].max(),
           color="grey", ls=":", lw=1.2, label="best static baseline")
ax.set_xscale("log"); ax.set_xlabel("P(abusive session ever confirmed)")
ax.set_ylabel("average DI"); ax.set_title("How much label confirmation is needed")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[3]
ax.plot(-gp.index.values, gp.bots_left.values, "o-", color="crimson", label="bots left")
ax.axhline(surv.loc["DQN (oracle reward)", "bots_left"], color="tab:green", ls="--",
           lw=1.2, label="oracle-trained DQN")
ax.axvline(need, color="k", ls=":", lw=1, label="predicted break-even")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("abuse penalty magnitude"); ax.set_ylabel("bots left alive")
ax.set_title("Weighting, not label coverage, is the binding constraint")
ax.legend(fontsize=7); ax.grid(alpha=.3)

plt.tight_layout(); plt.savefig(OUT / "proxy_reward.pdf", dpi=300)
plt.savefig(OUT / "proxy_reward.png", dpi=110); plt.close()

with pd.ExcelWriter(OUT / "proxy_reward.xlsx", engine="openpyxl") as xl:
    amb.to_excel(xl, sheet_name="Observability", index=False)
    head.to_excel(xl, sheet_name="Headline")
    summ.to_excel(xl, sheet_name="Proxy vs oracle")
    g.to_excel(xl, sheet_name="Label probability sweep")
    gp.to_excel(xl, sheet_name="Abuse penalty sweep")
    comp.to_excel(xl, sheet_name="Raw runs", index=False)
print("Notebook 07 complete.")
