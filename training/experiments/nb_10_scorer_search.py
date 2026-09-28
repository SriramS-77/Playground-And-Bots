# %% [markdown]
# # 10 — Finding the right scorer: architecture and hyperparameters
#
# The published scorer has **304,049 parameters fitted on 206 windows** — roughly 1,170
# parameters per training sample. That is not a model that can be expected to behave
# reproducibly, and it did not: rerunning identical code moved its window AUC from 0.910
# to 0.948.
#
# This notebook searches for a scorer matched to the amount of data that actually
# exists, evaluated by **5-fold cross-validation grouped by recording** over all 156
# sessions. Out-of-fold pooling means every recording is scored exactly once by a model
# that never saw it, giving ~520 evaluation windows instead of the 167 a single held-out
# split provided.
#
# ## Selection rule, fixed before looking at any result
#
# Written down first, deliberately, because choosing the criterion after seeing the
# numbers is how selection bias gets in — and I made exactly that mistake earlier in this
# project.
#
# 1. **Primary:** session-level AUC. This is what the simulation consumes — the scorer
#    averages over a chunk's windows, so chunk/session level is the operating point,
#    not window level.
# 2. **Among configurations within 1 standard error of the best session AUC**, prefer the
#    lowest post-temperature-scaling ECE. Calibration is not cosmetic here: the
#    multi-threshold baseline maps the score directly onto a threat level with
#    `int(score * 10)`.
# 3. **Among those**, prefer fewer parameters. Parsimony is the whole point of the
#    exercise.
#
# **One amendment, made before running anything and for a stated reason.** Notebook 09
# found that a generic model loses 0.126 session AUC and sees its ECE rise from 0.055 to
# 0.220 when tested on a collection sitting it never saw. That is a property of the
# *data* — three participants confounded with two campaigns — not of any particular
# architecture, so acting on it is not selection bias. Stage C therefore also evaluates
# each finalist leave-one-campaign-out, and a configuration that collapses there is
# rejected even if it tops the cross-validated ranking. A scorer that only works on the
# sitting it was trained on is not usable for the RL work.

# %%
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import expkit
from expkit.partition import index_sessions
from expkit.paths import RESULTS
from expkit.scorer_search import ArchSpec, PUBLISHED, cross_validate, save, to_frame

OUT = RESULTS
REFS = index_sessions()
EPOCHS, PATIENCE, N_SPLITS = 90, 12, 5
print(f"{len(REFS)} recordings, {N_SPLITS}-fold grouped by recording")

# %% [markdown]
# ## 10.1 Stage A — capacity and input representation
#
# Six architectures crossed with four input representations. The published model is
# included so every later comparison has it as a reference point rather than a memory.
#
# The representations:
#
# | name | features | what it tests |
# |---|---|---|
# | `xy` | x, y | the published input |
# | `xy_dt` | x, y, Δt | does timing help at all? |
# | `dxdy` | Δx, Δy | translation-invariant, no timing — the control |
# | `dxdy_dt` | Δx, Δy, Δt | translation-invariant plus timing |
# | `kinematic` | Δx, Δy, Δt, speed, accel | the features the manuscript claims to use |

# %%
ARCHS = [
    ArchSpec("lstm16",      (16,),      (8,)),
    ArchSpec("lstm16_8",    (16, 8),    (16,)),
    ArchSpec("lstm32",      (32,),      (32,)),
    ArchSpec("lstm32_16",   (32, 16),   (32,)),
    ArchSpec("lstm64_32",   (64, 32),   (64,)),
    PUBLISHED,
]
REPS = ["xy", "xy_dt", "dxdy", "dxdy_dt", "kinematic"]

results_a, t0 = [], time.time()
for arch in ARCHS:
    for rep in REPS:
        r = cross_validate(REFS, arch, rep, n_splits=N_SPLITS, group="session",
                           seed=0, epochs=EPOCHS, patience=PATIENCE)
        results_a.append(r)
        print(f"  {arch.name:12s} {rep:10s} params={r.params:>7,}  "
              f"winAUC={r.window_auc:.3f} sessAUC={r.session_auc:.3f} "
              f"ECE={r.ece:.3f}->{r.ece_after_T:.3f}  [{(time.time()-t0)/60:.0f}m]")
save(results_a, "stage_a")
A = to_frame(results_a)
A.to_csv(OUT / "search_stage_a.csv", index=False)

# %%
print("\nsession AUC by architecture x representation:")
print(A.pivot_table(index="spec", columns="representation",
                    values="session_auc").round(3).to_string())
print("\npost-scaling ECE:")
print(A.pivot_table(index="spec", columns="representation",
                    values="ece_after_T").round(3).to_string())
print("\nparameters per training sample (~250 windows):")
print((A.groupby("spec").params.first() / 250).round(1).to_string())

# %% [markdown]
# ## 10.2 Stage B — learning rate and regularisation
#
# The three architectures with the best mean session AUC across representations, each
# with its own best representation, swept over learning rate and dropout.

# %%
rank_arch = A.groupby("spec").session_auc.mean().sort_values(ascending=False)
top_specs = [s for s in rank_arch.index[:3]]
best_rep = {s: A[A.spec == s].sort_values("session_auc", ascending=False)
                .representation.iloc[0] for s in top_specs}
print("carried into stage B:", {s: best_rep[s] for s in top_specs})

BY_NAME = {a.name: a for a in ARCHS}
results_b = []
t0 = time.time()
for s in top_specs:
    base = BY_NAME[s]
    for lr in (3e-4, 1e-3, 3e-3):
        for drop in (0.0, 0.25):
            import dataclasses
            spec = dataclasses.replace(base, name=f"{s}_lr{lr:g}_d{drop:g}",
                                       lr=lr, dropout=drop)
            r = cross_validate(REFS, spec, best_rep[s], n_splits=N_SPLITS,
                               group="session", seed=0, epochs=EPOCHS, patience=PATIENCE)
            results_b.append(r)
            print(f"  {spec.name:26s} {best_rep[s]:10s} "
                  f"sessAUC={r.session_auc:.3f} ECE={r.ece_after_T:.3f} "
                  f"[{(time.time()-t0)/60:.0f}m]")
save(results_b, "stage_b")
B = to_frame(results_b)
B.to_csv(OUT / "search_stage_b.csv", index=False)
print("\n", B[["spec", "representation", "params", "session_auc", "window_auc",
               "ece", "ece_after_T", "temperature"]].round(3)
        .sort_values("session_auc", ascending=False).to_string(index=False))

# %% [markdown]
# ## 10.3 Stage C — is the winner stable across seeds?
#
# The single most important check, given that this project has already been bitten by
# seed instability once. A configuration that only wins on seed 0 has not won.

# %%
ALL = pd.concat([A, B], ignore_index=True)
finalists = (ALL.sort_values("session_auc", ascending=False)
                .drop_duplicates("spec").head(4))
print("finalists:\n", finalists[["spec", "representation", "params", "session_auc",
                                 "ece_after_T"]].round(3).to_string(index=False))

SPECS_BY_NAME = {a.name: a for a in ARCHS}
for r in results_b:
    import dataclasses
    base_name = r.spec.split("_lr")[0]
    lr = float(r.spec.split("_lr")[1].split("_d")[0])
    drop = float(r.spec.split("_d")[1])
    SPECS_BY_NAME[r.spec] = dataclasses.replace(SPECS_BY_NAME[base_name],
                                                name=r.spec, lr=lr, dropout=drop)

results_c, t0 = [], time.time()
for _, row in finalists.iterrows():
    for seed in (0, 1, 2):
        r = cross_validate(REFS, SPECS_BY_NAME[row.spec], row.representation,
                           n_splits=N_SPLITS, group="session", seed=seed,
                           epochs=EPOCHS, patience=PATIENCE)
        results_c.append(r)
    sub = [x for x in results_c if x.spec == row.spec]
    print(f"  {row.spec:26s} sessAUC={np.mean([x.session_auc for x in sub]):.3f}"
          f" +/-{np.std([x.session_auc for x in sub]):.3f}  "
          f"ECE={np.mean([x.ece_after_T for x in sub]):.3f}  [{(time.time()-t0)/60:.0f}m]")
save(results_c, "stage_c")
C = to_frame(results_c)
C.to_csv(OUT / "search_stage_c.csv", index=False)

stab = C.groupby(["spec", "representation"]).agg(
    params=("params", "first"),
    sess_auc=("session_auc", "mean"), sess_auc_sd=("session_auc", "std"),
    win_auc=("window_auc", "mean"), win_auc_sd=("window_auc", "std"),
    ece=("ece", "mean"), ece_T=("ece_after_T", "mean"), ece_T_sd=("ece_after_T", "std"),
    T=("temperature", "mean")).reset_index().sort_values("sess_auc", ascending=False)
print("\n", stab.round(3).to_string(index=False))

# %% [markdown]
# ### Robustness gate — leave-one-campaign-out
#
# Each finalist re-fitted with a whole collection sitting held out. This is the closest
# proxy available for an unseen participant and setup.

# %%
loco = []
for _, row in stab.iterrows():
    r = cross_validate(REFS, SPECS_BY_NAME[row.spec], row.representation,
                       n_splits=2, group="campaign", seed=0,
                       epochs=EPOCHS, patience=PATIENCE)
    loco.append({"spec": row.spec, "representation": row.representation,
                 "loco_win_auc": r.window_auc, "loco_sess_auc": r.session_auc,
                 "loco_ece": r.ece, "loco_ece_T": r.ece_after_T})
    print(f"  {row.spec:26s} LOCO sessAUC={r.session_auc:.3f} "
          f"ECE={r.ece:.3f}->{r.ece_after_T:.3f}")
L = pd.DataFrame(loco)
L.to_csv(OUT / "search_loco.csv", index=False)
stab = stab.merge(L, on=["spec", "representation"], how="left")
print("\ncross-validated vs held-out-campaign, same models:")
print(stab[["spec", "representation", "sess_auc", "loco_sess_auc",
            "ece_T", "loco_ece_T"]].round(3).to_string(index=False))
stab["optimism"] = (stab.sess_auc - stab.loco_sess_auc).round(3)

# %% [markdown]
# ## 10.4 Applying the selection rule

# %%
# The gate is applied to ALL finalists FIRST. Applying it inside the 1-SE band, as an
# earlier version did, meant that when every member of the band failed the gate the
# fallback quietly reinstated them -- the gate did nothing in exactly the case it was
# written for.
gate = stab.loco_sess_auc.max() - 0.05
survivors = stab[stab.loco_sess_auc >= gate].copy()
rejected = stab[stab.loco_sess_auc < gate]
print(f"robustness gate: LOCO session AUC >= {gate:.3f} "
      f"(best LOCO {stab.loco_sess_auc.max():.3f} minus 0.05)")
if len(rejected):
    print("\nrejected -- strong cross-validated but collapses on an unseen sitting:")
    print(rejected[["spec", "representation", "sess_auc",
                    "loco_sess_auc"]].round(3).to_string(index=False))

best = survivors.iloc[0]
se = best.sess_auc_sd / np.sqrt(3)
within = survivors[survivors.sess_auc >= best.sess_auc - se].copy()
print(f"\nbest surviving session AUC {best.sess_auc:.3f}, 1 SE = {se:.3f}  ->  "
      f"{len(within)} configuration(s) within it")
within = within.sort_values(["ece_T", "params"])
print(within[["spec", "representation", "params", "sess_auc", "sess_auc_sd",
              "ece_T", "loco_sess_auc", "optimism"]].round(3).to_string(index=False))
choice = within.iloc[0]
print(f"\nSELECTED: {choice.spec} on {choice.representation}  "
      f"({int(choice.params):,} params, session AUC {choice.sess_auc:.3f} "
      f"+/-{choice.sess_auc_sd:.3f}, ECE after scaling {choice.ece_T:.3f})")

pub_row = stab[stab.spec == "published"]
if len(pub_row) == 0:
    pub_row = ALL[ALL.spec == "published"].sort_values("session_auc",
                                                       ascending=False).head(1)
print(f"\nfor reference, published architecture: "
      f"{int(pub_row.iloc[0]['params']):,} params, "
      f"session AUC {pub_row.iloc[0].get('sess_auc', pub_row.iloc[0].get('session_auc')):.3f}")

json.dump({"spec": choice.spec, "representation": choice.representation,
           "params": int(choice.params), "session_auc": float(choice.sess_auc),
           "ece_after_T": float(choice.ece_T),
           "loco_session_auc": float(choice.loco_sess_auc),
           "loco_ece_after_T": float(choice.loco_ece_T),
           "rule": "max session AUC; within 1 SE apply the leave-one-campaign-out "
                   "robustness gate, then prefer lowest post-scaling ECE, then "
                   "fewest parameters"},
          open(OUT / "scorer_arch_choice.json", "w"), indent=2)

# %% [markdown]
# ## 10.5 Does timing actually help? — the question notebook 02 could not answer
#
# Notebook 02 compared representations on a single 52-session holdout and every interval
# straddled zero. With 5-fold cross-validation the evaluation set is ~3x larger, so the
# intervals should be ~1.8x tighter. This is where we find out whether that is enough.

# %%
rep_cmp = (A.groupby("representation")
             .agg(mean_sess_auc=("session_auc", "mean"),
                  best_sess_auc=("session_auc", "max"),
                  mean_ece_T=("ece_after_T", "mean")).round(3))
print("averaged over all six architectures:\n", rep_cmp.to_string())
print("\nper architecture, session AUC of each representation minus `xy`:")
delta = A.pivot_table(index="spec", columns="representation", values="session_auc")
print((delta.sub(delta["xy"], axis=0)).round(3).to_string())
print("""
Read the sign consistency, not the size. If a representation beats `xy` for every
architecture, that is evidence even when each individual gap is small; if the sign flips
between architectures, the differences are noise.""")

# %% [markdown]
# ## 10.6 Figures

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))

for rep in REPS:
    s = A[A.representation == rep].sort_values("params")
    ax[0].plot(s.params, s.session_auc, "o-", ms=5, label=rep)
ax[0].set_xscale("log"); ax[0].set_xlabel("parameters"); ax[0].set_ylabel("session AUC")
ax[0].set_title("Capacity vs performance"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].scatter(A.session_auc, A.ece_after_T, s=30, c=np.log10(A.params), cmap="viridis")
for _, r in A.iterrows():
    if r.params > 100000 or r.session_auc > A.session_auc.quantile(.9):
        ax[1].annotate(r.spec[:9], (r.session_auc, r.ece_after_T), fontsize=6)
ax[1].set_xlabel("session AUC"); ax[1].set_ylabel("ECE after temperature scaling")
ax[1].set_title("Discrimination vs calibration (colour = log params)")
ax[1].grid(alpha=.3)

st = stab.head(6)
x = np.arange(len(st))
ax[2].bar(x, st.sess_auc, yerr=st.sess_auc_sd, capsize=3)
ax[2].set_xticks(x); ax[2].set_xticklabels(st.spec, rotation=30, ha="right", fontsize=6.5)
ax[2].set_ylim(0.5, 1.0); ax[2].set_ylabel("session AUC (3 seeds)")
ax[2].set_title("Stability of the finalists"); ax[2].grid(alpha=.3, axis="y")

plt.tight_layout(); plt.savefig(OUT / "scorer_search.pdf", dpi=300)
plt.savefig(OUT / "scorer_search.png", dpi=110); plt.close()

with pd.ExcelWriter(OUT / "scorer_search.xlsx", engine="openpyxl") as xl:
    A.to_excel(xl, sheet_name="Stage A arch x rep", index=False)
    B.to_excel(xl, sheet_name="Stage B lr x dropout", index=False)
    stab.to_excel(xl, sheet_name="Stage C seeds", index=False)
    rep_cmp.to_excel(xl, sheet_name="Representation")
print("Notebook 10 complete.")
