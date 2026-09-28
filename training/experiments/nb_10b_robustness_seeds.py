# %% [markdown]
# # 10b — Choosing between a strong scorer and a robust one, with error bars on both
#
# Notebook 10's robustness gate did its job and rejected the best cross-validated model:
#
# | config | CV session AUC | LOCO session AUC |
# |---|---|---|
# | `lstm32` / `dxdy_dt` | 0.973 ± 0.011 | 0.696 |
# | `lstm16_8_lr3e-4_d0.25` / `dxdy` | 0.914 ± 0.089 | 0.792 |
#
# But the LOCO figures came from **one run each**, with no uncertainty. Selecting on a
# single noisy measurement is precisely the error this project has already made three
# times. And the surviving model has a cross-validated seed spread of ±0.089 — it is
# itself unstable, so "robust" may be the wrong word for it.
#
# This notebook measures **both** axes with three seeds each, for a wider candidate set,
# and then re-applies the gate that was declared in notebook 10 — unchanged. Fixing a
# noisy *measurement* is legitimate; changing the *rule* after seeing which model it
# favours would not be.
#
# Leave-one-campaign-out has only two folds, so a "seed" here changes the weight
# initialisation and the validation slice, not the fold boundaries. That captures
# training variability but not pool variability — with two campaigns there is no way to
# capture the latter, which is itself worth stating.

# %%
import dataclasses, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import expkit
from expkit.partition import index_sessions
from expkit.paths import RESULTS
from expkit.scorer_search import ArchSpec, PUBLISHED, cross_validate

OUT = RESULTS
REFS = index_sessions()
SEEDS = (0, 1, 2)

BASE = {a.name: a for a in [
    ArchSpec("lstm16", (16,), (8,)), ArchSpec("lstm16_8", (16, 8), (16,)),
    ArchSpec("lstm32", (32,), (32,)), ArchSpec("lstm32_16", (32, 16), (32,)),
    ArchSpec("lstm64_32", (64, 32), (64,)), PUBLISHED]}

# Candidates: the notebook-10 finalists plus the strongest stage-A cells that stage B
# never reached, so the comparison is not confined to one corner of the grid.
CANDIDATES = [
    (dataclasses.replace(BASE["lstm32"], name="lstm32"), "dxdy_dt"),
    (dataclasses.replace(BASE["lstm32"], name="lstm32"), "dxdy"),
    (dataclasses.replace(BASE["lstm16_8"], name="lstm16_8"), "dxdy"),
    (dataclasses.replace(BASE["lstm16_8"], name="lstm16_8_lr3e-4_d0.25",
                         lr=3e-4, dropout=0.25), "dxdy"),
    (dataclasses.replace(BASE["lstm32_16"], name="lstm32_16"), "kinematic"),
    (dataclasses.replace(BASE["lstm64_32"], name="lstm64_32"), "kinematic"),
    (dataclasses.replace(BASE["published"], name="published"), "xy"),
]
print(f"{len(CANDIDATES)} candidates x {len(SEEDS)} seeds x 2 schemes")

# %% [markdown]
# ## 10b.1 Cross-validated and held-out-campaign, three seeds each

# %%
rows, t0 = [], time.time()
for spec, rep in CANDIDATES:
    for seed in SEEDS:
        for group, n_splits in (("session", 5), ("campaign", 2)):
            r = cross_validate(REFS, spec, rep, n_splits=n_splits, group=group,
                               seed=seed, epochs=90, patience=12)
            rows.append({"spec": spec.name, "representation": rep, "seed": seed,
                         "scheme": "CV" if group == "session" else "LOCO",
                         "params": r.params, "session_auc": r.session_auc,
                         "window_auc": r.window_auc, "ece": r.ece,
                         "ece_T": r.ece_after_T})
    print(f"  {spec.name:22s} {rep:10s}  [{(time.time()-t0)/60:.0f}m]")

rb = pd.DataFrame(rows)
rb.to_csv(OUT / "robustness_seeds.csv", index=False)

piv = rb.pivot_table(index=["spec", "representation", "params"], columns="scheme",
                     values="session_auc", aggfunc=["mean", "std"]).round(3)
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv.reset_index().sort_values("mean_CV", ascending=False)
ece = rb.pivot_table(index=["spec", "representation"], columns="scheme",
                     values="ece_T", aggfunc="mean").round(3)
ece.columns = [f"ece_{c}" for c in ece.columns]
tab = piv.merge(ece.reset_index(), on=["spec", "representation"])
tab["optimism"] = (tab.mean_CV - tab.mean_LOCO).round(3)
print("\n", tab.to_string(index=False))

# %% [markdown]
# ## 10b.2 Re-applying the notebook-10 gate to the better measurement
#
# Rule unchanged: reject anything whose held-out-campaign session AUC is more than 0.05
# below the best; among survivors take those within 1 SE of the best cross-validated
# AUC; then lowest post-scaling ECE, then fewest parameters.

# %%
gate = tab.mean_LOCO.max() - 0.05
surv = tab[tab.mean_LOCO >= gate].copy()
rej = tab[tab.mean_LOCO < gate]
print(f"gate: LOCO session AUC >= {gate:.3f}\n")
if len(rej):
    print("rejected:")
    print(rej[["spec", "representation", "mean_CV", "mean_LOCO",
               "std_LOCO"]].to_string(index=False))
print(f"\n{len(surv)} survivor(s):")
print(surv[["spec", "representation", "params", "mean_CV", "std_CV",
            "mean_LOCO", "std_LOCO", "ece_CV"]].to_string(index=False))

best = surv.sort_values("mean_CV", ascending=False).iloc[0]
se = best.std_CV / np.sqrt(len(SEEDS))
band = surv[surv.mean_CV >= best.mean_CV - se].sort_values(["ece_CV", "params"])
print(f"\nwithin 1 SE ({se:.3f}) of the best survivor: {len(band)}")
choice = band.iloc[0]
print(f"\nSELECTED: {choice.spec} on {choice.representation} "
      f"({int(choice.params):,} params)")
print(f"  cross-validated : {choice.mean_CV:.3f} +/- {choice.std_CV:.3f}")
print(f"  unseen campaign : {choice.mean_LOCO:.3f} +/- {choice.std_LOCO:.3f}")
print(f"  ECE (CV / LOCO) : {choice.ece_CV:.3f} / {choice.ece_LOCO:.3f}")

json.dump({"spec": choice.spec, "representation": choice.representation,
           "params": int(choice.params),
           "cv_session_auc": float(choice.mean_CV), "cv_sd": float(choice.std_CV),
           "loco_session_auc": float(choice.mean_LOCO), "loco_sd": float(choice.std_LOCO),
           "ece_cv": float(choice.ece_CV), "ece_loco": float(choice.ece_LOCO),
           "rule": "notebook-10 gate, re-applied to 3-seed measurements"},
          open(OUT / "scorer_arch_choice.json", "w"), indent=2)

# %% [markdown]
# ## 10b.3 Does timing help? — the control now settles it
#
# `dxdy_dt` changes two things against `xy` at once: it drops absolute screen position
# **and** adds inter-event timing. `dxdy` isolates the first. Comparing the three across
# all six architectures separates them.

# %%
A = pd.read_csv(OUT / "search_stage_a.csv")
d = A.pivot_table(index="spec", columns="representation", values="session_auc")
cmp = pd.DataFrame({
    "dxdy - xy  (drop absolute position)": d["dxdy"] - d["xy"],
    "dxdy_dt - dxdy  (add timing on top)": d["dxdy_dt"] - d["dxdy"],
    "xy_dt - xy  (add timing, keep position)": d["xy_dt"] - d["xy"],
}).round(3)
print(cmp.to_string())
print("\nsummary across the six architectures:")
for c in cmp.columns:
    v = cmp[c]
    print(f"  {c:42s} mean {v.mean():+.3f}   positive in {int((v > 0).sum())}/6")
cmp.to_csv(OUT / "timing_vs_position.csv")
print("""
Read the consistency column. A change that helps for every architecture is evidence even
when each gap is modest; a change whose sign flips between architectures is noise.""")

# %% [markdown]
# ## 10b.4 Figures

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))

ax[0].errorbar(tab.mean_LOCO, tab.mean_CV, xerr=tab.std_LOCO, yerr=tab.std_CV,
               fmt="o", ms=6, capsize=3)
for _, r in tab.iterrows():
    ax[0].annotate(f"{r.spec[:12]}/{r.representation[:7]}", (r.mean_LOCO, r.mean_CV),
                   fontsize=6, xytext=(4, 3), textcoords="offset points")
lim = [min(tab.mean_LOCO.min(), tab.mean_CV.min()) - .05, 1.0]
ax[0].plot(lim, lim, "k--", lw=1, label="no optimism")
ax[0].axvline(gate, color="crimson", ls=":", lw=1.2, label="robustness gate")
ax[0].set_xlabel("session AUC, unseen campaign"); ax[0].set_ylabel("session AUC, cross-validated")
ax[0].set_title("Strong vs robust"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

x = np.arange(len(cmp))
w = 0.26
for i, c in enumerate(cmp.columns):
    ax[1].bar(x + (i - 1) * w, cmp[c], w, label=c.split("  ")[0])
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels(cmp.index, rotation=30, ha="right", fontsize=6.5)
ax[1].set_ylabel("Δ session AUC"); ax[1].set_title("Position vs timing")
ax[1].legend(fontsize=6.5); ax[1].grid(alpha=.3, axis="y")

t = tab.sort_values("params")
ax[2].errorbar(t.params, t.mean_CV, yerr=t.std_CV, fmt="o-", capsize=3, label="cross-validated")
ax[2].errorbar(t.params, t.mean_LOCO, yerr=t.std_LOCO, fmt="s--", capsize=3, label="unseen campaign")
ax[2].set_xscale("log"); ax[2].set_xlabel("parameters"); ax[2].set_ylabel("session AUC")
ax[2].set_title("Capacity vs both measures"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig(OUT / "robustness_seeds.pdf", dpi=300)
plt.savefig(OUT / "robustness_seeds.png", dpi=110); plt.close()
print("Notebook 10b complete.")
