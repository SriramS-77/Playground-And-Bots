# %% [markdown]
# # 2 — Behavioural scorer: timestamps, perturbation, and calibration
#
# Three reviewer threads, one notebook.
#
# **R1.7 (a) — timestamps.** The published model is fed `[[m['x'], m['y']] for m in mouse]`
# and nothing else, yet the manuscript describes acceleration and jerk features. The
# recordings *do* carry timestamps. We retrain with them and measure what they buy.
#
# **R1.7 (b) — calibration.** The score is used as a probability (the multi-threshold
# baseline maps it straight onto a threat level with `int(score * 10)`), but only
# accuracy and AUC were reported. Those measure discrimination, not calibration. We
# report ECE, MCE, Brier with its Murphy decomposition, reliability curves, and fit
# temperature scaling on held-out data.
#
# **Perturbation.** `perturb_mouse_data` applies ONE integer offset in {-1,0,1} to the
# whole session — a rigid one-pixel translation. So two simulated users drawn from the
# same recording are essentially the same user, and the effective sample size is 73, not
# 1100. We retrain with per-movement jitter instead and measure the difference.
#
# Every variant shares one protocol: trained on the `lstm` pool only, early-stopped on a
# session-disjoint slice of it, standardised with training-pool statistics, evaluated on
# the `eval` pool that neither the scorer nor the RL agent ever sees.

# %%
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import expkit
from expkit import calibration as cal
from expkit.features import N_FEATURES
from expkit.partition import load_partition
from expkit.paths import RESULTS
from expkit.scorer_train import (build_windows, load_scorer, predict_windows,
                                 split_by_session, train_scorer)

OUT = RESULTS
partition = load_partition()
print({k: len(v) for k, v in partition.items()})

# %% [markdown]
# ## 2.1 The variants
#
# | name | representation | augmentation | tests |
# |---|---|---|---|
# | `xy` | x, y | none | the published input, under a clean protocol |
# | `xy_dt` | x, y, dt | none | does timing help? |
# | `dxdy_dt` | dx, dy, dt | none | translation invariance — is it reading *position*? |
# | `kinematic` | dx, dy, dt, speed, accel | none | the features the paper claims |
# | `xy__rigid` | x, y | rigid x4 | the published augmentation |
# | `xy__per_move` | x, y | per-movement x4 | the augmentation the paper describes |

# %%
VARIANTS = [
    dict(representation="xy",        perturbation="none",     n_copies=1, name="xy"),
    dict(representation="xy_dt",     perturbation="none",     n_copies=1, name="xy_dt"),
    dict(representation="dxdy_dt",   perturbation="none",     n_copies=1, name="dxdy_dt"),
    dict(representation="kinematic", perturbation="none",     n_copies=1, name="kinematic"),
    dict(representation="xy",        perturbation="rigid",    n_copies=4, name="xy__rigid"),
    dict(representation="xy",        perturbation="per_move", n_copies=4, name="xy__per_move"),
]

# Each variant is trained THREE times. An earlier single-seed pass of this notebook gave
# xy__rigid a window AUC of 0.910; a rerun of identical code gave 0.948. Training on 206
# windows is not stable run to run, so a single fit cannot support a claim that one
# representation beats another -- and with six variants, picking the winner from one run
# each is mostly picking the luckiest seed.
SEEDS = [0, 1, 2]

metas = {}          # (variant, seed) -> ScorerResult
for v in VARIANTS:
    for sd in SEEDS:
        t0 = time.time()
        key = (v["name"], sd)
        metas[key] = train_scorer(partition["lstm"], seed=sd, epochs=80, patience=15,
                                  verbose=0, **{**v, "name": f"{v['name']}__s{sd}"})
        m = metas[key]
        print(f"{v['name']:14s} seed={sd} windows={m.n_train_windows:5d} "
              f"epochs={m.epochs_run:3d} params={m.params:,}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## 2.2 Evaluation on the held-out pool
#
# Window level is what the model optimises; session level is what the simulation
# consumes (the scorer averages over a chunk's windows). Both are reported.

# %%
def session_scores(p, groups, y):
    df = pd.DataFrame({"p": p, "g": groups, "y": y})
    agg = df.groupby("g").agg(p=("p", "mean"), y=("y", "first"))
    return agg.p.values, agg.y.values

rows, curves = [], {}
for (name, sd), meta in metas.items():
    model, std, _ = load_scorer(f"{name}__s{sd}")
    ws = build_windows(partition["eval"], meta.representation, "none", 1, seed=99)
    p = predict_windows(model, std, ws)

    win = cal.report(p, ws.y, name=name)
    sp, sy = session_scores(p, ws.groups, ws.y)
    ses = cal.report(sp, sy, name=name)

    # Temperature fitted on the lstm pool's held-out validation slice, never on eval.
    _, va_refs = split_by_session(partition["lstm"], seed=sd + 7)
    va = build_windows(va_refs, meta.representation, "none", 1, seed=1)
    pv = predict_windows(model, std, va)
    T = cal.fit_temperature(pv, va.y)
    pt = cal.apply_temperature(p, T)
    win_T = cal.report(pt, ws.y, name=name + " (T)")

    curves[(name, sd)] = (p, ws.y, pt, T)
    rows.append({
        "variant": name, "seed": sd, "features": N_FEATURES[meta.representation],
        "win_acc": win["accuracy"], "win_AUC": win["AUC"],
        "sess_acc": ses["accuracy"], "sess_AUC": ses["AUC"],
        "Brier": win["Brier"], "ECE": win["ECE_quantile"], "MCE": win["MCE_quantile"],
        "reliability": win["reliability"], "resolution": win["resolution"],
        "T": T, "ECE_after_T": win_T["ECE_quantile"], "Brier_after_T": win_T["Brier"],
        "n_eval_windows": win["n"],
    })

per_seed = pd.DataFrame(rows)
per_seed.round(4).to_csv(OUT / "scorer_variants_per_seed.csv", index=False)

res = (per_seed.groupby("variant")
       .agg(features=("features", "first"),
            win_AUC=("win_AUC", "mean"), win_AUC_sd=("win_AUC", "std"),
            sess_AUC=("sess_AUC", "mean"), sess_AUC_sd=("sess_AUC", "std"),
            win_acc=("win_acc", "mean"), sess_acc=("sess_acc", "mean"),
            Brier=("Brier", "mean"), ECE=("ECE", "mean"), ECE_sd=("ECE", "std"),
            ECE_after_T=("ECE_after_T", "mean"),
            n_eval_windows=("n_eval_windows", "first"))
       .reset_index()
       .sort_values(["sess_AUC", "ECE"], ascending=[False, True]))
res.round(4).to_csv(OUT / "scorer_variants.csv", index=False)
print("Mean over 3 training seeds (sd in the *_sd columns):\n")
print(res.round(3).to_string(index=False))
print(f"\nheld-out pool: {int(res.n_eval_windows.iloc[0])} windows from "
      f"{len(partition['eval'])} sessions -- small, see 2.3a")
print("\nSpread ACROSS TRAINING SEEDS for the same variant:")
print(per_seed.groupby("variant")[["win_AUC", "sess_AUC", "ECE"]]
      .agg(["min", "max"]).round(3).to_string())

# %% [markdown]
# ### 2.3a Are any of these differences real?
#
# The held-out pool is 167 windows from 52 sessions. Differences of a few points of AUC
# on a sample that size are not obviously distinguishable from noise, and Reviewer 1 asks
# us not to say "significantly" without testing. A paired bootstrap **over sessions**
# (not windows — windows from one session are not independent) gives the honest answer.

# %%
def variant_session_scores(name):
    """Session scores averaged over the variant's three training seeds.

    Averaging first removes seed-to-seed wobble from the comparison, so the bootstrap
    measures the effect of the REPRESENTATION rather than of one lucky fit.
    """
    ws = build_windows(partition["eval"], metas[(name, SEEDS[0])].representation,
                       "none", 1, seed=99)
    mats = [session_scores(curves[(name, sd)][0], ws.groups, ws.y) for sd in SEEDS]
    return np.mean([m[0] for m in mats], axis=0), mats[0][1]


def paired_bootstrap(name_a, name_b, n_boot=2000, seed=0):
    """Session-level AUC difference between two variants, resampling SESSIONS."""
    rng = np.random.default_rng(seed)
    sa, ya_ = variant_session_scores(name_a)
    sb, yb_ = variant_session_scores(name_b)
    out, n = [], len(sa)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(ya_[idx])) < 2:
            continue
        out.append(cal.roc_auc(sa[idx], ya_[idx]) - cal.roc_auc(sb[idx], yb_[idx]))
    out = np.array(out)
    return float(out.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


base = "xy"
print(f"Session-level AUC difference vs the published representation ({base}),")
print("3-seed ensemble, paired bootstrap over sessions, 95% CI:")
print()
boot_rows = []
for name in res.variant:
    if name == base:
        continue
    d, lo, hi = paired_bootstrap(name, base)
    verdict = "distinguishable" if (lo > 0 or hi < 0) else "not distinguishable"
    print(f"  {name:14s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    boot_rows.append({"variant": name, "delta_sess_AUC": d, "ci_lo": lo, "ci_hi": hi,
                      "distinguishable": bool(lo > 0 or hi < 0)})
pd.DataFrame(boot_rows).round(4).to_csv(OUT / "scorer_bootstrap.csv", index=False)
print("""
Read this before quoting any number above as an improvement. Two sources of noise stack
here: 52 held-out sessions, and unstable training on 206 windows. Where the interval
straddles zero the honest statement is that this dataset cannot separate the variants --
which is itself a finding, and belongs in the paper as one.""")

# %% [markdown]
# ## 2.3 Reference point: the published checkpoint
#
# `big_model.keras` consumes raw pixel coordinates (its `handle_data` adapts a
# `Normalization` layer and then returns the *unadapted* array). Scored here on the same
# held-out pool, it is not directly comparable — the eval pool overlaps its own training
# campaign — but it anchors the scale.

# %%
import tensorflow as tf
from rlcaptcha.config import HUMANITY_SCORER
from expkit.features import rep_xy, windows as mk_windows

pub = tf.keras.models.load_model(str(HUMANITY_SCORER))
ws = build_windows(partition["eval"], "xy", "none", 1, seed=99)
p_pub = pub.predict(ws.X, batch_size=256, verbose=0).ravel()   # raw, unstandardised
pub_win = cal.report(p_pub, ws.y, name="published (raw input)")
sp, sy = session_scores(p_pub, ws.groups, ws.y)
pub_ses = cal.report(sp, sy, name="published session")
print(pd.Series({
    "window acc": pub_win["accuracy"], "window AUC": pub_win["AUC"],
    "session acc": pub_ses["accuracy"], "session AUC": pub_ses["AUC"],
    "Brier": pub_win["Brier"], "ECE (quantile)": pub_win["ECE_quantile"],
}).round(3).to_string())
print("\nNOTE: part of this pool was in the published model's own training campaign,")
print("so these numbers are optimistic. They are a scale anchor, not a comparison.")

# %% [markdown]
# ## 2.4 Shortcut control — is the LSTM reading dynamics or screen position?
#
# If four summary statistics of raw position (mean and sd of x and y) get close to the
# LSTM, then the "behavioural biometrics" claim is really a claim about where on the page
# bots move. Grouped 5-fold by session so no session straddles the folds.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

Xs = np.stack([ws.X[:, :, 0].mean(1), ws.X[:, :, 1].mean(1),
               ws.X[:, :, 0].std(1),  ws.X[:, :, 1].std(1)], axis=1)
pred = np.zeros(len(ws.y))
for tr, te in GroupKFold(n_splits=5).split(Xs, ws.y, ws.groups):
    pred[te] = LogisticRegression(max_iter=2000).fit(Xs[tr], ws.y[tr]).predict_proba(Xs[te])[:, 1]

best = res.iloc[0]
print(f"position-summary baseline : AUC={roc_auc_score(ws.y, pred):.3f}  "
      f"acc={((pred >= .5) == (ws.y == 1)).mean():.3f}")
print(f"best LSTM variant ({best.variant:10s}): AUC={best.win_AUC:.3f}  acc={best.win_acc:.3f}")
print("""
A large gap means the LSTM is using sequence structure rather than absolute position,
which is the pre-emptive answer to the obvious 'it memorised screen regions' objection.""")

# %% [markdown]
# ## 2.5 Figures

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
for name in res.variant:
    p = np.mean([curves[(name, sd)][0] for sd in SEEDS], axis=0)
    y = curves[(name, SEEDS[0])][1]
    conf, obs, w = cal.reliability_curve(p, y, n_bins=10)
    ax.plot(conf, obs, "o-", ms=4, lw=1.4, label=name)
ax.set_xlabel("mean predicted P(bot)"); ax.set_ylabel("observed fraction of bots")
ax.set_title("Reliability — held-out pool, equal-mass bins")
ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = axes[1]
x = np.arange(len(res))
ax.bar(x - .2, res.ECE, .4, label="ECE raw")
ax.bar(x + .2, res.ECE_after_T, .4, label="ECE after temperature scaling")
ax.set_xticks(x); ax.set_xticklabels(res.variant, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("expected calibration error"); ax.legend(); ax.grid(alpha=.3, axis="y")
ax.set_title("Calibration before and after scaling")
plt.tight_layout(); plt.savefig(OUT / "scorer_calibration.pdf", dpi=300)
plt.savefig(OUT / "scorer_calibration.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.38
x = np.arange(len(res))
ax.bar(x - w/2, res.win_AUC, w, label="window AUC")
ax.bar(x + w/2, res.sess_AUC, w, label="session AUC")
ax.axhline(0.5, color="k", lw=.8, ls=":")
ax.set_xticks(x); ax.set_xticklabels(res.variant, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0.4, 1.02); ax.set_ylabel("AUC"); ax.legend(); ax.grid(alpha=.3, axis="y")
ax.set_title("Scorer variants on the held-out pool")
plt.tight_layout(); plt.savefig(OUT / "scorer_variants.pdf", dpi=300)
plt.savefig(OUT / "scorer_variants.png", dpi=110); plt.close()
print("figures written")

# %% [markdown]
# ## 2.6 Which scorer the rest of the series uses
#
# Selection is on **session-level AUC** — the quantity the simulation actually consumes,
# since the scorer averages over a chunk's windows — with **ECE** as the tie-break. AUC
# alone would be the wrong rule here: the top variants are within a point or two of each
# other on discrimination while differing threefold on calibration, and the multi-
# threshold baseline maps the score straight onto a threat level, so calibration is not
# cosmetic.

# %%
variant = res.iloc[0].variant
# Within the winning variant, take the seed whose session AUC is closest to that
# variant's mean -- a typical fit, not the luckiest one.
sub = per_seed[per_seed.variant == variant]
best_seed = int(sub.loc[(sub.sess_AUC - sub.sess_AUC.mean()).abs().idxmin(), "seed"])
choice = f"{variant}__s{best_seed}"

json.dump({"chosen": choice, "variant": variant, "seed": best_seed,
           "rule": "max mean session AUC over 3 seeds, ties on ECE; "
                   "then the seed closest to that variant's mean",
           "table": res.round(5).to_dict(orient="records")},
          open(OUT / "scorer_choice.json", "w"), indent=2)
print(f"chosen variant for notebooks 03+: {variant} (seed {best_seed}) -> {choice}")
print(f"baseline for comparison         : xy__s0")

# %% [markdown]
# ## 2.7 What to write in the paper
#
# **Timestamps (R1.7).** The published model never receives them. Adding them (`xy_dt`)
# changes window accuracy and calibration noticeably but session-level AUC only a
# little, and the full kinematic representation — the `dx, dy, dt, speed, accel` that the
# manuscript's acceleration-and-jerk language implies — is *not* better than raw
# coordinates on this data. Two honest conclusions follow, and both should be stated:
#
# 1. the acceleration/jerk description must be removed or rewritten, because the reported
#    model cannot compute those features from its inputs;
# 2. adding timing is a modest, not transformative, improvement here — which is itself
#    worth reporting, since it bounds how much of the signal is temporal.
#
# **Per-movement vs rigid perturbation.** This is the more consequential result. The
# published `perturb_mouse_data` applies one integer offset to the whole session, so
# augmented copies are near-duplicates. Switching to independent per-movement jitter
# changes the trained model materially — compare the `xy__rigid` and `xy__per_move` rows,
# especially their calibration. It also invalidates the humanity-score cache, because two
# users drawn from one recording genuinely differ; `expkit.xscoring.PerMoveScoreCache`
# handles that at the cost of scaling with population size.
#
# **Calibration (R1.7).** Reported above as ECE, MCE and Brier with its reliability /
# resolution split, plus temperature scaling fitted on held-out data. Whichever variant
# the paper adopts, quote the calibrated scorer or state the miscalibration.
#
# **Sample size.** 167 held-out windows from 52 sessions. Every comparison in this
# notebook should be read with the bootstrap intervals from 2.3a, not the point
# estimates. Notebook 08 pursues this.

# %%
print("Notebook 02 complete.")
