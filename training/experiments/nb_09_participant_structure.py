# %% [markdown]
# # 9 — Three participants: what that does to the evidence
#
# The 44 human recordings come from **3 participants**. That is now a known fact rather
# than an inference, and it changes what the data can support. This notebook works out
# how, and decides the cross-validation scheme the rest of the scorer work uses.
#
# ## The concern, stated precisely
#
# The worry is *not* that three people produce unnaturally **high** variance between
# sessions. It is the opposite. Two things follow from a small participant pool, and
# they pull in the same direction:
#
# 1. **Sessions from one person are not independent.** Treating 44 recordings as 44
#    independent human samples overstates the evidence. The effective number of
#    independent human samples is closer to 3.
# 2. **The human class becomes a few behavioural prototypes.** A classifier can learn
#    "human = one of these three motion styles" and score well on held-out sessions from
#    the same three people while failing on a fourth. Held-out accuracy then measures
#    *within-person* generalisation, which is an easier problem than the one the paper
#    claims to solve.
#
# Neither is fatal. Both need measuring, and both change how results should be reported.

# %%
import json, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import expkit
from expkit.partition import index_sessions
from expkit.paths import RESULTS
from expkit.scorer_search import ArchSpec, cross_validate

OUT = RESULTS
N_PARTICIPANTS = 3

# %% [markdown]
# ## 9.1 Behavioural fingerprint per recording
#
# Fifteen features describing *how* the mouse moved, not where. If three participants
# produce three distinguishable styles, these should cluster into three groups.

# %%
def features(mv):
    x = np.array([m["x"] for m in mv], float)
    y = np.array([m["y"] for m in mv], float)
    t = np.array([m["timestamp"] for m in mv], float)
    dx, dy = np.diff(x), np.diff(y)
    dt = np.maximum(np.diff(t), 1.0)
    step = np.hypot(dx, dy)
    speed = step / dt
    acc = np.diff(speed) / dt[1:] if len(speed) > 1 else np.array([0.0])
    ang = np.arctan2(dy, dx)
    dang = np.abs(np.diff(np.unwrap(ang))) if len(ang) > 1 else np.array([0.0])
    path, net = step.sum(), np.hypot(x[-1] - x[0], y[-1] - y[0])
    return dict(
        sp_mean=speed.mean(), sp_std=speed.std(), sp_p90=np.percentile(speed, 90),
        step_mean=step.mean(), step_std=step.std(), acc_absmean=np.abs(acc).mean(),
        dt_median=np.median(dt), dt_std=dt.std(), pause_frac=(dt > 100).mean(),
        curv_mean=dang.mean(), curv_std=dang.std(),
        straight=net / path if path > 0 else 0.0,
        x_range=np.ptp(x), y_range=np.ptp(y), n_moves=len(mv),
    )

rows = []
for r in index_sessions():
    if r.is_bot:
        continue
    mv = json.loads(Path(r.path).read_text()).get("mouse_movements") or []
    if len(mv) < 30:
        continue
    rows.append({"name": r.name, "campaign": r.campaign, **features(mv)})
hum = pd.DataFrame(rows)
FEATS = [c for c in hum.columns if c not in ("name", "campaign")]
print(f"human recordings analysed: {len(hum)}  "
      f"(campaign A {(hum.campaign=='A').sum()}, campaign B {(hum.campaign=='B').sum()})")
print(f"stated participants: {N_PARTICIPANTS}  ->  ~{len(hum)/N_PARTICIPANTS:.0f} sessions each")

# %% [markdown]
# ## 9.2 Do the recordings separate into three groups?
#
# Ward hierarchical clustering, scored by **silhouette** — a number in [-1, 1] measuring
# how much closer each point is to its own cluster than to the nearest other cluster.
# Above ~0.5 is strong structure; 0.25-0.5 is weak; near 0 means the clusters are
# arbitrary cuts through one blob.

# %%
X = StandardScaler().fit_transform(hum[FEATS].values)
Z = linkage(X, method="ward")

sil = []
for k in range(2, 8):
    lab = fcluster(Z, k, criterion="maxclust")
    sil.append({"k": k, "silhouette": silhouette_score(X, lab),
                "sizes": list(np.bincount(lab)[1:])})
sil_df = pd.DataFrame(sil)
print(sil_df.to_string(index=False))

hum["cluster3"] = fcluster(Z, 3, criterion="maxclust")
print("\nk=3 clustering vs collection campaign:")
print(pd.crosstab(hum.cluster3, hum.campaign).to_string())

# %% [markdown]
# Read the silhouette table carefully. The k=2 score is high only because it isolates
# two outlier recordings against everything else — a size-2 cluster is outlier
# detection, not a participant split. At k=3 the score is weak (~0.25), and the split
# lines up substantially with **collection campaign**, not with anything we can
# independently verify as person identity.
#
# So: with three participants recorded across two sittings three months apart,
# **participant and session-date are confounded.** We cannot tell "this is participant 2"
# from "this is the November setup" — different screen, browser, pointing device and
# sampling rate all move the same features.

# %%
print("features separating the k=3 clusters most strongly "
      "(between-group / within-group variance):")
ratios = {}
for c in FEATS:
    gm = hum[c].mean()
    between = sum(((hum[hum.cluster3 == g][c].mean() - gm) ** 2) * (hum.cluster3 == g).sum()
                  for g in (1, 2, 3)) / 2
    within = sum(((hum[hum.cluster3 == g][c] - hum[hum.cluster3 == g][c].mean()) ** 2).sum()
                 for g in (1, 2, 3)) / (len(hum) - 3)
    ratios[c] = between / within if within > 0 else np.inf
rat = pd.Series(ratios).sort_values(ascending=False)
print(rat.head(6).round(1).to_string())
print("""
These are sampling-rate and pausing features -- how often the mouse reported its
position and how long it sat still. Those change with hardware and browser as readily as
with the person, which is exactly the confound above.""")

# %% [markdown]
# ## 9.3 What it costs: session-level vs campaign-level validation
#
# This is the measurement that matters. The same model, evaluated two ways:
#
# * **session-level 5-fold** — a fold boundary is a recording. Held-out recordings come
#   from the *same three people* the model trained on. Optimistic.
# * **leave-one-campaign-out** — a fold boundary is an entire collection sitting. The
#   model is tested on a setup it never saw. Pessimistic.
#
# The gap between them is the part of reported performance that comes from having seen
# these particular people and this particular setup.

# %%
refs = index_sessions()
spec = ArchSpec("probe", lstm_units=(32, 16), dense_units=(32,), lr=1e-3)

probe = {}
for group, label in (("session", "session-level 5-fold"),
                     ("campaign", "leave-one-campaign-out")):
    r = cross_validate(refs, spec, "xy", n_splits=5, group=group, seed=0, epochs=60)
    probe[label] = r
    print(f"{label:24s} win AUC={r.window_auc:.3f}  sess AUC={r.session_auc:.3f}  "
          f"ECE={r.ece:.3f}  ({r.seconds:.0f}s)")

gap_w = probe["session-level 5-fold"].window_auc - probe["leave-one-campaign-out"].window_auc
gap_s = probe["session-level 5-fold"].session_auc - probe["leave-one-campaign-out"].session_auc
print(f"\noptimism of session-level splitting: {gap_w:+.3f} window AUC, "
      f"{gap_s:+.3f} session AUC")

# %% [markdown]
# ## 9.4 Effective sample size
#
# A blunt but useful way to state the limitation. For the bot side, 112 recordings come
# from 4 scripted generators, so the number of independent *behaviours* is 4, not 112.
# For the human side it is 3, not 44.

# %%
inv = pd.read_csv(OUT / "session_inventory.csv") if (OUT / "session_inventory.csv").exists() else None
eff = pd.DataFrame([
    {"class": "human", "recordings": int(len(hum)), "independent sources": N_PARTICIPANTS,
     "recordings per source": round(len(hum) / N_PARTICIPANTS, 1)},
    {"class": "bot", "recordings": 112, "independent sources": 4,
     "recordings per source": 28.0},
])
print(eff.to_string(index=False))
eff.to_csv(OUT / "effective_sample_size.csv", index=False)
print("""
Consequences to carry into the write-up:

1. Confidence intervals bootstrapped over RECORDINGS are still optimistic, because
   recordings within a participant are correlated. The truthful resampling unit is the
   participant, and with 3 of them no useful interval exists. Report recording-level
   intervals and state this explicitly rather than implying they are the final word.
2. 'Generalises to human users' is not supportable at n=3. 'Distinguishes these
   participants from these four bot generators' is.
3. Leave-one-campaign-out is the closest available proxy for a genuinely new user and
   setup, so it belongs in the paper alongside the session-level number.""")

# %% [markdown]
# ## 9.5 Is the human class unnaturally homogeneous?
#
# The specific failure mode to check: if three people produce three tight clusters, the
# classifier's job is easier than the paper implies. Compare the spread of human
# recordings against the spread of bot recordings in the same feature space — if humans
# are much tighter, the task is partly "recognise these three styles".

# %%
bot_rows = []
for r in index_sessions():
    if not r.is_bot:
        continue
    mv = json.loads(Path(r.path).read_text()).get("mouse_movements") or []
    if len(mv) < 30:
        continue
    bot_rows.append({"name": r.name, "family": r.family, **features(mv)})
bots = pd.DataFrame(bot_rows)

scaler = StandardScaler().fit(pd.concat([hum[FEATS], bots[FEATS]]).values)
Xh, Xb = scaler.transform(hum[FEATS].values), scaler.transform(bots[FEATS].values)

def spread(M):
    c = M.mean(0)
    return float(np.mean(np.linalg.norm(M - c, axis=1)))

print(f"mean distance to class centroid (standardised feature space):")
print(f"  humans ({len(Xh):3d} recordings, {N_PARTICIPANTS} people)  {spread(Xh):.2f}")
print(f"  bots   ({len(Xb):3d} recordings, 4 generators)  {spread(Xb):.2f}")
for fam in sorted(bots.family.unique()):
    m = scaler.transform(bots[bots.family == fam][FEATS].values)
    if len(m) > 1:
        print(f"    {fam:14s} ({len(m):2d})  {spread(m):.2f}")
print("""
If the human spread is comparable to a single bot family's spread, then 'human' is
behaving like one more narrow generator rather than like a population -- which is the
honest reading of a 3-participant sample.""")

# %% [markdown]
# ## 9.6 Figures

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

ax[0].plot(sil_df.k, sil_df.silhouette, "o-")
ax[0].axvline(N_PARTICIPANTS, color="crimson", ls="--", lw=1,
              label=f"{N_PARTICIPANTS} participants")
ax[0].set_xlabel("number of clusters"); ax[0].set_ylabel("silhouette")
ax[0].set_title("Do human recordings form 3 groups?")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

dendrogram(Z, ax=ax[1], no_labels=True,
           link_color_func=lambda _: "tab:blue")
ax[1].set_title("Ward dendrogram, human recordings")
ax[1].set_ylabel("merge distance")

from sklearn.decomposition import PCA
pc = PCA(n_components=2).fit(np.vstack([Xh, Xb]))
ph, pb = pc.transform(Xh), pc.transform(Xb)
for fam in sorted(bots.family.unique()):
    m = pc.transform(scaler.transform(bots[bots.family == fam][FEATS].values))
    ax[2].scatter(m[:, 0], m[:, 1], s=18, alpha=.6, label=fam)
ax[2].scatter(ph[:, 0], ph[:, 1], s=30, c="k", marker="^", label="human (3 people)")
ax[2].set_title("Feature space (PCA)"); ax[2].legend(fontsize=6.5); ax[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig(OUT / "participant_structure.pdf", dpi=300)
plt.savefig(OUT / "participant_structure.png", dpi=110); plt.close()

hum.to_csv(OUT / "human_session_features.csv", index=False)
sil_df.to_csv(OUT / "human_silhouette.csv", index=False)
pd.DataFrame([{"scheme": k, "window_auc": v.window_auc, "session_auc": v.session_auc,
               "ece": v.ece} for k, v in probe.items()]).to_csv(
    OUT / "cv_scheme_comparison.csv", index=False)

# %% [markdown]
# ## 9.7 Decision for the scorer work
#
# * **Primary scheme: session-level 5-fold over all 156 recordings.** Out-of-fold
#   pooling means every recording is scored once by a model that never saw it, so the
#   evaluation set is ~520 windows instead of the 167 a single held-out split gave us.
#   That is the largest honest gain available without collecting more data, and it is
#   what makes model comparison possible at all.
# * **Reported alongside: leave-one-campaign-out**, as the pessimistic bound and the
#   closest proxy for an unseen participant and setup.
# * **Not claimed: generalisation to human users at large.** n=3.

# %%
print("Notebook 09 complete.")
