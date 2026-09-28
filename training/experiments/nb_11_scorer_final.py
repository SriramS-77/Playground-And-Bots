# %% [markdown]
# # 11 — Augmentation, calibration, and the scorer the RL work will use
#
# Notebook 10 picked an architecture and an input representation. This one settles the
# remaining question — how to augment — then produces a fully characterised, calibrated
# scorer and exports it for the reinforcement-learning experiments.
#
# ## The augmentation question
#
# The published `perturb_mouse_data` draws **one** integer offset in {-1, 0, +1} and
# applies it to every point of a session: a rigid one-pixel translation of the whole
# trajectory. Notebook 08 measured what that does to the score — **nothing**, to within
# 1e-5. Two "different" simulated users drawn from one recording are literally the same
# sample.
#
# Independent per-movement jitter is a genuinely different perturbation, and notebook 02
# hinted it trains a more stable model. Here we test it properly, at several magnitudes,
# against Gaussian noise.

# %%
import dataclasses, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import expkit
from expkit import calibration as cal
from expkit.features import N_FEATURES, PERTURBATIONS
from expkit.partition import index_sessions, load_partition
from expkit.paths import RESULTS, SCORERS
from expkit.scorer_search import ArchSpec, PUBLISHED, cross_validate, save, to_frame
from expkit.scorer_train import Standardiser, build_windows, split_by_session

OUT = RESULTS
REFS = index_sessions()
partition = load_partition()
choice = json.load(open(OUT / "scorer_arch_choice.json"))
print("architecture chosen in notebook 10:", choice)

ARCHS = {a.name: a for a in [
    ArchSpec("lstm16", (16,), (8,)), ArchSpec("lstm16_8", (16, 8), (16,)),
    ArchSpec("lstm32", (32,), (32,)), ArchSpec("lstm32_16", (32, 16), (32,)),
    ArchSpec("lstm64_32", (64, 32), (64,)), PUBLISHED]}

base_name = choice["spec"].split("_lr")[0]
SPEC = ARCHS[base_name]
if "_lr" in choice["spec"]:
    lr = float(choice["spec"].split("_lr")[1].split("_d")[0])
    dr = float(choice["spec"].split("_d")[1])
    SPEC = dataclasses.replace(SPEC, name=choice["spec"], lr=lr, dropout=dr)
REP = choice["representation"]
print(f"using {SPEC.name} on {REP} ({SPEC.param_count(N_FEATURES[REP]):,} params)")

# %% [markdown]
# ## 11.1 Augmentation sweep
#
# Each augmented setting produces 4 copies of every training recording. `magnitude` is
# the half-range of the integer offset (so 1 means {-1,0,+1}); `sigma` is the standard
# deviation of the Gaussian in pixels.

# %%
AUGS = [
    dict(tag="none",            augmentation="none",     n_copies=1),
    dict(tag="rigid +/-1 (published)", augmentation="rigid", n_copies=4, magnitude=1),
    dict(tag="per_move +/-1",   augmentation="per_move", n_copies=4, magnitude=1),
    dict(tag="per_move +/-2",   augmentation="per_move", n_copies=4, magnitude=2),
    dict(tag="per_move +/-3",   augmentation="per_move", n_copies=4, magnitude=3),
    dict(tag="gaussian s=1",    augmentation="gaussian", n_copies=4, sigma=1.0),
    dict(tag="gaussian s=2",    augmentation="gaussian", n_copies=4, sigma=2.0),
]

rows, t0 = [], time.time()
for a in AUGS:
    tag = a.pop("tag")
    for seed in (0, 1, 2):
        r = cross_validate(REFS, SPEC, REP, n_splits=5, group="session", seed=seed,
                           epochs=90, patience=12, **a)
        rows.append({"augmentation": tag, "seed": seed, "window_auc": r.window_auc,
                     "session_auc": r.session_auc, "ece": r.ece,
                     "ece_after_T": r.ece_after_T, "brier": r.brier,
                     "temperature": r.temperature})
    sub = [x for x in rows if x["augmentation"] == tag]
    print(f"  {tag:24s} sessAUC={np.mean([x['session_auc'] for x in sub]):.3f}"
          f" +/-{np.std([x['session_auc'] for x in sub]):.3f}  "
          f"ECE_T={np.mean([x['ece_after_T'] for x in sub]):.3f}  "
          f"[{(time.time()-t0)/60:.0f}m]")
    a["tag"] = tag

aug = pd.DataFrame(rows)
aug.to_csv(OUT / "augmentation_sweep.csv", index=False)
aug_s = aug.groupby("augmentation").agg(
    sess_auc=("session_auc", "mean"), sess_auc_sd=("session_auc", "std"),
    win_auc=("window_auc", "mean"), win_auc_sd=("window_auc", "std"),
    ece=("ece", "mean"), ece_T=("ece_after_T", "mean"), ece_T_sd=("ece_after_T", "std"),
    brier=("brier", "mean")).sort_values("sess_auc", ascending=False)
print("\n", aug_s.round(4).to_string())

# %% [markdown]
# ### Selection, same rule as notebook 10
#
# Best mean session AUC; among settings within 1 standard error of it, the lowest
# post-scaling ECE. Note the **sd columns** as much as the means — an augmentation that
# makes training reproducible is worth more here than one that wins on a single fit,
# given this project has already been bitten by seed instability.

# %%
best = aug_s.iloc[0]
se = best.sess_auc_sd / np.sqrt(3)
near = aug_s[aug_s.sess_auc >= best.sess_auc - se].sort_values("ece_T")
print(f"best {best.name}: session AUC {best.sess_auc:.3f}, 1 SE = {se:.3f}")
print(f"{len(near)} setting(s) within 1 SE:\n")
print(near[["sess_auc", "sess_auc_sd", "ece_T", "win_auc_sd"]].round(4).to_string())
AUG_CHOICE = near.index[0]
aug_kw = next(a for a in AUGS if a["tag"] == AUG_CHOICE)
print(f"\nSELECTED AUGMENTATION: {AUG_CHOICE}")

# %% [markdown]
# ## 11.2 Why rigid translation is not augmentation
#
# Direct measurement: take one chunk, produce eight perturbed copies under each scheme,
# and look at the spread of the resulting scores. An augmentation that leaves the score
# unchanged has not created a new training example.

# %%
import random as _rnd
from expkit.simx import sessions_from_refs
from expkit.xscoring import XScorer

ev_h, ev_b = sessions_from_refs(partition["eval"])
probe_chunks = [c for s in (ev_h[:5] + ev_b[:5]) for c in s.chunks if len(c) >= 20][:40]

# a quick model purely to measure score sensitivity
from keras.optimizers import Adam
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
tr_refs, _ = split_by_session(partition["lstm"], seed=7)
tw = build_windows(tr_refs, REP, "none", 1, seed=0)
sd_ = Standardiser().fit(tw.X)
probe_model = SPEC.build(N_FEATURES[REP])
probe_model.compile(optimizer=Adam(SPEC.lr), loss="bce")
probe_model.fit(sd_(tw.X), tw.y, epochs=25, batch_size=SPEC.batch_size, verbose=0)
probe_scorer = XScorer(probe_model, sd_, REP)

sens = []
for tag, fn_name, kw in [("rigid +/-1", "rigid", {"magnitude": 1}),
                         ("per_move +/-1", "per_move", {"magnitude": 1}),
                         ("per_move +/-2", "per_move", {"magnitude": 2}),
                         ("per_move +/-3", "per_move", {"magnitude": 3}),
                         ("gaussian s=1", "gaussian", {"sigma": 1.0}),
                         ("gaussian s=2", "gaussian", {"sigma": 2.0})]:
    fn = PERTURBATIONS[fn_name]
    spreads = []
    for ch in probe_chunks:
        rng = _rnd.Random(0)
        sc = [probe_scorer.score_chunk(fn(ch, rng, **kw)) for _ in range(8)]
        sc = [v for v in sc if v is not None]
        if len(sc) > 1:
            spreads.append(np.std(sc))
    sens.append({"perturbation": tag, "score_sd_across_copies": float(np.mean(spreads)),
                 "max": float(np.max(spreads))})
sens_df = pd.DataFrame(sens)
sens_df.to_csv(OUT / "perturbation_sensitivity_final.csv", index=False)
print(sens_df.round(5).to_string(index=False))
print("""
A rigid shift moves every point by the same one pixel, so the shape of the trajectory --
which is what a recurrent net reads -- is unchanged. Per-movement jitter perturbs the
step-to-step differences, which is exactly the signal the model uses, so it produces a
genuinely different example.""")

# %% [markdown]
# ## 11.3 The final scorer, fully characterised
#
# Trained on the **`lstm` pool only** (73 recordings). The `rl` and `eval` pools stay
# unseen, which is what keeps the three-way partition intact for the RL work — notebook
# 03b showed the train/eval leak flows specifically through the scorer's outputs, so the
# scorer must not have seen the RL evaluation recordings.
#
# Performance is reported from cross-validation, not from this final fit.

# %%
from keras.callbacks import EarlyStopping

fit_refs, val_refs = split_by_session(partition["lstm"], seed=7)
tr = build_windows(fit_refs, REP, aug_kw.get("augmentation", "none"),
                   aug_kw.get("n_copies", 1), seed=0,
                   magnitude=aug_kw.get("magnitude", 1), sigma=aug_kw.get("sigma", 2.0))
va = build_windows(val_refs, REP, "none", 1, seed=1)
std = Standardiser().fit(tr.X)

tf.keras.utils.set_random_seed(0)
model = SPEC.build(N_FEATURES[REP])
model.compile(optimizer=Adam(SPEC.lr), loss="bce", metrics=["accuracy"])
n_pos = int(tr.y.sum()); n_neg = len(tr.y) - n_pos; tot = n_pos + n_neg
hist = model.fit(std(tr.X), tr.y, validation_data=(std(va.X), va.y), epochs=120,
                 batch_size=SPEC.batch_size, verbose=0,
                 class_weight={0: tot/(2*max(n_neg,1)), 1: tot/(2*max(n_pos,1))},
                 callbacks=[EarlyStopping(monitor="val_loss", patience=15,
                                          restore_best_weights=True)])
print(f"trained {len(hist.history['loss'])} epochs on {len(tr)} windows "
      f"from {len(fit_refs)} recordings")

pv = model.predict(std(va.X), batch_size=512, verbose=0).ravel()
T = cal.fit_temperature(pv, va.y)
print(f"temperature fitted on the held-out validation slice: T = {T:.3f}")

# %% [markdown]
# ### Calibration on the unseen pools
#
# The `eval` pool has never been touched by this model.

# %%
te = build_windows(partition["eval"], REP, "none", 1, seed=99)
p_raw = model.predict(std(te.X), batch_size=512, verbose=0).ravel()
p_cal = cal.apply_temperature(p_raw, T)

def sess(p, g, y):
    d = pd.DataFrame({"p": p, "g": g, "y": y}).groupby("g").agg(p=("p", "mean"),
                                                                y=("y", "first"))
    return d.p.values, d.y.values

report_rows = []
for tag, p in (("raw", p_raw), ("temperature-scaled", p_cal)):
    w = cal.report(p, te.y, tag)
    sp, sy = sess(p, te.groups, te.y)
    s = cal.report(sp, sy, tag)
    report_rows.append({"scores": tag, "window_acc": w["accuracy"], "window_AUC": w["AUC"],
                        "session_acc": s["accuracy"], "session_AUC": s["AUC"],
                        "Brier": w["Brier"], "ECE_uniform": w["ECE_uniform"],
                        "ECE_quantile": w["ECE_quantile"], "MCE": w["MCE_quantile"],
                        "reliability": w["reliability"], "resolution": w["resolution"]})
final_rep = pd.DataFrame(report_rows)
final_rep.to_csv(OUT / "final_scorer_calibration.csv", index=False)
print(final_rep.round(4).to_string(index=False))

# %% [markdown]
# ### Per bot family
#
# A single aggregate number hides whether the scorer solves one easy sub-problem and one
# hard one. NaiveBot recordings in particular carry only a handful of mouse movements.

# %%
fam = {r.name: r.family for r in partition["eval"]}
fam_df = pd.DataFrame({"p": p_cal, "g": te.groups, "y": te.y})
fam_df["family"] = fam_df.g.map(fam)
hum_mask = fam_df.family == "human"
per_fam = []
for f in sorted(fam_df.family.unique()):
    if f == "human":
        continue
    sub = pd.concat([fam_df[fam_df.family == f], fam_df[hum_mask]])
    per_fam.append({"family": f, "n_windows": int((fam_df.family == f).sum()),
                    "AUC vs humans": cal.roc_auc(sub.p.values, sub.y.values),
                    "mean score": float(fam_df[fam_df.family == f].p.mean())})
per_fam.append({"family": "human", "n_windows": int(hum_mask.sum()),
                "AUC vs humans": np.nan,
                "mean score": float(fam_df[hum_mask].p.mean())})
pf = pd.DataFrame(per_fam)
pf.to_csv(OUT / "final_scorer_per_family.csv", index=False)
print(pf.round(3).to_string(index=False))

# %% [markdown]
# ### Pessimistic check: leave-one-campaign-out
#
# Tests the scorer on a collection sitting it never saw — the closest available proxy
# for a new participant and a new setup, given three participants confounded with two
# campaigns (notebook 09).

# %%
camp = cross_validate(REFS, SPEC, REP, n_splits=2, group="campaign", seed=0,
                      epochs=90, patience=12, **{k: v for k, v in aug_kw.items()
                                                 if k != "tag"})
print(f"leave-one-campaign-out: window AUC {camp.window_auc:.3f}, "
      f"session AUC {camp.session_auc:.3f}, ECE {camp.ece:.3f} -> {camp.ece_after_T:.3f}")
print("Report this number in the paper next to the cross-validated one.")

# %% [markdown]
# ## 11.4 Export for the RL experiments

# %%
name = "final_rl_scorer"
model.save(str(SCORERS / f"{name}.keras"))
meta = {"name": name, "representation": REP, "spec": SPEC.name,
        "lstm_units": list(SPEC.lstm_units), "dense_units": list(SPEC.dense_units),
        "lr": SPEC.lr, "dropout": SPEC.dropout,
        "augmentation": aug_kw.get("augmentation", "none"),
        "n_copies": aug_kw.get("n_copies", 1),
        "magnitude": aug_kw.get("magnitude", 1), "sigma": aug_kw.get("sigma", 2.0),
        "params": int(model.count_params()), "temperature": float(T),
        "trained_on": "lstm pool only (rl and eval pools unseen)",
        "n_train_windows": int(len(tr)), "epochs": len(hist.history["loss"]),
        "model_path": str(SCORERS / f"{name}.keras"),
        "standardiser": std.to_dict()}
(SCORERS / f"{name}.json").write_text(json.dumps(meta, indent=2))
print(json.dumps({k: v for k, v in meta.items() if k != "standardiser"}, indent=2))

# %% [markdown]
# ### Verify the batched path, and measure what per-movement jitter now costs in RL
#
# `score_many` batches every chunk into one `predict` call. A single call carries ~150 ms
# of graph-dispatch overhead *regardless of model size*, so scoring chunk-by-chunk was
# never limited by the network — it was limited by how many times we called it.

# %%
from expkit.xscoring import XScorer as XS

final_scorer = XS.from_name(name)
chunks = [c for s in (ev_h + ev_b) for c in s.chunks if c][:300]

t0 = time.time(); one_by_one = [final_scorer.score_chunk(c) for c in chunks[:60]]
per_call = (time.time() - t0) / 60
t0 = time.time(); batched = final_scorer.score_many(chunks)
per_batched = (time.time() - t0) / len(chunks)

agree = np.max(np.abs(np.array(one_by_one) - np.array(batched[:60])))
print(f"max |per-chunk - batched| = {agree:.2e}  (must be ~0)")
print(f"per-chunk call : {per_call*1000:7.1f} ms")
print(f"batched        : {per_batched*1000:7.3f} ms   ({per_call/per_batched:.0f}x faster)")

EP_USERS, N_CHUNKS = 600, 12
per_ep = EP_USERS * N_CHUNKS * per_batched
print(f"\nper-movement jitter in RL, batched per episode:")
print(f"  {EP_USERS} users x {N_CHUNKS} chunks = {EP_USERS*N_CHUNKS:,} scores "
      f"= {per_ep:.1f}s per episode")
for eps in (100, 150):
    print(f"  {eps} episodes -> {per_ep*eps/60:.0f} min of scorer time")
print(f"  (chunk-by-chunk this would be {EP_USERS*N_CHUNKS*per_call*100/3600:.0f} h "
      f"for 100 episodes)")

# %% [markdown]
# ## 11.5 Figures

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))

ax = axes[0]
x = np.arange(len(aug_s))
ax.bar(x - .2, aug_s.sess_auc, .4, yerr=aug_s.sess_auc_sd, capsize=3, label="session AUC")
ax.bar(x + .2, 1 - aug_s.ece_T * 5, .4, yerr=aug_s.ece_T_sd * 5, capsize=3,
       label="1 - 5x ECE (higher better)")
ax.set_xticks(x); ax.set_xticklabels(aug_s.index, rotation=30, ha="right", fontsize=6.5)
ax.set_title("Augmentation"); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

ax = axes[1]
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
for tag, p in (("raw", p_raw), ("temperature-scaled", p_cal)):
    conf, obs, w = cal.reliability_curve(p, te.y, n_bins=10)
    ax.plot(conf, obs, "o-", ms=5, label=tag)
ax.set_xlabel("mean predicted P(bot)"); ax.set_ylabel("observed fraction bots")
ax.set_title("Reliability, unseen eval pool"); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axes[2]
ax.bar(range(len(sens_df)), sens_df.score_sd_across_copies)
ax.set_yscale("log")
ax.set_xticks(range(len(sens_df)))
ax.set_xticklabels(sens_df.perturbation, rotation=30, ha="right", fontsize=6.5)
ax.set_ylabel("score sd across 8 copies of one chunk")
ax.set_title("Does the perturbation create a new sample?"); ax.grid(alpha=.3, axis="y")

plt.tight_layout(); plt.savefig(OUT / "scorer_final.pdf", dpi=300)
plt.savefig(OUT / "scorer_final.png", dpi=110); plt.close()

with pd.ExcelWriter(OUT / "scorer_final.xlsx", engine="openpyxl") as xl:
    aug_s.to_excel(xl, sheet_name="Augmentation")
    final_rep.to_excel(xl, sheet_name="Calibration", index=False)
    pf.to_excel(xl, sheet_name="Per family", index=False)
    sens_df.to_excel(xl, sheet_name="Perturbation sensitivity", index=False)
print("Notebook 11 complete.")
