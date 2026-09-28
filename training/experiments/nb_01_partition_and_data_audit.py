# %% [markdown]
# # 1 — Data audit and the clean three-way partition
#
# **Reviewer 1, point 2:** *"Sessions are described as ~120 seconds, split into 10-second
# samples, then segmented into overlapping 100-movement sequences — but it's not stated
# whether the train/test split happened before or after this."*
#
# This notebook establishes three things:
#
# 1. **What the published split actually was.** It is cleaner than the review assumes:
#    the LSTM trained on one recording campaign and tested on another, three months
#    apart, with zero session overlap. That needs saying in the paper.
# 2. **Where the real leak is.** The RL agent trained and evaluated on the same 73
#    recordings, which are also the LSTM's test set — and which selected the LSTM
#    checkpoint through `EarlyStopping(restore_best_weights=True)`.
# 3. **A fix.** A session-level three-way partition, stratified by class, bot family and
#    campaign, used by every notebook that follows.
#
# Nothing in `rlcaptcha/` is modified anywhere in this series.

# %%
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

import expkit
from expkit.partition import (index_sessions, make_partition, save_partition,
                              summarise, SPLITS)
from expkit.paths import CAMPAIGN_A, CAMPAIGN_B, RESULTS

pd.set_option("display.width", 160)
OUT = RESULTS
print("campaign A (LSTM train in the published run):", CAMPAIGN_A)
print("campaign B (LSTM test + all RL work)        :", CAMPAIGN_B)

# %% [markdown]
# ## 1.1 The published split, verified
#
# `LSTM_Train_Analysis.ipynb` cell 2 reads:
#
# ```python
# x_train, Y_train = load(max_length=100, path="../train_data")   # -> 237 windows
# x_test,  Y_test  = load(max_length=100, path="../data")         # -> 283 windows
# ```
#
# So the split is by directory, and the directories are two separate recording campaigns.
# Below: file-name overlap, content-hash overlap, and the date range of each.

# %%
def digest(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

a_files = sorted(CAMPAIGN_A.glob("*.json"))
b_files = sorted(CAMPAIGN_B.glob("*.json"))
a_names, b_names = {p.name for p in a_files}, {p.name for p in b_files}
a_hash, b_hash = {digest(p) for p in a_files}, {digest(p) for p in b_files}

def dates(files):
    ds = sorted(f.stem.split("_data_")[-1][:10] for f in files)
    return f"{ds[0]} .. {ds[-1]}"

audit = pd.DataFrame([
    {"campaign": "A (train_data)", "files": len(a_files), "dates": dates(a_files),
     "humans": sum(1 for f in a_files if f.name.startswith("human")),
     "bots": sum(1 for f in a_files if f.name.startswith("bot"))},
    {"campaign": "B (data)", "files": len(b_files), "dates": dates(b_files),
     "humans": sum(1 for f in b_files if f.name.startswith("human")),
     "bots": sum(1 for f in b_files if f.name.startswith("bot"))},
])
print(audit.to_string(index=False))
print()
print(f"filename overlap     : {len(a_names & b_names)}")
print(f"content-hash overlap : {len(a_hash & b_hash)}")
assert not (a_names & b_names) and not (a_hash & b_hash)
print("\n=> The published LSTM split is session-level AND campaign-level.")
print("   No window from a training session can appear in the test set.")

# %% [markdown]
# **Finding 1.** The published LSTM train/test split is clean, and stronger than a random
# window split: it is a temporal holdout across two collection campaigns. The manuscript
# simply never says so. This is a documentation fix, not an experiment.
#
# **Finding 2.** Two real problems remain, and both need a re-run rather than a sentence:
#
# * the test set was passed as `validation_data` alongside
#   `EarlyStopping(restore_best_weights=True)`, so it selected the checkpoint;
# * `offline/offline_training.z.py` line 13 sets `DATA_DIR = "data"`, so the RL agent
#   trained on, and was evaluated on, campaign B — the LSTM's test set.

# %% [markdown]
# ## 1.2 Session inventory
#
# Everything the stratifier sees: class, bot family, campaign, and the raw size of each
# recording. The bot families are the four scripted generators; `human` is the
# participant pool.

# %%
refs = index_sessions()
rows = []
for r in refs:
    raw = json.loads(Path(r.path).read_text())
    mv = raw.get("mouse_movements") or []
    ts = raw.get("timestamps", {})
    dur = (ts.get("end", 0) - ts.get("start", 0)) / 1000 if ts else np.nan
    rows.append({"name": r.name, "campaign": r.campaign, "family": r.family,
                 "is_bot": r.is_bot, "movements": len(mv),
                 "duration_s": round(dur, 1) if dur == dur else np.nan,
                 "windows": max(1, len(mv) // 100) + (1 if len(mv) % 100 or len(mv) >= 100 else 0)})
inv = pd.DataFrame(rows)
inv.to_csv(OUT / "session_inventory.csv", index=False)

print(inv.groupby(["campaign", "family"]).agg(
    n=("name", "size"), movements=("movements", "median"),
    duration_s=("duration_s", "median")).round(1).to_string())
print("\n(medians: one campaign-A human recording has a corrupt end timestamp that")
print(" inflates the mean duration to ~7,000 s.)")
print(f"\nTOTAL: {len(inv)} sessions, {inv.movements.sum():,} mouse movements")

# %% [markdown]
# ### The participant question
#
# Reviewer 1 asks for participant count and recruitment. The recordings cannot answer it
# — there is no participant field — but the timestamps are suggestive, and the paper will
# have to state the true number.

# %%
human = inv[~inv.is_bot].copy()
h_times = sorted(Path(n).stem.split("_data_")[-1] for n in human.name)
print("Human recordings by campaign and clock time:\n")
for camp in ("A", "B"):
    sub = sorted(human[human.campaign == camp].name)
    if not sub:
        continue
    stamps = [s.split("_data_")[-1].replace(".json", "") for s in sub]
    print(f"  campaign {camp}: {len(sub)} sessions, {stamps[0]} -> {stamps[-1]}")
print("""
Campaign A's 12 human recordings fall inside one 3.5-hour window; campaign B's 32 inside
one 2-hour window. Both are single sittings, which is consistent with a very small number
of participants -- possibly one. Notebook 08 quantifies what that costs statistically.
The paper must report the true count, and if it is small the split has to be at
PARTICIPANT level, not session level, and the external-validity claim has to be narrowed
accordingly.""")

print("\nData-quality note -- the NaiveBot recordings are nearly empty:")
print(inv[inv.family == "NaiveBot"].groupby("campaign").movements.describe()[
    ["count", "min", "50%", "max"]].to_string())
print("""
A NaiveBot session carries a median of ~4-10 mouse movements, i.e. a single padded
window of repeated points. The behavioural scorer has almost nothing to read on this
family, so 'NaiveBot' is effectively detected by absence of movement rather than by
movement dynamics. That is worth stating in the paper rather than leaving for a reader
to discover.""")

# %% [markdown]
# ## 1.3 The clean partition
#
# Pooled across both campaigns and cut three ways at session level, stratified by
# `(campaign, family)`:
#
# | pool | purpose |
# |---|---|
# | `lstm` | trains the behavioural scorer (a slice is held out for early stopping) |
# | `rl`   | populates the RL replay buffer |
# | `eval` | Table 2 and Table 4 — never seen by either learner |

# %%
partition = make_partition(refs)
save_partition(partition)
summary = summarise(partition)
summary.to_csv(OUT / "partition_summary.csv", index=False)
print(summary.to_string(index=False))

names = {s: {r.name for r in partition[s]} for s in SPLITS}
for a in SPLITS:
    for b in SPLITS:
        if a < b:
            assert not (names[a] & names[b]), f"{a}/{b} overlap"
print("\nAll three pools are pairwise disjoint at session level.")
print(f"Saved -> {OUT / 'partition.json'}")

# %% [markdown]
# Every pool draws from both campaigns and every bot family, so no pool is a temporal or
# generator-specific slice. That matters: a partition that put campaign A entirely in
# `lstm` would confound the clean-split result with a three-month distribution shift.

# %%
mix = pd.crosstab([inv.set_index("name").loc[
    [r.name for s in SPLITS for r in partition[s]], "campaign"].values],
    [s for s in SPLITS for _ in partition[s]])
print("campaign x pool:\n", mix.to_string())

# %% [markdown]
# ## 1.4 What each notebook in this series does
#
# | notebook | reviewer point | output |
# |---|---|---|
# | 01 (this one) | R1.2 | the partition, the data audit |
# | 02 | R1.7 (timestamps, calibration) | retrained scorer variants |
# | 03 | R1.2 | Table 4 on a clean partition |
# | 04 | R1.3, R2.4 | reward sensitivity sweep |
# | 05 | R1.5 | independent human/bot traffic |
# | 06 | R1.1 | probabilistic action model |
# | 07 | R1.1 | proxy reward for deployment |
# | 08 | — | is 156 recordings enough? |

# %%
print("Notebook 01 complete.")
