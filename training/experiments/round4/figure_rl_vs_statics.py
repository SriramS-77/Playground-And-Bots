"""Exploratory figure (not in the paper): DI under calibration drift, RL vs static policies.

Rows: bots look human / humans look bot-like. Columns: static policies (raw score, posterior with
fixed share, posterior with EM share); DQN; PPO. RL panels show offline-frozen and online-trained
(oracle / score-only / posterior reward) curves from round 4, with the best static policies as grey
references. All on the eval block, 100 humans x bots 20..1000, 20 eval seeds, mean over seeds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
R4 = HERE / "deliverables_fold2" / "drift" / "drift_curves.csv"
PS = HERE.parent / "results" / "round4" / "fold2" / "drift" / "posterior_statics_runs.csv"
OUT = HERE.parent / "results" / "round4" / "fold2" / "drift" / "rl_vs_statics.png"
ROWS = {"bots": "Bots look human (covert bots)", "humans": "Humans look bot-like"}

d = pd.read_csv(R4)
ps = pd.read_csv(PS)
base = ps[ps.eval_direction == "none"]


def static_curve(pol, direction):
    f = pd.concat([base, ps[ps.eval_direction == direction]])
    return f[f.policy == pol].groupby("eval_rho").DI.mean()


def r4(panel, policy, direction, drop0=False):
    f = d[(d.panel == panel) & (d.policy == policy) & (d.eval_direction == direction)].sort_values("eval_rho")
    return f[f.eval_rho > 0] if drop0 else f


STATICS = {"Static Single-Threshold": ("#555555", "-", "raw score, single"),
           "Static Multi-Threshold": ("#999999", "-", "raw score, multi"),
           "Posterior single (fixed share)": ("#7b3fb5", "-", "posterior single, fixed share"),
           "Posterior multi (fixed share)": ("#b48ae0", "-", "posterior multi, fixed share"),
           "Posterior single (EM share)": ("#7b3fb5", ":", "posterior single, EM share"),
           "Posterior multi (EM share)": ("#b48ae0", ":", "posterior multi, EM share")}
ARMS = {"oracle": "#2a78d6", "score-only": "#eb6834", "posterior": "#1baf7a"}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#bbbbbb", "legend.frameon": False})
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
for r, dr in enumerate(ROWS):
    a = ax[r, 0]
    for pol, (c, ls, lab) in STATICS.items():
        s = static_curve(pol, dr)
        a.plot(s.index, s.values, color=c, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
    for col, algo in ((1, "DQN"), (2, "PPO")):
        a = ax[r, col]
        for pol, ls in (("Static Single-Threshold", "--"), ("Posterior single (fixed share)", "--")):
            s = static_curve(pol, dr)
            a.plot(s.index, s.values, color=STATICS[pol][0], ls=ls, lw=1.1, label=STATICS[pol][2] + " (ref.)")
        off = r4("offline", algo, dr)
        a.plot(off.eval_rho, off.DI, color="black", lw=1.8, marker="s", ms=3, label="offline-trained, frozen")
        for arm, c in ARMS.items():
            on = r4("online", f"{algo} {arm}", dr, drop0=True)
            a.plot(on.eval_rho, on.DI, color=c, lw=1.6, marker="o", ms=3, label=f"online, {arm} reward")
    for c in range(3):
        a = ax[r, c]
        a.axhline(0, color="#888", lw=0.6)
        a.grid(axis="y", color="#e6e6e6")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        if r == 1:
            a.set_xlabel(r"shift level $\rho$")
    ax[r, 0].set_ylabel(ROWS[dr] + "\nDI (%)")
ax[0, 0].set_title("Static policies", loc="left", fontweight="bold")
ax[0, 1].set_title("DQN", loc="left", fontweight="bold")
ax[0, 2].set_title("PPO", loc="left", fontweight="bold")
ax[0, 0].legend(fontsize=7, loc="lower left")
ax[0, 2].legend(fontsize=7, loc="lower left")
fig.suptitle("Calibration drift: RL (offline / online) vs static policies (raw score / posterior) -- eval block, grounded world",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
