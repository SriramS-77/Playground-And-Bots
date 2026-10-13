"""Paper figure: DI vs scorer calibration error (round 4, fold 2, eval block, 5 seeds).

Source: training/experiments/round4/deliverables_fold2/drift/drift_curves.csv (verified).
Writes latex_project/images/drift.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "training/experiments/round4/deliverables_fold2/drift/drift_curves.csv"
OUT = HERE.parent / "images" / "drift.pdf"

INK, INK2, GRID = "#1a1a1a", "#555555", "#e2e2e2"
OFFLINE = {"DQN": ("#2a78d6", "o"), "PPO": ("#eb6834", "s"), "Thompson Sampling": ("#1baf7a", "^"),
           "Static Single-Threshold": ("#555555", "x"), "Static Multi-Threshold": ("#9a9a9a", "+")}
ONLINE = {"oracle": ("#2a78d6", "o"), "score-only": ("#eb6834", "s"), "posterior": ("#1baf7a", "^")}
ROWS = {"bots": "Bots look human", "humans": "Humans look bot-like"}

d = pd.read_csv(SRC)
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), sharey=True)


def series(panel, policy, direction):
    f = d[(d.panel == panel) & (d.policy == policy) & (d.eval_direction.isin([direction]))]
    if panel != "online":   # rho = 0 rows carry direction "bots"/"humans" for offline/control
        pass
    return f.sort_values("eval_rho")


for r, direction in enumerate(ROWS):
    ax = axes[r, 0]
    for pol, (c, m) in OFFLINE.items():
        s = series("offline", pol, direction)
        ax.errorbar(s.eval_rho, s.DI, yerr=s.DI_se, color=c, marker=m, ms=3.5, lw=1.2,
                    capsize=1.5, label=pol)
    for col, algo in ((1, "DQN"), (2, "PPO")):
        ax = axes[r, col]
        ctl = series("control", f"{algo} oracle", direction)
        ax.plot(ctl.eval_rho, ctl.DI, color="#888888", ls=(0, (3, 2)), lw=1.1,
                label="no shift in training (oracle)")
        for arm, (c, m) in ONLINE.items():
            s = series("online", f"{algo} {arm}", direction)
            s = s[s.eval_rho > 0]  # rho=0 online rows are not the no-shift protocol; omitted
            if s.empty:
                s = series("control", f"{algo} {arm}", direction)
            ax.errorbar(s.eval_rho, s.DI, yerr=s.DI_se, color=c, marker=m, ms=3.5, lw=1.2,
                        capsize=1.5, label=f"online, {arm} reward")
    for c in range(3):
        ax = axes[r, c]
        ax.axhline(0, color=INK2, lw=0.6)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([0, 0.5, 1.0])
        if r == 1:
            ax.set_xlabel(r"Shift level $\rho$")
    axes[r, 0].set_ylabel(f"{ROWS[direction]}\nDI (%)")

axes[0, 0].set_title("Offline-trained, frozen", loc="left", fontweight="bold", color=INK)
axes[0, 1].set_title("DQN, online training", loc="left", fontweight="bold", color=INK)
axes[0, 2].set_title("PPO, online training", loc="left", fontweight="bold", color=INK)
h0, l0 = axes[0, 0].get_legend_handles_labels()
h1, l1 = axes[0, 1].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=5, fontsize=6.8)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
print("wrote", OUT)
