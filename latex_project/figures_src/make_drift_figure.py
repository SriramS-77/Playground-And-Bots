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
           "LinUCB": ("#8c564b", "d"),
           "Static Single-Threshold": ("#555555", "x"), "Static Multi-Threshold": ("#9a9a9a", "+")}
LABEL = {"Static Single-Threshold": "Static threshold", "Static Multi-Threshold": "Static multi-threshold"}
DRIFT = HERE.parents[1] / "training/experiments/results/round4/fold2/drift"
NP_SRC = DRIFT / "np_drift_curve.csv"          # round4/np_drift_curve.py (alpha = 0.10)
ONLINE = {"oracle": ("#6a3d9a", "D"), "score-only": ("#b8860b", "v"), "posterior": ("#d62728", "P")}
ARM_LABEL = {"oracle": "label-based (oracle)", "score-only": "score-only", "posterior": "posterior (EM share)"}
ROWS = {"bots": "Covert bots"}
AUDIT = {"audit_recal": ("#17becf", "o", "audit recal., 2% labels"),
         "audit_ch": ("#e377c2", "s", "audit recal., level-9 challenge")}
_a = pd.concat([pd.read_csv(f) for f in (DRIFT / "np_audit").glob("online__*__audit_*__bots__*.csv")],
               ignore_index=True)
_a = _a[(_a.eval_direction == "bots") & _a.arm.isin(list(AUDIT))]
_a = _a.groupby(["algo", "arm", "eval_rho", "train_seed"], as_index=False).DI.mean()
AUD = _a.groupby(["algo", "arm", "eval_rho"], as_index=False).DI.agg(
    DI="mean", se=lambda x: x.std(ddof=1) / len(x) ** 0.5)

d = pd.read_csv(SRC)
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.3), sharey=True, squeeze=False)


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
                    capsize=1.5, label=LABEL.get(pol, pol))
    for col, algo in ((1, "DQN"), (2, "PPO")):
        ax = axes[r, col]
        ctl = series("control", f"{algo} oracle", direction)
        ax.plot(ctl.eval_rho, ctl.DI, color="#888888", ls=(0, (3, 2)), lw=1.1,
                label="label-based, trained without the shift")
        for arm, (c, m) in ONLINE.items():
            s = series("online", f"{algo} {arm}", direction)
            s = s[s.eval_rho > 0]  # rho=0 online rows are not the no-shift protocol; omitted
            if s.empty:
                s = series("control", f"{algo} {arm}", direction)
            ax.errorbar(s.eval_rho, s.DI, yerr=s.DI_se, color=c, marker=m, ms=3.5, lw=1.2,
                        capsize=1.5, label=ARM_LABEL[arm])
        # audit-recalibrated rewards: trained online only at rho = 0.5 and 1 (round4/np_audit.py)
        for arm, (c, m, lab) in AUDIT.items():
            a = AUD[(AUD.algo == algo.lower()) & (AUD.arm == arm)]
            if a.empty:
                continue
            ax.errorbar(a.eval_rho, a.DI, yerr=a.se, color=c, marker=m, ms=4.5, lw=0.9,
                        ls=(0, (1, 1.5)), capsize=1.5, label=lab, zorder=4)
    for c in range(3):
        ax = axes[r, c]
        ax.axhline(0, color=INK2, lw=0.6)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([0, 0.5, 1.0])
        if r == len(ROWS) - 1:
            ax.set_xlabel(r"Shift level $\rho$")
    axes[r, 0].set_ylabel(f"{ROWS[direction]}\nDI (%)")

axes[0, 0].set_title("Offline-trained, frozen", loc="left", fontweight="bold", color=INK)
axes[0, 1].set_title("DQN, online training", loc="left", fontweight="bold", color=INK)
axes[0, 2].set_title("PPO, online training", loc="left", fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.20, 1, 1))
h0, l0 = axes[0, 0].get_legend_handles_labels()
h1, l1 = axes[0, 1].get_legend_handles_labels()
p0, p1, p2 = (a.get_position() for a in axes[0])
fig.legend(h0, l0, loc="upper center", bbox_to_anchor=((p0.x0 + p0.x1) / 2, 0.205), ncol=2,
           fontsize=6.6, title="Policy (offline-trained, frozen)", title_fontsize=6.8,
           columnspacing=0.8, handlelength=1.6)
fig.legend(h1, l1, loc="upper center", bbox_to_anchor=((p1.x0 + p2.x1) / 2, 0.205), ncol=2,
           fontsize=6.6, title="Reward used in online training (DQN and PPO)", title_fontsize=6.8,
           columnspacing=0.8, handlelength=1.6)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
print("wrote", OUT)
