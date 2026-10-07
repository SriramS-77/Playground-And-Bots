"""DI against the scorer's calibration error -- the round-4 drift figure (HANDOFF_ROUND4 §4).

Rows: shift direction (bots look human / humans look bot-like). Columns: offline-trained
policies evaluated frozen; online training with DQN; online training with PPO. Online
panels: solid = trained through the shift, dashed = the same arm trained without it (the
control). One y-axis (DI) for every panel. Points are means over training seeds, whiskers SE.

Palette: the dataviz reference categorical order, validated (light surface: CVD worst adjacent
dE 9.1, normal-vision floor 19.6). Slots 3-5 are under 3:1 on the surface, so every series also
has its own marker shape and a legend, and the numbers ship as a table (drift_curves.csv).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
OFFLINE = ["DQN", "PPO", "Thompson Sampling", "LinUCB", "DQN without H-Score",
           "Static Single-Threshold", "Static Multi-Threshold"]
ARMS = ["oracle", "posterior", "posterior (floored)", "posterior+labels",
        "posterior+labels (floored)", "score-only"]
ROWS = {"bots": "Shift: bots look human\n(advanced web-bots replace a fraction of bots)",
        "humans": "Shift: humans look bot-like\n(Balabit users replace a fraction of humans)"}
XLAB = {"bots": "Calibration error of bots: mean(1 - P(bot))",
        "humans": "Calibration error of humans: mean P(bot)"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def _series(ax, d, color, marker, label, dashed=False):
    d = d.sort_values("citl_error")
    ax.errorbar(d.citl_error, d.DI, yerr=d.DI_se.fillna(0), color=color, lw=1.2 if dashed else 2,
                ls=(0, (4, 3)) if dashed else "-", marker=marker, ms=4 if dashed else 6,
                mfc=SURF if dashed else color, mec=color, mew=1.2, elinewidth=1, capsize=2,
                alpha=0.8 if dashed else 1.0, label=label, zorder=2 if dashed else 3)


def _axes(ax, m, direction):
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    t = m.sort_values("rho")
    t = t.iloc[::2] if len(t) > 6 else t
    ax.set_xticks(t.citl_error)
    ax.set_xticklabels([f"{e:.2f}\nAUC {a:.2f}" for e, a in zip(t.citl_error, t.pool_auc)], fontsize=7)
    ax.set_xlabel(XLAB[direction], fontsize=8)


def make_figure(curves_csv, metrics_csv, out_png):
    cur = pd.read_csv(curves_csv)
    met = pd.read_csv(metrics_csv)
    met = met[met.pool == "eval"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharey=True)
    for r, direction in enumerate(("bots", "humans")):
        m = met[((met.direction == direction) | (met.direction == "none")) & (met["class"] == direction)]
        c = cur[cur.eval_direction == direction]
        # offline panel
        ax = axes[r, 0]
        for i, pol in enumerate(OFFLINE):
            d = c[(c.panel == "offline") & (c.policy == pol)]
            if len(d):
                _series(ax, d, SLOTS[i], MARKERS[i], pol)
        _axes(ax, m, direction)
        ax.set_ylabel(ROWS[direction] + "\n\nDI (humans kept % - bots through %)", fontsize=9)
        if r == 0:
            ax.set_title("Offline-trained, evaluated frozen (deterministic world)",
                         loc="left", fontweight="bold", fontsize=10)
        else:
            ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2,
                      bbox_to_anchor=(0.5, -0.24))
        # online panels
        for col, algo in ((1, "DQN"), (2, "PPO")):
            ax = axes[r, col]
            for i, arm in enumerate(ARMS):
                pol = f"{algo} {arm}"
                on = c[(c.panel == "online") & (c.policy == pol)]
                ctl = c[(c.panel == "control") & (c.policy == pol)]
                if len(ctl):
                    _series(ax, ctl, SLOTS[i], MARKERS[i], f"{arm}, no online training", dashed=True)
                if len(on):
                    # rho = 0 is the control itself: start the online line from it
                    z = ctl[ctl.eval_rho == 0]
                    _series(ax, pd.concat([z, on[on.eval_rho > 0]]), SLOTS[i], MARKERS[i],
                            f"{arm}, trained through the shift")
            _axes(ax, m, direction)
            if r == 0:
                ax.set_title(f"Online training, {algo} (grounded world)", loc="left",
                             fontweight="bold", fontsize=10)
            else:
                h, lab = ax.get_legend_handles_labels()
                keep = [k for k, s in enumerate(lab) if "trained through" in s]
                ax.legend([h[k] for k in keep], [lab[k].replace(", trained through the shift", "")
                                                 for k in keep],
                          fontsize=7.5, frameon=False, loc="upper center", ncol=2,
                          bbox_to_anchor=(0.5, -0.24),
                          title="solid: trained through the shift\ndashed: same arm, no shift in training",
                          title_fontsize=7)
    fig.suptitle("DI as the humanity scorer's calibration error grows -- fold 2, round 4 "
                 "(mean over training seeds, whiskers SE; tick labels: calibration error / pool AUC)",
                 fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=2.5, w_pad=1.5)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)
