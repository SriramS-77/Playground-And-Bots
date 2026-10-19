"""Paper figure: humans lost vs bots stopped (Pareto view) at no drift and under covert bots.

Source: training/experiments/results/round4/fold2/drift/pareto_points.csv (round4/pareto.py).
Static NP threshold: a curve over its human-challenge budget alpha. RL: one marker per method at
the mean over the five false-positive penalties (each a mean of 3 training seeds); bars span the
range over penalties. Grey step line: non-dominated frontier. Panels zoom to their own y-range.
Writes latex_project/images/pareto.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "training/experiments/results/round4/fold2/drift/pareto_points.csv"
OUT = HERE.parent / "images" / "pareto.pdf"

d = pd.read_csv(SRC)
d["lost"] = 100 * (1 - d.human_survival)
d["stopped"] = 100 * (1 - d.bot_survival)
PANELS = [("none", 0.0, "No drift\nscorer chunk AUC 0.994", (85, 101)),       # round4/chunk_auc.py
          ("bots", 0.5, "Covert bots, $\\rho=0.5$\nscorer chunk AUC 0.969", (40, 102)),
          ("bots", 1.0, "Covert bots, $\\rho=1$\nscorer chunk AUC 0.942", (0, 103))]
NP_LABELLED = (0.01, 0.1, 0.2)   # 0.05 sits on top of 0.01: small mark only if on the frontier
RL = {"PPO frozen": ("#2a78d6", "o", "PPO, offline-trained (frozen)"),
      "DQN frozen": ("#2a78d6", "s", "DQN, offline-trained (frozen)"),
      "PPO online": ("#b8860b", "o", "PPO, online (score-only)"),
      "DQN online": ("#b8860b", "s", "DQN, online (score-only)")}


def frontier(x, y):
    pts = sorted(zip(x, y), key=lambda p: (p[0], -p[1]))
    out, best = [], -1
    for px, py in pts:
        if py > best:
            out.append((px, py))
            best = py
    return np.array(out)


plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#bbbbbb", "legend.frameon": False})
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
for ax, (dr, rho, title, ylim) in zip(axes, PANELS):
    f = d[(d.eval_direction == dr) & (d.eval_rho == rho)]
    allx, ally = [], []
    # Grey line: best static trade-off, the non-dominated points over every single-threshold
    # policy tested (NP budgets 0.01..0.3 and raw score thresholds 0.1..0.9), joined straight:
    # only these thresholds were evaluated, so nothing between them is known
    st = f[f.policy.isin(["NP threshold", "Raw threshold"])]
    fr = frontier(st.lost, st.stopped)
    fr = fr[: int(np.argmax(fr[:, 1] >= fr[:, 1].max() - 0.5)) + 1]   # stop where it reaches its top
    # anchored at "never challenge" (no human lost, no bot stopped): the threshold above every score
    line = np.vstack([[0.0, 0.0], fr])
    ax.plot(line[:, 0], line[:, 1], color="#9a9a9a", lw=1.0, zorder=1, clip_on=True,
            label="best static threshold (frontier over all thresholds)")
    # small markers on the frontier's vertices: the evaluated thresholds that make its bends
    vert = [st[(abs(st.lost - x) < 1e-9) & (abs(st.stopped - y) < 1e-9)].iloc[0] for x, y in fr]
    for v in vert:
        np_pt = v.policy == "NP threshold"
        if np_pt and v.knob in NP_LABELLED:
            continue
        lab_np = f[(f.policy == "NP threshold") & f.knob.isin(NP_LABELLED)]
        if (abs(lab_np.lost - v.lost) < 1.5).any():
            continue                    # on the near-vertical stretch of labelled NP points: no visible bend
        ax.scatter(v.lost, v.stopped, s=7, zorder=4, color="#7b3fb5" if np_pt else "#7a7a7a",
                   marker="D" if np_pt else "o")
        if not np_pt and not (abs(lab_np.lost - v.lost) < 1.5).any():   # no label in a crowded spot
            ax.annotate(f"t={v.knob:g}", (v.lost, v.stopped), xytext=(4, 2),
                        textcoords="offset points", fontsize=5.2, color="#7a7a7a")
    s = f[(f.policy == "NP threshold") & f.knob.isin(NP_LABELLED)].sort_values("knob")
    ax.scatter(s.lost, s.stopped, color="#7b3fb5", marker="D", s=16, zorder=5,
               label=r"Static NP threshold (one point per $\alpha$)")
    for x, y, a in zip(s.lost, s.stopped, s.knob):
        dy = -9
        ax.annotate(rf"$\alpha$={a:g}", (x, y), xytext=(4, dy), textcoords="offset points",
                    fontsize=5.5, color="#7b3fb5", zorder=6)
    allx += list(s.lost); ally += list(s.stopped)
    for pol, (c, m, lab) in RL.items():
        g = f[f.policy == pol]
        mx, my = g.lost.mean(), g.stopped.mean()
        ax.errorbar(mx, my, xerr=[[mx - g.lost.min()], [g.lost.max() - mx]],
                    yerr=[[my - g.stopped.min()], [g.stopped.max() - my]],
                    fmt=m, ms=5.5, color=c, mfc=c if pol.startswith("PPO") else "white",
                    mec=c, ecolor=c, elinewidth=0.8, capsize=1.5, zorder=3, label=lab)
        allx.append(mx); ally.append(my)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Humans lost (%)")
    ax.set_xlim(-1.5, 55)
    ax.set_ylim(*ylim)
    ax.grid(color="#e6e6e6", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Bots stopped (%)")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, columnspacing=1.2)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=160)
print("wrote", OUT)
