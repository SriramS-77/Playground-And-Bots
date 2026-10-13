"""Tables and figures for the round-4 sensitivity check, one per state version (v2 world).

Rows of each figure: shift direction; columns: DQN, PPO, Thompson (mog2). x = shift level rho.
  black solid   offline-trained (oracle reward, 200 episodes), evaluated frozen
  grey dashed   online protocol without the shift (control), oracle reward
  coloured      online training through the shift: oracle / score-only / posterior reward
Seed 0 (both states ran it); the eval block is never touched (rl_val + held-out external halves).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results" / "round4" / "fold2"
OUT = ROOT / "sens_report"
STATES = {"state_v1": ROOT / "sens", "state_v2": ROOT / "sens_statev2"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = {"oracle": ("#2a78d6", "s"), "score-only": ("#eb6834", "^"), "posterior": ("#1baf7a", "D")}
ALGOS = {"dqn": ("DQN", "DQN"), "ppo": ("PPO", "PPO"), "thompson": ("Thompson (mog2)", "Thompson (mog2)")}
ROWS = {"bots": "Covert bots (advanced web-bots)", "humans": "Bot-like humans (Balabit)"}


def load(state):
    d = STATES[state]
    off = pd.read_csv(d / "offline.csv")
    off = off[(off.world == "v2") & (off.bots > 0)]
    on = pd.read_csv(d / "online.csv")
    on = on[(on.bots > 0) & (on.train_seed == 0) & (on.arm != "posterior (fixed share)")]
    return off, on


def curve(frame, direction):
    f = frame[frame.eval_direction.isin([direction, "none"])]
    return f.groupby("eval_rho")[["DI", "human_survival", "bot_survival", "friction_s"]].mean()


def tables(state):
    off, on = load(state)
    rows = []
    for algo, (label, _) in ALGOS.items():
        o = off[off.policy == label]
        for d in ("bots", "humans"):
            for rho, r in curve(o, d).iterrows():
                rows.append({"algorithm": label, "training": "offline (frozen)", "reward": "oracle",
                             "direction": d, "rho": rho, **r.round(3).to_dict()})
            for arm in ARMS:
                a = on[(on.algo == algo) & (on.arm == arm)]
                ctl = curve(a[a.panel == "control"], d)
                for rho, r in ctl.iterrows():
                    rows.append({"algorithm": label, "training": "online, no shift (control)",
                                 "reward": arm, "direction": d, "rho": rho, **r.round(3).to_dict()})
                sh = a[(a.panel == "online") & (a.eval_direction == d)]
                for rho, r in sh.groupby("eval_rho")[["DI", "human_survival", "bot_survival",
                                                     "friction_s"]].mean().iterrows():
                    rows.append({"algorithm": label, "training": "online, through the shift",
                                 "reward": arm, "direction": d, "rho": rho, **r.round(3).to_dict()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"{state}_table.csv", index=False)
    return t


def figure(state, t):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                         "axes.facecolor": SURF})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), sharey=True, sharex=True)
    for r, d in enumerate(("bots", "humans")):
        for c, (algo, (label, _)) in enumerate(ALGOS.items()):
            ax = axes[r, c]
            x = t[(t.algorithm == label) & (t.direction == d)]
            off = x[x.training == "offline (frozen)"]
            ax.plot(off.rho, off.DI, color=INK, lw=2, marker="o", ms=6, label="Offline-trained, frozen")
            ctl = x[(x.training == "online, no shift (control)") & (x.reward == "oracle")]
            ax.plot(ctl.rho, ctl.DI, color="#8a8985", lw=1.6, ls=(0, (4, 3)), marker="o", ms=4,
                    mfc=SURF, label="No shift in training (oracle)")
            for arm, (col, mk) in ARMS.items():
                z = x[(x.training == "online, no shift (control)") & (x.reward == arm) & (x.rho == 0)]
                sh = x[(x.training == "online, through the shift") & (x.reward == arm)]
                s = pd.concat([z, sh]).sort_values("rho")
                ax.plot(s.rho, s.DI, color=col, lw=2, marker=mk, ms=7, label=f"Online, {arm} reward")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xticks([0, 0.5, 1.0])
            if r == 0:
                ax.set_title(label, loc="left", fontweight="bold", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(ROWS[d] + "\nDI (humans kept % - bots through %)")
            if r == 1:
                ax.set_xlabel("Shift level rho (fraction of the class replaced)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    name = {"state_v1": "current state (5 features)", "state_v2": "new state (7 features)"}[state]
    fig.suptitle(f"Drift sensitivity, v2 world, {name} -- seed 0, rl_val + held-out external sessions",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT / f"{state}_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for st in STATES:
        figure(st, tables(st))
        print("wrote", OUT / f"{st}_drift.png")
