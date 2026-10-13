"""Round 5 -- fold 2: the drift study in the v2 world, with the new RL state (HANDOFF_ROUND5).

PRE-DECLARED (fixed before any run; from the local pilot `round4/sens_world.py`):
  world      GROUNDED_V2 / GROUNDED_V2_CFG (WORLD_V2.md): abandonment when a visible challenge
             is shown (Searles et al. 2023 rates), a leaver costs -100, level 10 a hard block.
  state      --state v2 (primary): 7 features -- score, running score, challenges solved
             (levels >= 1), steps passed, last level / 10, highest level solved / 10, active
             users / 300. --state v1 (sensitivity): round 1-4's 5 features.
  scorer     cross-fit seed 1 (round 4's drift scorer, R1), loaded from round 4's drift stage.
  archs      round 4's arch_choice: dqn 128-64, ppo 128-64, thompson h128_64_e64 (posterior
             mog2 -- chosen on rl_val by the pilot, all three still reported offline), linucb.
  data       train on the rl block (+ external INJECT halves when shifted); evaluate on the
             EVAL BLOCK (+ HELD-OUT halves) -- opened and logged by every evaluating task.
  shift      round 4's: a fraction rho of bots -> advanced web-bots, or of humans -> Balabit;
             LEVELS 0.1..1.0 (state v2) / {0.5, 1.0} (state v1); x-axis = calibration error.
  kinds      metrics   the x-axis (state v2 only)
             offline   DQN, PPO, Thompson, LinUCB, DQN without H-score (5 seeds) + statics,
                       oracle reward, 200 episodes, evaluated frozen at every point
             control   online protocol (300 episodes) without the shift, at every point
             online    shift on from episode 100; evaluated at its point and at rho = 0
             retrain   ceiling (state v2, DQN / PPO): oracle reward on the shifted pool
                       from episode 0, evaluated at its point
             online arms: oracle, score-only, posterior; algorithms DQN, PPO, Thompson
  contrast   primary (state v2): per (algorithm, arm, direction), mean DI over the shifted
             levels, online - control, Welch over seeds, Holm over the 18 contrasts.

    python round5/r5_fold2.py --state v2 --list | --task N | --collect | --smoke
    python round5/r5_fold2.py --state v1 --list | --task N | --collect
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _state():
    for i, a in enumerate(sys.argv):
        if a == "--state" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "v2"


STATE = _state()
if STATE not in ("v1", "v2"):
    raise SystemExit("--state v1|v2")
os.environ["RLC_STATE"] = STATE                      # read by expkit.trainer at import
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "round4")]
_argv = list(sys.argv)
import sens_world as S                               # noqa: E402  (imports round 4 + round 3)
sys.argv = _argv

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

R4, C, R = S.R4, S.C, S.R
OUT = HERE.parent / "results" / "round5" / "fold2" / f"state_{STATE}"
LEVELS = R4.LEVELS if STATE == "v2" else (0.5, 1.0)
SEEDS = tuple(range(R.SEEDS))                        # 5
ALGOS = ("dqn", "ppo", "thompson")
ARMS = ("oracle", "score-only", "posterior")
OFFLINE = ("dqn", "ppo", "thompson", "linucb", "ablation")

# configure the shared runner for the final evaluation
S.OUT = OUT
S.LEVELS = LEVELS
S.EVAL_SEEDS, S.VOLUMES = C.EVAL_SEEDS, R4.SCORED_VOLUMES
S.offline_points = lambda: R4.points(LEVELS)
S.best_ts_posterior = lambda: ("mog2", None)        # declared from the pilot
_REASON = {"text": "R5"}


def _final_pools():
    from expkit.external import load_sessions
    p = C.pools(final=True, reason=_REASON["text"])  # opens the eval block -- logged
    p.cache._table.update(R4.ext_scores())
    p.ext = load_sessions()
    return p


S.pools = _final_pools


def tasks():
    t = [{"kind": "metrics"}] if STATE == "v2" else []
    t += [{"kind": "offline", "world": "v2", "family": f, "seed": s} for f in OFFLINE for s in SEEDS]
    t += [{"kind": "offline", "world": "v2", "family": "statics", "seed": 0}]
    t += [{"kind": "control", "algo": a, "arm": arm, "seed": s} for a in ALGOS for arm in ARMS
          for s in SEEDS]
    t += [{"kind": "online", "algo": a, "arm": arm, "direction": d, "rho": rho, "seed": s}
          for a in ALGOS for arm in ARMS for d in R4.DIRECTIONS for rho in LEVELS for s in SEEDS]
    if STATE == "v2":
        t += [{"kind": "retrain", "algo": a, "arm": "oracle", "direction": d, "rho": rho, "seed": s}
              for a in ("dqn", "ppo") for d in R4.DIRECTIONS for rho in LEVELS for s in SEEDS]
    return t


def name(task):
    return task["kind"] + "__" + S.name(task)


def run_task(i):
    S.no_power_throttling()
    task = tasks()[i]
    out = OUT / "tasks" / f"{name(task)}.csv"
    if out.exists():
        print("exists:", out)
        return
    _REASON["text"] = f"R5 state {STATE} task {i} {name(task)}"
    t0 = time.time()
    if task["kind"] == "metrics":
        R4.drift_metrics(_final_pools(), LEVELS).to_csv(OUT / "drift_metrics.csv", index=False)
        print("wrote drift_metrics.csv")
        return
    df = S.run_offline(task) if task["kind"] == "offline" else S.run_online(task)
    df["state"] = STATE
    if task["kind"] == "offline":
        df["panel"] = "offline"
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(task)}: {(time.time() - t0) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #

def collect():
    from scipy.stats import ttest_ind
    files = sorted((OUT / "tasks").glob("*.csv"))
    expected = len([t for t in tasks() if t["kind"] != "metrics"])
    if len(files) < expected:
        print(f"WARNING: {len(files)} task CSVs, expected {expected} -- rerun the missing ids")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs = runs[runs.bots > 0]
    key = ["panel", "policy", "eval_direction", "eval_rho", "train_seed"]
    cols = ["DI", "human_survival", "bot_survival", "friction_s", "abandonment_rate"]
    ps = runs.groupby(key)[cols].mean().reset_index()
    zero = ps[ps.eval_direction == "none"]
    ps = pd.concat([ps[ps.eval_direction != "none"]] + [zero.assign(eval_direction=d) for d in R4.DIRECTIONS],
                   ignore_index=True)
    mpath = OUT.parent / "state_v2" / "drift_metrics.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        m = m[m.pool == "eval"]
        xx = pd.concat([m[((m.direction == d) | (m.direction == "none")) & (m["class"] == d)].assign(direction=d)
                        for d in R4.DIRECTIONS])[["direction", "rho", "citl_error", "pool_auc"]]
        ps = ps.merge(xx.rename(columns={"direction": "eval_direction", "rho": "eval_rho"}),
                      on=["eval_direction", "eval_rho"], how="left")
    ps.to_csv(OUT / "per_seed.csv", index=False)
    g = ps.groupby(["panel", "policy", "eval_direction", "eval_rho"])
    t = g[[c for c in cols + ["citl_error", "pool_auc"] if c in ps]].mean()
    t[[c + "_se" for c in cols]] = g[cols].sem().to_numpy()
    t["n_seeds"] = g.size()
    t.to_csv(OUT / "curves.csv")
    rows = []
    on = ps[(ps.panel == "online") & (ps.eval_rho > 0)]
    ctl = ps[(ps.panel == "control") & (ps.eval_rho > 0) & ps.eval_rho.isin(LEVELS)]
    for (pol, d), a in on.groupby(["policy", "eval_direction"]):
        a = a.groupby("train_seed").DI.mean()
        b = ctl[(ctl.policy == pol) & (ctl.eval_direction == d)].groupby("train_seed").DI.mean()
        if len(a) > 1 and len(b) > 1:
            rows.append({"policy": pol, "direction": d, "online_DI": a.mean(), "control_DI": b.mean(),
                         "diff": a.mean() - b.mean(), "p": ttest_ind(a, b, equal_var=False).pvalue,
                         "n_online": len(a), "n_control": len(b)})
    if rows:
        ct = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
        k = len(ct)
        ct["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(ct.p)]))
        ct.to_csv(OUT / "contrasts.csv", index=False)
        print(ct.round(4).to_string(index=False))
    figure(t.reset_index())


def figure(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    arm_style = {"oracle": ("#2a78d6", "s"), "score-only": ("#eb6834", "^"), "posterior": ("#1baf7a", "D")}
    labels = {"dqn": ("DQN", "DQN"), "ppo": ("PPO", "PPO"), "thompson": ("Thompson (mog2)", "Thompson (mog2)")}
    x = "citl_error" if "citl_error" in t and t.citl_error.notna().any() else "eval_rho"
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for r, d in enumerate(R4.DIRECTIONS):
        for c, (algo, (off_name, on_prefix)) in enumerate(labels.items()):
            ax = axes[r, c]
            q = t[t.eval_direction == d]

            def line(sel, **kw):
                s = sel.sort_values(x)
                ax.errorbar(s[x], s.DI, yerr=s.DI_se.fillna(0), capsize=2, elinewidth=1, **kw)
            line(q[(q.panel == "offline") & (q.policy == off_name)], color=INK, lw=2, marker="o",
                 ms=5, label="Offline-trained, frozen")
            ctl = q[(q.panel == "control") & (q.policy == f"{on_prefix} oracle")]
            line(ctl, color="#8a8985", lw=1.6, ls=(0, (4, 3)), marker="o", ms=4, mfc=SURF,
                 label="No shift in training (oracle)")
            ret = q[(q.panel == "retrain") & (q.policy == f"{on_prefix} oracle")]
            if len(ret):
                line(pd.concat([ctl[ctl.eval_rho == 0], ret]), color=INK, lw=1.4, ls=(0, (1, 1.5)),
                     marker="s", ms=4, label="Ceiling: retrained on shifted data, true labels")
            for arm, (col, mk) in arm_style.items():
                on = q[(q.panel == "online") & (q.policy == f"{on_prefix} {arm}")]
                z = q[(q.panel == "control") & (q.policy == f"{on_prefix} {arm}") & (q.eval_rho == 0)]
                if len(on):
                    line(pd.concat([z, on[on.eval_rho > 0]]), color=col, lw=2, marker=mk, ms=6,
                         label=f"Online, {arm} reward")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(off_name, loc="left", fontweight="bold", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel({"bots": "Covert bots", "humans": "Bot-like humans"}[d]
                              + "\nDI (humans kept % - bots through %)")
            ax.set_xlabel({"citl_error": f"Calibration error of the shifted class ({d})",
                           "eval_rho": "Shift level rho"}[x])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"Round 5, fold 2, v2 world, state {STATE} -- eval block, mean over seeds, whiskers SE",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(OUT / "drift_figure.png", dpi=150)
    print("wrote", OUT / "drift_figure.png")


def smoke():
    """Toy budget on rl_fit / rl_val (never the eval block): one task of each kind."""
    S.pools = _rl_val_pools
    S.EVAL_SEEDS, S.VOLUMES = 1, (20,)
    R4.TOTAL, R4.INJECT_EP, R.EPISODES = 4, 2, 3
    global OUT
    OUT = OUT.parent / f"smoke_state_{STATE}"
    S.OUT = OUT
    log = R4.OUT / "eval_access.log"
    before = log.read_text() if log.exists() else ""
    want = ["offline", "control", "online", "retrain"]
    for i, t in enumerate(tasks()):
        if t["kind"] in want and (t.get("family", "dqn") in ("dqn", "statics")) and t.get("algo", "dqn") in ("dqn", "thompson"):
            want.remove(t["kind"])
            print("[smoke]", t, flush=True)
            df = S.run_offline(t) if t["kind"] == "offline" else S.run_online(t)
            (OUT / "tasks").mkdir(parents=True, exist_ok=True)
            if t["kind"] == "offline":
                df["panel"] = "offline"
            df.assign(state=STATE).to_csv(OUT / "tasks" / f"{name(t)}.csv", index=False)
    after = log.read_text() if log.exists() else ""
    assert after == before, "smoke opened the eval block"
    print("smoke: OK (eval block untouched)")


def _rl_val_pools():
    from expkit.external import load_sessions
    p = C.pools(final=False, smoke=True)
    p.cache._table.update(R4.ext_scores())
    p.ext = load_sessions()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="v2", choices=("v1", "v2"))
    for f in ("list", "collect", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    if a.list:
        ts = tasks()
        for i, t in enumerate(ts):
            print(i, t)
        print(f"\n{len(ts)} tasks -> sbatch --array=0-{len(ts) - 1}%25 slurm/r5.sh {STATE}")
    elif a.collect:
        collect()
    elif a.smoke:
        smoke()
    elif a.task is not None:
        run_task(a.task)
    else:
        ap.error("one of --list, --task N, --collect, --smoke")


if __name__ == "__main__":
    main()
