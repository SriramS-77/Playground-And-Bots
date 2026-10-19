"""Paper headline in the grounded world -- fold 2, run locally (round 6).

Trains every policy exactly as round 4's drift-stage OFFLINE panel does (`r4_fold2.run_offline`):
rl block, GROUNDED world (stochastic solves, human failures, leave-after-pass tau = 45 s), oracle
reward, 200 episodes, round 4's arch_choice, cross-fit seed-1 scorer, training seeds 0-4. The
agent is saved, then evaluated on the EVAL BLOCK with no shift on two traffic grids:

  fixed    100 humans x bots {20, 100, 200, 500, 1000}                 (round 3's published grid)
  varying  humans U(50, 250) x bot share {5, 15, 30, 43, 60, 75} %     (round 3's realistic grid)

50 evaluation seeds per grid point (round 3's headline budget). Each task opens the eval block
once, logged in results/round4/fold2/drift/eval_access.log by `context()`.

Consistency check: eval seeds 0-19 of the fixed grid are the cells round 4's offline panel
evaluated at rho = 0, so `--collect` compares them with `drift_per_seed.csv`.

    python round4/grounded_headline.py --list
    python round4/grounded_headline.py --smoke            # toy budget on rl_val, eval block shut
    python round4/grounded_headline.py --task N
    python round4/grounded_headline.py --run-all 6        # every missing task, 6 at a time
    python round4/grounded_headline.py --collect
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_argv = sys.argv
sys.argv = [sys.argv[0], "--stage", "drift"]      # r4_fold2 resolves its stage at import
import r4_fold2 as D                              # noqa: E402
sys.argv = _argv

import pandas as pd                               # noqa: E402

C, R = D.C, D.R
from expkit.simx import evaluate_x, run_x         # noqa: E402

OUT = R.OUT / "headline"
SMOKE_OUT = R.SMOKE_OUT / "headline"
FAMILIES = ("dqn", "ppo", "thompson", "linucb", "ablation")
EVAL_SEEDS = 50
NAMES = {"dqn": "DQN", "ppo": "PPO", "linucb": "LinUCB", "thompson": "Thompson Sampling",
         "ablation": "DQN without H-Score"}


def no_power_throttling():
    """Windows: opt this process out of EcoQoS throttling (copied from sens_world.py)."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class STATE(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]
    st = STATE(1, 0x1, 0x0)
    k = ctypes.windll.kernel32
    k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))


def free_mb():
    """Free physical memory in MB (Windows; elsewhere a large number)."""
    if os.name != "nt":
        return 1e9
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] +                    [(n, ctypes.c_ulonglong) for n in ("tot", "avail", "tpf", "apf", "tv", "av", "x")]
    m = MS(); m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2 ** 20


def tasks():
    t = [{"family": f, "seed": s} for f in FAMILIES for s in range(C.SEEDS)]
    return t + [{"family": "statics", "seed": -1}]


def name(task):
    return f"{task['family']}_s{task['seed']}"


def evaluate(pol, label, p, eval_seeds, tags, volumes=D.SCORED_VOLUMES, fractions=R.BOT_FRACTIONS):
    rows = []
    grid = [("fixed", v) for v in volumes] + [("varying", f) for f in fractions]
    for g, level in grid:
        for s in range(eval_seeds):
            nh, nb = (100, level) if g == "fixed" else R._population(level, s)
            res, _ = run_x(pol, p.ev_h, p.ev_b, nb, cache=p.cache, n_humans=nh, seed=s,
                           solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
            m = evaluate_x(res)
            rows.append({"policy": label, "grid": g, "level": level, "eval_seed": s,
                         "n_humans": nh, "n_bots": nb, "DI": m.DI, "BOS": m.BOS,
                         "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                         "human_survival": res.surviving_humans / nh,
                         "bot_survival": res.surviving_bots / max(nb, 1),
                         "friction_s": res.mean_human_friction_seconds,
                         "false_positive_rate": res.false_positive_rate,
                         "abandonment_rate": res.abandonment_rate, **tags})
    return pd.DataFrame(rows)


def run_task(i, smoke=False):
    no_power_throttling()
    task = tasks()[i]
    out = (SMOKE_OUT if smoke else OUT) / f"{name(task)}.csv"
    if out.exists():
        print("exists:", out)
        return
    t0 = time.time()
    fam, seed = task["family"], task["seed"]
    p = C.pools(final=not smoke, smoke=smoke, reason=f"R6 grounded headline {name(task)}")
    eval_seeds = 2 if smoke else EVAL_SEEDS
    vol = (20, 100) if smoke else D.SCORED_VOLUMES
    fr = R.BOT_FRACTIONS[:2] if smoke else R.BOT_FRACTIONS
    if fam == "statics":
        df = pd.concat([evaluate(pol, pol.name, p, eval_seeds, {"family": "statics", "train_seed": -1},
                                 vol, fr) for pol in C.statics()], ignore_index=True)
    else:
        ch = D.choice(smoke)
        base = "dqn" if fam == "ablation" else fam
        spec = ch[base].split(":", 1)[1]
        post = [spec.split(":")[1]] if base in ("linucb", "thompson") else None
        variants, _ = R.train_candidate(fam, spec, seed, p.train_h, p.train_b, p.cache,
                                        3 if smoke else R.EPISODES, posteriors=post,
                                        solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
        (tag, pol), = variants.items()
        if hasattr(pol, "eval_mode"):
            pol.eval_mode()
        ck = out.parent / "agents"
        ck.mkdir(parents=True, exist_ok=True)
        pol.save(ck / f"{name(task)}.pt")
        df = evaluate(pol, NAMES[fam], p, eval_seeds,
                      {"family": fam, "candidate": tag, "train_seed": seed}, vol, fr)
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(task)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_all(jobs, ts_jobs=3):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T = tasks()
    done = lambda i: (OUT / f"{name(T[i])}.csv").exists()
    is_ts = lambda i: T[i]["family"] == "thompson"
    todo = sorted((i for i in range(len(T)) if not done(i)), key=lambda i: not is_ts(i))
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running, tries = {}, {i: 0 for i in todo}
    while todo or running:
        n_ts = sum(is_ts(i) for i in running)
        for i in list(todo):
            if len(running) >= jobs:
                break
            if is_ts(i) and (n_ts >= ts_jobs or free_mb() < 2500):
                continue
            todo.remove(i)
            n_ts += is_ts(i)
            tries[i] += 1
            log = open(logs / f"{i}.log", "w")
            running[i] = (subprocess.Popen([sys.executable, __file__, "--task", str(i)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env), log)
        for i, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[i]
                print(f"task {i} {name(T[i])} exit {proc.returncode}"
                      f"{'' if done(i) else ' NO CSV'} ({len(todo)} queued)", flush=True)
                if not done(i) and tries[i] <= 2:
                    todo.insert(0, i)
        time.sleep(5)


def collect():
    import numpy as np
    from scipy import stats

    files = sorted(OUT.glob("*.csv"))
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(files) != len(tasks()):
        print(f"WARNING: {len(files)} of {len(tasks())} tasks present")
    runs.to_csv(OUT.parent / "headline_runs.csv", index=False)
    cols = ["DI", "BOS", "SP_F1", "SI_F1", "human_survival", "bot_survival", "friction_s",
            "false_positive_rate", "abandonment_rate"]
    rows = []
    for (g, pol), f in runs.groupby(["grid", "policy"]):
        per_seed = f.groupby("train_seed")[cols].mean()
        n = len(per_seed)
        half = (stats.t.ppf(0.975, n - 1) * per_seed.DI.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({"grid": g, "policy": pol, "n_seeds": n, **per_seed.mean().to_dict(),
                     "DI_ci95": half, "DI_seed_min": per_seed.DI.min()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT.parent / "headline_summary.csv", index=False)
    by_level = runs[runs.grid == "fixed"].groupby(["policy", "level"]).DI.mean().unstack()
    by_level.to_csv(OUT.parent / "headline_fixed_by_volume.csv")
    by_frac = runs[runs.grid == "varying"].groupby(["policy", "level"]).DI.mean().unstack()
    by_frac.to_csv(OUT.parent / "headline_varying_by_share.csv")
    print(summary.round(3).to_string(index=False))
    print("\nfixed grid, DI by bot volume\n", by_level.round(1).to_string())
    print("\nvarying grid, DI by bot share\n", by_frac.round(1).to_string())
    # Consistency with round 4's offline panel at rho = 0 (eval seeds 0-19, same cells).
    ref = HERE / "deliverables_fold2" / "drift" / "drift_per_seed.csv"
    if ref.exists():
        r4 = pd.read_csv(ref)
        r4 = r4[(r4.panel == "offline") & (r4.eval_rho == 0) & (r4.eval_direction == "bots")]
        mine = runs[(runs.grid == "fixed") & (runs.eval_seed < 20)]
        mine = mine.groupby(["policy", "train_seed"]).DI.mean().rename("DI_now")
        cmp_ = r4.groupby(["policy", "train_seed"]).DI.mean().rename("DI_round4").to_frame().join(mine, how="inner")
        cmp_["diff"] = cmp_.DI_now - cmp_.DI_round4
        cmp_.to_csv(OUT.parent / "headline_vs_round4.csv")
        print("\nvs round 4 offline rho=0 (eval seeds 0-19)\n", cmp_.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.list:
        for i, t in enumerate(tasks()):
            print(i, t)
        return
    if a.collect:
        return collect()
    if a.smoke:
        log = R.OUT / "eval_access.log"
        before = log.read_text() if log.exists() else ""
        for i in (0, 5, 10, 15, 20, 25):
            run_task(i, smoke=True)
        after = log.read_text() if log.exists() else ""
        assert after == before, "smoke opened the eval block"
        print("smoke: OK (eval block untouched)")
        return
    if a.run_all:
        return run_all(a.run_all)
    if a.task is not None:
        return run_task(a.task)
    ap.error("one of --list, --smoke, --task N, --run-all J, --collect")


if __name__ == "__main__":
    main()
