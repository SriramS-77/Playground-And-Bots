"""Thompson Sampling posterior variants on the held-out evaluation block (round 6).

Same protocol as grounded_headline.py's thompson family (h128_64_e64 network, 200 episodes,
5 training seeds, 50 eval seeds, fixed and varying traffic). Each seed trains the network once
and refits all four posteriors (gaussian diag, gaussian full, mog2, mog3) on the same weights
and buffer, so the variants differ in the posterior alone. mog3 reproduces Table 4's row.

    python round4/ts_variants.py --list | --task N | --run-all J | --collect
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
import grounded_headline as G                     # noqa: E402

import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
OUT = G.OUT.parent / "ts_variants"
POSTS = ("gaussian", "gaussian_full", "mog2", "mog3")


def tasks():
    return [{"seed": s} for s in range(C.SEEDS)]


def run_task(i):
    G.no_power_throttling()
    seed = tasks()[i]["seed"]
    out = OUT / f"ts_s{seed}.csv"
    if out.exists():
        return
    t0 = time.time()
    p = C.pools(final=True, smoke=False, reason=f"R6 TS posterior variants s{seed}")
    spec = D.choice(False)["thompson"].split(":", 1)[1].rsplit(":", 1)[0]   # h128_64_e64
    variants, _ = R.train_candidate("thompson", spec + ":mog3", seed, p.train_h, p.train_b, p.cache,
                                    R.EPISODES, posteriors=list(POSTS),
                                    solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
    dfs = []
    for tag, pol in variants.items():
        post = next(x for x in sorted(POSTS, key=len, reverse=True) if tag.endswith(x))
        dfs.append(G.evaluate(pol, f"TS {post}", p, G.EVAL_SEEDS,
                              {"family": "thompson", "posterior": post, "candidate": tag,
                               "train_seed": seed}))
        print(f"s{seed} {post} done {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def run_all(jobs):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    T_ = tasks()
    done = lambda i: (OUT / f"ts_s{T_[i]['seed']}.csv").exists()
    todo = [i for i in range(len(T_)) if not done(i)]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            log = open(OUT / "logs" / f"{i}.log", "w")
            running[i] = (subprocess.Popen([sys.executable, __file__, "--task", str(i)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env), log)
        for i, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[i]
                print(f"task {i} exit {proc.returncode}{'' if done(i) else ' NO CSV'}", flush=True)
        time.sleep(5)


def collect():
    r = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv"))], ignore_index=True)
    r.to_csv(OUT.parent / "ts_variants_runs.csv", index=False)
    f = r[r.grid == "fixed"]
    vol = f.groupby(["posterior", "level"]).DI.mean().unstack().round(1)
    seed = f.groupby(["posterior", "train_seed"]).DI.mean()
    ci = seed.groupby("posterior").agg(["mean", "std", "min", "count"])
    ci["ci95"] = 2.776 * ci["std"] / ci["count"] ** 0.5
    met = f.groupby("posterior")[["BOS", "SP_F1", "SI_F1", "human_survival", "friction_s"]].mean().round(3)
    vs = r[r.grid == "varying"].groupby(["posterior", "train_seed"]).DI.mean().groupby("posterior")
    var = vs.agg(["mean", "std"]); var["ci95"] = 2.776 * var["std"] / 5 ** 0.5
    print(vol.to_string()); print(ci.round(2).to_string()); print(met.to_string()); print(var.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.list:
        print(tasks())
    elif a.collect:
        collect()
    elif a.run_all:
        run_all(a.run_all)
    elif a.task is not None:
        run_task(a.task)


if __name__ == "__main__":
    main()
