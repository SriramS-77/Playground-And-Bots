"""Round 4's drift study rerun with leave-after-pass switched off (RLC_ABANDON=none), into a
separate results root (ROUND4_OUT=results/round4_noabandon/fold2), so the paper's runs are untouched.

Only the drift tasks the paper reports: metrics, statics, offline families (5 seeds), control and
online training for the oracle, score-only and posterior rewards -- covert bots at every rho, bot-like
humans at rho 0.5 and 1. Each task is `r4_fold2.py --stage drift --task i`, J at a time.

    python round4/noabandon_drift.py --list | --run-all J
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP / "results" / "round4_noabandon" / "fold2"
ENV = {"ROUND4_OUT": str(ROOT), "RLC_ABANDON": "none"}
ARMS = ("oracle", "score-only", "posterior")


def selected():
    os.environ.update(ENV)
    sys.argv = [sys.argv[0], "--stage", "drift"]
    sys.path.insert(0, str(HERE))
    import r4_fold2 as D
    keep = []
    for i, t in enumerate(D.drift_tasks()[:D.ROUND4_TASKS]):
        k = t["kind"]
        if k in ("metrics", "offline"):
            keep.append(i)
        elif k == "control" and t["arm"] in ARMS:
            keep.append(i)
        elif k == "online" and t["arm"] in ARMS and (t["direction"] == "bots" or t["rho"] in (0.5, 1.0)):
            keep.append(i)
    return keep, D.drift_tasks()


def run_all(jobs):
    keep, T = selected()
    logs = ROOT / "drift" / "noabandon_logs"
    logs.mkdir(parents=True, exist_ok=True)
    done = lambda i: (logs / f"{i}.done").exists()
    todo = [i for i in keep if not done(i)]
    todo.sort(key=lambda i: T[i].get("algo") != "dqn")         # long DQN runs first
    env = {**os.environ, **ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            log = open(logs / f"{i}.log", "w")
            running[i] = (subprocess.Popen([sys.executable, str(HERE / "r4_fold2.py"), "--stage", "drift",
                                            "--task", str(i)], stdout=log, stderr=subprocess.STDOUT,
                                           env=env, cwd=str(EXP)), log)
        for i, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[i]
                if proc.returncode == 0:
                    (logs / f"{i}.done").touch()
                print(f"task {i} {T[i]} exit {proc.returncode}", flush=True)
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run-all", type=int)
    a = ap.parse_args()
    if a.list:
        keep, T = selected()
        for i in keep:
            print(i, T[i])
        print(len(keep), "tasks")
    elif a.run_all:
        run_all(a.run_all)


if __name__ == "__main__":
    main()
