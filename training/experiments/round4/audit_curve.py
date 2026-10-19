"""Audit-recalibrated rewards over Fig. 4's full covert-bot grid (rho 0.1..1.0), 5 seeds, PPO and DQN.
Runs only the missing np_audit tasks (online, bots direction, audit_recal / audit_ch).
    N_SEEDS=5 AUDIT_LEVELS=0.1,...,1.0 python round4/audit_curve.py J
"""
import os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import np_audit as NA

jobs = [(i, NA.OUT / f"{NA.name(t)}.csv") for i, t in enumerate(NA.tasks())
        if t["kind"] == "online" and t.get("direction") == "bots" and t["arm"] in ("audit_recal", "audit_ch")]
jobs = [j for j in jobs if not j[1].exists()]
jobs.sort(key=lambda j: "dqn" not in j[1].name)      # long DQN runs first
print(len(jobs), "tasks", flush=True)
env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
running, J = [], int(sys.argv[1])
while jobs or running:
    while jobs and len(running) < J:
        i, out = jobs.pop(0)
        log = open(NA.OUT / "logs" / f"curve_{i}.log", "w")
        running.append((subprocess.Popen([sys.executable, str(HERE / "np_audit.py"), "--task", str(i)],
                                         stdout=log, stderr=subprocess.STDOUT, env=env), log, out))
    for r in list(running):
        if r[0].poll() is not None:
            r[1].close(); running.remove(r)
            print(r[2].name, "exit", r[0].returncode, "" if r[2].exists() else "NO CSV", flush=True)
    time.sleep(5)
