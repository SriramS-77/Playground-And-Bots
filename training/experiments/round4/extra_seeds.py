"""Seeds 3 and 4 for Tables 5 and 6 (and Fig. 4's audit markers): runs only the missing tasks.

np_audit: PPO and DQN audit_recal / audit_ch (control, bots, and PPO humans) ; gated: all arms.
Run with N_SEEDS=5 so task indices match the scripts' 5-seed task lists.
    N_SEEDS=5 python round4/extra_seeds.py J
"""
import os, subprocess, sys, time
from pathlib import Path
assert os.environ.get("N_SEEDS") == "5"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import np_audit as NA, gated as GA

jobs = []
for i, t in enumerate(NA.tasks()):
    if t.get("arm") in ("audit_recal", "audit_ch") and t["seed"] >= 3 and \
            (t["algo"] == "ppo" or t.get("direction", "bots") == "bots"):
        jobs.append((HERE / "np_audit.py", i, NA.OUT / f"{NA.name(t)}.csv"))
for i, t in enumerate(GA.tasks()):
    if t["seed"] >= 3:
        jobs.append((HERE / "gated.py", i, GA.OUT / f"{GA.name(t)}.csv"))
jobs = [j for j in jobs if not j[2].exists()]
print(len(jobs), "tasks", flush=True)
env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
running, J = [], int(sys.argv[1])
while jobs or running:
    while jobs and len(running) < J:
        f, i, out = jobs.pop(0)
        log = open(out.parent / "logs" / f"x5_{f.stem}_{i}.log", "w")
        running.append((subprocess.Popen([sys.executable, str(f), "--task", str(i)], stdout=log,
                                         stderr=subprocess.STDOUT, env=env), log, out))
    for r in list(running):
        if r[0].poll() is not None:
            r[1].close(); running.remove(r)
            print(r[2].name, "exit", r[0].returncode, "" if r[2].exists() else "NO CSV", flush=True)
    time.sleep(5)
