"""Round-4 sensitivity check (local): does the drift story survive a different human model?

WORLDS
  grounded   round 3/4's: humans may leave AFTER passing (P = 1 - exp(-s/45 s)); level 10 is
             a very hard challenge (humans pass 52 %, strong bots sometimes).
  v2         `GROUNDED_V2` / `GROUNDED_V2_CFG`: humans may leave when a VISIBLE challenge is
             SHOWN (and again before the retry), never after passing, at Searles et al.'s
             (USENIX Sec 2023) per-type rates and front-loaded per-position multipliers
             (`expkit.stochastic.PRESENTATION_LEAVE` / `PRESENTATION_POSITION`); a leaver costs
             what a wrongly blocked human costs (-100); level 0 = no challenge, level 10 = hard
             block (nobody passes). Levels 1-9 keep the round-3 ladder.

DATA  rotation 0's rl_fit (train) and rl_val (evaluate), plus the external sets' INJECT halves
      (train) and HELD-OUT halves (evaluate). The eval block is NEVER opened. Scorer: cross-fit
      seed 1 (the drift stage's). Architectures: round 4's arch_choice.

PHASE offline   every family trained offline (oracle reward, 200 episodes) in each world, seeds
                0-1, evaluated frozen at rho = 0 and at rho = 1 in each direction.
                Thompson's posterior (gaussian / mog2 / mog3) for the online phase is the best
                mean DI at rho = 0 in world v2.
PHASE online    world v2 only. DQN, PPO, Thompson (that posterior); arms oracle, score-only,
                posterior, posterior (fixed share); 300 episodes, shift on from episode 100 at
                rho in LEVELS, or never (control); seeds 0-1. Evaluated at the training point
                and at rho = 0 (controls at every point).

    python round4/sens_world.py --list offline|online
    python round4/sens_world.py --run offline|online --jobs 8
    python round4/sens_world.py --task offline|online N
    python round4/sens_world.py --collect
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv_saved = list(sys.argv)
sys.argv = [sys.argv[0], "--stage", "drift"]          # r4_fold2 reads its stage at import
sys.path[:0] = [str(HERE)]
import r4_fold2 as R4                                   # noqa: E402
sys.argv = sys.argv_saved

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

C, R = R4.C, R4.R
OUT = R4.R4_ROOT / ("sens_statev2" if os.environ.get("RLC_STATE") == "v2" else "sens")   # state version
SEEDS = tuple(int(x) for x in os.environ.get("SENS_SEEDS", "0,1").split(","))   # SENS_SEEDS=0: one seed
LEVELS = (0.5, 1.0)
EVAL_SEEDS, VOLUMES = 10, (20, 100, 500)
OFFLINE_FAMILIES = ("dqn", "ppo", "thompson", "linucb", "ablation", "statics")
ONLINE_ALGOS = ("dqn", "ppo", "thompson")
ONLINE_ARMS = tuple(a for a in ("oracle", "score-only", "posterior", "posterior (fixed share)")
                    if os.environ.get("SENS_NO_FIXED") != "1" or a != "posterior (fixed share)")
TS_POSTERIORS = ("gaussian", "mog2", "mog3")


def world(name):
    from expkit.stochastic import GROUNDED_V2, GROUNDED_V2_CFG
    return {"grounded": (C.GROUNDED, C.GROUNDED_CFG), "v2": (GROUNDED_V2, GROUNDED_V2_CFG)}[name]


def offline_tasks():
    return [{"phase": "offline", "world": w, "family": f, "seed": s}
            for w in (("v2",) if os.environ.get("RLC_STATE") == "v2" else ("v2", "grounded"))
            for f in OFFLINE_FAMILIES
            for s in (SEEDS if f != "statics" else (0,))]


def online_tasks():
    t = []
    for a in ONLINE_ALGOS:
        for arm in ONLINE_ARMS:
            for s in SEEDS:
                t.append({"phase": "online", "kind": "control", "algo": a, "arm": arm, "seed": s})
                t += [{"phase": "online", "kind": "online", "algo": a, "arm": arm, "direction": d,
                       "rho": rho, "seed": s} for d in R4.DIRECTIONS for rho in LEVELS]
    return t


def tasks(phase):
    return offline_tasks() if phase == "offline" else online_tasks()


def name(task):
    keys = ("world", "family", "kind", "algo", "arm", "direction", "rho", "seed")
    return "__".join(str(task[k]) for k in keys if k in task).replace(" ", "_").replace("(", "").replace(")", "")


# --------------------------------------------------------------------------- #

def pools():
    from expkit.external import load_sessions
    p = C.pools(final=False, smoke=True)                 # rl_fit / rl_val -- never the eval block
    p.cache._table.update(R4.ext_scores())
    p.ext = load_sessions()
    return p


def evaluate(policy, label, p, pts, solve, cfg, tags):
    frames = []
    for d, rho in pts:
        h, b, real = R4.shifted(p, d, rho, held_out=True)   # rl_val + HELD-OUT external halves
        frames.append(C.evaluate(policy, label, h, b, p.cache, EVAL_SEEDS, VOLUMES, solve=solve,
                                 cfg=cfg, eval_direction=d, eval_rho=rho, **tags))
    return pd.concat(frames, ignore_index=True)


def offline_points():
    """Overridden by round 5 (every level)."""
    return [("none", 0.0), ("bots", 1.0), ("humans", 1.0)]


def run_offline(task):
    p = pools()
    solve, cfg = world(task["world"])
    pts = offline_points()
    tags = {"phase": "offline", "world": task["world"], "family": task["family"],
            "train_seed": task["seed"]}
    if task["family"] == "statics":
        return pd.concat([evaluate(pol, pol.name, p, pts, solve, cfg, tags) for pol in C.statics()],
                         ignore_index=True)
    ch = R4.choice(False)
    fam = task["family"]
    base = "dqn" if fam == "ablation" else fam
    spec = ch[base].split(":", 1)[1]
    post = list(TS_POSTERIORS) if fam == "thompson" else ["gaussian"] if fam == "linucb" else None
    if fam in ("thompson", "linucb"):
        spec = spec.split(":")[0]
    variants, _ = R.train_candidate(fam, spec, task["seed"], p.train_h, p.train_b, p.cache,
                                    R.EPISODES, posteriors=post, solve=solve, cfg=cfg)
    frames = []
    for tag, pol in variants.items():
        if hasattr(pol, "eval_mode"):
            pol.eval_mode()
        label = {"dqn": "DQN", "ppo": "PPO", "linucb": "LinUCB", "ablation": "DQN without H-Score",
                 "thompson": f"Thompson ({tag.rsplit(':', 1)[-1]})"}[fam]
        frames.append(evaluate(pol, label, p, pts, solve, cfg, {**tags, "candidate": tag}))
    return pd.concat(frames, ignore_index=True)


def best_ts_posterior():
    f = OUT / "offline.csv"
    if not f.exists():
        raise SystemExit("run --collect after the offline phase first")
    d = pd.read_csv(f)
    d = d[(d.world == "v2") & (d.family == "thompson") & (d.eval_direction == "none") & (d.bots > 0)]
    by = d.groupby("policy").DI.mean()
    return by.idxmax().split("(")[1].rstrip(")"), by


def run_online(task):
    """E9 / round 4's online protocol in world v2; Thompson via a hooked `bandits_x.run_x`."""
    import dataclasses
    import e7_proxy_reward as E
    import expkit.bandits_x as BX
    import expkit.posterior_reward as PR
    import expkit.trainer as T
    from expkit.posterior_reward import OutcomeModel
    from expkit.proxy import OBSERVABLE

    solve, cfg = world("v2")
    algo, arm, seed = task["algo"], task["arm"], task["seed"]
    retrain = task["kind"] == "retrain"          # the ceiling: shifted pool from episode 0, oracle
    online = task["kind"] in ("online", "retrain")
    assert not retrain or arm == "oracle"
    direction, rho = (task["direction"], task["rho"]) if online else ("none", 0.0)
    p = pools()
    after_h, after_b, real = R4.shifted(p, direction, rho, held_out=False)
    fixed = arm == "posterior (fixed share)"
    spec = dict(E.CONFIGS["posterior" if fixed else arm])
    hook = spec.pop("hook", None)
    spec.pop("episodes", None)
    om = calib = None
    if hook in E.POSTERIOR_HOOKS:
        om = OutcomeModel(solve, cfg)
        calib = E.calibration_for(hook, E.fit_calibration(p) if hook != "score_only" else None)
    real_pr = PR.posterior_rewards

    def pr_fixed(transitions, om_, calib_, **kw):
        kw["share"] = calib.pi_cal
        return real_pr(transitions, om_, calib_, **kw)

    st = {"ep": 0}
    real_run_x, real_bx_run_x = T.run_x, BX.run_x

    def shift(humans, bots, kw):
        if kw.get("collect") and humans is p.train_h:
            if online and st["ep"] >= (0 if retrain else R4.INJECT_EP):
                humans, bots = after_h, after_b
            st["ep"] += 1
        return humans, bots

    def run_x_hooked(policy, humans, bots, n_bots, **kw):
        humans, bots = shift(humans, bots, kw)
        return real_run_x(policy, humans, bots, n_bots, **kw)

    rng_relabel = __import__("random").Random(seed + 7)

    def bx_run_x_hooked(policy, humans, bots, n_bots, **kw):
        training = kw.get("collect") and humans is p.train_h
        humans, bots = shift(humans, bots, kw)
        res, tr = real_bx_run_x(policy, humans, bots, n_bots, **kw)
        if training and hook is not None:
            # the bandit learns from the immediate reward: relabel it exactly as DQN / PPO are,
            # and balance its replay on the OBSERVABLE outcome, never the class (as `train_dqn`)
            tr, _ = T.relabel(tr, spec.get("proxy_cfg"), rng_relabel)
            tr = [dataclasses.replace(t, is_bot=OBSERVABLE[t.outcome] == "failed_gone") for t in tr]
        return res, tr

    t0 = time.time()
    if fixed:     # BEFORE `hooked`: it binds posterior_rewards when entered
        PR.posterior_rewards = pr_fixed
    with E.hooked(hook, spec.get("proxy_cfg"), None, om=om, calib=calib):
        T.run_x, BX.run_x = run_x_hooked, bx_run_x_hooked
        try:
            if algo == "thompson":
                from exp_round3_fold2 import BANDIT_ARCHS
                post, _ = best_ts_posterior()
                arch_name = R4.choice(False)["thompson"].split(":")[1]
                arch = next(a for a in BANDIT_ARCHS if a.name == arch_name)
                agent, _ = BX.train_bandit("thompson", p.train_h, p.train_b, p.cache, arch=arch,
                                           episodes=R4.TOTAL, seed=seed,
                                           human_counts=R.TRAIN_HUMAN_RANGE,
                                           bot_choices=R4.BOT_VOLUMES, verbose=0, solve=solve, cfg=cfg)
                if post != "gaussian":
                    agent = BX.fit_posteriors_from(agent, agent._buffer, post)
                label = f"Thompson ({post}) {arm}"
            else:
                agent, _ = R4.train_online(algo, p.train_h, p.train_b, p.cache, seed, R4.TOTAL,
                                           False, solve=solve, cfg=cfg, **spec)
                label = f"{algo.upper()} {arm}"
        finally:
            T.run_x, BX.run_x = real_run_x, real_bx_run_x
            PR.posterior_rewards = real_pr
    assert st["ep"] == R4.TOTAL, f"{st['ep']} training episodes seen, expected {R4.TOTAL}"
    agent.eval_mode()
    pts = ([(direction, rho)] if retrain else [("none", 0.0), (direction, rho)] if online else
           [("none", 0.0)] + [(d, r) for d in R4.DIRECTIONS for r in LEVELS])
    tags = {"phase": "online", "world": "v2", "panel": task["kind"], "algo": algo, "arm": arm,
            "train_direction": direction, "train_rho": rho, "train_seed": seed,
            "train_min": (time.time() - t0) / 60}
    return evaluate(agent, label, p, pts, solve, cfg, tags)


def no_power_throttling():
    """Windows: opt this process out of EcoQoS power throttling (background processes otherwise
    run at reduced clocks / on efficiency cores). Affects only this process."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class STATE(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]
    st = STATE(1, 0x1, 0x0)      # EXECUTION_SPEED: control it, and turn throttling off
    k = ctypes.windll.kernel32
    k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))


def run_task(phase, i):
    no_power_throttling()
    task = tasks(phase)[i]
    out = OUT / phase / f"{name(task)}.csv"
    if out.exists():
        print("exists:", out)
        return
    t0 = time.time()
    df = run_offline(task) if phase == "offline" else run_online(task)
    df["wall_min"] = (time.time() - t0) / 60
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name(task)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def free_mb():
    """Free physical memory in MB (Windows; elsewhere a large number)."""
    if os.name != "nt":
        return 1e9
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("tot", "avail", "tpf", "apf", "tv", "av", "x")]
    m = MS(); m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2 ** 20


def already_running(phase):
    """Task ids of this phase already running in another process (a restarted runner adopts
    them instead of launching duplicates)."""
    if os.name != "nt":
        return set()
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout
    import re
    return {int(m) for m in re.findall(rf"--task {phase} (\d+)", out)}


def run_all(phase, jobs, ts_jobs=int(os.environ.get("SENS_TS_JOBS", 4)), retries=2):
    """Run every missing task, `jobs` at a time but at most `ts_jobs` Thompson tasks (their
    statistics refit allocates ~1 GB; 11 at once ran a 16 GB laptop out of memory). A task
    that exits without its CSV is retried up to `retries` times."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    is_ts = lambda i: tasks(phase)[i].get("algo") == "thompson" or tasks(phase)[i].get("family") == "thompson"
    done = lambda i: (OUT / phase / f"{name(tasks(phase)[i])}.csv").exists()
    adopted = already_running(phase)                      # left by a previous runner
    todo = [i for i in range(len(tasks(phase))) if not done(i) and i not in adopted]
    todo.sort(key=lambda i: not is_ts(i))                 # longest first
    tries = {i: 0 for i in todo}
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    last_check = time.time()
    while todo or running or adopted:
        if adopted and time.time() - last_check > 60:
            adopted &= already_running(phase)
            last_check = time.time()
        n_ts = sum(is_ts(i) for i in [*running, *adopted])
        for i in list(todo):
            if len(running) + len(adopted) >= jobs:
                break
            if is_ts(i) and (n_ts >= ts_jobs or free_mb() < 2500):
                continue
            todo.remove(i)
            n_ts += is_ts(i)
            tries[i] += 1
            log = open(OUT / "logs" / f"{phase}_{i}.log", "w")
            running[i] = (subprocess.Popen([sys.executable, __file__, "--task", phase, str(i)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env), log)
        for i, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[i]
                ok = done(i)
                print(f"[{phase}] task {i} exit {proc.returncode}{'' if ok else ' NO CSV'} "
                      f"(try {tries[i]}, {len(todo)} queued)", flush=True)
                if not ok and tries[i] <= retries:
                    todo.insert(0, i)
        time.sleep(5)


def collect():
    for phase in ("offline", "online"):
        files = sorted((OUT / phase).glob("*.csv"))
        if files:
            d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            d.to_csv(OUT / f"{phase}.csv", index=False)
            print(f"{phase}: {len(files)} task CSVs")
    if (OUT / "offline.csv").exists():
        post, by = best_ts_posterior()
        print("Thompson posterior for the online phase:", post, by.round(2).to_dict())
        (OUT / "ts_choice.json").write_text(json.dumps({"posterior": post, "DI_v2_rho0": by.to_dict()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", choices=("offline", "online"))
    ap.add_argument("--run", choices=("offline", "online"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--task", nargs=2)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, t in enumerate(tasks(a.list)):
            print(i, t)
    elif a.run:
        run_all(a.run, a.jobs)
    elif a.task:
        run_task(a.task[0], int(a.task[1]))
    elif a.collect:
        collect()


if __name__ == "__main__":
    main()
