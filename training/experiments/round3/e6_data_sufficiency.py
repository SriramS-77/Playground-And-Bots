"""E6 -- is the dataset big enough, and where does the uncertainty live? (Reviewer 1,
point 4; HANDOFF_ROUND3 §5 Priority 8.)

1. BOOTSTRAP over recordings. The headline's checkpoints (seed 0 of each learned family)
   and the two statics, each run on N_POOLS resampled eval pools -- humans and bots
   resampled separately, with replacement -- at SIM_SEEDS simulation seeds per pool. The
   CI is the percentile interval of the pool means, which is the uncertainty a reader
   cares about: seed-level intervals are tight however small the pool is.
   The same runs give a one-way variance decomposition per policy:
       between-pool variance (pool means, corrected for within-pool noise / SIM_SEEDS)
       vs within-pool (simulation) variance
   reported PER POLICY. Round 2 quoted one pooled "30 %", which was an average dominated
   by Static Single (0.78) while the DQN's own share was 0.04.

2. LEARNING CURVE. The DQN winner trained on stratified subsets of fold 2's rl block
   (25 / 50 / 75 / 100 %), 5 seeds each, evaluated on the eval block.

    tasks: 5 policies x (N_POOLS / POOLS_PER_TASK) bootstrap chunks  +  4 fractions x 5 seeds
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

import common as C
from expkit.crossfit import assign_parts
from expkit.simx import sessions_from_refs

EXP = "e6"
N_POOLS = 200
POOLS_PER_TASK = 20
SIM_SEEDS = 3
BOOT_VOLUMES = (100, 500)
POLICIES = ("dqn_s0", "thompson_s0", "linucb_s0", "Static Single-Threshold", "Static Multi-Threshold")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def tasks():
    t = [{"kind": "bootstrap", "policy": p, "chunk": c}
         for p in POLICIES for c in range(N_POOLS // POOLS_PER_TASK)]
    return t + [{"kind": "curve", "fraction": f, "seed": s} for f in FRACTIONS for s in range(C.SEEDS)]


def smoke_tasks():
    return [{"kind": "bootstrap", "policy": "dqn_s0", "chunk": 0},
            {"kind": "bootstrap", "policy": "Static Multi-Threshold", "chunk": 0},
            {"kind": "curve", "fraction": 0.5, "seed": 0}]


def _policy(name, smoke):
    from exp_round3_fold2 import load_policy
    for s in C.statics():
        if s.name == name:
            return s
    agents = C.headline_agents(smoke)
    if name not in agents:
        raise RuntimeError(f"checkpoint {name} not found among {sorted(agents)}")
    return load_policy(agents[name])


def run_task(task, smoke):
    episodes, eval_seeds, volumes = C.budget(smoke)
    if task["kind"] == "bootstrap":
        p = C.pools(final=True, smoke=smoke, reason=f"E6 bootstrap {task['policy']} chunk {task['chunk']}")
        pol = _policy(task["policy"], smoke)
        n_pools = 2 if smoke else POOLS_PER_TASK
        sim_seeds = 1 if smoke else SIM_SEEDS
        vols = volumes[:1] if smoke else BOOT_VOLUMES
        rows = []
        for i in range(n_pools):
            pool_id = task["chunk"] * POOLS_PER_TASK + i
            rng = random.Random(50_000 + pool_id)          # same pools for every policy
            bh = [rng.choice(p.ev_h) for _ in p.ev_h]
            bb = [rng.choice(p.ev_b) for _ in p.ev_b]
            df = C.evaluate(pol, pol.name, bh, bb, p.cache, sim_seeds, vols,
                            seed_offset=1000 * pool_id, pool=pool_id)
            rows.append(df)
        C.save(pd.concat(rows, ignore_index=True), EXP,
               f"boot_{task['policy'].replace(' ', '_')}_c{task['chunk']}", smoke)
        return
    # learning curve
    p = C.pools(final=True, smoke=smoke, reason=f"E6 curve f={task['fraction']} seed {task['seed']}")
    rl = list(p.fold.rl) if not smoke else C.R.rl_rotations(p.fold)[0][0]
    part = assign_parts(rl, 4, seed=7)
    keep = [r for r in rl if part[r.name] < round(4 * task["fraction"])]
    th, tb = sessions_from_refs(keep)
    agent, _ = C.train_winner(th, tb, p.cache, seed=task["seed"], episodes=episodes, smoke=smoke)
    df = C.evaluate(agent.eval_mode(), "DQN", p.ev_h, p.ev_b, p.cache, eval_seeds, volumes,
                    fraction=task["fraction"], n_recordings=len(keep), train_seed=task["seed"])
    C.save(df, EXP, f"curve_f{task['fraction']}_s{task['seed']}", smoke)


def collect(smoke):
    runs = C.load_tasks(EXP, smoke)
    d = C.out_dir(EXP, smoke)
    if "pool" in runs:
        b = runs[runs.pool.notna()]
        rows = []
        for (pol, nb), g in b.groupby(["policy", "bots"]):
            pool_means = g.groupby("pool").DI.mean()
            within = g.groupby("pool").DI.var(ddof=1).mean() if g.groupby("pool").size().min() > 1 else np.nan
            k = g.groupby("pool").size().mean()
            between = max(pool_means.var(ddof=1) - (within / k if np.isfinite(within) else 0), 0.0)
            total = between + (within if np.isfinite(within) else 0)
            rows.append({"policy": pol, "bots": nb, "n_pools": len(pool_means),
                         "DI_mean": pool_means.mean(),
                         "ci_lo": pool_means.quantile(0.025), "ci_hi": pool_means.quantile(0.975),
                         "var_between_pools": between, "var_within_pool": within,
                         "share_between_pools": between / total if total else np.nan})
        t = pd.DataFrame(rows)
        t.to_csv(d / "e6_bootstrap.csv", index=False)
        print(t.round(3).to_string(index=False))
    if "fraction" in runs:
        c = runs[runs.fraction.notna() & (runs.bots > 0)]
        per_seed = c.groupby(["fraction", "train_seed"])[["DI", "human_survival", "bot_survival", "friction_s"]].mean()
        t = per_seed.groupby("fraction").agg(["mean", "std"])
        t.to_csv(d / "e6_learning_curve.csv")
        print(f"\nlearning curve over rl-block size:\n{t.round(3).to_string()}")


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
