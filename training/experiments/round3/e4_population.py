"""E4 -- does the policy read the attack level off total population? (Reviewer 1, point 5;
HANDOFF_ROUND3 §5 Priority 4.)

The published design had corr(total population, bots) = 1.0, so a policy could detect an
attack by counting users. Two stages:

  train   the DQN winner retrained on a varied grid -- humans ~ U(25, 400) independently
          of bots in BOT_VOLUMES -- 5 seeds, checkpoints saved. No evaluation.
  eval    each checkpoint on the eval block over humans {25, 50, 100, 200, 400} x bots
          {0, 20, 100, 200, 500, 1000}, with the population input
              unfrozen   as trained
              true       wrapper passing the TRUE n_active  -- the self-test
              c          frozen at c in {25, 100, 200, 400, 800, 1400}
              random     n_active ~ U(25, 1400) per decision -- OUT OF DISTRIBUTION;
                         reported, never used as evidence

The wrapper is tested before it is trusted: `true` must reproduce `unfrozen` EXACTLY
(asserted in-task on three cells, and on every cell in --collect). In round 2, five of six
frozen constants returned DI 0.00 with 100 % of humans AND 100 % of bots surviving -- a
policy that never blocks -- and that was reported as "freezing costs nothing". Here every
variant also reports the share of decisions that were challenges, so a policy that stops
acting is visible as exactly that.

Human survival, bot survival and friction are reported separately: DI conflates "ignores
the feature" with "became more permissive". No verdict is hardcoded.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

import common as C

EXP = "e4"
TRAIN_HUMANS = (25, 400)
GRID_HUMANS = (25, 50, 100, 200, 400)
GRID_BOTS = (0, 20, 100, 200, 500, 1000)
CONSTANTS = (25, 100, 200, 400, 800, 1400)
VARIANTS = ["unfrozen", "true", *map(str, CONSTANTS), "random"]


class PopulationInput:
    """Wraps any policy; replaces the n_active it sees. Delegates everything else."""

    def __init__(self, inner, mode, seed=0):
        self.inner, self.mode = inner, mode
        self.rng = random.Random(seed)
        self.name = f"{inner.name} [n_active={mode}]"

    def __getattr__(self, attr):                  # needs_bot_score, flags, features, ...
        return getattr(self.inner, attr)

    def _n(self, n_active):
        if self.mode == "true":
            return n_active
        if self.mode == "random":
            return self.rng.randint(25, 1400)
        return int(self.mode)

    def select_action(self, user, n_active):
        return self.inner.select_action(user, self._n(n_active))

    def features(self, user, n_active):
        return self.inner.features(user, self._n(n_active))


def tasks():
    t = [{"kind": "train", "seed": s} for s in range(C.SEEDS)]
    t += [{"kind": "eval", "seed": s, "variant": v} for s in range(C.SEEDS) for v in VARIANTS]
    return t + [{"kind": "statics"}]


def smoke_tasks():
    return [{"kind": "train", "seed": 0}, {"kind": "eval", "seed": 0, "variant": "unfrozen"},
            {"kind": "eval", "seed": 0, "variant": "true"},
            {"kind": "eval", "seed": 0, "variant": "200"}, {"kind": "statics"}]


def _ckpt(seed, smoke):
    return C.out_dir(EXP, smoke) / "agents" / f"dqn_grid_s{seed}.pt"


def _grid(policy, name, p, eval_seeds, cells, **tags):
    frames = []
    for nh in cells["humans"]:
        frames.append(C.evaluate(policy, name, p.ev_h, p.ev_b, p.cache, eval_seeds,
                                 cells["bots"], n_humans=nh, **tags))
    return pd.concat(frames, ignore_index=True)


def run_task(task, smoke):
    from exp_round3_fold2 import load_policy

    episodes, eval_seeds, _ = C.budget(smoke)
    cells = ({"humans": (25, 100), "bots": (0, 100)} if smoke
             else {"humans": GRID_HUMANS, "bots": GRID_BOTS})
    if task["kind"] == "train":
        p = C.pools(final=False, smoke=smoke)                  # training only
        agent, _ = C.train_winner(p.train_h, p.train_b, p.cache, seed=task["seed"],
                                  episodes=episodes, smoke=smoke, human_counts=TRAIN_HUMANS)
        path = _ckpt(task["seed"], smoke)
        path.parent.mkdir(parents=True, exist_ok=True)
        agent.eval_mode().save(path)
        return
    if task["kind"] == "eval" and not _ckpt(task["seed"], smoke).exists():
        raise RuntimeError(f"{_ckpt(task['seed'], smoke)} missing -- E4's train tasks (0-4) must "
                           "finish first: sbatch --array=0-4 ..., then --array=5-50 "
                           "--dependency=afterok:<jobid>")
    p = C.pools(final=True, smoke=smoke, reason=f"E4 eval {task}")
    if task["kind"] == "statics":
        df = pd.concat([_grid(pol, pol.name, p, eval_seeds, cells, variant="(static)", train_seed=-1)
                        for pol in C.statics()])
        C.save(df, EXP, "statics", smoke)
        return
    base = load_policy(_ckpt(task["seed"], smoke))
    v = task["variant"]
    pol = base if v == "unfrozen" else PopulationInput(base, v, seed=task["seed"])
    if v == "true":                                         # the wrapper's self-test
        for nh, nb, s in ((25, 0, 0), (100, 100, 1), (cells["humans"][-1], cells["bots"][-1], 2)):
            a, _ = C.run_x(base, p.ev_h, p.ev_b, nb, cache=p.cache, n_humans=nh, seed=s)
            b, _ = C.run_x(pol, p.ev_h, p.ev_b, nb, cache=p.cache, n_humans=nh, seed=s)
            assert (a.surviving_humans, a.surviving_bots, a.action_hist) == \
                   (b.surviving_humans, b.surviving_bots, b.action_hist), \
                   "wrapper with the TRUE n_active changed the policy -- the wrapper is broken"
        print("  [ok] wrapper self-test: n_active=true reproduces the unwrapped policy exactly")
    df = _grid(pol, "DQN (grid-trained)", p, eval_seeds, cells, variant=v, train_seed=task["seed"])
    C.save(df, EXP, f"s{task['seed']}_{v}", smoke)


def collect(smoke):
    runs = C.load_tasks(EXP, smoke)
    d = C.out_dir(EXP, smoke)
    cols = ["human_survival", "bot_survival", "friction_s", "challenge_share"]

    # The self-test on every cell, not just three.
    key = ["train_seed", "n_humans", "bots", "eval_seed"]
    u = runs[runs.variant == "unfrozen"].set_index(key)[cols].sort_index()
    t = runs[runs.variant == "true"].set_index(key)[cols].sort_index()
    common_idx = u.index.intersection(t.index)
    if len(u) and len(t):
        # Never let the self-test skip itself: if both variants ran, they must overlap.
        assert len(common_idx), "'unfrozen' and 'true' share no cells -- the self-test did not run"
        diff = (u.loc[common_idx] - t.loc[common_idx]).abs().to_numpy().max()
        assert diff == 0, f"variant 'true' differs from 'unfrozen' by {diff} -- wrapper broken"
        print(f"wrapper self-test on all {len(common_idx)} cells: identical")
    elif len(u) or len(t):
        print("WARNING: only one of 'unfrozen' / 'true' present -- the self-test could not run")

    table = runs.groupby("variant")[cols].mean()
    table["DI (bots>0)"] = runs[runs.bots > 0].groupby("variant").DI.mean()
    order = [v for v in ["unfrozen", "true", *map(str, CONSTANTS), "random", "(static)"] if v in table.index]
    table = table.reindex(order)
    table["note"] = ["out of distribution -- not evidence" if v == "random" else "" for v in table.index]
    table.to_csv(d / "e4_variants.csv")
    print(table.round(3).to_string())

    num = table[cols].astype(float)              # `note` makes a row's dtype object
    froz = [str(c) for c in CONSTANTS if str(c) in num.index]
    if "unfrozen" in num.index and froz:
        delta = (num.loc[froz] - num.loc["unfrozen"]).abs().max()
        print("\nlargest |frozen constant - unfrozen| per metric (derive the conclusion from these):")
        print(delta.round(3).to_string())
    by_cell = runs[runs.variant.isin(["unfrozen", "(static)"] + [str(c) for c in CONSTANTS])] \
        .groupby(["variant", "n_humans", "bots"])[["DI", "human_survival", "bot_survival", "friction_s"]].mean()
    by_cell.to_csv(d / "e4_by_cell.csv")


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
