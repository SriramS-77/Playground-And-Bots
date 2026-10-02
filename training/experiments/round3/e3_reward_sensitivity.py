"""E3 -- are the conclusions an artefact of hand-chosen reward constants? (Reviewer 1,
point 3; HANDOFF_ROUND3 §5 Priority 5.)

Three parts.

1. RETRAINED sensitivity. The DQN winner retrained under each setting, 5 seeds, and
   evaluated in the published environment. Re-scoring a fixed policy under a new reward
   is provably a no-op here -- survival depends only on blocking and abandonment, and the
   reward constants enter neither -- so retraining is the only meaningful test.
   Settings vary ONE constant from the published one at a time:

       published        friction 2^T, R_leak -150, overkill 2.5, underestimation 5
       gentle / harsh   friction base 1.5 / 2.5
       cheap / costly   R_leak -50 / -300
       overkill 2.0
       under 2.5 / 10   underestimation
       linear friction  2 reward per measured second instead of 2^T

2. Fixed-policy LANDSCAPE: which policy does each of 18 reward settings rank highest?
   A property of the reward, measured on fold 2's rl block -- the eval block is not
   needed for it, so it is not opened.

3. ABANDONMENT curve -- the one parameter that moves the metrics directly. The headline's
   checkpoints and the statics under `paper`, empirical tau in {22, 45, 80}, and `none`,
   on the eval block.

--collect writes every table, the Pareto frontier (bots blocked vs friction seconds), and
computes -- never asserts -- how many retrained settings beat every static baseline.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd

import common as C
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT

EXP = "e3"
P = PAPER_EQUIVALENT
SETTINGS = {
    "published": P,
    "gentle friction": dataclasses.replace(P, friction_base=1.5),
    "harsh friction": dataclasses.replace(P, friction_base=2.5),
    "cheap leak": dataclasses.replace(P, leakage_penalty=-50.0),
    "costly leak": dataclasses.replace(P, leakage_penalty=-300.0),
    "overkill 2.0": dataclasses.replace(P, overkill=2.0),
    "under 2.5": dataclasses.replace(P, underestimation=2.5),
    "under 10.0": dataclasses.replace(P, underestimation=10.0),
    "linear friction": dataclasses.replace(P, friction_base=None, friction_per_second=2.0),
}
LANDSCAPE = [dataclasses.replace(P, friction_base=f, leakage_penalty=l, overkill=o)
             for f, l, o in itertools.product((1.5, 2.0, 2.5), (-50.0, -150.0, -300.0), (2.0, 2.5))]
ABANDONMENT = {"paper": P,
               **{f"empirical tau={t}": dataclasses.replace(P, abandonment="empirical",
                                                         abandonment_tau=float(t))
                  for t in (22, 45, 80)},
               "none": dataclasses.replace(P, abandonment="none")}


def tasks():
    t = [{"kind": "retrain", "setting": s, "seed": i} for s in SETTINGS for i in range(C.SEEDS)]
    t += [{"kind": "statics"}]
    t += [{"kind": "landscape", "cell": i} for i in range(len(LANDSCAPE))]   # one per setting
    return t + [{"kind": "abandonment", "curve": c} for c in ABANDONMENT]


def smoke_tasks():
    return [{"kind": "retrain", "setting": "published", "seed": 0},
            {"kind": "retrain", "setting": "linear friction", "seed": 0},
            {"kind": "statics"}, {"kind": "landscape", "cell": 0}, {"kind": "landscape", "cell": 9},
            {"kind": "abandonment", "curve": "empirical tau=45"}]


def _agents(smoke):
    """Seed 0 of each learned family, plus the statics. The landscape and the abandonment
    curves are properties of the reward and the environment, not of seed spread; all 15
    checkpoints x 18 settings would be ~30,000 episodes, bandits at 1000 bots included."""
    from exp_round3_fold2 import load_policy
    pols = [load_policy(p) for k, p in C.headline_agents(smoke).items()
            if k.endswith("_s0") and not k.startswith("ablation")]
    return pols + C.statics()


def run_task(task, smoke):
    episodes, eval_seeds, volumes = C.budget(smoke)
    k = task["kind"]
    if k == "retrain":
        p = C.pools(final=True, smoke=smoke, reason=f"E3 retrain {task['setting']} seed {task['seed']}")
        agent, _ = C.train_winner(p.train_h, p.train_b, p.cache, seed=task["seed"],
                                  episodes=episodes, smoke=smoke, cfg=SETTINGS[task["setting"]])
        df = C.evaluate(agent.eval_mode(), "DQN", p.ev_h, p.ev_b, p.cache, eval_seeds, volumes,
                        setting=task["setting"], train_seed=task["seed"])
        name = f"retrain_{task['setting'].replace(' ', '_')}_s{task['seed']}"
    elif k == "statics":
        p = C.pools(final=True, smoke=smoke, reason="E3 statics")
        df = pd.concat([C.evaluate(pol, pol.name, p.ev_h, p.ev_b, p.cache, eval_seeds, volumes,
                                   setting="(static)") for pol in C.statics()])
        name = "statics"
    elif k == "landscape":
        p = C.pools(final=False, smoke=smoke)                # rl block: eval stays closed
        rows = []
        i = task["cell"]
        for cfg in (LANDSCAPE[i],):
            for pol in _agents(smoke):
                for nb in volumes:
                    for s in range(eval_seeds):
                        res, _ = C.run_x(pol, p.train_h, p.train_b, nb, cache=p.cache,
                                         n_humans=100, seed=s, solve=DETERMINISTIC, cfg=cfg)
                        n = res.n_humans + res.n_bots
                        rows.append({"cell": i, "friction_base": cfg.friction_base,
                                     "leakage_penalty": cfg.leakage_penalty, "overkill": cfg.overkill,
                                     "policy": pol.name, "bots": nb, "eval_seed": s,
                                     "reward_per_user": (res.human_session_reward * res.n_humans
                                                         + res.bot_session_reward * res.n_bots) / n})
        df, name = pd.DataFrame(rows), f"landscape_c{i}"
    else:
        p = C.pools(final=True, smoke=smoke, reason=f"E3 abandonment {task['curve']}")
        cfg = ABANDONMENT[task["curve"]]
        df = pd.concat([C.evaluate(pol, pol.name, p.ev_h, p.ev_b, p.cache, eval_seeds, volumes,
                                   cfg=cfg, curve=task["curve"]) for pol in _agents(smoke)])
        name = f"abandonment_{task['curve'].replace(' ', '_').replace('=', '')}"
    C.save(df, EXP, name, smoke)


def pareto(t: pd.DataFrame) -> pd.Series:
    """Non-dominated on (more bots blocked, less friction)."""
    b, f = 1 - t["bot_survival"], t["friction_s"]
    return pd.Series([not ((b >= b[i]) & (f <= f[i]) & ((b > b[i]) | (f < f[i]))).any()
                      for i in t.index], index=t.index)


def collect(smoke):
    runs = C.load_tasks(EXP, smoke)
    d = C.out_dir(EXP, smoke)
    cols = ["DI", "human_survival", "bot_survival", "friction_s"]

    rt = runs[runs.get("setting").notna()] if "setting" in runs else pd.DataFrame()
    if len(rt):
        rt = rt[rt.bots > 0]
        per_seed = rt.groupby(["setting", "policy", "train_seed"], dropna=False)[cols].mean()
        t = rt.groupby(["setting", "policy"])[cols].mean()
        t["DI_seed_sd"] = per_seed.groupby(["setting", "policy"]).DI.std()
        t = t.reset_index()
        t["pareto"] = pareto(t.reset_index(drop=True)).values
        t.to_csv(d / "e3_retrained.csv", index=False)
        print(t.round(3).to_string(index=False))
        st = t[t.setting == "(static)"]
        if len(st):
            best_static = st.DI.max()
            learned = t[t.setting != "(static)"]
            beat = int((learned.DI > best_static).sum())
            print(f"\n{beat} of {len(learned)} retrained settings beat every static baseline "
                  f"(best static mean DI {best_static:.1f}) -- computed, not asserted")

    if "cell" in runs:
        ls = runs[runs.cell.notna()]
        rank = ls.groupby(["cell", "friction_base", "leakage_penalty", "overkill", "policy"]) \
                 .reward_per_user.mean().unstack("policy")
        rank["preferred"] = rank.idxmax(axis=1)
        rank.to_csv(d / "e3_landscape.csv")
        share = rank.preferred.value_counts(normalize=True)
        print(f"\nfixed-policy landscape -- share of {len(rank)} reward settings preferring each "
              f"policy:\n{share.round(3).to_string()}")

    if "curve" in runs:
        ab = runs[runs.curve.notna() & (runs.bots > 0)]
        t = ab.groupby(["curve", "policy"])[cols].mean().unstack("curve")
        t.to_csv(d / "e3_abandonment.csv")
        print(f"\nabandonment curve (DI):\n{t['DI'].round(1).to_string()}")


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
