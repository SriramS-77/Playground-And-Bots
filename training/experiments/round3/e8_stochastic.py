"""E8 -- usability modelled, not assumed: a probabilistic action model. (Reviewer 2;
HANDOFF_ROUND3 §5 Priority 6.)

The published rule T > B_s is the alpha -> infinity limit of a two-parameter logistic
item-response model, P(solve) = sigma(alpha * (B_s + 0.5 - T)), so the stochastic world is a
continuous deformation of the published one -- verified 144/144 by `--prepare`'s
determinism check. Humans pass each level with the measured probability of the level
ladder (`expkit.stochastic.HUMAN_PASS`) and can fail, which the deterministic simulator
cannot express at all: false positives exist only here.

1. ALPHA SWEEP: every headline checkpoint and both statics, evaluated in the grounded
   world at alpha in {0.5, 1, 1.5, 3, 6, 12} and the deterministic-bots limit
   (`DETERMINISTIC_FALLIBLE_HUMANS`). How do policies trained in the deterministic world
   degrade as the world becomes stochastic?
2. RETRAINED: the DQN winner trained IN the grounded world (alpha 1.5), 5 seeds, matched
   budget -- round 1's 90-episode attempt produced an uninterpretable bimodal policy.

Reported throughout: DI, false-positive rate, abandonment rate, friction seconds.
"""

from __future__ import annotations

import pandas as pd

import common as C
from expkit.stochastic import DETERMINISTIC_FALLIBLE_HUMANS, SolveModel

EXP = "e8"
ALPHAS = (0.5, 1.0, 1.5, 3.0, 6.0, 12.0, "deterministic")


def _solve(alpha):
    return DETERMINISTIC_FALLIBLE_HUMANS if alpha == "deterministic" else SolveModel(alpha=float(alpha))


def tasks():
    from exp_round3_fold2 import headline_tasks
    names = [f"{t['policy']}_s{t['seed']}" for t in headline_tasks() if t["policy"] != "statics"]
    t = [{"kind": "sweep", "policy": n} for n in names] + [{"kind": "sweep", "policy": "statics"}]
    return t + [{"kind": "retrain", "seed": s} for s in range(C.SEEDS)]


def smoke_tasks():
    return [{"kind": "sweep", "policy": "dqn_s0"}, {"kind": "sweep", "policy": "statics"},
            {"kind": "retrain", "seed": 0}]


def run_task(task, smoke):
    from exp_round3_fold2 import load_policy

    episodes, eval_seeds, volumes = C.budget(smoke)
    if task["kind"] == "sweep":
        p = C.pools(final=True, smoke=smoke, reason=f"E8 alpha sweep {task['policy']}")
        pols = C.statics() if task["policy"] == "statics" else \
            [load_policy(C.headline_agents(smoke)[task["policy"]])]
        alphas = ALPHAS[2:3] + ALPHAS[-1:] if smoke else ALPHAS
        frames = [C.evaluate(pol, pol.name, p.ev_h, p.ev_b, p.cache, eval_seeds, volumes,
                             solve=_solve(a), cfg=C.GROUNDED_CFG, alpha=str(a),
                             checkpoint=task["policy"])
                  for pol in pols for a in alphas]
        C.save(pd.concat(frames, ignore_index=True), EXP, f"sweep_{task['policy']}", smoke)
        return
    p = C.pools(final=True, smoke=smoke, reason=f"E8 retrain grounded seed {task['seed']}")
    agent, _ = C.train_winner(p.train_h, p.train_b, p.cache, seed=task["seed"], episodes=episodes,
                              smoke=smoke, solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
    df = C.evaluate(agent.eval_mode(), "DQN (grounded-trained)", p.ev_h, p.ev_b, p.cache,
                    eval_seeds, volumes, solve=C.GROUNDED, cfg=C.GROUNDED_CFG, alpha="1.5",
                    checkpoint=f"grounded_s{task['seed']}", train_seed=task["seed"])
    C.save(df, EXP, f"retrain_s{task['seed']}", smoke)


def collect(smoke):
    runs = C.load_tasks(EXP, smoke)
    runs = runs[runs.bots > 0]
    d = C.out_dir(EXP, smoke)
    cols = ["DI", "false_positive_rate", "abandonment_rate", "friction_s", "human_survival", "bot_survival"]
    sweep = runs.groupby(["alpha", "policy"])[cols].mean()
    sweep.to_csv(d / "e8_alpha_sweep.csv")
    print("alpha sweep -- DI\n", sweep["DI"].unstack("policy").round(1).to_string())
    print("\nalpha sweep -- false-positive rate\n",
          sweep["false_positive_rate"].unstack("policy").round(3).to_string())
    g = runs[runs.alpha == "1.5"].groupby("policy")[cols].mean().sort_values("DI", ascending=False)
    g.to_csv(d / "e8_grounded.csv")
    print(f"\ngrounded world (alpha 1.5): retrained vs deterministic-trained vs statics\n{g.round(3).to_string()}")


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
