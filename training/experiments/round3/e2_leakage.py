"""E2 -- how much does a policy gain from being judged on recordings it trained on?
(Reviewer 1, point 2; HANDOFF_ROUND3 §5 Priority 7.)

Cross-fitting already removed the scorer-level leak, so what is left is the policy-level
one. Role reversal inside fold 2's rl block, never the eval block:

    split fold.rl into halves P1 / P2 (stratified by family), 10 seeded splits
    agent A trains on P1, agent B on P2 -- both on out-of-fold scores
    seen   = A on P1  +  B on P2
    unseen = A on P2  +  B on P1
    gap    = seen - unseen

Pool difficulty enters both sides and cancels. For a policy that does not train, "A" and
"B" are the same policy, so seen and unseen are literally the same runs and the gap must
be EXACTLY 0.000 -- asserted in --collect, because it is what proves the design works.

The no-H-Score ablation sees nothing that identifies a recording, so it is the control
for the mechanism: if the full agent leaks and the ablation does not, the leak flows
through the scorer's outputs.

    tasks: 10 splits x 2 halves x {DQN, ablation} trainings  +  10 static-baseline tasks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from expkit.crossfit import assign_parts
from expkit.simx import sessions_from_refs

EXP = "e2"
N_SPLITS = 10


def tasks():
    t = [{"kind": "train", "split": s, "half": h, "policy": p}
         for s in range(N_SPLITS) for h in (0, 1) for p in ("dqn", "ablation")]
    return t + [{"kind": "statics", "split": s} for s in range(N_SPLITS)]


def smoke_tasks():
    """Both halves of one split, so the gap arithmetic itself is exercised for a learned
    policy -- one half alone leaves seen/unseen incomplete."""
    return [{"kind": "train", "split": 0, "half": h, "policy": "dqn"} for h in (0, 1)] + \
           [{"kind": "statics", "split": 0}]


def halves(fold, split):
    part = assign_parts(fold.rl, 2, seed=1000 + split)
    return [[r for r in fold.rl if part[r.name] == h] for h in (0, 1)]


def run_task(task, smoke):
    episodes, eval_seeds, volumes = C.budget(smoke)
    p = C.pools(final=False, smoke=smoke)            # out-of-fold scores; no eval block
    if smoke:                                        # smoke: halves of rotation-0 rl_fit
        fit, _ = C.R.rl_rotations(p.fold)[0]
        p.fold = type("F", (), {"rl": fit})()
    H = [sessions_from_refs(h) for h in halves(p.fold, task["split"])]
    frames = []
    if task["kind"] == "train":
        th, tb = H[task["half"]]
        agent, _ = C.train_winner(th, tb, p.cache, seed=task["split"], episodes=episodes,
                                  smoke=smoke, use_score=(task["policy"] == "dqn"))
        pol = agent.eval_mode()
        for eh in (0, 1):
            frames.append(C.evaluate(pol, pol.name, *H[eh], p.cache, eval_seeds, volumes,
                                     split=task["split"], train_half=task["half"],
                                     eval_half=eh))
    else:
        for pol in C.statics():
            for eh in (0, 1):
                frames.append(C.evaluate(pol, pol.name, *H[eh], p.cache, eval_seeds, volumes,
                                         split=task["split"], train_half=-1, eval_half=eh))
    name = (f"s{task['split']}_h{task['half']}_{task['policy']}" if task["kind"] == "train"
            else f"s{task['split']}_statics")
    C.save(pd.concat(frames, ignore_index=True), EXP, name, smoke)


def collect(smoke):
    runs = C.load_tasks(EXP, smoke)
    runs = runs[runs.bots > 0]
    rows = []
    for (pol, split), g in runs.groupby(["policy", "split"]):
        cell = g.groupby(["train_half", "eval_half"]).DI.mean()
        if pol.startswith("Static"):
            seen = (cell.get((-1, 0)) + cell.get((-1, 1))) / 2      # same runs both sides
            unseen = (cell.get((-1, 1)) + cell.get((-1, 0))) / 2
        else:
            if not {(0, 0), (0, 1), (1, 0), (1, 1)} <= set(cell.index):
                print(f"WARNING: {pol} split {split} incomplete -- skipped")
                continue
            seen = (cell[(0, 0)] + cell[(1, 1)]) / 2
            unseen = (cell[(0, 1)] + cell[(1, 0)]) / 2
        rows.append({"policy": pol, "split": split, "seen": seen, "unseen": unseen,
                     "gap": seen - unseen})
    per = pd.DataFrame(rows)
    for pol, g in per.groupby("policy"):
        if pol.startswith("Static"):
            worst = float(g.gap.abs().max())
            assert worst == 0.0, f"{pol}: gap {worst} -- must be exactly 0; the design is broken"
    summ = per.groupby("policy").gap.agg(["mean", "std", "count"])
    summ["se"] = summ["std"] / np.sqrt(summ["count"])
    summ["ci_lo"], summ["ci_hi"] = summ["mean"] - 1.96 * summ["se"], summ["mean"] + 1.96 * summ["se"]
    d = C.out_dir(EXP, smoke)
    per.to_csv(d / "e2_gaps_per_split.csv", index=False)
    summ.to_csv(d / "e2_gap_summary.csv")
    print(summ.round(3).to_string())
    print("\nstatic baselines: gap exactly 0.000 on every split (asserted)")


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
