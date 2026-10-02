"""Shared plumbing for the round-3 experiment scripts (E2-E8), fold 2 only.

Every E-script imports this and nothing else from the pipeline, so all of them:
  * score with the SAME cross-fitted scorers (`exp_round3_fold2.context`);
  * train the SAME policy -- fold 2's DQN winner from `arch_choice.json`, 5 seeds x 200
    episodes, the headline's training sampler (TRAIN_HUMAN_RANGE x BOT_VOLUMES);
  * open the eval block only through `pools(final=True, reason=...)`, which writes one
    line to `eval_access.log` and is refused before `arch_choice.json` exists;
  * have a `--smoke` mode that trains and evaluates on rotation 0's rl_fit / rl_val,
    so development never touches the eval block.

CLI, identical for every script:
    python round3/eN_name.py --list            # task list and the --array line
    python round3/eN_name.py --task N          # one array task
    python round3/eN_name.py --collect         # tables from the task CSVs
    python round3/eN_name.py --smoke           # toy budget, rl_val only, then --collect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

import exp_round3_fold2 as R  # noqa: E402  -- also puts training/ on sys.path
import preflight              # noqa: E402
from exp_policy_arch_search import BOT_VOLUMES, SCORED_VOLUMES  # noqa: E402
from expkit.partition import load_rotation  # noqa: E402
from expkit.paths import RESULTS  # noqa: E402
from expkit.simx import evaluate_x, run_x, sessions_from_refs  # noqa: E402
from expkit.stochastic import (DETERMINISTIC, GROUNDED, PAPER_EQUIVALENT,  # noqa: E402
                               StochasticRewardConfig)
from expkit.trainer import train_dqn  # noqa: E402

#: The stochastic "grounded" world of E7 and E8 (nb_06 / nb_07): humans can fail a
#: challenge (false positives exist), abandonment follows measured solve times.
GROUNDED_CFG = StochasticRewardConfig(abandonment="empirical", abandonment_tau=45.0,
                                      false_positive_penalty=-100.0)

SEEDS = R.SEEDS                   # 5
EPISODES = R.EPISODES             # 200
EVAL_SEEDS = 20                   # E-experiments; the headline uses 50
SMOKE = SimpleNamespace(seeds=1, episodes=2, eval_seeds=2, volumes=(20, 100))


# --------------------------------------------------------------------------- #
# What to train, and on what
# --------------------------------------------------------------------------- #

def dqn_hidden(smoke: bool = False) -> tuple:
    """Fold 2's DQN winner -- never the published 128-64-32."""
    path = R.OUT / "arch_choice.json"
    if not path.exists():
        if smoke:
            return (64,)
        raise RuntimeError(f"{path} missing -- run the search and --collect-search first")
    spec = json.loads(path.read_text())["choice"]["dqn"].split(":", 1)[1]
    return tuple(int(x) for x in spec.split("-"))


def pools(final: bool, reason: str | None = None, smoke: bool = False):
    """Training and evaluation pools, with one score cache covering both.

    final=False   train on fold 2's rl block; nothing to evaluate on (E2 builds its own).
    final=True    train on the rl block, evaluate on the eval block -- opens it, logged.
    smoke=True    train on rotation 0's rl_fit, "evaluate" on its rl_val; never the eval
                  block, whatever `final` says.
    """
    if smoke:
        ctx = R.context()
        fit, val = R.rl_rotations(ctx.fold)[0]
        (th, tb), (eh, eb) = sessions_from_refs(fit), sessions_from_refs(val)
        return SimpleNamespace(fold=ctx.fold, cache=ctx.cache, train_h=th, train_b=tb,
                               ev_h=eh, ev_b=eb, ctx=ctx)
    if final:
        ctx = R.context(with_eval=True, reason=reason)
        return SimpleNamespace(fold=ctx.fold, cache=ctx.cache, train_h=ctx.rl_h,
                               train_b=ctx.rl_b, ev_h=ctx.ev_h, ev_b=ctx.ev_b, ctx=ctx)
    ctx = R.context()
    return SimpleNamespace(fold=ctx.fold, cache=ctx.cache, train_h=ctx.rl_h,
                           train_b=ctx.rl_b, ev_h=None, ev_b=None, ctx=ctx)


def train_winner(humans, bots, cache, seed: int, episodes: int, smoke: bool = False,
                 use_score: bool = True, human_counts=None, **kw):
    """The DQN winner, trained exactly as the headline trains it unless `kw` overrides
    (reward_source, proxy_cfg, solve, cfg, bot_choices)."""
    kw.setdefault("bot_choices", BOT_VOLUMES)
    return train_dqn(humans, bots, cache, episodes=episodes, seed=seed, verbose=0,
                     hidden=dqn_hidden(smoke), use_score=use_score,
                     human_counts=human_counts or R.TRAIN_HUMAN_RANGE,
                     name="DQN" if use_score else "DQN without H-Score", **kw)


def headline_agents(smoke: bool = False) -> dict:
    """{name: path} of the headline's saved checkpoints (smoke: tiny ones, trained now)."""
    d = (R.SMOKE_OUT if smoke else R.OUT) / "agents"
    if smoke and not any(d.glob("*.pt")):
        p = pools(final=False, smoke=True)
        d.mkdir(parents=True, exist_ok=True)
        for fam, spec in (("dqn", "64"), ("linucb", "h64_e32"), ("thompson", "h64_e32")):
            post = None if fam == "dqn" else (["mog2"] if fam == "thompson" else ["gaussian"])
            (_, pol), = R.train_candidate(fam, spec, 0, p.train_h, p.train_b, p.cache, 1,
                                          posteriors=post)[0].items()
            pol.save(d / f"{fam}_s0.pt")
    found = {f.stem: f for f in sorted(d.glob("*.pt"))}
    if not found:
        raise RuntimeError(f"no checkpoints in {d} -- run the headline first")
    return found


def statics():
    from rlcaptcha.policies.static import MultiThresholdPolicy, SingleThresholdPolicy
    return [SingleThresholdPolicy(), MultiThresholdPolicy()]


class NeverChallenge:
    """The do-nothing baseline: level 0 for everyone. The floor of the gap-closed ratio."""
    name = "never challenge"
    needs_bot_score = False
    acts_when_exhausted = False
    updates_last_threat = True
    initial_last_threat = 0

    def features(self, user, n_active):
        return np.zeros(3, dtype=np.float32)

    def select_action(self, user, n_active):
        return 0


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(policy, name, humans, bots, cache, eval_seeds, volumes=SCORED_VOLUMES,
             n_humans=100, solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT, seed_offset=0, **tags):
    """One row per (volume, eval seed): DI and the quantities DI hides, side by side."""
    rows = []
    for nb in volumes:
        for s in range(eval_seeds):
            nh = n_humans(nb, s) if callable(n_humans) else n_humans
            res, _ = run_x(policy, humans, bots, nb, cache=cache, n_humans=nh,
                           seed=seed_offset + s, solve=solve, cfg=cfg)
            m = evaluate_x(res)
            rows.append({"policy": name, "bots": nb, "n_humans": nh, "eval_seed": s,
                         "DI": m.DI, "BOS": m.BOS, "SI_F1": m.SI_F1,
                         "human_survival": res.surviving_humans / nh,
                         "bot_survival": res.surviving_bots / max(nb, 1),
                         "friction_s": res.mean_human_friction_seconds,
                         "false_positive_rate": res.false_positive_rate,
                         "abandonment_rate": res.abandonment_rate,
                         "challenge_share": res.challenges_issued / max(sum(res.action_hist.values()), 1),
                         **tags})
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, by, cols=("DI", "human_survival", "bot_survival", "friction_s")):
    """Means with the zero-bot cell excluded -- the rule for every headline number."""
    f = preflight.headline(frame) if "bots" in frame else frame
    return f.groupby(by)[list(cols)].mean()


# --------------------------------------------------------------------------- #
# Output and CLI
# --------------------------------------------------------------------------- #

def out_dir(exp: str, smoke: bool) -> Path:
    d = (R.SMOKE_OUT if smoke else R.OUT) / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(df: pd.DataFrame, exp: str, name: str, smoke: bool):
    path = out_dir(exp, smoke) / "tasks" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


#: Columns that are LABELS. pandas parses each CSV on its own, so a file whose `variant` is
#: only "true" comes back as booleans and one that is only "200" as integers -- and a
#: comparison against the string then silently matches nothing. Caught in E4's smoke, where
#: it made the wrapper self-test skip itself; E8's "1.5" vs "deterministic" had the same bug.
LABEL_COLS = ("policy", "variant", "alpha", "setting", "curve", "config", "checkpoint",
              "observation", "candidate", "arm", "injection", "eval_condition")


def load_tasks(exp: str, smoke: bool) -> pd.DataFrame:
    files = sorted((out_dir(exp, smoke) / "tasks").glob("*.csv"))
    if not files:
        raise SystemExit(f"no task CSVs for {exp}")
    dtype = {c: str for c in LABEL_COLS}
    return pd.concat([pd.read_csv(f, dtype=dtype) for f in files], ignore_index=True)


def cli(exp: str, tasks, run_task, collect, smoke_tasks=None):
    """`tasks()` -> list of dicts; `run_task(task, smoke)`; `collect(smoke)`;
    `smoke_tasks` -> the subset run by --smoke (default: the first task of each kind)."""
    ap = argparse.ArgumentParser(description=f"round 3, fold 2 -- {exp}")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.list:
        tl = tasks()
        for i, t in enumerate(tl):
            print(f"{i:4d}  {t}")
        print(f"\n{len(tl)} tasks  ->  sbatch --array=0-{len(tl) - 1}%25 slurm/r3_e.sh {exp}")
        return
    if a.collect:
        return collect(False)
    preflight.check(hardware=False)
    if a.smoke:
        log = R.OUT / "eval_access.log"
        before = log.read_text() if log.exists() else ""
        chosen = smoke_tasks() if smoke_tasks else _first_of_each(tasks())
        for t in chosen:
            print(f"[smoke] {exp} {t}", flush=True)
            run_task(t, True)
        collect(True)
        after = log.read_text() if log.exists() else ""
        assert after == before, f"{exp} --smoke opened the eval block"
        print(f"{exp} smoke: OK (eval block untouched)")
        return
    if a.task is None:
        ap.error("one of --list, --task N, --collect, --smoke")
    preflight.check_budget(n_train_seeds=SEEDS, n_episodes=EPISODES, n_eval_seeds=EVAL_SEEDS)
    run_task(tasks()[a.task], False)


def _first_of_each(task_list):
    seen, out = set(), []
    for t in task_list:
        key = t.get("kind", "task")
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def budget(smoke: bool):
    """(episodes, eval_seeds, volumes) for this run."""
    if smoke:
        return SMOKE.episodes, SMOKE.eval_seeds, SMOKE.volumes
    return EPISODES, EVAL_SEEDS, SCORED_VOLUMES
