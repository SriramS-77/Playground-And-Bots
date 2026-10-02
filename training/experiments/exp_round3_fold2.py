"""Round 3 -- fold 2 only. See HANDOFF_ROUND3.md; this is its reference implementation.

    python exp_round3_fold2.py --prepare              # cross-fitted scorers + every gate (1 job)
    python exp_round3_fold2.py --smoke                # toy RL budget, NEVER touches fold.eval
    python exp_round3_fold2.py --list-search          # prints the --array line
    python exp_round3_fold2.py --search-task N        # one (family, arch, seed, rotation)
    python exp_round3_fold2.py --collect-search       # selection rule -> arch_choice.json
    python exp_round3_fold2.py --list-headline
    python exp_round3_fold2.py --headline-task N      # retrain a winner on fold.rl, eval once
    python exp_round3_fold2.py --collect-headline

E2-E8 import `context()` from here, so every experiment scores with the same cross-fitted
scorers and nothing re-derives them.

Discipline, enforced in code where it can be:
  * the policy trains and is selected on out-of-fold scores of fold 2's rl block;
  * the eval block is scored only by the refit scorer and opened only by headline tasks,
    each of which appends a line to `eval_access.log` -- the audit trail of every look;
  * `--smoke` evaluates on rl_val, so development never prints a fold-2 eval number.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import expkit  # noqa: F401  -- puts `training/` on sys.path, so import it first
import preflight
from expkit.bandits_x import POSTERIORS, BanditArch, fit_posteriors_from, train_bandit
from expkit.crossfit import OOF_CANARY_MIN_HUMANS, assign_parts, fit_crossfit
from expkit.partition import load_rotation
from expkit.paths import RESULTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from expkit.trainer import train_dqn
from exp_policy_arch_search import BANDIT_ARCHS, BOT_VOLUMES, DQN_HIDDEN, n_params, score_policy

# =========================================================================== #
# THE USER'S DECISIONS -- one line each. HANDOFF_ROUND3.md §1 explains every one.
# =========================================================================== #

#: D1. Scorer config. Phase 0's own selection rule picked this on fold 2 (scorer + rl only):
#: lr 3e-3, dropout 0.25, recurrent dropout 0.2 -- AUC 0.968 vs the baseline's 0.951,
#: baseline ranked 19th of 52. Plus start_from_epoch=30 (0/30 collapses vs 1/30).
#: To keep the round-2 baseline instead: drop the lr / dropout / recurrent_dropout overrides.
SCORER_CFG = dataclasses.replace(preflight.FINAL_SCORER_CFG, lr=3e-3, dropout=0.25,
                                 recurrent_dropout=0.2, start_from_epoch=30)

#: D2. Selection among candidates within 1 SE of the best mean DI: "friction" picks the one
#: imposing the least friction on rl_val (then fewest params); "params" is round 2's rule.
#: The friction tiebreak only considers band members that are no worse than the best-DI
#: candidate on EITHER side of the trade-off: human survival >= best - HUMANS_MARGIN (friction
#: is per ENROLLED human, so driving humans away looks cheap) AND bot survival <= best +
#: BOTS_MARGIN (a policy that waves bots through is cheap too -- the single-seed check saw
#: mixture variants at 0-3 s of friction letting 17-37 % of bots through).
SELECTION_TIEBREAK = "friction"
HUMANS_MARGIN = 0.05
BOTS_MARGIN = 0.05

#: D3. Selection-set rotations. 1 = round 2's single rl_fit/rl_val split (75 tasks): its
#: rl_val is 18 recordings, 5 humans, all campaign B. 3 = three disjoint rl_val thirds of
#: fold 2's rl block, every candidate trained on each (225 tasks, round 2's whole budget).
VAL_ROTATIONS = 3

#: D4. When a scorer gate trips in --prepare: "stop" (report and exit) or "retry" (refit
#: the whole cross-fit with the next seed, at most MAX_GATE_RETRIES retries, every attempt
#: recorded in prepare_log.json). Gates use training-side data only, so a pre-registered
#: retry is not selection on results. The user chose "retry".
ON_GATE_FAILURE = "retry"
MAX_GATE_RETRIES = 2

#: D5. Humans per TRAINING episode (bots always from BOT_VOLUMES, 0-1000). (50, 250) is local
#: testing's range and covers the realistic evaluation grid (humans U(50, 250)); round 2's
#: (80, 100) leaves that grid's human counts out of the training distribution, and makes
#: total population a near-perfect proxy for the bot count during training.
TRAIN_HUMAN_RANGE = (50, 250)

# =========================================================================== #

FOLD = 2
K = 5
SEEDS = 5
EPISODES = 200
SEARCH_EVAL_SEEDS = 20
HEADLINE_EVAL_SEEDS = 50
#: A fresh directory, so round 2's `results/fold2_scorer/` can never be picked up.
OUT = Path(os.environ.get("ROUND3_OUT", RESULTS / "round3" / f"fold{FOLD}"))
SMOKE_OUT = OUT.parent / f"{OUT.name}_smoke"

#: The realistic-traffic grid (local testing; Imperva 2025: 37 % bad bots; Akamai: 43 % of
#: logins credential abuse). Humans drawn from a range, independently of the bot fraction.
BOT_FRACTIONS = (0.05, 0.15, 0.30, 0.43, 0.60, 0.75)
HUMAN_RANGE = (50, 250)


# --------------------------------------------------------------------------- #
# Shared context -- E2-E8 use this, nothing else
# --------------------------------------------------------------------------- #

def _crossfit(refit_stale=False):
    # The seed --prepare settled on, from its own log -- which also proves prepare passed.
    # Without this a task could load scorers from an attempt whose gates failed.
    plog = OUT / "prepare_log.json"
    if not plog.exists():
        raise RuntimeError(f"{plog} missing -- run --prepare first")
    seed = json.loads(plog.read_text()).get("seed_used")
    if seed is None:
        raise RuntimeError(f"{plog}: no attempt passed the scorer gates -- stop and report")
    fold = load_rotation(RESULTS / "rotation.json")[FOLD]
    return fold, fit_crossfit(fold, SCORER_CFG, OUT / "scorer", k=K, seed=seed,
                              refit_stale=refit_stale, verbose=False)


def context(with_eval: bool = False, reason: str | None = None):
    """Fold 2, its cross-fitted scorers, and its sessions.

    `cache` scores the whole rl block out-of-fold. Ask for `with_eval=True` only in a
    final evaluation -- it adds the eval block, scored by the refit, and writes ONE line
    to `eval_access.log` (pass `reason`, e.g. "E7 final evaluation").
    """
    if with_eval:
        # Opening the eval block before selection is final is how test-set selection
        # happens; an unlabelled opening cannot be audited. Both are refused here.
        if not reason:
            raise ValueError("context(with_eval=True) needs reason='<experiment> task <i>'")
        if not (OUT / "arch_choice.json").exists():
            raise RuntimeError("the eval block opens only after --collect-search has "
                               "written arch_choice.json")
    fold, cf = _crossfit()
    rl_h, rl_b = sessions_from_refs(fold.rl)
    ctx = SimpleNamespace(fold=fold, crossfit=cf, rl_h=rl_h, rl_b=rl_b)
    if with_eval:
        ctx.ev_h, ctx.ev_b = sessions_from_refs(fold.eval)
        ctx.cache = cf.cache(rl_h + rl_b, ctx.ev_h + ctx.ev_b)
        log_eval_access(f"{reason}  [{Path(sys.argv[0]).name}]")
    else:
        ctx.cache = cf.cache(rl_h + rl_b)
    return ctx


def log_eval_access(what: str):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "eval_access.log", "a", encoding="utf-8") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}  {what}\n")


def rl_rotations(fold, n: int = VAL_ROTATIONS):
    """[(rl_fit, rl_val)] -- n disjoint rl_val parts over fold.rl, stratified by family.
    n = 1 reproduces round 2's split exactly."""
    if n == 1:
        return [(list(fold.rl_fit), list(fold.rl_val))]
    part = assign_parts(fold.rl, n, seed=0)
    return [([r for r in fold.rl if part[r.name] != i], [r for r in fold.rl if part[r.name] == i])
            for i in range(n)]


# --------------------------------------------------------------------------- #
# --prepare
# --------------------------------------------------------------------------- #

def prepare():
    fold = load_rotation(RESULTS / "rotation.json")[FOLD]
    preflight.check_rotation(load_rotation(RESULTS / "rotation.json"))
    preflight.check_determinism()
    n_attempts = 1 + (MAX_GATE_RETRIES if ON_GATE_FAILURE == "retry" else 0)
    log = {"on_gate_failure": ON_GATE_FAILURE, "max_retries": MAX_GATE_RETRIES,
           "scorer_cfg": dataclasses.asdict(SCORER_CFG), "attempts": [],
           "seed_used": None, "retries_used": None}
    OUT.mkdir(parents=True, exist_ok=True)

    def write():                     # after EVERY attempt, so even a crash leaves a record
        (OUT / "prepare_log.json").write_text(json.dumps(log, indent=2))

    for attempt in range(n_attempts):
        seed = attempt                # pre-registered: attempt i uses cross-fit seed i
        t = time.time()
        rec = {"attempt": attempt, "seed": seed, "status": "running",
               "started": _dt.datetime.now().isoformat(timespec="seconds")}
        log["attempts"].append(rec)
        write()                       # a crash or node kill mid-fit still leaves "running"
        try:
            cf = fit_crossfit(fold, SCORER_CFG, OUT / "scorer", k=K, seed=seed, refit_stale=True)
            rl_h, rl_b = sessions_from_refs(fold.rl)
            cache = cf.cache(rl_h + rl_b)
            kept = preflight.check_threshold_canary(cache, rl_h, rl_b,
                                                    min_human_survival=OOF_CANARY_MIN_HUMANS)
            rec.update(ok=True, canary_humans_kept=kept, scorers=cf.report)
        except AssertionError as exc:
            msg = str(exc)
            gate = ("canary" if msg.startswith("canary") else
                    "held-out-part AUC" if "held-out part" in msg else
                    "validation AUC" if "validation AUC" in msg else
                    "best epoch" if "early stopping restored" in msg else "other")
            rec.update(ok=False, failed_gate=gate, error=msg)
            print(f"GATE FAILED -- attempt {attempt}, seed {seed}, gate '{gate}': {msg}",
                  file=sys.stderr, flush=True)
        rec["minutes"] = round((time.time() - t) / 60, 1)
        rec["status"] = "passed" if rec["ok"] else "failed"
        write()
        if rec["ok"]:
            log["seed_used"], log["retries_used"] = seed, attempt
            write()
            break
        if attempt + 1 < n_attempts:
            print(f"  retrying with cross-fit seed {attempt + 1} "
                  f"({n_attempts - attempt - 1} retr{'y' if n_attempts - attempt - 1 == 1 else 'ies'} left)",
                  flush=True)

    if not log["attempts"][-1]["ok"]:
        raise SystemExit(f"prepare: scorer gates failed on all {n_attempts} attempt(s) -- stop "
                         f"and report prepare_log.json (HANDOFF_ROUND3 §3)")
    note = (f"after {log['retries_used']} RETRY(IES) -- report this, see prepare_log.json"
            if log["retries_used"] else "first attempt")
    print(f"prepare: OK -- cross-fit seed {log['seed_used']}, {note}; scorers in {OUT / 'scorer'}")


# --------------------------------------------------------------------------- #
# Training one candidate -- shared by search and headline
# --------------------------------------------------------------------------- #

def train_candidate(family: str, spec: str, seed: int, humans, bots, cache, episodes: int,
                    posteriors=None):
    """-> {candidate_name: policy}, {candidate_name: n_params}"""
    if family in ("dqn", "ablation"):
        hidden = tuple(int(x) for x in spec.split("-"))
        agent, _ = train_dqn(humans, bots, cache, episodes=episodes, seed=seed,
                             human_counts=TRAIN_HUMAN_RANGE, bot_choices=BOT_VOLUMES,
                             verbose=0, hidden=hidden, use_score=(family == "dqn"),
                             name=("DQN" if family == "dqn" else "DQN without H-Score"))
        assert agent.hidden == hidden
        tag = f"{family}:{spec}"
        return {tag: agent.eval_mode()}, {tag: n_params(agent.policy_net)}

    arch_name = spec.split(":")[0]
    arch = next(a for a in BANDIT_ARCHS if a.name == arch_name)
    policy, _ = train_bandit(family, humans, bots, cache, arch=arch, episodes=episodes,
                             seed=seed, human_counts=TRAIN_HUMAN_RANGE,
                             bot_choices=BOT_VOLUMES, verbose=0)
    wanted = posteriors or (POSTERIORS if family == "thompson" else ("gaussian",))
    variants = {}
    for post in wanted:
        tag = f"{family}:{arch_name}:{post}"
        if post == "gaussian":
            variants[tag] = policy
            continue
        try:
            variants[tag] = fit_posteriors_from(policy, policy._buffer, post)
        except torch.linalg.LinAlgError as exc:      # keep the others; collect reports gaps
            print(f"  POSTERIOR FAILED {tag} seed={seed}: {exc}", file=sys.stderr, flush=True)
    return variants, {k: n_params(v.net) for k, v in variants.items()}


def load_policy(path):
    """Any checkpoint from `agents/` -- DQN, ablation, LinUCB or Thompson (any posterior) --
    as a frozen policy. E6 and E8 evaluate these instead of retraining."""
    from expkit.bandits_x import load_bandit
    from expkit.trainer import TrainableDQN

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if "policy_net_state_dict" in ck:
        score = ck["state_size"] == 5
        return TrainableDQN(name="DQN" if score else "DQN without H-Score", use_score=score,
                            hidden=tuple(ck["hidden"]), device="cpu").load(path).eval_mode()
    return load_bandit(path)


def check_roundtrip(humans, bots, cache, episodes=1):
    """save -> load must reproduce a policy's behaviour exactly, for every family."""
    import tempfile

    trained = {}
    trained.update(train_candidate("dqn", "64", 0, humans, bots, cache, episodes)[0])
    trained.update(train_candidate("ablation", "64", 0, humans, bots, cache, episodes)[0])
    trained.update(train_candidate("linucb", "h64_e32", 0, humans, bots, cache, episodes)[0])
    trained.update(train_candidate("thompson", "h64_e32", 0, humans, bots, cache, episodes,
                                   posteriors=("gaussian", "mog2"))[0])
    with tempfile.TemporaryDirectory() as d:
        for tag, pol in trained.items():
            path = Path(d) / (tag.replace(":", "_") + ".pt")
            pol.save(path)
            back = load_policy(path)
            a, _ = run_x(pol, humans, bots, 100, cache=cache, n_humans=100, seed=7)
            b, _ = run_x(back, humans, bots, 100, cache=cache, n_humans=100, seed=7)
            assert (a.surviving_humans, a.surviving_bots, a.action_hist) == \
                   (b.surviving_humans, b.surviving_bots, b.action_hist), f"{tag}: reload differs"
            print(f"  [ok] {tag}: save -> load reproduces the episode exactly")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def search_tasks():
    out = []
    for family, specs in (("dqn", ["-".join(map(str, h)) for h in DQN_HIDDEN]),
                          ("linucb", [a.name for a in BANDIT_ARCHS]),
                          ("thompson", [a.name for a in BANDIT_ARCHS])):
        for spec in specs:
            for seed in range(SEEDS):
                for rot in range(VAL_ROTATIONS):
                    out.append({"family": family, "spec": spec, "seed": seed, "rotation": rot})
    return out


def run_search_task(task, episodes=EPISODES, eval_seeds=SEARCH_EVAL_SEEDS, out=OUT):
    ctx = context()                                   # out-of-fold scores, no eval block
    fit, val = rl_rotations(ctx.fold)[task["rotation"]]
    fit_h, fit_b = sessions_from_refs(fit)
    val_h, val_b = sessions_from_refs(val)
    variants, params = train_candidate(task["family"], task["spec"], task["seed"],
                                       fit_h, fit_b, ctx.cache, episodes)
    rows = []
    for tag, pol in variants.items():
        df = score_policy(pol, val_h, val_b, ctx.cache, eval_seeds)
        df["candidate"], df["family"], df["params"] = tag, task["family"], params[tag]
        df["train_seed"], df["rotation"] = task["seed"], task["rotation"]
        rows.append(df)
    frame = pd.concat(rows, ignore_index=True)
    dest = out / "search"
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / f"{task['family']}_{task['spec']}_s{task['seed']}_r{task['rotation']}.csv",
                 index=False)
    return frame


def select3(sub: pd.DataFrame, tiebreak: str = SELECTION_TIEBREAK):
    """Best mean DI; within 1 SE of it, `tiebreak` ('friction' -> least friction, then
    fewest params; 'params' -> fewest params); then name. Per-seed means first (averaged
    over volumes, eval seeds AND rotations), then the spread across training seeds."""
    per = (sub.groupby(["candidate", "train_seed"])
              [["DI", "friction_s", "human_survival", "bot_survival"]].mean().reset_index())
    g = per.groupby("candidate").agg(DI=("DI", "mean"), DI_sd=("DI", "std"), n=("DI", "size"),
                                     friction_s=("friction_s", "mean"),
                                     human_survival=("human_survival", "mean"),
                                     bot_survival=("bot_survival", "mean")).reset_index()
    g["se"] = g["DI_sd"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1))
    g["params"] = g["candidate"].map(sub.groupby("candidate")["params"].first())
    g = g.sort_values("DI", ascending=False).reset_index(drop=True)
    best = g.iloc[0]
    band = g[g["DI"] >= best["DI"] - best["se"]]
    if tiebreak == "friction":
        # Never let the tiebreak reward driving humans away, or letting bots through.
        band = band[(band["human_survival"] >= best["human_survival"] - HUMANS_MARGIN)
                    & (band["bot_survival"] <= best["bot_survival"] + BOTS_MARGIN)]
        keys = ["friction_s", "params", "candidate"]
    else:
        keys = ["params", "candidate"]
    return str(band.sort_values(keys).iloc[0]["candidate"]), g


def collect_search(out=OUT):
    files = sorted((out / "search").glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {out / 'search'}")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs = runs[runs.bots > 0]                       # zero-bot cell excluded, always
    runs.to_csv(out / "search_runs.csv", index=False)
    expected = len(search_tasks())
    if len(files) != expected:
        print(f"WARNING: {len(files)} task CSVs, grid expects {expected}")
    cells = runs.groupby("candidate")[["train_seed", "rotation"]].apply(
        lambda g: len(g.drop_duplicates()))
    short = cells[cells < SEEDS * VAL_ROTATIONS]
    if len(short):
        print(f"WARNING: candidates short of {SEEDS * VAL_ROTATIONS} (seed, rotation) cells:\n"
              f"{short.to_string()}")
    choice = {}
    for family in ("dqn", "linucb", "thompson"):
        pick, table = select3(runs[runs.family == family])
        table.to_csv(out / f"search_{family}.csv", index=False)
        choice[family] = pick
        print(f"\n[{family}] -> {pick}\n{table.round(3).to_string(index=False)}")
    payload = {"fold": FOLD, "selected_on": f"rl_val, {VAL_ROTATIONS} rotation(s)",
               "rule": f"best mean DI (bots>0), SE over training seeds; within 1 SE: "
                       + (f"least friction among those with human survival >= best's - "
                          f"{HUMANS_MARGIN} and bot survival <= best's + {BOTS_MARGIN}, then "
                          f"fewest params" if SELECTION_TIEBREAK == "friction"
                          else "fewest params"),
               "train_human_range": list(TRAIN_HUMAN_RANGE),
               "scorer_cfg": dataclasses.asdict(SCORER_CFG), "choice": choice}
    (out / "arch_choice.json").write_text(json.dumps(payload, indent=2))
    print("\nwrote", out / "arch_choice.json")
    return payload


# --------------------------------------------------------------------------- #
# Headline -- the one place the eval block is opened
# --------------------------------------------------------------------------- #

def headline_tasks():
    tasks = [{"policy": p, "seed": s} for p in ("dqn", "linucb", "thompson", "ablation")
             for s in range(SEEDS)]
    return tasks + [{"policy": "statics", "seed": 0}]


def _population(fraction, seed):
    rng = np.random.default_rng(10_000 + seed)
    n_h = int(rng.integers(HUMAN_RANGE[0], HUMAN_RANGE[1] + 1))
    return n_h, int(round(n_h * fraction / (1 - fraction)))


def evaluate(policy, name, humans, bots, cache, eval_seeds):
    rows = []
    grids = [("published", b, 100) for b in BOT_VOLUMES]
    for frac in BOT_FRACTIONS:
        grids.append(("realistic", frac, None))
    for grid, level, n_h_fixed in grids:
        for s in range(eval_seeds):
            n_h, n_b = (n_h_fixed, level) if grid == "published" else _population(level, s)
            res, _ = run_x(policy, humans, bots, n_b, cache=cache, n_humans=n_h, seed=s,
                           solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT)
            m = evaluate_x(res)
            rows.append({"policy": name, "grid": grid, "level": level, "eval_seed": s,
                         "n_humans": n_h, "n_bots": n_b, "DI": m.DI, "BOS": m.BOS,
                         "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                         "human_survival": res.surviving_humans / n_h,
                         "bot_survival": res.surviving_bots / max(n_b, 1),
                         "friction_s": res.mean_human_friction_seconds})
    return pd.DataFrame(rows)


def run_headline_task(task, smoke=False, episodes=EPISODES, eval_seeds=HEADLINE_EVAL_SEEDS):
    from rlcaptcha.policies.static import MultiThresholdPolicy, SingleThresholdPolicy

    out = SMOKE_OUT if smoke else OUT
    if smoke:
        ctx = context()                               # smoke evaluates on rl_val, never eval
        fit, val = rl_rotations(ctx.fold)[0]
        train_h, train_b = sessions_from_refs(fit)
        ev_h, ev_b = sessions_from_refs(val)
        choice = {"dqn": "dqn:64", "linucb": "linucb:h64_e32:gaussian",
                  "thompson": "thompson:h64_e32:mog2"}
    else:
        choice = json.loads((OUT / "arch_choice.json").read_text())["choice"]
        ctx = context(with_eval=True, reason=f"headline task {task}")
        train_h, train_b, ev_h, ev_b = ctx.rl_h, ctx.rl_b, ctx.ev_h, ctx.ev_b

    p, seed = task["policy"], task["seed"]
    if p == "statics":
        frames = [evaluate(pol, pol.name, ev_h, ev_b, ctx.cache, eval_seeds)
                  for pol in (SingleThresholdPolicy(), MultiThresholdPolicy())]
    else:
        fam = "dqn" if p == "ablation" else p          # the ablation is of the DQN winner
        spec = choice[fam].split(":", 1)[1]            # "64" | "h128_e64:mog3"
        post = [spec.split(":")[1]] if fam != "dqn" else None
        variants, _ = train_candidate(p, spec, seed, train_h, train_b, ctx.cache, episodes,
                                      posteriors=post)
        if len(variants) != 1:
            raise RuntimeError(f"headline {task}: expected the one selected variant {choice[fam]}, "
                               f"got {sorted(variants)} -- the posterior refit failed; report it")
        (tag, pol), = variants.items()
        ck = out / "agents"
        ck.mkdir(parents=True, exist_ok=True)
        pol.save(ck / f"{p}_s{seed}.pt")
        name = {"dqn": "DQN", "linucb": "LinUCB", "thompson": "Thompson Sampling",
                "ablation": "DQN without H-Score"}[p]
        df = evaluate(pol, name, ev_h, ev_b, ctx.cache, eval_seeds)
        df["candidate"], df["train_seed"] = tag, seed
        frames = [df]
    frame = pd.concat(frames, ignore_index=True)
    dest = out / "headline"
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / f"{p}_s{seed}.csv", index=False)
    return frame


def collect_headline(out=OUT):
    files = sorted((out / "headline").glob("*.csv"))
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs.to_csv(out / "headline_runs.csv", index=False)
    if len(files) != len(headline_tasks()):
        print(f"WARNING: {len(files)} of {len(headline_tasks())} headline tasks present")
    pub = preflight.headline(runs[runs.grid == "published"].rename(columns={"level": "bots"}))
    for grid, frame in (("published (bots > 0)", pub), ("realistic", runs[runs.grid == "realistic"])):
        t = frame.groupby("policy").agg(DI=("DI", "mean"), humans=("human_survival", "mean"),
                                        bots_through=("bot_survival", "mean"),
                                        friction_s=("friction_s", "mean"), SI_F1=("SI_F1", "mean"))
        if "train_seed" in frame:
            seed_di = frame.dropna(subset=["train_seed"]).groupby(["policy", "train_seed"]).DI.mean()
            t["seed_DI_sd"] = seed_di.groupby("policy").std()
            t["seed_DI_min"] = seed_di.groupby("policy").min()
        print(f"\n== {grid}\n{t.sort_values('DI', ascending=False).round(3).to_string()}")
        t.to_csv(out / f"headline_{grid.split()[0]}.csv")

        # Each learned policy against Static Multi, seed by seed (Static Multi has no seed).
        cols = ["DI", "human_survival", "bot_survival", "friction_s"]
        ref = frame[frame.policy == "Static Multi-Threshold"][cols].mean()
        learned = frame.dropna(subset=["train_seed"])
        if len(learned) and not ref.isna().all():
            vs = learned.groupby(["policy", "train_seed"])[cols].mean() - ref
            vs.round(3).to_csv(out / f"headline_{grid.split()[0]}_vs_static_multi.csv")
            print(f"-- minus Static Multi, per training seed\n{vs.round(3).to_string()}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    for flag in ("prepare", "smoke", "list-search", "collect-search", "list-headline",
                 "collect-headline"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--search-task", type=int)
    ap.add_argument("--headline-task", type=int)
    a = ap.parse_args()

    if a.list_search or a.list_headline:
        tasks = search_tasks() if a.list_search else headline_tasks()
        for i, t in enumerate(tasks):
            print(f"{i:4d}  {t}")
        print(f"\n{len(tasks)} tasks  ->  #SBATCH --array=0-{len(tasks) - 1}%25")
        return
    if a.collect_search:
        return collect_search()
    if a.collect_headline:
        return collect_headline()

    preflight.check()
    if a.prepare:
        return prepare()
    if a.smoke:
        # Toy RL budget on the REAL prepared scorers; every evaluation is on rl_val.
        log = OUT / "eval_access.log"
        before = log.read_text() if log.exists() else ""
        t = search_tasks()
        for i in (0, len(t) // 3, 2 * len(t) // 3):       # one DQN, LinUCB, Thompson task
            run_search_task(t[i], episodes=3, eval_seeds=2, out=SMOKE_OUT)
        for task in ({"policy": "dqn", "seed": 0}, {"policy": "thompson", "seed": 0},
                     {"policy": "ablation", "seed": 0}, {"policy": "statics", "seed": 0}):
            run_headline_task(task, smoke=True, episodes=3, eval_seeds=2)
        collect_search(out=SMOKE_OUT)
        collect_headline(out=SMOKE_OUT)
        ctx = context()
        val_h, val_b = sessions_from_refs(rl_rotations(ctx.fold)[0][1])
        check_roundtrip(val_h, val_b, ctx.cache)          # E6 / E8 depend on reloading
        after = log.read_text() if log.exists() else ""
        assert after == before, "smoke opened the eval block -- it must not"
        print("smoke: OK (eval_access.log unchanged: no fold-2 eval data touched)")
        return
    preflight.check_budget(n_train_seeds=SEEDS, n_episodes=EPISODES,
                           n_eval_seeds=SEARCH_EVAL_SEEDS)
    if a.search_task is not None:
        run_search_task(search_tasks()[a.search_task])
    elif a.headline_task is not None:
        run_headline_task(headline_tasks()[a.headline_task])


if __name__ == "__main__":
    main()
