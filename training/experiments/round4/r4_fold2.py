"""Round 4 -- fold 2 only. Two stages, each with its own output directory and scorer.

STAGE ppo     PPO architecture search, under ROUND 3's scorer (cross-fit seed 0), exactly as
              round 3 searched DQN and the bandits: same rl_val rotations, sampler, budget,
              deterministic world, oracle reward and selection rule (D2). Only the policy
              family is new. -> results/round4/fold2/ppo/
STAGE drift   DI against scorer calibration error, offline-trained (frozen) and online-trained
              policies, under OUR scorer (cross-fit seed 1, `round4/scorer_seed1/`, a declared
              post-hoc choice -- HANDOFF_ROUND4 §1). -> results/round4/fold2/drift/

    python round4/r4_fold2.py --stage ppo   --setup          # scorer -> OUT, verify it LOADS
    python round4/r4_fold2.py --stage ppo   --list | --task N | --collect | --smoke
    python round4/r4_fold2.py --stage drift --setup          # needs ppo/arch_choice.json
    python round4/r4_fold2.py --stage drift --list | --task N | --collect | --smoke | --figure

PRE-DECLARED for the drift stage (fixed before any run; do not revise):
  shift       two directions, never mixed in one run:
                "bots"    a fraction RHO of the BOT pool is advanced web-bots (Iliou et al. 2021)
                "humans"  a fraction RHO of the HUMAN pool is Balabit users (Fulop et al. 2016)
              RHO in LEVELS = 0.1 .. 1.0 (10 levels) plus 0 (no shift). Training pools draw from
              the INJECT halves of the external sets, evaluation pools from the HELD-OUT halves
              (`expkit.external` splits; disjoint by user for Balabit). A fraction is realised by
              replication (`mix`) and the realised value is recorded. The bot:human ratio is the
              evaluation grid's own (100 humans, SCORED_VOLUMES bots) at every level, so the shift
              never changes the base rate (Pendlebury et al. 2019 C3; Arp et al. 2022 P8).
  x-axis      calibration-in-the-large error of the SHIFTED class on the evaluation pool, from
              the scorer's chunk-level P(bot) as the policy sees it (first 12 chunks):
                bots    mean(1 - P(bot)) over bot chunks;  humans  mean(P(bot)) over human chunks.
              Secondary, reported beside it: session-level AUC (humans vs bots of the pool), the
              shifted class's Brier score, balanced accuracy at 0.5. AUC is NOT the x-axis: on
              the same-source web-bot data it stays ~0.998 while the advanced bots' P(bot) sits
              at 0.30-0.39 -- ranking intact, calibration broken (Davis et al. 2017).
  world       GROUNDED (stochastic solves, human failures and abandonment; E7/E8/E9's world) for
              BOTH panels, training and evaluation -- so the panels differ only in offline vs
              online training. Decided by the user before any run (HANDOFF_ROUND4 R6).
  offline     policies trained OFFLINE (rl block, GROUNDED world, oracle reward, 200 episodes,
              arch_choice): DQN, PPO, Thompson, LinUCB, DQN without H-score, 5 seeds; Static
              Single / Multi. Evaluated FROZEN at all 21 shift points, 100 humans x
              SCORED_VOLUMES x 20 seeds.
  online      E9's protocol, GROUNDED world: 300 episodes on the rl block; from episode 100 the
              training pool of the shifted class is mixed at RHO with INJECT sessions; evaluated
              at episode 300 on the same (direction, RHO) point and on "seen" (RHO = 0).
              algorithms DQN (round-3 winner) and PPO (this round's winner);
              arms oracle, posterior, posterior (floored), posterior+labels,
                   posterior+labels (floored), score-only (E7's configs, unchanged); 5 seeds.
  control     the same algorithm x arm x seed trained 300 episodes with NO shift, evaluated at all
              21 points: the "no online training through the shift" comparator.
  contrast    primary, per (algorithm, arm, direction): mean DI over the 10 shifted levels,
              online minus control, Welch over the 5 seeds; Holm correction over the 24
              contrasts. Per-level differences are descriptive only.
  figure      DI (bots > 0) against the x-axis; rows = direction; columns = offline (frozen),
              online DQN, online PPO; solid = online-trained, dashed = its control; whiskers = SE
              over seeds. Secondary x labels give the pool AUC.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_RESULTS = EXP_DIR / "results"


def _stage_from_argv() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--stage" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--stage="):
            return a.split("=", 1)[1]
    raise SystemExit("--stage ppo|drift is required")


STAGE = _stage_from_argv()
if STAGE not in ("ppo", "drift"):
    raise SystemExit(f"unknown stage {STAGE!r}")
# Every round-3 helper reads its output directory from ROUND3_OUT at import time, so the stage
# decides it BEFORE anything from round 3 is imported: context(), the scorer cross-fit,
# prepare_log.json and eval_access.log all resolve inside this stage's directory.
R4_ROOT = Path(os.environ.get("ROUND4_OUT", REPO_RESULTS / "round4" / "fold2"))
os.environ["ROUND3_OUT"] = str(R4_ROOT / STAGE)
sys.path[:0] = [str(EXP_DIR), str(EXP_DIR / "round3")]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

import common as C            # noqa: E402  (round 3's plumbing; imports exp_round3_fold2 as C.R)
import preflight              # noqa: E402
from exp_policy_arch_search import BOT_VOLUMES, DQN_HIDDEN, SCORED_VOLUMES  # noqa: E402

R = C.R
OUT, SMOKE_OUT = R.OUT, R.SMOKE_OUT

#: Where each stage's scorer comes from (committed in the repository).
SCORER_SRC = {
    "ppo": (EXP_DIR / "round3" / "deliverables_fold2" / "results" / "scorer",
            EXP_DIR / "round3" / "deliverables_fold2" / "results" / "prepare_log.json", 0),
    "drift": (HERE / "scorer_seed1", HERE / "scorer_seed1" / "prepare_log.json", 1),
}
ROUND3_CHOICE = EXP_DIR / "round3" / "deliverables_fold2" / "results" / "arch_choice.json"
ROUND3_SEARCH_DQN = EXP_DIR / "round3" / "deliverables_fold2" / "results" / "search_dqn.csv"

# --------------------------------------------------------------------------- #
# Setup: put the declared scorer in OUT and prove it LOADS (never fits)
# --------------------------------------------------------------------------- #

def setup():
    src, plog, seed = SCORER_SRC[STAGE]
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "scorer").exists():
        shutil.copytree(src, OUT / "scorer", ignore=shutil.ignore_patterns("prepare_log.json"))
    shutil.copyfile(plog, OUT / "prepare_log.json")
    if json.loads((OUT / "prepare_log.json").read_text())["seed_used"] != seed:
        raise SystemExit(f"{plog}: seed_used is not {seed}")
    if STAGE == "drift":
        src_choice = R4_ROOT / "ppo" / "arch_choice.json"
        if src_choice.exists():
            shutil.copyfile(src_choice, OUT / "arch_choice.json")
        else:      # --smoke works without it; real tasks refuse (and the eval block stays shut)
            print(f"WARNING: {src_choice} missing -- finish the ppo stage (--collect), then rerun "
                  "this --setup before launching the drift array", flush=True)
    t = time.time()
    fold, cf = R._crossfit(refit_stale=False)       # raises if any scorer would need a refit
    if time.time() - t > 600:
        print("WARNING: loading took > 10 min -- was something fitted?", flush=True)
    print(f"setup[{STAGE}]: cross-fit seed {seed} LOADED from {OUT / 'scorer'} "
          f"({len(cf.report)} scorers, none refitted)")
    if STAGE == "drift":
        ext_scores(build=True)


# --------------------------------------------------------------------------- #
# STAGE ppo -- the search
# --------------------------------------------------------------------------- #

PPO_SPECS = ["-".join(map(str, h)) for h in DQN_HIDDEN]   # the DQN grid, layer for layer


def ppo_tasks():
    return [{"family": "ppo", "spec": spec, "seed": s, "rotation": r}
            for spec in PPO_SPECS for s in range(R.SEEDS) for r in range(R.VAL_ROTATIONS)]


def ppo_collect(out=None):
    out = out or OUT
    files = sorted((out / "search").glob("ppo_*.csv"))
    if not files:
        raise SystemExit(f"no PPO search CSVs in {out / 'search'}")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs = runs[runs.bots > 0]
    runs.to_csv(out / "search_runs_ppo.csv", index=False)
    if out == OUT and len(files) != len(ppo_tasks()):
        print(f"WARNING: {len(files)} task CSVs, the grid expects {len(ppo_tasks())}")
    pick, table = R.select3(runs)
    table.to_csv(out / "search_ppo.csv", index=False)
    print(f"[ppo] -> {pick}\n{table.round(3).to_string(index=False)}")
    choice = json.loads(ROUND3_CHOICE.read_text())
    payload = dict(choice)
    payload["choice"] = {**choice["choice"], "ppo": pick}
    payload["round4"] = ("ppo selected in round 4 under round 3's scorer (cross-fit seed 0) with "
                         "round 3's rule; dqn / linucb / thompson carried over from round 3")
    (out / "arch_choice.json").write_text(json.dumps(payload, indent=2))
    if ROUND3_SEARCH_DQN.exists():
        dqn = pd.read_csv(ROUND3_SEARCH_DQN)
        both = pd.concat([table.assign(family="ppo"), dqn.assign(family="dqn")], ignore_index=True)
        both.to_csv(out / "search_ppo_vs_dqn.csv", index=False)
        print("\nPPO next to round 3's DQN search (same scorer, rotations, sampler, budget):\n"
              + both.sort_values("DI", ascending=False).round(3).to_string(index=False))
    print("\nwrote", out / "arch_choice.json")


def ppo_smoke():
    log = OUT / "eval_access.log"
    before = log.read_text() if log.exists() else ""
    t = ppo_tasks()
    for i in (0, len(t) - 1):
        R.run_search_task(t[i], episodes=3, eval_seeds=2, out=SMOKE_OUT)
    ppo_collect(out=SMOKE_OUT)
    after = log.read_text() if log.exists() else ""
    assert after == before, "smoke opened the eval block"
    print("ppo smoke: OK (eval block untouched)")


# --------------------------------------------------------------------------- #
# STAGE drift -- pools and the x-axis
# --------------------------------------------------------------------------- #

LEVELS = tuple(round(0.1 * i, 1) for i in range(1, 11))
DIRECTIONS = ("bots", "humans")
ALGOS = ("dqn", "ppo")
ARMS = ("oracle", "posterior", "posterior (floored)", "posterior+labels",
        "posterior+labels (floored)", "score-only")
OFFLINE_FAMILIES = ("dqn", "ppo", "thompson", "linucb", "ablation")
TOTAL, INJECT_EP = 300, 100
MIX_MAX_COPIES = 400      # replication bound when a fraction is realised exactly (`mix_counts`)
MIX_TOL = 0.02            # |realised - declared rho| above this fails the task
SMOKE_TOTAL, SMOKE_INJECT = 4, 2
SMOKE_LEVELS = (0.5, 1.0)
EXT_SCORES = "ext_scores.pkl"


def points(levels=LEVELS):
    """The 21 evaluation points: (direction, rho), rho = 0 once."""
    return [("none", 0.0)] + [(d, r) for d in DIRECTIONS for r in levels]


def mix_counts(n_base, n_ext, rho, tol=0.005, max_copies=MIX_MAX_COPIES):
    """(m, k): m copies of the base pool and k of the external one, so that a uniform draw is
    external with probability k*n_ext / (m*n_base + k*n_ext) within `tol` of rho -- the
    smallest such pool. Rounding a fixed pool size instead collapses levels when the pools'
    sizes differ a lot (150 Balabit sessions against 14 eval humans)."""
    best = None
    for m in range(1, max_copies + 1):
        k = max(1, round(rho * m * n_base / ((1 - rho) * n_ext)))
        real = k * n_ext / (m * n_base + k * n_ext)
        if best is None or abs(real - rho) < abs(best[2] - rho):
            best = (m, k, real)
        if abs(real - rho) <= tol:
            break
    return best


def mix(base, ext, rho):
    """A pool in which a fraction rho of draws comes from `ext`. Returns (pool, realised)."""
    if rho <= 0:
        return list(base), 0.0
    if rho >= 1:
        return list(ext), 1.0
    m, k, real = mix_counts(len(base), len(ext), rho)
    if abs(real - rho) > MIX_TOL:
        raise AssertionError(f"mix: rho {rho} realised as {real:.3f} (> {MIX_TOL} off)")
    return list(base) * m + list(ext) * k, real


def ext_scores(build=False):
    """{(session key, chunk): P(bot)} for every external session, by THIS stage's refit (never
    retrained). Built once by --setup; every task loads it instead of re-scoring."""
    import pickle
    path = OUT / EXT_SCORES
    if path.exists() and not build:
        return pickle.load(open(path, "rb"))
    from expkit.crossfit import _score_sessions
    from expkit.external import load_sessions
    _, cf = R._crossfit()
    ext = load_sessions()
    table = _score_sessions(cf.refit, [s for v in ext.values() for s in v])
    pickle.dump(table, open(path, "wb"))
    print(f"  scored {sum(len(v) for v in ext.values())} external sessions -> {path}")
    return table


def pools(smoke, reason):
    from expkit.external import load_sessions
    p = C.pools(final=not smoke, smoke=smoke, reason=reason)
    p.cache._table.update(ext_scores())
    p.ext = load_sessions()
    return p


def shifted(p, direction, rho, held_out: bool):
    """(humans, bots, realised rho) -- base pools are the training (rl) or evaluation pools."""
    h, b = (p.ev_h, p.ev_b) if held_out else (p.train_h, p.train_b)
    half = "heldout" if held_out else "inject"
    if direction == "bots":
        b, real = mix(b, p.ext[f"webbot_advanced_{half}"], rho)
    elif direction == "humans":
        h, real = mix(h, p.ext[f"balabit_{half}"], rho)
    else:
        real = 0.0
    return h, b, real


def _chunk_scores(cache, sessions, max_chunks=12):
    out = []
    for s in sessions:
        v = [cache._table.get((s.key, i)) for i in range(min(len(s.chunks), max_chunks))]
        out.append([z for z in v if z is not None])
    return out


def drift_metrics(p, levels=LEVELS):
    """The x-axis and its companions, per point, on the evaluation pools (and, for reference,
    on the training pools after the switch)."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for held_out in (True, False):
        for direction, rho in points(levels):
            h, b, real = shifted(p, direction, rho, held_out)
            hs, bs = _chunk_scores(p.cache, h), _chunk_scores(p.cache, b)
            hc = np.concatenate([np.asarray(v) for v in hs if v])
            bc = np.concatenate([np.asarray(v) for v in bs if v])
            hm = np.array([np.mean(v) for v in hs if v])
            bm = np.array([np.mean(v) for v in bs if v])
            auc = roc_auc_score(np.r_[np.zeros(len(hm)), np.ones(len(bm))], np.r_[hm, bm])
            cls = direction if direction != "none" else "bots"
            for c, chunks in (("bots", bc), ("humans", hc)):
                err = np.mean(1 - chunks) if c == "bots" else np.mean(chunks)
                brier = np.mean((1 - chunks) ** 2) if c == "bots" else np.mean(chunks ** 2)
                acc = np.mean(chunks >= 0.5) if c == "bots" else np.mean(chunks < 0.5)
                rows.append({"pool": "eval" if held_out else "train", "direction": direction,
                             "rho": rho, "rho_realised": real, "class": c,
                             "shifted_class": c == cls, "citl_error": float(err),
                             "brier": float(brier), "accuracy_at_0.5": float(acc),
                             "pool_auc": float(auc)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def choice(smoke):
    path = OUT / "arch_choice.json"
    if not path.exists():
        if smoke:
            return {"dqn": "dqn:64", "ppo": "ppo:64", "linucb": "linucb:h64_e32:gaussian",
                    "thompson": "thompson:h64_e32:mog2"}
        raise RuntimeError(f"{path} missing -- run --setup for the drift stage")
    return json.loads(path.read_text())["choice"]


def hidden_of(spec):
    return tuple(int(x) for x in spec.split(":", 1)[1].split("-"))


def train_online(algo, humans, bots, cache, seed, episodes, smoke, **kw):
    """DQN or PPO, trained exactly as round 3's E-experiments train the DQN winner."""
    from expkit.ppo import train_ppo
    from expkit.trainer import train_dqn
    hidden = hidden_of(choice(smoke)[algo])
    kw.setdefault("bot_choices", BOT_VOLUMES)
    fn = train_ppo if algo == "ppo" else train_dqn
    return fn(humans, bots, cache, episodes=episodes, seed=seed, verbose=0, hidden=hidden,
              use_score=True, human_counts=R.TRAIN_HUMAN_RANGE, name=algo.upper(), **kw)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

def drift_tasks():
    t = [{"kind": "metrics"}, {"kind": "offline", "family": "statics", "seed": 0}]
    t += [{"kind": "offline", "family": f, "seed": s} for f in OFFLINE_FAMILIES for s in range(R.SEEDS)]
    t += [{"kind": "control", "algo": a, "arm": arm, "seed": s}
          for a in ALGOS for arm in ARMS for s in range(R.SEEDS)]
    t += [{"kind": "online", "algo": a, "arm": arm, "direction": d, "rho": rho, "seed": s}
          for a in ALGOS for d in DIRECTIONS for rho in LEVELS for arm in ARMS for s in range(R.SEEDS)]
    return t


def smoke_drift_tasks():
    return [{"kind": "metrics"}, {"kind": "offline", "family": "statics", "seed": 0},
            {"kind": "offline", "family": "ppo", "seed": 0},
            {"kind": "control", "algo": "dqn", "arm": "posterior (floored)", "seed": 0},
            {"kind": "online", "algo": "dqn", "arm": "posterior (floored)", "direction": "bots",
             "rho": 1.0, "seed": 0},
            {"kind": "online", "algo": "ppo", "arm": "oracle", "direction": "humans",
             "rho": 0.5, "seed": 0}]


def _name(task):
    parts = [task["kind"]] + [str(task[k]) for k in ("family", "algo", "arm", "direction", "rho", "seed")
                              if k in task]
    return "__".join(parts).replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")


def _evaluate_points(policy, name, p, pts, smoke, solve, cfg, tags):
    _, eval_seeds, volumes = C.budget(smoke)
    frames = []
    for direction, rho in pts:
        h, b, real = shifted(p, direction, rho, held_out=True)
        frames.append(C.evaluate(policy, name, h, b, p.cache, eval_seeds, volumes, solve=solve,
                                 cfg=cfg, eval_direction=direction, eval_rho=rho,
                                 eval_rho_realised=real, **tags))
    return pd.concat(frames, ignore_index=True)


def run_offline(task, smoke):
    fam, seed = task["family"], task["seed"]
    p = pools(smoke, f"R4 drift offline {fam} seed {seed}")
    pts = points(SMOKE_LEVELS if smoke else LEVELS)
    episodes = 3 if smoke else R.EPISODES
    if fam == "statics":
        frames = [_evaluate_points(pol, pol.name, p, pts, smoke, C.GROUNDED, C.GROUNDED_CFG,
                                   {"panel": "offline", "family": "statics", "train_seed": -1})
                  for pol in C.statics()]
        df = pd.concat(frames, ignore_index=True)
    else:
        ch = choice(smoke)
        base = "dqn" if fam == "ablation" else fam
        spec = ch[base].split(":", 1)[1]
        post = [spec.split(":")[1]] if base in ("linucb", "thompson") else None
        variants, _ = R.train_candidate(fam, spec, seed, p.train_h, p.train_b, p.cache, episodes,
                                        posteriors=post, solve=C.GROUNDED,
                                        cfg=C.GROUNDED_CFG)
        (tag, pol), = variants.items()
        if hasattr(pol, "eval_mode"):
            pol.eval_mode()
        name = {"dqn": "DQN", "ppo": "PPO", "linucb": "LinUCB", "thompson": "Thompson Sampling",
                "ablation": "DQN without H-Score"}[fam]
        df = _evaluate_points(pol, name, p, pts, smoke, C.GROUNDED, C.GROUNDED_CFG,
                              {"panel": "offline", "family": fam, "candidate": tag, "train_seed": seed})
    C.save(df, "drift", _name(task), smoke)


def run_trained(task, smoke):
    """kind "online" (shift injected at INJECT_EP) or "control" (never shifted)."""
    import e7_proxy_reward as E
    import expkit.trainer as T
    from expkit.posterior_reward import OutcomeModel
    algo, arm, seed = task["algo"], task["arm"], task["seed"]
    online = task["kind"] == "online"
    direction, rho = (task["direction"], task["rho"]) if online else ("none", 0.0)
    total, inject = (SMOKE_TOTAL, SMOKE_INJECT) if smoke else (TOTAL, INJECT_EP)
    p = pools(smoke, f"R4 drift {task['kind']} {algo} {arm} {direction} {rho} seed {seed}")
    after_h, after_b, real = shifted(p, direction, rho, held_out=False)
    spec = dict(E.CONFIGS[arm])
    hook = spec.pop("hook", None)
    spec.pop("episodes", None)
    om = calib = None
    if hook in E.POSTERIOR_HOOKS:
        om = OutcomeModel(C.GROUNDED, C.GROUNDED_CFG)
        calib = E.calibration_for(hook, E.fit_calibration(p) if hook != "score_only" else None)
    st = {"ep": 0}
    real_run_x = T.run_x

    def run_x_hooked(policy, humans, bots, n_bots, **kw):
        if kw.get("collect") and humans is p.train_h:        # a training episode
            if online and st["ep"] >= inject:
                humans, bots = after_h, after_b
            st["ep"] += 1
        return real_run_x(policy, humans, bots, n_bots, **kw)

    t0 = time.time()
    with E.hooked(hook, spec.get("proxy_cfg"), None, om=om, calib=calib):
        T.run_x = run_x_hooked
        try:
            agent, _ = train_online(algo, p.train_h, p.train_b, p.cache, seed, total, smoke,
                                    solve=C.GROUNDED, cfg=C.GROUNDED_CFG, **spec)
        finally:
            T.run_x = real_run_x
    assert st["ep"] == total, f"{st['ep']} training episodes seen, expected {total}"
    agent.eval_mode()
    lv = SMOKE_LEVELS if smoke else LEVELS
    pts = points(lv) if not online else [("none", 0.0), (direction, rho)]
    tags = {"panel": "online" if online else "control", "algo": algo.upper(), "arm": arm,
            "train_direction": direction, "train_rho": rho, "train_rho_realised": real,
            "train_seed": seed}
    df = _evaluate_points(agent, f"{algo.upper()} {arm}", p, pts, smoke, C.GROUNDED,
                          C.GROUNDED_CFG, tags)
    C.save(df, "drift", _name(task), smoke)
    print(f"  {_name(task)}: {(time.time() - t0) / 60:.1f} min", flush=True)


def run_metrics(smoke):
    p = pools(smoke, "R4 drift metrics (scores only, no policy)")
    m = drift_metrics(p, SMOKE_LEVELS if smoke else LEVELS)
    d = C.out_dir("drift", smoke)
    m.to_csv(d / "drift_metrics.csv", index=False)
    print(m[m.shifted_class & (m.pool == "eval")].round(3).to_string(index=False))


def run_drift_task(task, smoke):
    {"metrics": lambda: run_metrics(smoke),
     "offline": lambda: run_offline(task, smoke),
     "control": lambda: run_trained(task, smoke),
     "online": lambda: run_trained(task, smoke)}[task["kind"]]()


# --------------------------------------------------------------------------- #
# Collect, contrasts, figure
# --------------------------------------------------------------------------- #

def drift_collect(smoke=False):
    from scipy.stats import ttest_ind
    d = C.out_dir("drift", smoke)
    runs = preflight.headline(C.load_tasks("drift", smoke))           # bots > 0
    runs = runs.copy()
    runs["eval_rho"] = runs.eval_rho.astype(float)
    m = pd.read_csv(d / "drift_metrics.csv")
    m = m[m.pool == "eval"]
    # x for direction dd: the error of class dd, at its shifted points AND at the shared rho = 0
    xx = pd.concat([m[((m.direction == dd) | (m.direction == "none")) & (m["class"] == dd)]
                    .assign(direction=dd) for dd in DIRECTIONS], ignore_index=True)
    xx = xx[["direction", "rho", "citl_error", "pool_auc", "brier"]]
    n_files = len(list((d / "tasks").glob("*.csv")))
    if not smoke and n_files < len(drift_tasks()) - 1:        # "metrics" writes no task CSV
        print(f"WARNING: {n_files} task CSVs, expected {len(drift_tasks()) - 1} -- rerun the missing ids")
    cols = ["DI", "human_survival", "bot_survival", "friction_s"]
    key = ["panel", "policy", "eval_direction", "eval_rho", "train_seed"]
    per_seed = runs.groupby(key, dropna=False)[cols].mean().reset_index()
    # rho = 0 is one point, shared by both directions: duplicate it into each for plotting
    zero = per_seed[per_seed.eval_direction == "none"]
    per_seed = pd.concat([per_seed[per_seed.eval_direction != "none"]]
                         + [zero.assign(eval_direction=dd) for dd in DIRECTIONS], ignore_index=True)
    per_seed = per_seed.merge(xx.rename(columns={"direction": "eval_direction", "rho": "eval_rho"}),
                              on=["eval_direction", "eval_rho"], how="left")
    per_seed.to_csv(d / "drift_per_seed.csv", index=False)
    g = per_seed.groupby(["panel", "policy", "eval_direction", "eval_rho"])
    t = g[cols + ["citl_error", "pool_auc"]].mean()
    t[[c + "_se" for c in cols]] = g[cols].sem().to_numpy()
    t["n_seeds"] = g.size()
    t.to_csv(d / "drift_curves.csv")
    pd.set_option("display.width", 240)
    print(t[["citl_error", "pool_auc", "DI", "DI_se", "n_seeds"]].round(3).to_string())

    # primary contrast: online vs control, mean over the 10 shifted levels, per seed
    rows = []
    online = per_seed[(per_seed.panel == "online") & (per_seed.eval_rho > 0)]
    control = per_seed[(per_seed.panel == "control") & (per_seed.eval_rho > 0)]
    for (pol, dd), a in online.groupby(["policy", "eval_direction"]):
        a = a.groupby("train_seed").DI.mean()
        b = control[(control.policy == pol) & (control.eval_direction == dd)].groupby("train_seed").DI.mean()
        if len(a) > 1 and len(b) > 1:
            rows.append({"policy": pol, "direction": dd, "online_DI": a.mean(), "control_DI": b.mean(),
                         "diff": a.mean() - b.mean(), "p": ttest_ind(a, b, equal_var=False).pvalue,
                         "n_online": len(a), "n_control": len(b)})
    if rows:
        ct = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
        k = len(ct)
        ct["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * pv for i, pv in enumerate(ct.p)]))
        ct.to_csv(d / "drift_contrasts.csv", index=False)
        print("\nPRIMARY CONTRAST (online - control, mean over shifted levels; Holm over "
              f"{k}):\n" + ct.round(4).to_string(index=False))
    return t


def drift_figure(smoke=False):
    from figure_drift import make_figure
    d = C.out_dir("drift", smoke)
    make_figure(d / "drift_curves.csv", d / "drift_metrics.csv", d / "drift_vs_calibration.png")


def drift_smoke():
    log = OUT / "eval_access.log"
    before = log.read_text() if log.exists() else ""
    for t in smoke_drift_tasks():
        print(f"[smoke] {t}", flush=True)
        run_drift_task(t, True)
    drift_collect(True)
    drift_figure(True)
    after = log.read_text() if log.exists() else ""
    assert after == before, "drift smoke opened the eval block"
    print("drift smoke: OK (eval block untouched)")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="round 4, fold 2")
    ap.add_argument("--stage", required=True, choices=("ppo", "drift"))
    for f in ("setup", "list", "collect", "smoke", "figure"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--task", type=int)
    a = ap.parse_args()
    tasks = ppo_tasks() if STAGE == "ppo" else drift_tasks()
    if a.list:
        for i, t in enumerate(tasks):
            print(f"{i:5d}  {t}")
        print(f"\n{len(tasks)} tasks  ->  sbatch --array=0-{len(tasks) - 1}%25 slurm/r4.sh {STAGE}")
        return
    if a.collect:
        return ppo_collect() if STAGE == "ppo" else drift_collect()
    if a.figure:
        return drift_figure()
    preflight.check(hardware=False)
    if a.setup:
        return setup()
    if a.smoke:
        return ppo_smoke() if STAGE == "ppo" else drift_smoke()
    if a.task is None:
        ap.error("one of --setup, --list, --task N, --collect, --smoke, --figure")
    preflight.check_budget(n_train_seeds=R.SEEDS, n_episodes=R.EPISODES, n_eval_seeds=C.EVAL_SEEDS)
    task = tasks[a.task]
    print(f"[{STAGE}] task {a.task}: {task}", flush=True)
    if STAGE == "ppo":
        R.run_search_task(task)
    else:
        run_drift_task(task, False)


if __name__ == "__main__":
    main()
