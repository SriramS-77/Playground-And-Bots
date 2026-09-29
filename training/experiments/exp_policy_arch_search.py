"""Round 2 -- policy architecture and Thompson posterior search, on one rotation fold.

Reference implementation. The GPU agent should adapt the grids and budgets to its own
cluster, but must not change the selection rule or the pool discipline.

What this does, and why each piece is the way it is
---------------------------------------------------

1. **Rotation fold.** `expkit.partition.make_rotation` cuts 156 recordings into three
   equal blocks and rotates the scorer / policy / evaluation roles, so every recording is
   evaluated exactly once across the three folds. Round 1's 73/31/52 allocation gave the
   RL agent 31 recordings, 8 of them human, and that is the leading candidate for its
   collapse at high bot volumes.

2. **The scorer is retrained per fold.** `results/final_scorer/` was fitted on the OLD
   `lstm` pool; under the new split those recordings land in the new rl and eval pools,
   which is exactly the leak E2 measured at +8.7 DI. Carry forward the finalised
   *config* -- kinematic / mask / ctx 32 / lstm(32) / dense(32) / window balancing --
   and refit the weights and the temperature on this fold's scorer block.

3. **Selection happens on `rl_val`, never on `eval`.** Choosing an architecture or a
   posterior by evaluation-pool DI is test-set selection, which is Reviewer 1's point 2
   all over again. The policy block is split 2:1 by recording; candidates are ranked on
   the held-out third; the winner is then retrained on the full policy block and
   evaluated once.

4. **One environment for every policy.** `expkit.bandits_x` fixes all five DQN/bandit
   mismatches, and every policy here gets the same `cfg`, the same episode count, the
   same seeds and the same bot-volume sampler. Round 1's inversion (DQN third behind both
   bandits on the clean partition) was produced by an unequal comparison.

Selection rule -- fixed before running, not to be changed afterwards
-------------------------------------------------------------------

    1. Primary: mean DI over the evaluation seeds, **zero-bot cells excluded**.
    2. Among candidates within 1 SE of the best: fewest parameters.
    3. Ties broken by name, for determinism.

Usage
-----

    python exp_policy_arch_search.py --smoke                # ~minutes, validates wiring
    python exp_policy_arch_search.py --fold 0 --seeds 5     # one Slurm array task
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import expkit  # noqa: F401  -- puts `training/` on sys.path, so import it first
from rlcaptcha.scoring import ScoreCache
from expkit.bandits_x import (POSTERIORS, BanditArch, FeatureNetX, LinUCBX,
                              MixturePosterior, ThompsonX, fit_posteriors_from,
                              train_bandit)
from expkit.humanity_scorer import HumanityScorer, ScorerConfig
from expkit.partition import index_sessions, make_rotation, save_rotation
from expkit.paths import RESULTS
from expkit.simx import evaluate_x, run_x, sessions_from_refs
from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT
from expkit.trainer import PUBLISHED_HIDDEN, QNet, TrainableDQN, train_dqn

OUT = RESULTS
BOT_VOLUMES = (0, 20, 100, 200, 500, 1000)
SCORED_VOLUMES = tuple(b for b in BOT_VOLUMES if b > 0)   # headline excludes zero-bot

# The scorer design settled in round 1. Weights are refit per fold; this is the config.
FINAL_SCORER_CFG = ScorerConfig(
    representation="kinematic", padding="mask", context=32,
    lstm_units=(32,), dense_units=(32,), class_balance="window", augmentation="none",
)

DQN_HIDDEN = [
    (64,),
    (128, 64),
    PUBLISHED_HIDDEN,            # (128, 64, 32) -- the published reference row
    (256, 128, 64),
    (128, 64, 32, 16),
]

BANDIT_ARCHS = [
    BanditArch("h128_e64", (128,), 64),           # published reference row
    BanditArch("h64_e32", (64,), 32),
    BanditArch("h256_e64", (256,), 64),
    BanditArch("h128_64_e64", (128, 64), 64),
    BanditArch("h256_128_e128", (256, 128), 128),
]


# --------------------------------------------------------------------------- #
# Smoke tests -- these run before anything expensive, because the cluster job is
# unattended and every one of these guards a number nobody downstream would question.
# --------------------------------------------------------------------------- #

def smoke_tests(verbose: bool = True):
    def ok(msg):
        if verbose:
            print(f"  [ok] {msg}")

    # 1. The default QNet is still byte-compatible with the published checkpoints.
    from rlcaptcha.config import DQN_CKPT
    net = QNet(5)
    assert [n for n, _ in net.named_parameters()][:2] == ["layer1.weight", "layer1.bias"]
    if Path(DQN_CKPT).exists():
        ckpt = torch.load(str(DQN_CKPT), map_location="cpu", weights_only=False)
        sd = ckpt.get("policy_net_state_dict", ckpt)
        net.load_state_dict(sd)
        ok("default QNet loads the published DQN checkpoint")
    assert QNet(5, hidden=(64, 32)).n_layers == 3
    ok("QNet depth is configurable")

    # 2. The published feature network is reproduced at the default arch.
    from rlcaptcha.policies.bandits import FeatureNet
    a = {k: tuple(v.shape) for k, v in FeatureNet(5, 64).state_dict().items()}
    b = {k: tuple(v.shape) for k, v in FeatureNetX(5, (128,), 64).state_dict().items()}
    assert a == b, (a, b)
    ok("FeatureNetX at hidden=(128,) matches the published FeatureNet")

    # 3. K=1 mixture == published Gaussian Thompson draw, bit-exactly.
    d, alpha = 8, 1.0
    post = MixturePosterior(d, 1, alpha, "diag")
    rng = np.random.default_rng(0)
    for _ in range(40):
        z = torch.tensor(rng.normal(size=d), dtype=torch.float32)
        post.update(z, float(rng.normal()))
    post.freeze()
    z = torch.tensor(rng.normal(size=d), dtype=torch.float32)

    invA = torch.linalg.inv(post.A[0])
    theta = invA @ post.b[0]
    sigma = alpha * torch.sqrt(torch.diag(invA))
    torch.manual_seed(7)
    want = float(z @ (theta + torch.randn_like(theta) * sigma))
    torch.manual_seed(7)
    got = post.thompson_score(z)
    assert abs(want - got) < 1e-9, (want, got)
    assert float(post.gate(z)[0]) == 1.0
    ok("K=1 posterior reproduces published Gaussian Thompson Sampling exactly")

    # 4. A mixture gate actually depends on the context (the whole point -- a
    #    context-free gate leaves E[r|z,a] linear in z and adds nothing).
    mog = MixturePosterior(d, 2, alpha, "diag", warmup=16)
    for i in range(80):
        z = torch.tensor(rng.normal(loc=(3.0 if i % 2 else -3.0), size=d),
                         dtype=torch.float32)
        mog.update(z, 50.0 if i % 2 else -50.0)
    mog.freeze()
    g1 = mog.gate(torch.full((d,), 3.0))
    g2 = mog.gate(torch.full((d,), -3.0))
    assert float((g1 - g2).abs().max()) > 0.1, (g1, g2)
    ok("mixture gate is context-dependent")

    # 4b. Batch EM actually fits a two-regime reward. `recompute_statistics` is the ONLY
    #     path that produces an evaluated mixture, so this is the one that matters --
    #     a single online sweep there leaves every assignment made against a `theta`
    #     fitted on the first `warmup` samples.
    th_a, th_b = rng.normal(size=d), rng.normal(size=d)

    def regime_data(n, separated):
        Z, R = [], []
        for _ in range(n):
            second = rng.random() < 0.5
            loc = (1.5 if second else -1.5) if separated else 0.0
            z = rng.normal(loc=loc, size=d)
            Z.append(z)
            R.append(float(z @ (th_b if second else th_a)) + rng.normal(scale=0.3))
        return np.array(Z, dtype=np.float32), np.array(R, dtype=np.float32)

    Ztr, Rtr = regime_data(600, True)
    Zte, Rte = regime_data(300, True)
    errs = {}
    for K in (1, 2):
        p = MixturePosterior(d, K, alpha, "diag", warmup=64).fit_batch(Ztr, Rtr)
        p.freeze()
        errs[K] = float(np.mean([(p.mean_score(torch.tensor(z)) - r) ** 2
                                 for z, r in zip(Zte, Rte)]))
    assert errs[2] < 0.25 * errs[1], errs
    ok(f"batch EM fits two reward regimes (K=1 MSE {errs[1]:.1f} -> K=2 {errs[2]:.2f})")

    # 5. Every aligned bandit carries the DQN's flags.
    for cls in (LinUCBX, ThompsonX):
        p = cls(seed=0)
        assert p.acts_when_exhausted is False
        assert p.updates_last_threat is True
        assert p.initial_last_threat == 0
    dqn = TrainableDQN(seed=0)
    for attr in ("acts_when_exhausted", "updates_last_threat", "initial_last_threat"):
        assert getattr(LinUCBX(seed=0), attr) == getattr(dqn, attr)
    ok("LinUCBX / ThompsonX are aligned with TrainableDQN on all four policy flags")

    # 6. The rotation is disjoint and evaluates each recording exactly once.
    folds = make_rotation(index_sessions())
    n_eval = sum(len(f.eval) for f in folds)
    assert n_eval == len(index_sessions()), n_eval
    ok(f"rotation: 3 folds, {n_eval} recordings, each evaluated exactly once")
    return folds


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def batched_cache(scorer: HumanityScorer, *groups, verbose: bool = True) -> ScoreCache:
    """Build a ScoreCache through the scorer's BATCHED path.

    `ScoreCache.precompute` calls `score_chunk` once per chunk, and a Keras `predict`
    costs ~59 ms of graph dispatch regardless of model size. Batched it is ~0.33 ms
    amortised, which is the difference between minutes and hours per fold.
    """
    cache = ScoreCache(scorer)
    keys, chunks = [], []
    for sessions in groups:
        for session in sessions:
            for i, chunk in enumerate(session.chunks):
                keys.append((session.key, i))
                chunks.append(chunk)
    scores = scorer.score_chunks(chunks)
    cache._table = dict(zip(keys, scores))
    if verbose:
        print(f"  score cache: {len(cache._table)} (session, chunk) entries")
    return cache


def fold_scorer(fold, seed: int = 0, val_fraction: float = 0.25,
                cfg: ScorerConfig = FINAL_SCORER_CFG) -> HumanityScorer:
    """Refit the finalised scorer design on this fold's scorer block only."""
    from expkit.partition import split_refs

    train_refs, val_refs = split_refs(fold.scorer, val_fraction, seed=seed)
    scorer = HumanityScorer(dataclasses.replace(cfg)).fit(train_refs, val_refs, seed=seed)
    # The gate that matters: the scorer must never have seen the policy or eval blocks.
    names = {r.name for r in train_refs} | {r.name for r in val_refs}
    assert not (names & {r.name for r in fold.rl}), "scorer saw the policy block"
    assert not (names & {r.name for r in fold.eval}), "scorer saw the evaluation block"
    return scorer


def score_policy(policy, humans, bots, cache, seeds: int, volumes=SCORED_VOLUMES,
                 solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT) -> pd.DataFrame:
    """Evaluate one policy. Zero-bot cells are excluded by default -- with no bots, DI is
    100 for any policy that lets everyone through, which is what made a proxy agent that
    challenges nobody look like it beat the oracle."""
    rows = []
    for n_bots in volumes:
        for seed in range(seeds):
            result, _ = run_x(policy, humans, bots, n_bots, cache=cache,
                              n_humans=100, seed=seed, solve=solve, cfg=cfg)
            m = evaluate_x(result)      # a `rlcaptcha.metrics.Metrics` dataclass
            rows.append({"policy": policy.name, "bots": n_bots, "seed": seed,
                         "DI": m.DI, "BOS": m.BOS, "SP_F1": m.SP_F1, "SI_F1": m.SI_F1,
                         "humans_left": result.surviving_humans,
                         "bots_left": result.surviving_bots})
    return pd.DataFrame(rows)


def select(frame: pd.DataFrame, params: dict[str, int]) -> tuple[str, pd.DataFrame]:
    """The rule, applied verbatim: best mean DI; then within 1 SE, fewest parameters.

    The SE is taken over **training seeds**, not over the ~500 raw rows. Pooling
    volumes x evaluation seeds x training seeds would shrink the SE by 2-4x and fold
    between-volume spread into it, so the 1-SE band would collapse and the
    fewest-parameters tiebreak would never fire. Per-seed means first, then the spread
    across seeds -- the same rule notebook 10b used.
    """
    per_seed = (frame.groupby(["candidate", "train_seed"])["DI"].mean()
                     .reset_index(name="DI"))
    g = per_seed.groupby("candidate")["DI"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"].fillna(0.0) / np.sqrt(g["count"].clip(lower=1))
    g["params"] = g["candidate"].map(params)
    g = g.sort_values("mean", ascending=False).reset_index(drop=True)
    best = g.iloc[0]
    band = g[g["mean"] >= best["mean"] - best["se"]].sort_values(["params", "candidate"])
    return str(band.iloc[0]["candidate"]), g


def n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

def fold_scorer_cached(fold, seed: int = 0) -> HumanityScorer:
    """Fit the fold's scorer once and reuse it.

    Every task in the array must score against the *same* scorer. If each refits its own,
    TF/oneDNN nondeterminism gives slightly different weights and candidates are then
    ranked against different yardsticks. Run `--prepare` first; tasks then load.
    """
    d = OUT / f"fold{fold.index}_scorer"
    if (d / "model.keras").exists():
        return HumanityScorer.load(d)
    scorer = fold_scorer(fold, seed=seed)
    scorer.save(d)
    return scorer


def task_list(seeds: int) -> list[dict]:
    """One training per entry, in a fixed order, so `--task N` is stable across nodes."""
    out = []
    for fold in range(3):
        for seed in range(seeds):
            for hidden in DQN_HIDDEN:
                out.append({"fold": fold, "family": "dqn",
                            "spec": "-".join(map(str, hidden)), "seed": seed})
            for kind in ("linucb", "thompson"):
                for arch in BANDIT_ARCHS:
                    out.append({"fold": fold, "family": kind,
                                "spec": arch.name, "seed": seed})
    return out


def run_task(task: dict, folds, episodes: int, eval_seeds: int) -> pd.DataFrame:
    """One (fold, family, arch, seed). Writes its own CSV, so array tasks never collide."""
    fold = folds[task["fold"]]
    scorer = fold_scorer_cached(fold)
    fit_h, fit_b = sessions_from_refs(fold.rl_fit)
    val_h, val_b = sessions_from_refs(fold.rl_val)
    cache = batched_cache(scorer, fit_h, fit_b, val_h, val_b, verbose=False)

    seed, rows = task["seed"], []
    if task["family"] == "dqn":
        hidden = tuple(int(x) for x in task["spec"].split("-"))
        agent, _ = train_dqn(fit_h, fit_b, cache, episodes=episodes, seed=seed,
                             bot_choices=BOT_VOLUMES, verbose=0, name=f"DQN {hidden}")
        variants = {f"dqn:{task['spec']}": agent.eval_mode()}
        params = {k: n_params(agent.policy_net) for k in variants}
    else:
        kind = task["family"]
        arch = next(a for a in BANDIT_ARCHS if a.name == task["spec"])
        policy, _ = train_bandit(kind, fit_h, fit_b, cache, arch=arch,
                                 episodes=episodes, seed=seed,
                                 bot_choices=BOT_VOLUMES, verbose=0)
        variants = {f"{kind}:{arch.name}:gaussian": policy}
        if kind == "thompson":
            # Same weights, same buffer -- the posterior is the only thing that varies.
            for name in POSTERIORS:
                if name != "gaussian":
                    variants[f"{kind}:{arch.name}:{name}"] = \
                        fit_posteriors_from(policy, policy._buffer, name)
        params = {k: n_params(v.net) for k, v in variants.items()}

    for tag, variant in variants.items():
        df = score_policy(variant, val_h, val_b, cache, eval_seeds)
        df["candidate"], df["train_seed"] = tag, seed
        df["fold"], df["family"] = fold.index, task["family"]
        df["params"] = params[tag]
        rows.append(df)

    frame = pd.concat(rows, ignore_index=True)
    dest = OUT / "arch_search"
    dest.mkdir(exist_ok=True)
    frame.to_csv(dest / f"f{fold.index}_{task['family']}_{task['spec']}_s{seed}.csv",
                 index=False)
    return frame


def collect() -> dict:
    """Concatenate every task CSV and apply the selection rule, per fold."""
    files = sorted((OUT / "arch_search").glob("*.csv"))
    if not files:
        raise SystemExit("no task CSVs in results/arch_search -- run the tasks first")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs.to_csv(OUT / "policy_arch_runs.csv", index=False)

    chosen: dict = {}
    for fold in sorted(runs.fold.unique()):
        chosen[int(fold)] = {}
        for group, mask in (("dqn", runs.family == "dqn"),
                            ("bandit", runs.family != "dqn")):
            sub = runs[mask & (runs.fold == fold)]
            if sub.empty:
                continue
            params = sub.groupby("candidate")["params"].first().to_dict()
            pick, table = select(sub, params)
            table["fold"], table["group"] = fold, group
            table.to_csv(OUT / f"policy_arch_{group}_fold{fold}.csv", index=False)
            chosen[int(fold)][group] = pick
            print(f"\nfold {fold} [{group}] -> {pick}")
            print(table.round(3).to_string(index=False))

    agree = {g: {f[g] for f in chosen.values() if g in f} for g in ("dqn", "bandit")}
    payload = {"rule": "best mean DI (bots>0), SE over training seeds; "
                       "within 1 SE, fewest parameters",
               "selected_on": "rl_val", "by_fold": chosen,
               "cross_fold_agreement": {g: sorted(v) for g, v in agree.items()}}
    (OUT / "policy_arch_choice.json").write_text(json.dumps(payload, indent=2))
    print("\ncross-fold agreement:", payload["cross_fold_agreement"])
    print("wrote", OUT / "policy_arch_choice.json")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true",
                    help="fit and cache the three per-fold scorers, then exit")
    ap.add_argument("--task", type=int, default=None,
                    help="run one training unit; index into --list")
    ap.add_argument("--list", action="store_true", help="print the task list and exit")
    ap.add_argument("--collect", action="store_true",
                    help="concatenate task CSVs and apply the selection rule")
    ap.add_argument("--seeds", type=int, default=5, help="training seeds per candidate")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.collect:
        collect()
        return

    print("smoke tests")
    folds = smoke_tests()
    save_rotation(folds, OUT / "rotation.json")

    if args.smoke:
        args.seeds, args.episodes, args.eval_seeds = 1, 4, 2
        global DQN_HIDDEN, BANDIT_ARCHS
        DQN_HIDDEN, BANDIT_ARCHS = DQN_HIDDEN[:1], BANDIT_ARCHS[:1]

    tasks = task_list(args.seeds)
    if args.list:
        for i, t in enumerate(tasks):
            print(f"{i:4d}  fold={t['fold']}  {t['family']:9s} {t['spec']:16s} seed={t['seed']}")
        print(f"\n{len(tasks)} tasks  ->  #SBATCH --array=0-{len(tasks) - 1}%25")
        return

    if args.prepare:
        for fold in folds:
            t = time.time()
            fold_scorer_cached(fold)
            print(f"  fold {fold.index} scorer ready [{time.time() - t:.0f}s]")
        return

    todo = [tasks[args.task]] if args.task is not None else tasks
    t0 = time.time()
    for i, task in enumerate(todo):
        run_task(task, folds, args.episodes, args.eval_seeds)
        print(f"  [{i + 1}/{len(todo)}] fold={task['fold']} {task['family']}:"
              f"{task['spec']} seed={task['seed']}  [{(time.time() - t0) / 60:.0f}m]",
              flush=True)
    if args.task is None:
        collect()


if __name__ == "__main__":
    main()
