"""Round 2 -- Phase 0: confirm the humanity-scorer design, per fold, at 5 seeds.

Why this script exists
----------------------

`nb_10_scorer_search.py` is **superseded** and must not be used for this. It searches at
the *session* operating point through `expkit.scorer_search`/`XScorer`, and the ranking
inverts at the chunk operating point the RL loop actually uses -- that discovery is what
produced the current design. `probe_decide.py` had the right operating point but
cross-validated over all 156 recordings with no fold discipline, so the evaluation block
took part in choosing the config.

This script does it properly:

* **Chunk operating point**, deduplicated chunks, grouped 5-fold CV by recording, with
  out-of-fold pooling -- the same measurement that selected the current design.
* **Per rotation fold, on `scorer + rl` only.** The evaluation block never participates in
  choosing a scorer config. If all three folds pick the same config, the design is clean
  and the paper can say so.
* **5 seeds**, as the handoff's budget floor requires.
* **Per-family recall at 0.5 with window counts, never per-family AUC.** AUC over a
  5-window class is meaningless -- it is what made NaiveBot look like a 0.975 failure when
  its recall was 16/16.

Selection rule -- fixed in writing before running, not to be changed afterwards
------------------------------------------------------------------------------

    1. Primary: chunk-level AUC, mean over the 5 seeds.
    2. Among configs within 1 SE of the best: lowest chunk-level ECE after temperature
       scaling.
    3. Among those: lowest |corr(score, padding_fraction)|.
    4. Among those: fewest parameters.

The SE is taken over seeds, not over pooled rows.

Expected outcome: confirmation. Round 1's independent probes already put
`kinematic` + `mask` + ctx 32 on top at chunk AUC 0.970 / ECE 0.014. If a sweep changes
the answer, that is a finding -- report it and re-save `results/final_scorer_cfg.json`,
and everything downstream uses the new config.

Usage
-----

    python exp_scorer_phase0.py --list                  # task list + the --array line
    python exp_scorer_phase0.py --smoke                 # minutes, validates wiring
    python exp_scorer_phase0.py --task $SLURM_ARRAY_TASK_ID
    python exp_scorer_phase0.py --audit                 # P0.6, cheap, no training
    python exp_scorer_phase0.py --collect               # apply the rule, write the choice
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import expkit  # noqa: F401  -- puts `training/` on sys.path
import preflight
from expkit.calibration import apply_temperature, fit_temperature, report
from expkit.humanity_scorer import (ScorerConfig, Standardiser, build_chunk_dataset,
                                    _sample_weights)
from expkit.partition import index_sessions, load_rotation, make_rotation, save_rotation
from expkit.paths import RESULTS
from expkit.scorer_search import _folds

OUT = RESULTS
PHASE0 = OUT / "phase0"

# The round-1 answer. Every sweep varies one axis away from this.
BASE = ScorerConfig(representation="kinematic", padding="mask", context=32,
                    lstm_units=(32,), dense_units=(32,), class_balance="window",
                    augmentation="none", epochs=100, patience=12)

REPRESENTATIONS = ("xy", "xy_dt", "dxdy", "dxdy_dt", "kinematic")
PADDINGS = ("repeat_point", "repeat_row", "mask", "zero_mask")


def _cfg(**kw) -> ScorerConfig:
    return dataclasses.replace(BASE, **kw)


def sweeps() -> dict[str, list[ScorerConfig]]:
    """Each entry is one handoff sweep id. Names are what the CSVs key on."""
    s: dict[str, list[ScorerConfig]] = {}

    # P0.1 representation x padding, at the chosen context.
    s["p01"] = [_cfg(representation=r, padding=p)
                for r, p in itertools.product(REPRESENTATIONS, PADDINGS)]

    # P0.2 capacity, on the two strongest representations. The last row is the published
    # architecture (304,049 parameters on ~206 windows) as the reference.
    archs = [((16,), (8,)), ((16, 8), (16,)), ((32,), (32,)),
             ((32, 16), (32,)), ((64, 32), (64,)), ((200, 100), (128, 64))]
    s["p02"] = [_cfg(representation=r, lstm_units=l, dense_units=d)
                for r in ("kinematic", "dxdy") for l, d in archs]

    # P0.3 regularisation -- matters more when inputs are half padding.
    s["p03"] = [_cfg(lr=lr, dropout=do, recurrent_dropout=rd)
                for lr, do, rd in itertools.product((3e-4, 1e-3, 3e-3), (0.0, 0.25),
                                                    (0.0, 0.2))]

    # P0.4 class balancing. The answer differed by pool size in round 1: family balancing
    # gained +0.003 at CV scale and lost 0.084 on the 59-recording pool, because
    # NaiveBot's 9 windows drew 10.5x a human window's weight.
    s["p04"] = [_cfg(class_balance=b) for b in ("window", "family")]

    # P0.5 augmentation, redone at chunk level -- `none` won on session windows, which
    # may not hold once inputs are padded.
    s["p05"] = [_cfg(augmentation="none")]
    s["p05"] += [_cfg(augmentation="rigid", magnitude=1, n_copies=4)]
    s["p05"] += [_cfg(augmentation="per_move", magnitude=m, n_copies=4) for m in (1, 2, 3)]
    s["p05"] += [_cfg(augmentation="gaussian", sigma=sg, n_copies=4) for sg in (1.0, 2.0)]

    # P0.7 context length, probing below 32 since the trend had not clearly turned.
    s["p07"] = [_cfg(context=c) for c in (16, 24, 32, 50, 64)]
    return s


def cfg_tag(cfg: ScorerConfig) -> str:
    """Short, stable, filename-safe identity for one config."""
    lstm = "-".join(map(str, cfg.lstm_units))
    dense = "-".join(map(str, cfg.dense_units))
    aug = cfg.augmentation if cfg.augmentation == "none" else \
        f"{cfg.augmentation}{cfg.magnitude if cfg.augmentation != 'gaussian' else cfg.sigma}x{cfg.n_copies}"
    # `_sfe` only when set, so every round-2 tag (and its CSV) keeps its identity.
    sfe = f"_sfe{cfg.start_from_epoch}" if getattr(cfg, "start_from_epoch", 0) else ""
    return (f"{cfg.representation}_{cfg.padding}_c{cfg.context}_l{lstm}_d{dense}"
            f"_lr{cfg.lr:g}_do{cfg.dropout:g}_rd{cfg.recurrent_dropout:g}"
            f"_{cfg.class_balance}_{aug}{sfe}")


# --------------------------------------------------------------------------- #
# One measurement
# --------------------------------------------------------------------------- #

def evaluate_cfg(cfg: ScorerConfig, refs, seed: int, n_splits: int = 5) -> dict:
    """Grouped CV by recording, out-of-fold pooling, at the chunk operating point.

    Temperature is fitted on the pooled *validation* predictions, never on the held-out
    fold -- otherwise the ECE being reported is the ECE of a fit to itself.
    """
    import tensorflow as tf
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam

    P, Y, F, PF, VP, VY = [], [], [], [], [], []
    params = 0
    for tr_refs, te_refs, _ in _folds(refs, n_splits, "session", seed):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tr_refs))
        nv = max(2, int(len(tr_refs) * 0.2))
        va = [tr_refs[i] for i in idx[:nv]]
        fit = [tr_refs[i] for i in idx[nv:]]

        tr = build_chunk_dataset(fit, cfg, dedup=True, augment_seed=seed)
        vv = build_chunk_dataset(va, _no_aug(cfg), dedup=True)
        te = build_chunk_dataset(te_refs, _no_aug(cfg), dedup=True)

        std = Standardiser().fit(tr.X)
        tf.keras.utils.set_random_seed(seed)
        model = cfg.build()
        model.compile(optimizer=Adam(cfg.lr), loss="bce")
        model.fit(std(tr.X), tr.y, sample_weight=_sample_weights(tr, cfg.class_balance),
                  validation_data=(std(vv.X), vv.y), epochs=cfg.epochs,
                  batch_size=cfg.batch_size, verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                           restore_best_weights=True)])
        params = int(model.count_params())
        P.append(model.predict(std(te.X), batch_size=1024, verbose=0).ravel())
        Y.append(te.y); F.append(te.families); PF.append(te.pad_fraction)
        VP.append(model.predict(std(vv.X), batch_size=1024, verbose=0).ravel())
        VY.append(vv.y)

    P, Y, F, PF = map(np.concatenate, (P, Y, F, PF))
    T = float(fit_temperature(np.concatenate(VP), np.concatenate(VY)))
    r, rT = report(P, Y), report(apply_temperature(P, T), Y)

    row = {"params": params, "n_windows": int(len(Y)), "temperature": T,
           "chunk_auc": r["AUC"], "acc": r["accuracy"],
           "ece": r["ECE_quantile"], "ece_T": rT["ECE_quantile"],
           "corr_pad": float(np.corrcoef(P, PF)[0, 1]), "mean_pad": float(PF.mean())}
    # Per-family RECALL at 0.5, with counts. Never per-family AUC (see the docstring).
    for fam in np.unique(F):
        m = F == fam
        hit = (P[m] < 0.5) if fam == "human" else (P[m] >= 0.5)
        row[f"rec_{fam}"] = float(hit.mean())
        row[f"n_{fam}"] = int(m.sum())
    return row


def _no_aug(cfg: ScorerConfig) -> ScorerConfig:
    return dataclasses.replace(cfg, augmentation="none", n_copies=1)


# --------------------------------------------------------------------------- #
# P0.6 -- the activity-volume audit. No training; run it once.
# --------------------------------------------------------------------------- #

def audit(refs) -> pd.DataFrame:
    """How much of the scorer's per-family performance does a movement count reproduce?

    Round 1: `movements < 10` gives NaiveBot recall 1.000 at 5.9% human false-positive
    rate, AUC 0.968. So for the low-activity families the network is detecting *absence
    of motion*, not motion quality. Quantifying that sharpens the paper's claim rather
    than weakening it -- say it explicitly instead of letting a reader find it.
    """
    from expkit.humanity_scorer import build_chunk_dataset

    cs = build_chunk_dataset(refs, BASE, dedup=True)
    counts = ((cs.X[..., -1] > 0).sum(axis=1) if BASE.padding in ("mask", "zero_mask")
              else np.full(len(cs), BASE.context))
    rows = []
    for thr in (6, 8, 10, 12, 16):
        pred = counts < thr
        for fam in np.unique(cs.families):
            m = cs.families == fam
            hit = (~pred[m]) if fam == "human" else pred[m]
            rows.append({"threshold": thr, "family": fam, "n": int(m.sum()),
                         "flagged_rate": float(pred[m].mean()),
                         "recall_by_count_rule": float(hit.mean())})
    df = pd.DataFrame(rows)
    PHASE0.mkdir(parents=True, exist_ok=True)
    df.to_csv(PHASE0 / "p06_activity_audit.csv", index=False)
    print(df.pivot_table(index="threshold", columns="family",
                         values="flagged_rate").round(3).to_string())
    return df


def validate_configs(refs, verbose: bool = True) -> int:
    """Build every config in every sweep, with no training. Seconds, not hours.

    868 tasks run unattended. Only `p04` has ever actually executed; `repeat_row`,
    `zero_mask`, `xy_dt`, the rigid and gaussian augmentations, `recurrent_dropout` and
    the (200,100)+dense(128,64) architecture have not. A typo in any of them surfaces as
    a wall of failed array tasks hours later. This catches it now.
    """
    sample = refs[:12]
    n = 0
    for name, grid in sweeps().items():
        for cfg in grid:
            cs = build_chunk_dataset(sample, cfg, dedup=True, augment_seed=0)
            assert cs.X.ndim == 3, (name, cfg_tag(cfg), cs.X.shape)
            assert cs.X.shape[1] == cfg.context, (name, cfg_tag(cfg), cs.X.shape)
            assert cs.X.shape[2] == cfg.n_features, (name, cfg_tag(cfg), cs.X.shape)
            model = cfg.build()
            assert model.count_params() > 0
            n += 1
    if verbose:
        print(f"  [ok] {n} configs build and produce well-shaped windows "
              f"({len(set(cfg_tag(c) for g in sweeps().values() for c in g))} distinct)")
    return n


# --------------------------------------------------------------------------- #
# Tasks, selection, driver
# --------------------------------------------------------------------------- #

def task_list(seeds: int, which: list[str]) -> list[dict]:
    """Deduplicated by config identity.

    Every sweep varies ONE axis away from the same baseline, so the baseline config
    appears in all six grids. Left alone it would collect 6x the rows of any competitor
    under one `cfg_tag`, shrinking its standard error by about sqrt(6) -- the 1-SE band
    would narrow around it and competitors would drop out of the ECE tiebreak. The
    "confirmation" would then be partly an artefact of the bookkeeping. Deduping also
    saves ~75 tasks.
    """
    grids = sweeps()
    out, seen = [], set()
    for fold in range(3):
        for name in which:
            for i, cfg in enumerate(grids[name]):
                key = (fold, cfg_tag(cfg))
                if key in seen:
                    continue
                seen.add(key)
                for seed in range(seeds):
                    out.append({"fold": fold, "sweep": name, "idx": i, "seed": seed})
    return out


def run_task(task: dict, folds, n_splits: int = 5) -> pd.DataFrame:
    fold = folds[task["fold"]]
    cfg = sweeps()[task["sweep"]][task["idx"]]
    # The evaluation block never takes part in choosing a scorer config.
    refs = list(fold.scorer) + list(fold.rl)
    assert not ({r.name for r in refs} & {r.name for r in fold.eval})

    row = evaluate_cfg(cfg, refs, task["seed"], n_splits)
    row.update({"fold": fold.index, "sweep": task["sweep"], "seed": task["seed"],
                "config": cfg_tag(cfg), "representation": cfg.representation,
                "padding": cfg.padding, "context": cfg.context,
                "lstm_units": str(cfg.lstm_units), "dense_units": str(cfg.dense_units),
                "lr": cfg.lr, "dropout": cfg.dropout,
                "recurrent_dropout": cfg.recurrent_dropout,
                "class_balance": cfg.class_balance, "augmentation": cfg.augmentation})
    frame = pd.DataFrame([row])
    PHASE0.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PHASE0 / f"f{fold.index}_{task['sweep']}_{task['idx']:02d}"
                          f"_s{task['seed']}.csv", index=False)
    return frame


def select(frame: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """The rule, verbatim. SE over per-seed means, not over pooled rows."""
    per_seed = (frame.groupby(["config", "seed"])
                     .agg(chunk_auc=("chunk_auc", "mean"), ece_T=("ece_T", "mean"),
                          corr_pad=("corr_pad", "mean"), params=("params", "first"))
                     .reset_index())
    g = (per_seed.groupby("config")
                 .agg(auc=("chunk_auc", "mean"), auc_sd=("chunk_auc", "std"),
                      n=("chunk_auc", "size"), ece_T=("ece_T", "mean"),
                      corr_pad=("corr_pad", "mean"), params=("params", "first"))
                 .reset_index())
    g["corr_pad_abs"] = g["corr_pad"].abs()
    g["se"] = g["auc_sd"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1))
    g = g.sort_values("auc", ascending=False).reset_index(drop=True)
    best = g.iloc[0]
    band = g[g["auc"] >= best["auc"] - best["se"]]
    band = band.sort_values(["ece_T", "corr_pad_abs", "params", "config"])
    return str(band.iloc[0]["config"]), g


def collect() -> dict:
    files = sorted(PHASE0.glob("f*_*.csv"))
    if not files:
        raise SystemExit(f"no task CSVs in {PHASE0} -- run the tasks first")
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    expected = len(task_list(int(runs.seed.nunique()), sorted(runs.sweep.unique())))
    if len(files) != expected:
        print(f"WARNING: {len(files)} task CSVs but the grid expects {expected}. "
              "Array tasks failed; selection would run on an incomplete grid.")
    runs.to_csv(OUT / "phase0_runs.csv", index=False)

    chosen = {}
    for fold in sorted(runs.fold.unique()):
        pick, table = select(runs[runs.fold == fold])
        table["fold"] = fold
        table.to_csv(OUT / f"phase0_table_fold{fold}.csv", index=False)
        chosen[int(fold)] = pick
        print(f"\nfold {fold} -> {pick}")
        print(table.head(10).round(4).to_string(index=False))

    agree = sorted(set(chosen.values()))
    payload = {"rule": "chunk AUC (SE over seeds); within 1 SE -> lowest ECE_T -> "
                       "lowest |corr(score, padding)| -> fewest params",
               "selected_on": "fold.scorer + fold.rl (never fold.eval)",
               "by_fold": chosen, "unanimous": len(agree) == 1,
               "distinct_choices": agree}
    (OUT / "phase0_choice.json").write_text(json.dumps(payload, indent=2))
    print(f"\nunanimous across folds: {payload['unanimous']}  ({len(agree)} distinct)")
    if not payload["unanimous"]:
        print("NOT unanimous -- report this. Use each fold's own winner downstream;\n"
              "that is proper nested CV, and the disagreement is itself a result.")
    print("wrote", OUT / "phase0_choice.json")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--audit", action="store_true", help="P0.6; no training")
    ap.add_argument("--validate", action="store_true",
                    help="build every config in every sweep; no training")
    ap.add_argument("--sweeps", default="p01,p02,p03,p04,p05,p07")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.collect:
        collect()
        return

    preflight.check(hardware=not args.list)
    rot = OUT / "rotation.json"
    folds = load_rotation(rot) if rot.exists() else make_rotation(index_sessions())
    if not rot.exists():
        save_rotation(folds, rot)
    preflight.check_rotation(folds)

    if args.audit:
        audit(list(folds[0].scorer) + list(folds[0].rl))
        return

    which = [s.strip() for s in args.sweeps.split(",") if s.strip()]
    if args.validate or args.smoke:
        validate_configs(list(folds[0].scorer))
        if args.validate:
            return
    if args.smoke:
        args.seeds, which = 1, ["p04"]          # 2 configs, 1 seed, 1 fold below
    tasks = task_list(args.seeds, which)
    if args.smoke:
        tasks = [t for t in tasks if t["fold"] == 0]

    if args.list:
        for i, t in enumerate(tasks):
            cfg = sweeps()[t["sweep"]][t["idx"]]
            print(f"{i:4d}  fold={t['fold']} {t['sweep']} seed={t['seed']}  {cfg_tag(cfg)}")
        print(f"\n{len(tasks)} tasks  ->  #SBATCH --array=0-{len(tasks) - 1}%25")
        return

    todo = [tasks[args.task]] if args.task is not None else tasks
    t0 = time.time()
    for i, task in enumerate(todo):
        r = run_task(task, folds, n_splits=3 if args.smoke else 5)
        print(f"  [{i + 1}/{len(todo)}] fold={task['fold']} {task['sweep']} "
              f"seed={task['seed']}  AUC={r.chunk_auc.iloc[0]:.3f} "
              f"ECE_T={r.ece_T.iloc[0]:.3f}  [{(time.time() - t0) / 60:.0f}m]", flush=True)
    if args.task is None:
        collect()


if __name__ == "__main__":
    main()
