"""Cross-fitted humanity scores for one rotation fold -- the round-3 scorer design.

The RL policy is the second stage of a two-stage pipeline, so it must train on scores the
scorer produced for recordings it never saw: stacked generalisation (Wolpert 1992;
Breiman 1996), cross-fitting (Chernozhukov et al. 2018), and what scikit-learn's
`StackingClassifier` implements. Measured on fold 0, one scorer fitted on `scorer + rl`
misscored **0.0 %** of human chunks on the rl block it trained on, against **6.7 %** on the
unseen eval block -- the policy never met a misscored human in training. Out-of-fold
scores misscored 7.5 %.

    scorer + rl (~102 recordings), split into K parts by recording, stratified by family
      part k's sub-scorer = fitted on the other K-1 parts
          -> scores the rl recordings in part k     (policy TRAINS and is SELECTED on these)
      refit scorer       = fitted on all of scorer + rl
          -> scores the eval block, and only that   (policy is EVALUATED on these)

No scorer ever scores a recording it was fitted on, and the eval block is unseen by all
K + 1 of them -- both asserted when the cache is built.

Gates, none of which touches `fold.eval`:
    every scorer      `preflight.check_scorer_health` -- validation AUC >= 0.85
    every sub-scorer  AUC on its own held-out part >= 0.85 -- cross-fitting's free
                      out-of-sample check (Ahrens, Chernozhukov et al.)
    assembled cache   `preflight.check_threshold_canary` on the policy block
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .humanity_scorer import HumanityScorer, ScorerConfig, _no_aug, build_chunk_dataset

K_DEFAULT = 5
MIN_HELDOUT_AUC = 0.85
#: Static Single keeps a human only if NO chunk scores above 0.5. On out-of-sample scores a
#: healthy scorer lets it keep 40-80 % of humans (single sub-scorers on fold 0's eval
#: block: 0.40-0.60; refit: 0.80); a collapsed one, 0 %. This threshold catches collapse
#: without false alarms on honest out-of-sample noise. (The 0.5 default is for seen data.)
OOF_CANARY_MIN_HUMANS = 0.25


def assign_parts(refs, k: int = K_DEFAULT, seed: int = 0) -> dict[str, int]:
    """Recording name -> part index, round-robin within each family after a seeded shuffle,
    so every part carries the same mix of humans and bot generators."""
    rng = random.Random(seed)
    by: dict[str, list] = {}
    for r in refs:
        by.setdefault(r.family, []).append(r)
    part = {}
    for fam in sorted(by):
        group = sorted(by[fam], key=lambda r: r.name)
        rng.shuffle(group)
        for i, r in enumerate(group):
            part[r.name] = i % k
    return part


def heldout_metrics(scorer: HumanityScorer, refs) -> dict:
    """Chunk AUC and human misscore rate on recordings the scorer never saw."""
    from .calibration import roc_auc

    ds = build_chunk_dataset(list(refs), _no_aug(scorer.cfg), dedup=True)
    p = scorer._calibrate(scorer.model.predict(scorer.std(ds.X), batch_size=1024,
                                               verbose=0).ravel())
    return {"auc": roc_auc(p, ds.y), "humans_misscored": float(np.mean(p[ds.y == 0] > 0.5)),
            "bots_missed": float(np.mean(p[ds.y == 1] <= 0.5)), "n_windows": int(len(ds.y))}


@dataclass
class CrossFit:
    fold_index: int
    cfg: ScorerConfig
    k: int
    seed: int
    part_of: dict          # recording name -> part, over scorer + rl
    subs: list             # K HumanityScorers; subs[i] never saw part i
    refit: HumanityScorer  # fitted on all of scorer + rl; scores fold.eval only
    report: dict           # per-scorer health and held-out metrics, for the deliverables

    def cache(self, policy_sessions, eval_sessions=()):
        """One ScoreCache: policy-block chunks from the sub-scorer that never saw that
        recording, eval-block chunks from the refit. Batched per scorer."""
        from rlcaptcha.scoring import ScoreCache

        groups: dict[int, list] = {}
        for s in policy_sessions:
            assert s.key in self.part_of, (
                f"{s.key} is not in this fold's scorer + rl pool -- a policy-block session "
                "must have an out-of-fold scorer")
            groups.setdefault(self.part_of[s.key], []).append(s)
        for s in eval_sessions:
            assert s.key not in self.part_of, (
                f"{s.key} is in scorer + rl, so the refit has seen it -- it cannot be "
                "scored as evaluation data")

        table = {}
        for part, sessions in groups.items():
            table.update(_score_sessions(self.subs[part], sessions))
        if eval_sessions:
            ev = _score_sessions(self.refit, eval_sessions)
            assert not (set(ev) & set(table)), "a chunk key was scored by two scorers"
            table.update(ev)
        cache = ScoreCache(self.refit)
        cache._table = table
        return cache


def _score_sessions(scorer: HumanityScorer, sessions) -> dict:
    keys, chunks = [], []
    for s in sessions:
        for i, c in enumerate(s.chunks):
            keys.append((s.key, i))
            chunks.append(c)
    return dict(zip(keys, scorer.score_chunks(chunks)))


def _provenance(fold, role, cfg, seed, k, train, val, held=()):
    return {"fold": int(fold.index), "pool": "scorer+rl", "role": role, "k": k,
            "seed": int(seed), "config": asdict(cfg),
            "train": sorted(r.name for r in train), "val": sorted(r.name for r in val),
            "held_out": sorted(r.name for r in held)}


def _matches(scorer, want: dict) -> str | None:
    p = scorer.provenance
    if not p:
        return "no provenance record"
    for key in ("fold", "pool", "role", "k", "seed", "train", "val", "held_out"):
        if p.get(key) != want[key]:
            return f"provenance field {key!r} differs"
    if p.get("config") != json.loads(json.dumps(want["config"])):
        return "scorer config differs"
    return None


def fit_crossfit(fold, cfg: ScorerConfig, out_dir, k: int = K_DEFAULT, seed: int = 0,
                 refit_stale: bool = False, verbose: bool = True) -> CrossFit:
    """Fit (or load) the K sub-scorers and the refit for `fold`, gate every one, save.

    Array tasks call this with `refit_stale=False`: a missing or mismatched scorer then
    fails the task instead of K + 1 tasks refitting concurrently into one directory. The
    `--prepare` step calls it with `refit_stale=True`.
    """
    import preflight
    from .partition import split_refs

    out = Path(out_dir)
    pool = list(fold.scorer) + list(fold.rl)
    part_of = assign_parts(pool, k, seed)
    by_name = {r.name: r for r in pool}
    assert not ({r.name for r in fold.eval} & set(part_of)), "eval block inside scorer + rl"

    fitted = []

    def one(role, train_pool, held):
        tr, va = split_refs(train_pool, 0.25, seed=seed)
        want = _provenance(fold, role, cfg, seed, k, tr, va, held)
        d = out / role.replace(" ", "")
        if (d / "model.keras").exists():
            s = HumanityScorer.load(d)
            why = _matches(s, want)
            if why is None:
                preflight.check_scorer_health(s)
                return s
            if not refit_stale:
                raise RuntimeError(f"{d} is stale ({why}). Run the --prepare step, which "
                                   "refits stale scorers, before launching the array.")
            if verbose:
                print(f"  {d.name}: stale ({why}) -- refitting")
        s = HumanityScorer(dataclasses.replace(cfg)).fit(tr, va, seed=seed)
        s.provenance = want
        preflight.check_scorer_health(s)
        s.save(d)
        fitted.append(role)
        return s

    report, subs = {}, []
    for i in range(k):
        held = [by_name[n] for n, p in part_of.items() if p == i]
        train_pool = [by_name[n] for n, p in part_of.items() if p != i]
        s = one(f"part {i}", train_pool, held)
        m = heldout_metrics(s, held)
        assert m["auc"] >= MIN_HELDOUT_AUC, (
            f"sub-scorer {i}: AUC {m['auc']:.3f} on its held-out part (< {MIN_HELDOUT_AUC}) "
            "-- collapsed or near-collapsed. Stop and report; see HANDOFF_ROUND3 §3.")
        report[f"part{i}"] = {**(s.training_summary or {}), **{f"heldout_{a}": b for a, b in m.items()}}
        if verbose:
            t = s.training_summary or {}
            print(f"  sub-scorer {i}: val AUC {t.get('val_auc', float('nan')):.3f}, best epoch "
                  f"{t.get('best_epoch')}/{t.get('epochs_run')} | held-out part "
                  f"({len(held)} recs) AUC {m['auc']:.3f}, humans misscored "
                  f"{m['humans_misscored']:.3f}", flush=True)
        subs.append(s)

    refit = one("refit", pool, ())
    report["refit"] = dict(refit.training_summary or {})
    if verbose:
        t = refit.training_summary or {}
        print(f"  refit: val AUC {t.get('val_auc', float('nan')):.3f}, best epoch "
              f"{t.get('best_epoch')}/{t.get('epochs_run')}", flush=True)

    # Written only when something was fitted, i.e. by --prepare. Array tasks only load, so
    # 25 concurrent tasks never race on this file while others are reading it.
    if fitted or not (out / "parts.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "parts.json.tmp"
        tmp.write_text(json.dumps(
            {"fold": int(fold.index), "k": k, "seed": seed, "config": asdict(cfg),
             "part_of": part_of, "report": report}, indent=2))
        tmp.replace(out / "parts.json")
    return CrossFit(fold.index, cfg, k, seed, part_of, subs, refit, report)
