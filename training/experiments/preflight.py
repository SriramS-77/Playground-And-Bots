"""Round-2 gates. Import this at the top of every experiment script:

    import preflight; preflight.check()

Each assertion here corresponds to a specific way round 1 produced a wrong number. The
cluster run is unattended, so these have to fail the job rather than print a warning.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expkit  # noqa: F401  -- puts `training/` on sys.path
from expkit.humanity_scorer import ScorerConfig
from expkit.paths import RESULTS

# The design settled in round 1. Weights and temperature are refit per fold, so this is a
# config identity, never a weights identity.
FINAL_SCORER_CFG = ScorerConfig(
    representation="kinematic", padding="mask", context=32,
    lstm_units=(32,), dense_units=(32,), class_balance="window", augmentation="none",
)


def report_hardware() -> dict:
    """Round 1 ran 18.3 hours on CPU on a node called `nvidiaserver` and nobody noticed
    until the logs were read afterwards. Print it, every job."""
    info = {"python": sys.version.split()[0], "cuda": False, "devices": []}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["devices"] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
    except Exception as exc:                                   # pragma: no cover
        info["torch_error"] = str(exc)
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["tf_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception as exc:                                   # pragma: no cover
        info["tf_error"] = str(exc)
    print("preflight/hardware:", json.dumps(info))
    if not info["cuda"]:
        print("preflight/hardware: NO CUDA -- this run is CPU-only. Say so in the run "
              "notes. Python 3.14 has no TensorFlow GPU wheels; pin 3.11 or 3.12.")
    try:
        # check=False suppresses a non-zero exit code but NOT a missing executable,
        # which is what happens on any machine without the driver installed.
        subprocess.run(["nvidia-smi"], check=False)
    except (FileNotFoundError, OSError):
        print("preflight/hardware: nvidia-smi not on PATH")
    return info


def check_no_round1_scorer_choice():
    """`nb_02` writes `results/scorer_choice.json` and `nb_03` onwards reads it. That is
    exactly how round 1 ran every RL experiment against the wrong scorer. The file is not
    in git -- it gets CREATED by running nb_02 -- so the real guard is: never run nb_02,
    and fail if its output appears."""
    stale = RESULTS / "scorer_choice.json"
    assert not stale.exists(), (
        f"{stale} exists. Something ran nb_02 / run_all.py. Round 1's entire RL chain "
        "read this file and used the wrong scorer. Delete it and do not run nb_02.")


def check_scorer(scorer, fold=None, train_refs=None, val_refs=None):
    """The scorer must be the settled DESIGN, refit on THIS fold's scorer block.

    Do not assert on the temperature: T = 1.376 belonged to the old 73-recording `lstm`
    pool, and `results/final_scorer/` was fitted on recordings that land in the new rl and
    eval blocks -- loading those weights reintroduces the +8.7 DI leak E2 measured.
    """
    cfg = scorer.cfg
    for field in ("representation", "padding", "context", "lstm_units", "dense_units",
                  "class_balance"):
        want, got = getattr(FINAL_SCORER_CFG, field), getattr(cfg, field)
        assert want == got, f"scorer.{field}: expected {want!r}, got {got!r}"

    if fold is not None and train_refs is not None:
        seen = {r.name for r in train_refs}
        if val_refs is not None:
            seen |= {r.name for r in val_refs}
        # The invariant is that the EVALUATION block is unseen -- by the scorer and by the
        # policy. The scorer is deliberately fitted on `scorer + rl` (~102 recordings):
        # on `fold.scorer` alone early stopping runs on a ~13-recording validation split
        # and is unstable. See `exp_policy_arch_search.fold_scorer`.
        assert not (seen & {r.name for r in fold.eval}), "scorer saw the evaluation block"


#: A healthy fold scorer sits at 0.95-0.98 on its own validation split. The two collapsed
#: fits measured on fold 0 (scorer-only pool) sat at 0.62 and 0.65.
MIN_SCORER_VAL_AUC = 0.85


def check_scorer_health(scorer, val_refs=None, min_val_auc: float = MIN_SCORER_VAL_AUC):
    """Fail on a scorer that trained and saved without error but learned nothing.

    Round 2's fold-0 scorer did exactly that: validation loss never beat epoch 1, early
    stopping restored the epoch-1 weights, and every recording scored 0.41-0.64. Nothing
    raised; every table built on it was measuring the scorer's failure.

    Uses the training record `HumanityScorer.fit` persists. For a scorer without one, pass
    `val_refs` and the validation AUC is recomputed. Never pass `fold.eval` here -- a gate
    that can reject a scorer is a selection step, and the evaluation block takes no part
    in selection.
    """
    summary = getattr(scorer, "training_summary", None)
    if summary:
        # The epoch check only means something without `start_from_epoch` -- with it, the
        # best epoch is >= start by construction. It is a secondary check either way: a
        # cross-fitted fit collapsed with best epoch 2 (val AUC 0.616), which it would have
        # passed. Validation AUC is the gate that caught it.
        if not summary.get("start_from_epoch"):
            assert summary["best_epoch"] > 1, (
                f"scorer: early stopping restored epoch {summary['best_epoch']} of "
                f"{summary['epochs_run']} (val loss {summary['val_loss_best']:.3f}; chance "
                f"is 0.693) -- the network is effectively untrained")
        auc = summary["val_auc"]
    elif val_refs is not None:
        from expkit.calibration import roc_auc
        from expkit.humanity_scorer import _no_aug, build_chunk_dataset
        va = build_chunk_dataset(list(val_refs), _no_aug(scorer.cfg), dedup=True)
        p = scorer.model.predict(scorer.std(va.X), batch_size=1024, verbose=0).ravel()
        auc = roc_auc(p, va.y)
    else:
        raise ValueError("check_scorer_health needs the scorer's training record or "
                         "val_refs to recompute it")
    assert auc >= min_val_auc, (
        f"scorer: validation AUC {auc:.3f} < {min_val_auc} -- collapsed or near-collapsed "
        f"fit. Do not run anything on it; report it.")
    print(f"preflight/scorer: val AUC {auc:.3f}"
          + (f", best epoch {summary['best_epoch']}/{summary['epochs_run']}" if summary else ""))
    return auc


def check_threshold_canary(scorer_cache, humans, bots, min_human_survival: float = 0.5):
    """Downstream check: Static Single-Threshold must keep at least half the humans.

    It sends any user whose chunk scores above 0.5 straight to level 10, so it is the most
    scorer-sensitive policy there is: 0 % of humans on round 2's collapsed fold 0 versus
    ~80 % on a healthy scorer. Run it on a policy-block pool, never on `fold.eval`.
    """
    from rlcaptcha.policies.static import SingleThresholdPolicy
    from expkit.simx import run_x
    from expkit.stochastic import DETERMINISTIC, PAPER_EQUIVALENT

    res, _ = run_x(SingleThresholdPolicy(), humans, bots, 100, cache=scorer_cache,
                   n_humans=100, seed=0, solve=DETERMINISTIC, cfg=PAPER_EQUIVALENT)
    kept = res.surviving_humans / 100
    assert kept >= min_human_survival, (
        f"canary: Static Single-Threshold kept {kept:.0%} of humans (< "
        f"{min_human_survival:.0%}). The scorer is scoring humans as bots; every "
        f"threshold policy and every learned policy downstream would inherit it.")
    print(f"preflight/canary: Static Single-Threshold kept {kept:.0%} of humans")
    return kept


def check_batched_scoring(scorer, chunks, tol: float = 1e-6):
    """The RL loop only ever uses `score_chunks`. If it disagrees with `score_chunk`,
    every cached score is wrong and nothing downstream would reveal it."""
    chunks = [c for c in chunks if c][:64]
    if not chunks:
        return
    batched = scorer.score_chunks(chunks)
    single = [scorer.score_chunk(c) for c in chunks]
    worst = max(abs(a - b) for a, b in zip(batched, single))
    assert worst < tol, f"batched vs per-chunk scoring differ by {worst:.2e}"


def check_rotation(folds):
    """Disjoint roles inside each fold, and every recording evaluated exactly once."""
    from expkit.partition import index_sessions

    seen = []
    for f in folds:
        seen.extend(r.name for r in f.eval)
        names = {k: {r.name for r in getattr(f, k)} for k in ("scorer", "rl", "eval")}
        for a, b in (("scorer", "rl"), ("scorer", "eval"), ("rl", "eval")):
            assert not (names[a] & names[b]), f"fold {f.index}: {a}/{b} overlap"
    assert len(seen) == len(set(seen)) == len(index_sessions()), \
        f"rotation evaluates {len(seen)} recordings ({len(set(seen))} distinct)"


def check_determinism():
    """The 144/144 assertion: the extended simulator must reproduce the published one
    exactly in the deterministic limit, for all 6 policies x 6 bot volumes x 4 seeds."""
    from expkit.simx import verify_against_published
    ok, total = verify_against_published()
    assert ok == total, f"deterministic-limit check: {ok}/{total}"
    print(f"preflight/determinism: {ok}/{total} exact matches")


def headline(frame, bots_col: str = "bots"):
    """Drop the zero-bot cell. With no bots, S_B is undefined and DI = 100 for any policy
    that lets everyone through -- averaging it in rewards permissiveness, and it is what
    made a proxy agent that challenges NOBODY appear to beat the oracle by 14%. Use this
    helper for every headline mean; report the all-six average separately if at all."""
    return frame[frame[bots_col] > 0]


def check_budget(n_train_seeds: int, n_episodes: int, n_eval_seeds: int):
    """Round 1 used 1 training seed and 90-150 episodes against floors of 5 and 200."""
    assert n_train_seeds >= 5, f"training seeds {n_train_seeds} < 5"
    assert n_episodes >= 200, f"episodes {n_episodes} < 200"
    assert n_eval_seeds >= 20, f"evaluation seeds {n_eval_seeds} < 20"


def check(hardware: bool = True, strict_budget: bool = False, **budget):
    """The standard opening call."""
    check_no_round1_scorer_choice()
    if hardware:
        report_hardware()
    if strict_budget:
        check_budget(**budget)
    print("preflight: OK")


if __name__ == "__main__":
    check()
