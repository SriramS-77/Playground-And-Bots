"""Is 156 recordings enough? (the question the reviewers circle but never name)

The seeded Table 4 varies the simulation RNG. That measures one thing only: how much
the *simulation* wobbles given a fixed set of recordings. It says nothing about the
uncertainty that matters for a claim about bots and humans in general, which is
uncertainty over the **recording pool** -- 44 human and 112 bot sessions, from four
scripted bot generators and an unreported number of participants.

Two analyses here:

* ``bootstrap_pool`` resamples SESSIONS with replacement and re-runs the simulation.
  The resulting spread is the session-level confidence interval. It is the number that
  belongs in the paper, and it is much wider than the seed-level spread.
* ``learning_curve`` subsamples K recordings for increasing K and reports how the metric
  and its spread move. If the curve has not flattened by K = all, the dataset is the
  binding constraint and no amount of extra seeds will fix it.

``variance_decomposition`` splits total variance into a session component and a seed
component, which is the compact way to report the finding.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from rlcaptcha.metrics import evaluate

from .simx import evaluate_x, run_x, sessions_from_refs


def _metric(result, key: str) -> float:
    m = evaluate_x(result)
    return getattr(m, key.replace("-", "_"))


def bootstrap_pool(
    policy, refs, n_bots: int, cache_factory, key: str = "DI",
    n_boot: int = 200, n_humans: int = 100, seed: int = 0, **run_kw
) -> np.ndarray:
    """Resample the recording pool with replacement; one simulation per resample.

    `cache_factory(humans, bots)` must return a ScoreCache for the resampled pool --
    scores are keyed on session name, so a cache built once over ALL sessions is valid
    for every resample and is what you normally pass.
    """
    rng = random.Random(seed)
    humans_all = [r for r in refs if not r.is_bot]
    bots_all = [r for r in refs if r.is_bot]

    values = []
    for b in range(n_boot):
        h = [rng.choice(humans_all) for _ in range(len(humans_all))]
        bt = [rng.choice(bots_all) for _ in range(len(bots_all))]
        humans, bots = sessions_from_refs(h + bt)
        cache = cache_factory(humans, bots)
        result, _ = run_x(policy, humans, bots, n_bots, cache=cache,
                          n_humans=n_humans, seed=rng.randrange(10**9), **run_kw)
        values.append(_metric(result, key))
    return np.array(values)


def learning_curve(
    policy, refs, n_bots: int, cache, sizes, key: str = "DI",
    n_repeats: int = 12, n_humans: int = 100, seed: int = 0, **run_kw
):
    """Metric vs. number of distinct recordings available to the simulator.

    `sizes` are fractions of the pool (e.g. 0.25 -> a quarter of humans and a quarter of
    bots). Each size is repeated `n_repeats` times with a different random subset.
    """
    rng = random.Random(seed)
    humans_all = [r for r in refs if not r.is_bot]
    bots_all = [r for r in refs if r.is_bot]
    rows = []

    for frac in sizes:
        n_h = max(2, round(len(humans_all) * frac))
        n_b = max(2, round(len(bots_all) * frac))
        for rep in range(n_repeats):
            h = rng.sample(humans_all, n_h)
            bt = rng.sample(bots_all, n_b)
            humans, bots = sessions_from_refs(h + bt)
            result, _ = run_x(policy, humans, bots, n_bots, cache=cache,
                              n_humans=n_humans, seed=rng.randrange(10**9), **run_kw)
            rows.append({
                "fraction": frac, "n_human_sessions": n_h, "n_bot_sessions": n_b,
                "repeat": rep, key: _metric(result, key),
                "surviving_humans": result.surviving_humans,
                "surviving_bots": result.surviving_bots,
            })
    import pandas as pd
    return pd.DataFrame(rows)


def variance_decomposition(
    policy, refs, n_bots: int, cache, key: str = "DI",
    n_pools: int = 15, n_seeds: int = 15, n_humans: int = 100, seed: int = 0, **run_kw
):
    """Nested design: `n_pools` bootstrap pools x `n_seeds` simulation seeds each.

    Returns (DataFrame of every run, dict of variance components). The between-pool
    component is the one the paper currently does not report.
    """
    rng = random.Random(seed)
    humans_all = [r for r in refs if not r.is_bot]
    bots_all = [r for r in refs if r.is_bot]
    rows = []

    for p in range(n_pools):
        h = [rng.choice(humans_all) for _ in range(len(humans_all))]
        bt = [rng.choice(bots_all) for _ in range(len(bots_all))]
        humans, bots = sessions_from_refs(h + bt)
        for s in range(n_seeds):
            result, _ = run_x(policy, humans, bots, n_bots, cache=cache,
                              n_humans=n_humans, seed=1000 * p + s, **run_kw)
            rows.append({"pool": p, "seed": s, key: _metric(result, key)})

    import pandas as pd
    df = pd.DataFrame(rows)
    group_means = df.groupby("pool")[key].mean()
    within = df.groupby("pool")[key].var(ddof=1).mean()
    between = group_means.var(ddof=1)
    # Between-pool mean variance is inflated by the within component / n_seeds.
    between_corrected = max(0.0, between - within / n_seeds)
    total = between_corrected + within
    return df, {
        "var_between_pools": float(between_corrected),
        "var_within_pool_seeds": float(within),
        "total": float(total),
        "share_session_level": float(between_corrected / total) if total else float("nan"),
        "sd_between_pools": float(np.sqrt(between_corrected)),
        "sd_within_pool": float(np.sqrt(within)),
    }


def ci(values, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def required_sessions(curve_df, key: str = "DI", tolerance: float = 0.05):
    """Crude sufficiency check: the smallest fraction whose mean is within `tolerance`
    (relative) of the full-pool mean AND whose spread has stopped shrinking materially.
    """
    g = curve_df.groupby("fraction")[key].agg(["mean", "std"]).sort_index()
    full = g["mean"].iloc[-1]
    ok = g[(g["mean"] - full).abs() <= abs(full) * tolerance]
    return {
        "full_pool_mean": float(full),
        "sd_at_full_pool": float(g["std"].iloc[-1]),
        "sd_at_half_pool": float(g["std"].iloc[len(g) // 2]),
        "smallest_sufficient_fraction": float(ok.index[0]) if len(ok) else None,
        "table": g,
    }
