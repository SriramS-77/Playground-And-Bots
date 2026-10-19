"""Unsupervised shift detectors on the traffic's score distribution: calibration (exploratory).

Traffic per episode as in training: humans U(50, 250), bots from {0, 20, 100, 200, 500, 1000},
first-chunk score of every user (what the scorer reports when a user arrives). Covert bots
(calibration shift: Phase-2 advanced web bots; replay shift: human recordings as bots, both from
the training-side pools) replace a fraction rho of bots from episode SHIFT_EP on.

Detectors (one statistic per episode, standardised by the mean / SD of the burn-in episodes):
  cusum_mean   lower one-sided CUSUM on the mean score (covert bots lower it)
  cusum_mid    upper one-sided CUSUM on the share of users scoring in MID = [0.25, 0.6]
  cusum_ks     upper one-sided CUSUM on the KS distance between the episode's scores and the
               pooled burn-in scores
CUSUM: S_t = max(0, S_{t-1} + z_t - K), alarm at S_t > h. h is set per detector so that at most
FA_TARGET of no-drift runs raise any alarm in the post-burn-in episodes.

    python round4/detector_calib.py    -> results/round4/fold2/drift/detector_calib.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402

C, R, D = G.C, G.R, G.D
CALIB_SHIFT = D.shifted
import auc_drift                                  # noqa: E402,F401
REPLAY_SHIFT = D.shifted
D.shifted = CALIB_SHIFT

BURN, SHIFT_EP, EPISODES, RUNS = 50, 100, 300, 40
MID = (0.25, 0.6)
K = 0.5
FA_TARGET = 0.05
BOT_VOLUMES = (0, 20, 100, 200, 500, 1000)
DETECTORS = ("cusum_mean", "cusum_mid", "cusum_ks")


def first_scores(cache, sessions):
    out = []
    for s in sessions:
        v = cache._table.get((s.key, 0))
        out.append(np.nan if v is None else v)
    return np.array(out)


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    xs = np.r_[a, b]
    return float(np.max(np.abs(np.searchsorted(a, xs, side="right") / len(a)
                               - np.searchsorted(b, xs, side="right") / len(b))))


def episode_stats(pool_h, pool_b, rng):
    nh, nb = rng.integers(50, 251), rng.choice(BOT_VOLUMES)
    s = np.r_[rng.choice(pool_h, nh), rng.choice(pool_b, nb) if nb else []]
    s = s[~np.isnan(s)]
    return s


def statistics(eps):
    """eps: list of score arrays -> dict of per-episode z-scored statistics (burn-in baseline)."""
    mean = np.array([e.mean() for e in eps])
    mid = np.array([np.mean((e >= MID[0]) & (e <= MID[1])) for e in eps])
    ref = np.concatenate(eps[:BURN])
    ksd = np.array([ks(e, ref) for e in eps])
    out = {}
    for k, x, sign in (("cusum_mean", mean, -1), ("cusum_mid", mid, 1), ("cusum_ks", ksd, 1)):
        mu, sd = x[:BURN].mean(), x[:BURN].std() + 1e-9
        out[k] = sign * (x - mu) / sd
    return out


def cusum_path(z):
    s, path = 0.0, np.zeros(len(z))
    for t in range(BURN, len(z)):
        s = max(0.0, s + z[t] - K)
        path[t] = s
    return path


def run(pool_h, pool_b, after_b, rho, seed):
    rng = np.random.default_rng(seed)
    eps = []
    for t in range(EPISODES):
        b = after_b if (rho > 0 and t >= SHIFT_EP) else pool_b
        eps.append(episode_stats(pool_h, b, rng))
    return {k: cusum_path(z) for k, z in statistics(eps).items()}


def main():
    p = D.pools(False, "R6 detector calibration (scores only)")
    h = first_scores(p.cache, p.train_h)
    b = first_scores(p.cache, p.train_b)
    rows = []
    null = [run(h, b, None, 0.0, 1000 + i) for i in range(RUNS)]
    thresholds = {}
    for k in DETECTORS:
        peaks = np.array([r[k][BURN:].max() for r in null])
        thresholds[k] = float(np.quantile(peaks, 1 - FA_TARGET))
    for shift, fn in (("calibration", CALIB_SHIFT), ("replay", REPLAY_SHIFT)):
        D.shifted = fn
        for rho in (0.25, 0.5, 1.0):
            _, ab, _ = D.shifted(p, "bots", rho, held_out=False)
            after = first_scores(p.cache, ab)
            for i in range(RUNS):
                r = run(h, b, after, rho, 2000 + i)
                for k in DETECTORS:
                    path = r[k]
                    early = path[BURN:SHIFT_EP].max() > thresholds[k]
                    hit = np.nonzero(path[SHIFT_EP:] > thresholds[k])[0]
                    rows.append({"shift": shift, "rho": rho, "detector": k, "run": i,
                                 "false_alarm_before_shift": early,
                                 "detected": len(hit) > 0,
                                 "delay": float(hit[0]) if len(hit) else np.nan})
        D.shifted = CALIB_SHIFT
    df = pd.DataFrame(rows)
    df.to_csv(R.OUT / "detector_calib.csv", index=False)
    t = df.groupby(["shift", "rho", "detector"]).agg(detect_rate=("detected", "mean"),
                                                     median_delay=("delay", "median"),
                                                     early_alarm=("false_alarm_before_shift", "mean"))
    print("thresholds:", {k: round(v, 2) for k, v in thresholds.items()})
    print(t.round(2).to_string())
    pd.Series(thresholds).to_json(R.OUT / "detector_thresholds.json")


if __name__ == "__main__":
    main()
