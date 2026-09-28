"""Calibration of the behavioural score (Reviewer 1, point 7).

The manuscript uses the Humanity Score as a probability -- the static multi-threshold
baseline literally maps it onto a threat level with ``int(score * 10)``, and the DQN
consumes it as a feature alongside its running mean. Accuracy and ROC-AUC measure
*discrimination*: whether the score orders humans below bots. They say nothing about
whether 0.7 means "70% likely a bot". That is calibration, and it is what a threshold
map depends on.

Reported here: expected calibration error (equal-width and equal-mass binning), maximum
calibration error, Brier score with its Murphy decomposition, and a reliability curve.
Temperature scaling is fitted on a held-out split so we can report both the raw and the
corrected scorer.
"""

from __future__ import annotations

import numpy as np


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    base = float(y.mean())
    reliability = resolution = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        w = m.mean()
        reliability += w * (p[m].mean() - y[m].mean()) ** 2
        resolution += w * (y[m].mean() - base) ** 2
    return {
        "reliability": float(reliability),   # lower is better
        "resolution": float(resolution),     # higher is better
        "uncertainty": float(base * (1 - base)),
        "brier": brier(p, y),
    }


def expected_calibration_error(p, y, n_bins: int = 10, strategy: str = "uniform") -> dict:
    """ECE and MCE.

    `strategy="quantile"` uses equal-mass bins, which is the honest choice when the
    scores pile up near 0 and 1 -- as a confident bot detector's do.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        conf, acc, w = float(p[m].mean()), float(y[m].mean()), float(m.mean())
        gap = abs(conf - acc)
        ece += w * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "confidence": conf, "observed": acc})
    return {"ECE": float(ece), "MCE": float(mce), "n_bins_used": len(bins), "bins": bins}


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling on the logit, by grid + local refine.

    Returns T; apply with `apply_temperature`. T > 1 softens over-confident scores.
    """
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

    def nll(T):
        q = 1.0 / (1.0 + np.exp(-logit / T))
        q = np.clip(q, eps, 1 - eps)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 240))
    best = min(grid, key=nll)
    fine = np.linspace(best * 0.7, best * 1.4, 200)
    return float(min(fine, key=nll))


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    logit = np.log(pc / (1 - pc))
    return 1.0 / (1.0 + np.exp(-logit / T))


def roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    p, y = np.asarray(p, float), np.asarray(y, int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def report(p, y, name: str = "", n_bins: int = 10) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = {
        "name": name,
        "n": int(len(y)),
        "accuracy": float(((p >= 0.5) == (y == 1)).mean()),
        "AUC": roc_auc(p, y),
        "Brier": brier(p, y),
    }
    out.update({k: v for k, v in brier_decomposition(p, y, n_bins).items() if k != "brier"})
    out["ECE_uniform"] = expected_calibration_error(p, y, n_bins, "uniform")["ECE"]
    q = expected_calibration_error(p, y, n_bins, "quantile")
    out["ECE_quantile"] = q["ECE"]
    out["MCE_quantile"] = q["MCE"]
    return out


def reliability_curve(p, y, n_bins: int = 10, strategy: str = "quantile"):
    """(confidence, observed, weight) per bin, for plotting."""
    b = expected_calibration_error(p, y, n_bins, strategy)["bins"]
    conf = np.array([d["confidence"] for d in b])
    obs = np.array([d["observed"] for d in b])
    w = np.array([d["n"] for d in b], dtype=float)
    return conf, obs, w / w.sum()
