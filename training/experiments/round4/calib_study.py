"""Humanity Scorer calibration: reliability diagram and post-hoc calibrators (round 6).

Scorer: the fold-2 refit every RL result uses. Calibrators are fitted on the refit's own
validation recordings (the split its early stopping and temperature used), at CHUNK level --
the unit the controller acts on -- and evaluated on the eval block's 262 chunks.

  none          mean of the windows' raw probabilities
  window T      current method: temperature fitted on WINDOWS, chunk = mean of calibrated windows
  chunk T       temperature fitted on chunk logits, logit(mean raw prob)
  Platt         temperature + bias, sigma(a z + b) on the chunk logit (Platt 1999)
  beta          sigma(a ln p - b ln(1-p) + c) (Kull et al. 2017)
  isotonic      monotone step function (Zadrozny & Elkan 2002)
  hist. binning 10 equal-mass bins, each mapped to its validation bot share

Metrics on the eval block: ECE (10 uniform bins and 10 equal-mass bins), Brier, NLL, AUC, with
bootstrap 95% intervals over chunks (resampling sessions would leave too few humans).

    python round4/calib_study.py -> results/round4/fold2/drift/calib_study.{csv,png}, calib_chunks.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grounded_headline as G                     # noqa: E402

import numpy as np                                # noqa: E402
import pandas as pd                               # noqa: E402
from sklearn.isotonic import IsotonicRegression   # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from expkit.calibration import expected_calibration_error, fit_temperature, roc_auc  # noqa: E402
from expkit.humanity_scorer import HumanityScorer, chunk_key  # noqa: E402
from expkit.simx import sessions_from_refs        # noqa: E402

R = G.R
OUT = R.OUT
EPS = 1e-6


def chunk_rows(sc, sessions):
    """-> rows (session index, mean raw window prob, mean window-T prob, y) per distinct chunk."""
    T = sc.temperature
    rows = []
    for i, s in enumerate(sessions):
        seen, chunks = set(), []
        for c in s.chunks:
            if not c:
                continue
            k = chunk_key(s.key, c)
            if k not in seen:
                seen.add(k)
                chunks.append(c)
        sc.temperature = 1.0
        raw = sc.score_chunks(chunks)
        sc.temperature = T
        cal = sc.score_chunks(chunks)
        rows += [(i, r, c, int(s.is_bot)) for r, c in zip(raw, cal) if r is not None]
    return np.array(rows, dtype=float)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_calibrators(p, y):
    z = logit(p)
    T = fit_temperature(p, y)
    platt = LogisticRegression(C=1e6).fit(z[:, None], y)
    beta_x = lambda q: np.c_[np.log(np.clip(q, EPS, 1)), -np.log(np.clip(1 - q, EPS, 1))]
    beta = LogisticRegression(C=1e6).fit(beta_x(p), y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = 0, 1
    idx = lambda q: np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)
    share = np.array([y[idx(p) == k].mean() if (idx(p) == k).any() else 0.5
                      for k in range(len(edges) - 1)])
    return {
        "chunk T": lambda q: sig(logit(q) / T),
        "Platt (T + bias)": lambda q: platt.predict_proba(logit(q)[:, None])[:, 1],
        "beta": lambda q: beta.predict_proba(beta_x(q))[:, 1],
        "isotonic": lambda q: iso.predict(q),
        "hist. binning": lambda q: share[idx(q)],
    }, {"chunk_T": T, "platt_a": float(platt.coef_[0, 0]), "platt_b": float(platt.intercept_[0])}


def metrics(p, y):
    q = np.clip(p, EPS, 1 - EPS)
    return {"ECE_uniform": expected_calibration_error(p, y)["ECE"],
            "ECE_mass": expected_calibration_error(p, y, strategy="quantile")["ECE"],
            "Brier": float(np.mean((p - y) ** 2)),
            "NLL": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "AUC": roc_auc(p, y)}


def main():
    ctx = R.context(with_eval=True, reason="R6 calibration study on the eval block")
    sc = HumanityScorer.load(OUT / "scorer" / "refit")
    val_refs = sc.provenance["val"] if hasattr(sc, "provenance") else None
    if val_refs is None:
        import json
        val_refs = json.loads((OUT / "scorer" / "refit" / "scorer.json").read_text())["provenance"]["val"]
    by_name = {Path(r.path).name: r for r in ctx.fold.scorer + ctx.fold.rl}
    vh, vb = sessions_from_refs([by_name[n] for n in val_refs])
    v = chunk_rows(sc, vh + vb)
    e = chunk_rows(sc, ctx.ev_h + ctx.ev_b)
    print(f"validation: {len(v)} chunks ({int(v[:, 3].sum())} bot); eval: {len(e)} chunks")
    cals, params = fit_calibrators(v[:, 1], v[:, 3])
    preds = {"none (raw)": e[:, 1], "window T (current)": e[:, 2]}
    preds.update({k: f(e[:, 1]) for k, f in cals.items()})
    y = e[:, 3]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    rows = []
    for k, p in preds.items():
        m = metrics(p, y)
        for key in ("ECE_uniform", "Brier"):
            b = [metrics(p[i], y[i])[key] for i in boots[:300]]
            m[key + "_lo"], m[key + "_hi"] = np.percentile(b, [2.5, 97.5])
        rows.append({"method": k, **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calib_study.csv", index=False)
    pd.DataFrame({"session": e[:, 0], "y": y, **preds}).to_csv(OUT / "calib_chunks.csv", index=False)
    print(params)
    print(df.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["none (raw)", "window T (current)", "Platt (T + bias)", "isotonic"]
    col = {"none (raw)": "#888888", "window T (current)": "#2a78d6",
           "Platt (T + bias)": "#eb6834", "isotonic": "#1baf7a"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 4.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, ls="--")
    for k in show:
        bins = expected_calibration_error(preds[k], y)["bins"]
        a1.plot([b["confidence"] for b in bins], [b["observed"] for b in bins], marker="o", ms=3,
                lw=1.1, color=col[k],
                label=f"{k} (ECE {df.set_index('method').loc[k, 'ECE_uniform']:.3f})")
    a1.set_ylabel("Observed bot share")
    a1.legend(fontsize=6, frameon=False, loc="upper left")
    a2.hist(preds["none (raw)"], bins=np.linspace(0, 1, 11), color="#888888", alpha=0.6,
            label="raw")
    a2.set_yscale("log")
    a2.set_xlabel("Predicted P(bot), chunk")
    a2.set_ylabel("Chunks")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "calib_study.png", dpi=160)
    fig.savefig(OUT / "calib_study.pdf")


if __name__ == "__main__":
    main()
