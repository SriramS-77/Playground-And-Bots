"""The four composite metrics of Section III-H.

These previously existed only as spreadsheet formulas, which is why Table 4 could not be
regenerated from code. The mapping from the old column headings is:

    Percentage Difference btw final humans and bots -> DI
    F1 Score (Basic)                                -> BOS
    F1 Score (Advanced)                             -> SP-F1
    F1 Score (Super Advanced)                       -> SI-F1
"""

from __future__ import annotations

import statistics as st
from dataclasses import asdict, dataclass

from .simulate import SimResult


def _f1(a: float, b: float) -> float:
    return 0.0 if (a + b) == 0 else 2 * a * b / (a + b)


@dataclass
class Metrics:
    n_bots: int
    surviving_humans: int
    surviving_bots: int
    DI: float       # Discrimination Index, percentage points
    BOS: float      # Balanced Orchestration Score
    SP_F1: float    # System Purity F1
    SI_F1: float    # Scale-Invariant F1


def evaluate(result: SimResult) -> Metrics:
    s_h = result.surviving_humans / result.n_humans
    s_b = result.surviving_bots / result.n_bots if result.n_bots else 0.0
    k_b = 1.0 - s_b

    denom = result.surviving_humans + result.surviving_bots
    precision = result.surviving_humans / denom if denom else 1.0
    p_norm = s_h / (s_h + s_b) if (s_h + s_b) else 1.0

    return Metrics(
        n_bots=result.n_bots,
        surviving_humans=result.surviving_humans,
        surviving_bots=result.surviving_bots,
        DI=(s_h - s_b) * 100.0,
        BOS=_f1(s_h, k_b),
        SP_F1=_f1(precision, s_h),
        SI_F1=_f1(p_norm, s_h),
    )


def summarise(sweep: dict[int, SimResult]) -> dict:
    """Per-simulation metrics plus the mean +/- std ACROSS simulation types.

    Note the spread in Table 4 is the standard deviation over the six bot volumes, not
    over repeated seeds. Reproduced here for consistency with the manuscript.
    """
    rows = [evaluate(r) for r in sweep.values()]
    out = {"per_simulation": [asdict(r) for r in rows], "average": {}}
    for key in ("DI", "BOS", "SP_F1", "SI_F1"):
        vals = [getattr(r, key) for r in rows]
        out["average"][key] = {"mean": st.mean(vals), "std": st.pstdev(vals)}
    return out


def table(sweeps: dict[str, dict[int, SimResult]]):
    """Build the Table 4 comparison as a pandas DataFrame."""
    import pandas as pd

    records = []
    for policy_name, sweep in sweeps.items():
        for n_bots, result in sweep.items():
            m = evaluate(result)
            records.append(
                {
                    "Policy": policy_name,
                    "Bots": n_bots,
                    "Remaining Humans": m.surviving_humans,
                    "Remaining Bots": m.surviving_bots,
                    "DI": round(m.DI, 1),
                    "BOS": round(m.BOS, 3),
                    "SP-F1": round(m.SP_F1, 3),
                    "SI-F1": round(m.SI_F1, 3),
                }
            )
    return pd.DataFrame.from_records(records)
