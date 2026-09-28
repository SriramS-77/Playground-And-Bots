"""Session-level three-way partition.

Reviewer 1, point 2 asks whether training and evaluation data are independent. In the
published work they are not fully: `offline_training.z.py` sets ``DATA_DIR = "data"``,
so the RL agent was trained on, and evaluated on, the same 73 recordings -- which are
also the LSTM's test set (and, because ``EarlyStopping(restore_best_weights=True)`` was
given ``validation_data=(x_test, y_test)``, the set that selected its checkpoint).

This module pools both recording campaigns and cuts them into three disjoint pools:

    lstm    -- trains the behavioural scorer     (~45%)
    rl      -- populates the RL replay buffer    (~20%)
    eval    -- Table 2 and Table 4, never seen by either learner (~35%)

The split is at SESSION level, stratified by (label, bot family, campaign), so no
window and no simulated user can straddle two pools. A held-out slice of the `lstm`
pool is used for early stopping, so the eval pool never selects a checkpoint either.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import CAMPAIGN_A, CAMPAIGN_B, PARTITION_JSON

SPLITS = ("lstm", "rl", "eval")
DEFAULT_FRACTIONS = {"lstm": 0.45, "rl": 0.20, "eval": 0.35}


@dataclass(frozen=True)
class SessionRef:
    """One recording, with everything the stratifier needs."""

    path: str
    name: str
    campaign: str      # "A" (Aug 2025) or "B" (Nov 2025)
    is_bot: bool
    family: str        # "human", "NaiveBot", "HumanishBot", "MimicBot", "FallibleBot"

    @property
    def stratum(self) -> str:
        return f"{self.campaign}:{self.family}"


def _family(name: str) -> str:
    if name.startswith("human"):
        return "human"
    # bot_<Family>_data_...
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else "UnknownBot"


def index_sessions() -> list[SessionRef]:
    """Every recording across both campaigns, sorted for determinism."""
    refs: list[SessionRef] = []
    for campaign, directory in (("A", CAMPAIGN_A), ("B", CAMPAIGN_B)):
        if not directory.exists():
            raise FileNotFoundError(f"Recording campaign {campaign} missing: {directory}")
        for path in sorted(directory.glob("*.json")):
            refs.append(
                SessionRef(
                    path=str(path),
                    name=path.name,
                    campaign=campaign,
                    is_bot=path.name.startswith("bot"),
                    family=_family(path.name),
                )
            )
    return refs


def make_partition(
    refs: list[SessionRef] | None = None,
    fractions: dict[str, float] = None,
    seed: int = 20260926,
) -> dict[str, list[SessionRef]]:
    """Stratified session-level split.

    Allocation inside each stratum is largest-remainder, so small strata (e.g. the four
    NaiveBot recordings in campaign B) still contribute to every pool rather than
    landing entirely in one.
    """
    refs = refs if refs is not None else index_sessions()
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    rng = random.Random(seed)
    strata: dict[str, list[SessionRef]] = defaultdict(list)
    for ref in refs:
        strata[ref.stratum].append(ref)

    out: dict[str, list[SessionRef]] = {s: [] for s in SPLITS}
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda r: r.name)
        rng.shuffle(members)
        n = len(members)

        exact = {s: n * fractions[s] for s in SPLITS}
        counts = {s: int(exact[s]) for s in SPLITS}
        # Largest remainder for the leftover seats.
        leftover = n - sum(counts.values())
        order = sorted(SPLITS, key=lambda s: (-(exact[s] - counts[s]), s))
        for i in range(leftover):
            counts[order[i % len(SPLITS)]] += 1

        cursor = 0
        for split in SPLITS:
            take = counts[split]
            out[split].extend(members[cursor:cursor + take])
            cursor += take

    for split in SPLITS:
        out[split].sort(key=lambda r: r.name)
    return out


def save_partition(partition: dict[str, list[SessionRef]], path: Path = PARTITION_JSON) -> Path:
    payload = {
        "seed_note": "produced by expkit.partition.make_partition",
        "splits": {s: [asdict(r) for r in partition[s]] for s in SPLITS},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_partition(path: Path = PARTITION_JSON) -> dict[str, list[SessionRef]]:
    payload = json.loads(Path(path).read_text())
    return {s: [SessionRef(**d) for d in payload["splits"][s]] for s in SPLITS}


def summarise(partition: dict[str, list[SessionRef]]):
    """A DataFrame of counts per split x family, plus a disjointness assertion."""
    import pandas as pd

    seen: dict[str, str] = {}
    for split, refs in partition.items():
        for ref in refs:
            if ref.name in seen:
                raise AssertionError(
                    f"{ref.name} appears in both {seen[ref.name]} and {split}"
                )
            seen[ref.name] = split

    rows = []
    for split in SPLITS:
        refs = partition[split]
        row = {"split": split, "sessions": len(refs),
               "humans": sum(1 for r in refs if not r.is_bot),
               "bots": sum(1 for r in refs if r.is_bot)}
        for fam in ("human", "NaiveBot", "HumanishBot", "MimicBot", "FallibleBot"):
            row[fam] = sum(1 for r in refs if r.family == fam)
        for camp in ("A", "B"):
            row[f"campaign_{camp}"] = sum(1 for r in refs if r.campaign == camp)
        rows.append(row)
    return pd.DataFrame(rows)
