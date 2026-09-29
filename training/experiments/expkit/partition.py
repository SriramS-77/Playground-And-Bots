"""Session-level three-way partition.

Reviewer 1, point 2 asks whether training and evaluation data are independent. In the
published work they are not fully: `offline_training.z.py` sets ``DATA_DIR = "data"``,
so the RL agent was trained on, and evaluated on, the same 73 recordings -- which are
also the LSTM's test set (and, because ``EarlyStopping(restore_best_weights=True)`` was
given ``validation_data=(x_test, y_test)``, the set that selected its checkpoint).

This module pools both recording campaigns and cuts them into three disjoint pools:

    lstm    -- trains the behavioural scorer
    rl      -- populates the RL replay buffer
    eval    -- Table 2 and Table 4, never seen by either learner

The split is at SESSION level, so no window and no simulated user can straddle two
pools. A held-out slice of the `lstm` pool is used for early stopping, so the eval pool
never selects a checkpoint either.

Two stratification modes:

* ``"campaign_family"`` -- the round-1 default, kept so the published partition can be
  regenerated bit-for-bit.
* ``"family"`` -- the round-2 default for new work. The two campaigns are two sittings by
  the SAME three participants three months apart, so campaign is a nuisance variable, not
  a population boundary. Stratifying on it merely balances it; it is still reported.

`DEFAULT_FRACTIONS` (45/20/35) starved the RL agent: 31 recordings, of which 8 human.
`EQUAL_THIRDS` is the round-2 allocation. Better still, `make_rotation` rotates the three
roles so every recording is evaluated exactly once and results pool over all 156.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import CAMPAIGN_A, CAMPAIGN_B, PARTITION_JSON

SPLITS = ("lstm", "rl", "eval")
DEFAULT_FRACTIONS = {"lstm": 0.45, "rl": 0.20, "eval": 0.35}
EQUAL_THIRDS = {"lstm": 1 / 3, "rl": 1 / 3, "eval": 1 / 3}
STRATIFY_MODES = ("campaign_family", "family")


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

    def stratum_for(self, mode: str) -> str:
        if mode == "campaign_family":
            return self.stratum
        if mode == "family":
            return self.family
        raise ValueError(f"stratify_by must be one of {STRATIFY_MODES}, got {mode!r}")


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
    stratify_by: str = "campaign_family",
) -> dict[str, list[SessionRef]]:
    """Stratified session-level split.

    Allocation inside each stratum is largest-remainder, so small strata (e.g. the four
    NaiveBot recordings in campaign B) still contribute to every pool rather than
    landing entirely in one.

    Defaults reproduce the round-1 partition exactly. Round-2 work passes
    ``fractions=EQUAL_THIRDS, stratify_by="family"``.
    """
    refs = refs if refs is not None else index_sessions()
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if stratify_by not in STRATIFY_MODES:
        raise ValueError(f"stratify_by must be one of {STRATIFY_MODES}")

    rng = random.Random(seed)
    strata: dict[str, list[SessionRef]] = defaultdict(list)
    for ref in refs:
        strata[ref.stratum_for(stratify_by)].append(ref)

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


@dataclass(frozen=True)
class RotationFold:
    """One assignment of the three blocks to the three roles.

    Three disjoint blocks B0, B1, B2 are cut once. Fold i uses B_i to train the scorer,
    B_(i+1) to train the policy and B_(i+2) to evaluate. Every recording is therefore
    evaluated exactly once across the three folds, and results pool over all 156 rather
    than over one arbitrary 52-recording slice.

    `rl_fit` / `rl_val` subdivide the policy block. **Architecture and posterior
    selection happens on `rl_val` only.** Selecting on `eval` would be test-set
    selection, which is the reviewer's R1.2 objection all over again.
    """

    index: int
    scorer: list[SessionRef]
    rl: list[SessionRef]
    eval: list[SessionRef]
    rl_fit: list[SessionRef] = field(default_factory=list)
    rl_val: list[SessionRef] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[SessionRef]]:
        """Adapter for code that expects the old {lstm, rl, eval} shape."""
        return {"lstm": self.scorer, "rl": self.rl, "eval": self.eval}


def split_refs(
    refs: list[SessionRef],
    fraction: float = 1 / 3,
    seed: int = 0,
    stratify_by: str = "family",
) -> tuple[list[SessionRef], list[SessionRef]]:
    """Cut one pool in two, stratified. Returns (larger, smaller)."""
    rng = random.Random(seed)
    strata: dict[str, list[SessionRef]] = defaultdict(list)
    for ref in refs:
        strata[ref.stratum_for(stratify_by)].append(ref)

    held: list[SessionRef] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda r: r.name)
        rng.shuffle(members)
        take = max(1, round(len(members) * fraction)) if len(members) > 1 else 0
        held.extend(members[:take])

    held_names = {r.name for r in held}
    kept = sorted((r for r in refs if r.name not in held_names), key=lambda r: r.name)
    return kept, sorted(held, key=lambda r: r.name)


def make_rotation(
    refs: list[SessionRef] | None = None,
    seed: int = 20260929,
    stratify_by: str = "family",
    rl_val_fraction: float = 1 / 3,
) -> list[RotationFold]:
    """Three folds that rotate the scorer / policy / evaluation roles.

    Cost is 3x a single split, but the three folds are completely independent -- run them
    as three Slurm array tasks and the wall-clock is unchanged.
    """
    blocks = make_partition(refs, fractions=EQUAL_THIRDS, seed=seed,
                            stratify_by=stratify_by)
    b = [blocks["lstm"], blocks["rl"], blocks["eval"]]

    folds: list[RotationFold] = []
    for i in range(3):
        rl_block = b[(i + 1) % 3]
        fit, val = split_refs(rl_block, rl_val_fraction, seed=seed + i,
                              stratify_by=stratify_by)
        folds.append(RotationFold(index=i, scorer=b[i], rl=rl_block,
                                  eval=b[(i + 2) % 3], rl_fit=fit, rl_val=val))

    # Every recording evaluated exactly once, and no role collision inside a fold.
    seen: list[str] = []
    for f in folds:
        seen.extend(r.name for r in f.eval)
        names = {"scorer": {r.name for r in f.scorer}, "rl": {r.name for r in f.rl},
                 "eval": {r.name for r in f.eval}}
        for x, y in (("scorer", "rl"), ("scorer", "eval"), ("rl", "eval")):
            if names[x] & names[y]:
                raise AssertionError(f"fold {f.index}: {x} and {y} overlap")
        if {r.name for r in f.rl_fit} & {r.name for r in f.rl_val}:
            raise AssertionError(f"fold {f.index}: rl_fit and rl_val overlap")
    if len(seen) != len(set(seen)):
        raise AssertionError("a recording is evaluated more than once")
    return folds


def save_rotation(folds: list[RotationFold], path: Path) -> Path:
    payload = {"note": "produced by expkit.partition.make_rotation",
               "folds": [{"index": f.index,
                          **{role: [asdict(r) for r in getattr(f, role)]
                             for role in ("scorer", "rl", "eval", "rl_fit", "rl_val")}}
                         for f in folds]}
    Path(path).write_text(json.dumps(payload, indent=2))
    return Path(path)


def load_rotation(path: Path) -> list[RotationFold]:
    payload = json.loads(Path(path).read_text())
    return [RotationFold(index=d["index"],
                         **{role: [SessionRef(**x) for x in d[role]]
                            for role in ("scorer", "rl", "eval", "rl_fit", "rl_val")})
            for d in payload["folds"]]


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
