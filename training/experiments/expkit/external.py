"""Unseen data for round 3's adversary tests (E9): Balabit humans, web-bot bots.

* **Balabit Mouse Dynamics Challenge** (Fulop et al. 2016) -- remote-desktop sessions of 10
  users. HUMANS only: its "illegal" sessions are other users' data mixed in, not attacks.
  Used as UNSEEN HUMANS.
* **web_bot_detection_dataset, phase 2** (Iliou et al., Digital Threats 2021) -- browser
  sessions of humans, "moderate" bots and "advanced" bots on one site and logger. The
  ADVANCED bots are the UNSEEN ADVERSARY BOTS; the humans are the SAME-SOURCE CONTROL (same
  logger as the bots, so a scorer that separates by recording source shows up); the moderate
  bots are kept only for the scorer report -- the scorer is inverted on them (same-source
  AUC 0.05 locally), which is why E9 does not inject them.

Neither dataset is in git. `build_from_raw` converts the raw files under the FIXED rules
below; `write_cache` stores the cleaned movements (never scores, never chunks) as one gzip
JSON whose SHA-256 is pinned in `CACHE_SHA256`; `load_sessions` re-chunks with rlcaptcha's
own `_split_into_chunks`, so a cluster run needs only the cache file.

RULES -- fixed before any scoring, never tuned to the scores:
  Balabit     test_files; rows with state Move or Drag; t = client_ts * 1000 (ms)
  web-bot     drop time entries < 1e12 ms (truncated values); keep a record only if the
              remaining times align one-to-one with its coordinates
  all         drop coordinates outside [0, 10000]; minimum gap 10 ms between kept events (also
              drops non-increasing times); keep the first 120 s from the first event; require
              >= 30 s and >= 20 events
  selection   Balabit: <= 30 sessions per user, drawn with random.Random(0) from the sorted
              files that pass the rules -- never by score
  splits      Balabit users sorted, first 5 -> INJECT, last 5 -> HELD-OUT; web-bot, per type:
              sorted session ids shuffled with random.Random(0), first half INJECT, rest HELD-OUT
  keys        "ext/balabit/<user>/<file>", "ext/webbot_<type>/<session_id>"
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
from pathlib import Path

from .paths import EXP_ROOT, PROJECT_ROOT

RAW_ROOT = Path(os.environ.get("EXTERNAL_RAW", PROJECT_ROOT / "offline" / "scoring_service"))
CACHE = Path(os.environ.get("EXTERNAL_CACHE", EXP_ROOT / "data" / "external_sessions.json.gz"))
#: SHA-256 of the committed cache. `read_cache` refuses any other file.
CACHE_SHA256 = "cb287ad6510d70e9a400341271559cf8d2fd528705ee7d884dd7783c59d28248"

MIN_GAP_MS, WINDOW_MS, MIN_SPAN_MS, MIN_EVENTS, PER_USER = 10, 120_000, 30_000, 20, 30
SETS = ("balabit_inject", "balabit_heldout",
        "webbot_humans_inject", "webbot_humans_heldout",
        "webbot_advanced_inject", "webbot_advanced_heldout",
        "webbot_moderate_inject", "webbot_moderate_heldout")


def clean(t, x, y):
    """(times ms, xs, ys) -> list of {"x", "y", "timestamp"} under the fixed rules, or None."""
    keep, last = [], None
    for ti, xi, yi in zip(t, x, y):
        if not (0 <= xi <= 10000 and 0 <= yi <= 10000):
            continue
        if last is not None and ti - last < MIN_GAP_MS:
            continue
        keep.append({"x": int(xi), "y": int(yi), "timestamp": int(ti)})
        last = ti
    if not keep:
        return None
    t0 = keep[0]["timestamp"]
    keep = [m for m in keep if m["timestamp"] - t0 <= WINDOW_MS]
    if len(keep) < MIN_EVENTS or keep[-1]["timestamp"] - t0 < MIN_SPAN_MS:
        return None
    return keep


def _balabit(raw_root: Path) -> dict:
    import glob

    import numpy as np
    import pandas as pd

    out = {}
    base = raw_root / "balabit_dataset" / "test_files"
    for user in sorted(os.listdir(base)):
        ok = []
        for fp in sorted(glob.glob(str(base / user / "*"))):
            df = pd.read_csv(fp)
            df.columns = ["rec", "cli", "button", "state", "x", "y"]
            df = df[df.state.isin(["Move", "Drag"])]
            mv = clean((df.cli.to_numpy() * 1000).round().astype(np.int64), df.x.to_numpy(), df.y.to_numpy())
            if mv:
                ok.append((Path(fp).name, mv))
        pick = random.Random(0).sample(ok, min(PER_USER, len(ok)))
        out[user] = [{"key": f"ext/balabit/{user}/{n}", "is_bot": False, "moves": mv} for n, mv in sorted(pick)]
    return out


def _webbot(path: Path, kind: str, is_bot: bool) -> list:
    import numpy as np

    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        t = np.array([int(v) for v in d["mousemove_times"].split(",") if v.strip()])
        xy = re.findall(r"m\((-?\d+),(-?\d+)\)", d["mousemove_total_behaviour"])
        t = t[t >= 1e12]
        if len(t) != len(xy):
            continue
        mv = clean(t, [int(a) for a, _ in xy], [int(b) for _, b in xy])
        if mv:
            rows.append({"key": f"ext/webbot_{kind}/{d['session_id']}", "is_bot": is_bot, "moves": mv})
    rows.sort(key=lambda r: r["key"])
    random.Random(0).shuffle(rows)
    return rows


def build_from_raw(raw_root: Path = RAW_ROOT) -> dict:
    """{set name: [{"key", "is_bot", "moves"}]} from the raw datasets, under the fixed rules."""
    bal = _balabit(raw_root)
    users = sorted(bal)
    sets = {"balabit_inject": [r for u in users[:5] for r in bal[u]],
            "balabit_heldout": [r for u in users[5:] for r in bal[u]]}
    web = raw_root / "web_bot_detection_dataset" / "phase2" / "data" / "mouse_movements"
    for kind, f, is_bot in (("humans", web / "humans" / "mouse_movements_humans.json", False),
                            ("advanced", web / "bots" / "mouse_movements_advanced_bots.json", True),
                            ("moderate", web / "bots" / "mouse_movements_moderate_bots.json", True)):
        rows = _webbot(f, kind, is_bot)
        sets[f"webbot_{kind}_inject"], sets[f"webbot_{kind}_heldout"] = rows[: len(rows) // 2], rows[len(rows) // 2:]
    keys = [r["key"] for v in sets.values() for r in v]
    assert len(keys) == len(set(keys)), "key collision"
    return sets


def write_cache(sets: dict, path: Path = CACHE) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({k: sets[k] for k in SETS}, separators=(",", ":"), sort_keys=True).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as f:          # mtime=0 -> byte-identical rebuilds
        f.write(blob)
    return sha256(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cache(path: Path = CACHE, check: bool = True) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- E9 needs the external-data cache. Copy it from the repository "
            f"(git add -f training/experiments/data/external_sessions.json.gz) or rebuild it from the raw "
            f"datasets: EXTERNAL_RAW=<dir with balabit_dataset/ and web_bot_detection_dataset/> "
            f"python -m expkit.external --build")
    if check and CACHE_SHA256 and sha256(path) != CACHE_SHA256:
        raise RuntimeError(f"{path}: SHA-256 {sha256(path)} != pinned {CACHE_SHA256} -- not the data "
                           "the handoff describes")
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def load_sessions(path: Path = CACHE) -> dict:
    """{set name: [rlcaptcha Session]}, chunked exactly as our own recordings are."""
    from rlcaptcha.config import MAX_CHUNKS, SESSION_CHUNK_MS
    from rlcaptcha.data import Session, _split_into_chunks

    out = {}
    for name, rows in read_cache(path).items():
        out[name] = []
        for r in rows:
            mv = r["moves"]
            raw = {"timestamps": {"start": mv[0]["timestamp"], "end": mv[-1]["timestamp"]},
                   "mouse_movements": mv}
            out[name].append(Session(key=r["key"], is_bot=r["is_bot"],
                                     chunks=_split_into_chunks(raw, SESSION_CHUNK_MS, MAX_CHUNKS)))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="convert the raw datasets and write the cache")
    ap.add_argument("--check", action="store_true", help="verify the cache against the pinned hash")
    a = ap.parse_args()
    if a.build:
        s = build_from_raw()
        h = write_cache(s)
        print({k: len(v) for k, v in s.items()})
        print(f"wrote {CACHE}  sha256={h}")
    if a.check:
        print(f"{CACHE}: sha256={sha256(CACHE)}  pinned={CACHE_SHA256}  "
              f"{'OK' if sha256(CACHE) == CACHE_SHA256 else 'MISMATCH'}")
