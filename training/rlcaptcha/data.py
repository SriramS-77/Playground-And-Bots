"""Session data loading and the simulated user population.

A "session" is one recorded 120s interaction with the playground blog site, either from
a human participant or from one of the four bot tiers. Sessions are split into 10s
chunks; each chunk is one decision point for the policy.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, MAX_CHUNKS, SESSION_CHUNK_MS

Movement = dict


@dataclass
class Session:
    """One recorded interaction, pre-split into fixed-length chunks."""

    key: str                      # stable id -> the humanity-score cache key
    is_bot: bool
    chunks: list[list[Movement]]


def _split_into_chunks(raw: dict, chunk_ms: int, target: int) -> list[list[Movement]]:
    """Split a session's movements into `chunk_ms` windows.

    Ported verbatim from ``get_session_chunks`` in the original benchmark notebooks,
    including the recursive top-up that repeats earlier chunks when a session is shorter
    than `target` windows. That top-up is what makes every user last up to 12 steps.
    """
    start = raw["timestamps"]["start"]
    movements = raw.get("mouse_movements", [])
    if not movements:
        return [[]]

    chunks: list[list[Movement]] = []
    current: list[Movement] = []
    end = start + chunk_ms

    for move in movements:
        if move["timestamp"] <= end:
            current.append(move)
            continue
        chunks.append(current)
        if len(chunks) == target:
            return chunks
        while move["timestamp"] > end + chunk_ms:
            chunks.append([])
            if len(chunks) == target:
                return chunks
            end += chunk_ms
        current = [move]
        end += chunk_ms

    chunks.append(current)
    if len(chunks) < target:
        chunks += _split_into_chunks(raw, chunk_ms, target - len(chunks))
    return chunks


def load_sessions(data_dir: Path | str = DATA_DIR) -> tuple[list[Session], list[Session]]:
    """Load every session under `data_dir`, returning (humans, bots).

    Files are labelled by filename prefix: ``human*`` or ``bot*``.
    """
    data_dir = Path(data_dir)
    humans: list[Session] = []
    bots: list[Session] = []

    for path in sorted(data_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        if not raw.get("mouse_movements"):
            continue
        is_bot = path.name.startswith("bot")
        session = Session(
            key=path.name,
            is_bot=is_bot,
            chunks=_split_into_chunks(raw, SESSION_CHUNK_MS, MAX_CHUNKS),
        )
        (bots if is_bot else humans).append(session)

    if not humans or not bots:
        raise RuntimeError(f"Need both human and bot sessions in {data_dir}")
    return humans, bots


@dataclass
class User:
    """One simulated visitor replaying a session.

    ``bot_score`` is the raw LSTM output. Note the sign: the classifier is trained with
    human=0 / bot=1, so a HIGH value means "looks like a bot". The manuscript's H (the
    Humanity Score) is ``1 - bot_score``. The original code called this field
    `humanity_score` while using it with bot semantics; renamed here for clarity, the
    arithmetic is unchanged.
    """

    session: Session
    is_bot: bool
    bot_strength: int | None          # B_s in {0..9}; bots survive unless threat > B_s
    jitter: tuple[int, int] = (0, 0)  # per-user (dx, dy) in {-1,0,1}, see scoring.py
    bot_score: float = 0.5
    avg_bot_score: float = 0.5
    captchas_solved: int = 0
    last_threat_level: int = 0
    exhausted: bool = False           # ran out of session data, or was terminated
    _n_scores: int = 0
    _chunk_index: int = 0

    def next_chunk(self) -> list[Movement] | None:
        """Advance one timestep. Returns None (and marks exhausted) at end of session."""
        if self._chunk_index >= len(self.session.chunks):
            self.exhausted = True
            return None
        chunk = self.session.chunks[self._chunk_index]
        self._chunk_index += 1
        return chunk

    @property
    def cache_key(self) -> tuple[str, int]:
        """Identifies the chunk just consumed by `next_chunk`."""
        return (self.session.key, self._chunk_index - 1)

    def observe(self, score: float | None) -> None:
        """Record this timestep's score.

        A chunk can be empty (the user did not move the mouse in that 10s window), in
        which case the scorer returns None and the policy falls back to the running
        average rather than a stale reading -- matching the original
        ``if h_score is None: h_score = user.average_humanity_score``.
        """
        if score is None:
            self.bot_score = self.avg_bot_score
            return
        self.bot_score = score
        self.avg_bot_score = (
            self.avg_bot_score * self._n_scores + score
        ) / (self._n_scores + 1)
        self._n_scores += 1

    def end_session(self) -> None:
        self.captchas_solved = 0
        self.exhausted = True


def build_population(
    humans: list[Session],
    bots: list[Session],
    n_humans: int,
    n_bots: int,
    rng: random.Random,
) -> list[User]:
    """Sample a population with replacement and shuffle it.

    Each user gets a private (dx, dy) jitter in {-1, 0, 1}, reproducing the original
    ``perturb_mouse_data`` which offset every coordinate of a session by one random
    integer pair. It is what makes two users drawn from the same recording distinct.
    """

    def jitter() -> tuple[int, int]:
        return (rng.randint(-1, 1), rng.randint(-1, 1))

    users = [
        User(
            session=rng.choice(humans),
            is_bot=False,
            bot_strength=None,
            jitter=jitter(),
        )
        for _ in range(n_humans)
    ]
    users += [
        User(
            session=rng.choice(bots),
            is_bot=True,
            bot_strength=rng.randint(0, 9),
            jitter=jitter(),
        )
        for _ in range(n_bots)
    ]
    rng.shuffle(users)
    return users
