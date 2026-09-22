"""The LSTM behavioural scorer, plus the cache that makes evaluation cheap.

WHY A CACHE IS SOUND
--------------------
The simulation looks like it needs a fresh LSTM inference for every user at every
timestep: 100 humans + up to 1000 bots, 12 steps each, six bot volumes. That is ~80k
forward passes per policy, and it dominated the original runtime by orders of magnitude.

Almost all of that work is redundant, because of how the population is built:

1. There are only 73 recorded sessions (32 human, 41 bot). Every simulated user is a
   copy of one of them -- with 1100 users, each recording is reused ~15 times.
2. Chunk boundaries depend only on timestamps, which copying does not touch. So user A
   and user B drawn from the same recording see byte-identical chunk sequences.
3. The only thing distinguishing two such users is ``perturb_mouse_data``, which adds a
   single random integer offset in {-1, 0, +1} to x and y -- the SAME offset for every
   point in the session. It is a one-pixel rigid translation, not per-point noise.
4. ``HumanityScorer.handle_data`` builds a Normalization layer, calls ``adapt`` on it,
   and then returns the *unadapted* array. The network therefore sees raw pixel
   coordinates in the hundreds, and a +/-1px translation moves the input by well under
   0.5% of its scale.

So the score is a function of (recording, chunk index) up to a perturbation far smaller
than the model's own sensitivity, and can be memoised on that key. This turns ~80k
inferences into 73 sessions x 12 chunks = at most 876, computed once and reused for
every policy and every seed.

Use ``exact=True`` to disable the cache and reproduce the original code path exactly,
jitter included. In practice the two agree to ~1e-4 per score; ``verify_cache`` below
measures it on your machine.
"""

from __future__ import annotations

import numpy as np

from .config import HUMANITY_SCORER, SCORER_CONTEXT
from .data import Session


def _windows(movements, length: int) -> np.ndarray:
    """Split one chunk into fixed-length windows, padding the tail by repetition.

    Ported verbatim from ``HumanityScorer.handle_data``. Note the tail window is
    appended unconditionally: when len(movements) is an exact multiple of `length` the
    original still emits a final window of `length` copies of the last point, and that
    stationary window pulls the mean. Dropping it would change every score.
    """
    out = []
    n = len(movements)
    full = n // length
    for i in range(full):
        out.append(list(movements[i * length:(i + 1) * length]))
    tail = list(movements[full * length:])
    tail += [movements[-1]] * (length - len(tail))
    out.append(tail)
    return np.array(out, dtype=np.float64)


class BotScorer:
    """Wraps the trained Keras LSTM. Output is P(bot): HIGH means bot-like.

    The manuscript's Humanity Score H is ``1 - score``. The training labels were
    human=0 / bot=1 (see the original ``LSTM.ipynb`` loader), and the RL code has always
    consumed the raw P(bot) value despite naming it `humanity_score`.
    """

    def __init__(self, model_path=HUMANITY_SCORER, context: int = SCORER_CONTEXT):
        import tensorflow as tf  # imported lazily: the ablation policy never needs it

        self.model = tf.keras.models.load_model(str(model_path))
        self.context = context

    def score_chunk(self, movements, jitter: tuple[int, int] = (0, 0)) -> float | None:
        """Mean P(bot) over the windows of one 10s chunk. None if the chunk is empty."""
        if not movements:
            return None
        pts = [[m["x"] + jitter[0], m["y"] + jitter[1]] for m in movements]
        batch = _windows(pts, self.context)
        preds = self.model.predict(batch, verbose=0)
        return float(np.mean(preds))


class ScoreCache:
    """Memoises BotScorer over (session key, chunk index).

    Build it once with `precompute`, then every policy and every seed reads from it.
    """

    def __init__(self, scorer: BotScorer | None = None, exact: bool = False):
        self.scorer = scorer
        self.exact = exact
        self._table: dict[tuple[str, int], float | None] = {}

    def precompute(self, *session_groups: list[Session], verbose: bool = True) -> "ScoreCache":
        if self.exact:
            return self
        if self.scorer is None:
            raise ValueError("precompute() needs a BotScorer")
        for sessions in session_groups:
            for session in sessions:
                for i, chunk in enumerate(session.chunks):
                    self._table[(session.key, i)] = self.scorer.score_chunk(chunk)
        if verbose:
            print(f"Humanity score cache: {len(self._table)} (session, chunk) entries")
        return self

    def __len__(self) -> int:
        return len(self._table)

    def score(self, user) -> float | None:
        """Score the chunk the user just consumed."""
        if self.exact:
            idx = user.cache_key[1]
            if idx < 0 or idx >= len(user.session.chunks):
                return None
            return self.scorer.score_chunk(user.session.chunks[idx], user.jitter)
        return self._table.get(user.cache_key)


def verify_cache(scorer: BotScorer, sessions: list[Session], n: int = 20) -> dict:
    """Quantify the cache approximation: cached score vs. jittered score.

    Returns max/mean absolute difference and how often the 0.5 decision boundary flips.
    """
    import random

    rng = random.Random(0)
    diffs, flips, checked = [], 0, 0
    for session in sessions:
        for i, chunk in enumerate(session.chunks):
            if not chunk or checked >= n:
                continue
            base = scorer.score_chunk(chunk)
            jit = scorer.score_chunk(chunk, (rng.randint(-1, 1), rng.randint(-1, 1)))
            if base is None or jit is None:
                continue
            diffs.append(abs(base - jit))
            flips += (base >= 0.5) != (jit >= 0.5)
            checked += 1
    return {
        "n_compared": len(diffs),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
        "decision_flips": flips,
    }
