"""Adapters so a retrained scorer can drive the existing simulation.

`rlcaptcha.scoring.ScoreCache` only needs an object with
``score_chunk(movements, jitter) -> float | None``. `XScorer` provides that for any
(representation, standardiser, keras model) triple, so notebook 03 onward can swap the
behavioural front end without touching `rlcaptcha`.

`PerMoveScoreCache` exists because the published cache is only valid under the rigid
one-pixel perturbation. Once each movement is jittered independently, two users drawn
from the same recording genuinely differ and the (recording, chunk) key is no longer
sufficient -- so that cache keys on the user as well, and is correspondingly slower.
"""

from __future__ import annotations

import numpy as np

from rlcaptcha.scoring import ScoreCache

from .features import PERTURBATIONS, REPRESENTATIONS, windows
from .scorer_train import CONTEXT, Standardiser, load_scorer


class XScorer:
    """Scores a chunk with a retrained model under a chosen input representation."""

    def __init__(self, model, standardiser: Standardiser, representation: str = "xy",
                 context: int = CONTEXT, batch_size: int = 256,
                 temperature: float = 1.0):
        self.model = model
        self.std = standardiser
        self.rep = REPRESENTATIONS[representation]
        self.representation = representation
        self.context = context
        self.batch_size = batch_size
        # Temperature scaling, fitted on held-out data. T > 1 softens over-confident
        # scores. This matters downstream: the multi-threshold baseline maps the score
        # straight onto a threat level with int(score * 10), so an uncalibrated score
        # silently shifts every one of its decisions.
        self.temperature = float(temperature)

    def _calibrate(self, p):
        if self.temperature == 1.0:
            return p
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        return 1.0 / (1.0 + np.exp(-np.log(pc / (1 - pc)) / self.temperature))

    @classmethod
    def from_name(cls, name: str, calibrated: bool = True, **kw) -> "XScorer":
        model, std, meta = load_scorer(name)
        T = float(meta.get("temperature", 1.0)) if calibrated else 1.0
        return cls(model, std, meta["representation"], temperature=T, **kw)

    def score_chunk(self, movements, jitter=(0, 0)) -> float | None:
        if not movements:
            return None
        if jitter != (0, 0):
            movements = [{**m, "x": m["x"] + jitter[0], "y": m["y"] + jitter[1]}
                         for m in movements]
        batch = windows(self.rep(movements), self.context)
        preds = self.model.predict(self.std(batch), batch_size=self.batch_size, verbose=0)
        return float(np.mean(self._calibrate(preds)))

    def score_many(self, chunks, batch_size: int = 4096):
        """Score many chunks in ONE batched pass. Returns a list aligned with `chunks`.

        A single `model.predict` call costs ~150 ms of graph-dispatch overhead
        regardless of model size, so scoring chunk-by-chunk is ~570x slower than
        batching. Since a user's behavioural scores do not depend on the agent's
        actions, a whole episode's score table can be built up front with this.
        """
        flat, owner = [], []
        for i, ch in enumerate(chunks):
            if not ch:
                continue
            w = windows(self.rep(ch), self.context)
            flat.append(w)
            owner.extend([i] * len(w))
        out = [None] * len(chunks)
        if not flat:
            return out
        preds = self.model.predict(self.std(np.concatenate(flat)),
                                   batch_size=batch_size, verbose=0).ravel()
        preds = self._calibrate(preds)
        owner = np.asarray(owner)
        for i in np.unique(owner):
            out[int(i)] = float(preds[owner == i].mean())
        return out


class PerMoveScoreCache(ScoreCache):
    """Cache keyed on (session, chunk, per-user perturbation seed).

    Under `perturb_per_move` every simulated user really is a different sample, so the
    published memoisation would be wrong. Each user gets one perturbation seed for the
    whole session; the cache therefore collapses repeated *timesteps* of the same user
    but not distinct users. Cost scales with the population, which is the honest price
    of dropping the rigid-translation assumption.
    """

    def __init__(self, scorer, perturbation: str = "per_move", magnitude: int = 3,
                 sigma: float = 2.0):
        super().__init__(scorer=scorer, exact=False)
        self.perturb = PERTURBATIONS[perturbation]
        self.kw = {"sigma": sigma} if perturbation == "gaussian" else {"magnitude": magnitude}
        self._table = {}

    def precompute(self, *session_groups, verbose: bool = True):
        # Nothing to precompute: keys depend on the users, which do not exist yet.
        if verbose:
            print("PerMoveScoreCache: lazy (keys depend on per-user perturbation)")
        return self

    def score(self, user):
        import random as _r

        key = (user.session.key, user.cache_key[1], getattr(user, "perturb_seed", 0))
        if key in self._table:
            return self._table[key]
        idx = user.cache_key[1]
        if idx < 0 or idx >= len(user.session.chunks):
            return None
        chunk = user.session.chunks[idx]
        if not chunk:
            self._table[key] = None
            return None
        rng = _r.Random(hash(key) & 0xFFFFFFFF)
        value = self.scorer.score_chunk(self.perturb(chunk, rng, **self.kw))
        self._table[key] = value
        return value


def build_cache(scorer, humans, bots, verbose: bool = True) -> ScoreCache:
    """The standard (rigid-perturbation) cache, for any scorer object."""
    return ScoreCache(scorer).precompute(humans, bots, verbose=verbose)
