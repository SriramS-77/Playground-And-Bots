"""Scorer-embedding state for the RL agents (round 6, opt-in via RLC_EMB; see trainer.py).

Each chunk's embedding is the PCA-reduced (K = 8) mean of the scorer's 32-d Dense-ReLU outputs
over the chunk's windows (round4/emb_prepare.py). Like q, the agent sees the current chunk's
vector and the running mean over the user's session; an empty chunk falls back to the running
mean, as `User.observe` does for q.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

K = 8
TABLE: dict = {}
_ZERO = np.zeros(K, dtype=np.float32)


def install(directory):
    """Load the embedding table and wrap `User.observe` so it also tracks the embedding."""
    from rlcaptcha.data import User
    TABLE.update(pickle.load(open(Path(directory) / "emb_table.pkl", "rb")))
    if getattr(User.observe, "_emb", False):
        return
    real = User.observe

    def observe(self, score):
        e = TABLE.get(self.cache_key) if score is not None else None
        n = getattr(self, "_n_emb", 0)
        avg = getattr(self, "emb_avg", _ZERO)
        if e is None:
            self.emb_cur = avg
        else:
            self.emb_cur = e
            self.emb_avg = (avg * n + e) / (n + 1)
            self._n_emb = n + 1
        return real(self, score)

    observe._emb = True
    User.observe = observe


def features(user) -> np.ndarray:
    return np.concatenate([getattr(user, "emb_cur", _ZERO), getattr(user, "emb_avg", _ZERO)]).astype(np.float32)
