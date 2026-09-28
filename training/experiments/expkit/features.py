"""Input representations for the behavioural scorer, and the perturbation models.

Two reviewer threads meet here.

**Timestamps (Reviewer 1, point 7).** The published scorer is fed
``[[m['x'], m['y']] for m in mouse]`` and nothing else. The recordings *do* carry a
``timestamp`` field, so the manuscript's talk of acceleration and jerk describes
features the network never receives. The representations below let us retrain with
timing and measure what it is worth.

**Perturbation.** ``perturb_mouse_data`` in the original code draws ONE integer offset
in {-1,0,1} and applies it to every point of a session -- a rigid one-pixel translation.
That is what makes the humanity-score cache sound, but it also means two simulated users
drawn from the same recording are almost the same user. ``per_move`` and ``gaussian``
perturb each movement independently, which is the behaviour the manuscript's augmentation
description implies and a far stronger test of the scorer.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Representations: list[movement dict] -> (T, F) float array
# --------------------------------------------------------------------------- #

def _arr(movements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([m["x"] for m in movements], dtype=np.float64)
    y = np.array([m["y"] for m in movements], dtype=np.float64)
    t = np.array([m.get("timestamp", i) for i, m in enumerate(movements)], dtype=np.float64)
    return x, y, t


def rep_xy(movements) -> np.ndarray:
    """[x, y] -- exactly what the published model consumes. The baseline."""
    x, y, _ = _arr(movements)
    return np.stack([x, y], axis=1)


def rep_xy_dt(movements) -> np.ndarray:
    """[x, y, dt_ms] -- absolute position plus inter-event timing."""
    x, y, t = _arr(movements)
    dt = np.diff(t, prepend=t[0])
    return np.stack([x, y, dt], axis=1)


def rep_dxdy_dt(movements) -> np.ndarray:
    """[dx, dy, dt_ms] -- translation invariant: pure motion, no absolute position.

    If this matches `xy`, the scorer is reading dynamics. If it collapses, the scorer
    was partly reading *where on the screen* the cursor sits, which would not survive a
    change of page layout.
    """
    x, y, t = _arr(movements)
    return np.stack([
        np.diff(x, prepend=x[0]),
        np.diff(y, prepend=y[0]),
        np.diff(t, prepend=t[0]),
    ], axis=1)


def rep_dxdy(movements) -> np.ndarray:
    """[dx, dy] -- translation invariant but with NO timing.

    The control that separates the two things `dxdy_dt` changes at once. If this matches
    `dxdy_dt`, the gain over `xy` came from dropping absolute screen position and timing
    contributes nothing. If `dxdy_dt` beats it, timing genuinely carries signal.
    """
    x, y, _ = _arr(movements)
    return np.stack([np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])], axis=1)


def rep_kinematic(movements) -> np.ndarray:
    """[dx, dy, dt, speed, accel] -- the features the manuscript actually claims.

    Speed is |displacement| / dt; acceleration its first difference. dt is floored at
    1 ms so a burst of same-millisecond events cannot divide by zero.
    """
    x, y, t = _arr(movements)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dt = np.diff(t, prepend=t[0])
    safe_dt = np.maximum(dt, 1.0)
    speed = np.hypot(dx, dy) / safe_dt
    accel = np.diff(speed, prepend=speed[0]) / safe_dt
    return np.stack([dx, dy, dt, speed * 100.0, accel * 1000.0], axis=1)


REPRESENTATIONS = {
    "xy": rep_xy,                 # published baseline
    "xy_dt": rep_xy_dt,
    "dxdy": rep_dxdy,
    "dxdy_dt": rep_dxdy_dt,
    "kinematic": rep_kinematic,
}

N_FEATURES = {"xy": 2, "xy_dt": 3, "dxdy": 2, "dxdy_dt": 3, "kinematic": 5}


# --------------------------------------------------------------------------- #
# Windowing -- byte-identical to rlcaptcha.scoring._windows
# --------------------------------------------------------------------------- #

def windows(mat: np.ndarray, length: int = 100) -> np.ndarray:
    """Split (T, F) into (n, length, F), padding the tail by repeating the last row.

    The tail window is appended unconditionally, matching the original
    ``HumanityScorer.handle_data``: when T is an exact multiple of `length` it still
    emits a final window of `length` copies of the last point.
    """
    n = len(mat)
    out = [mat[i * length:(i + 1) * length] for i in range(n // length)]
    tail = mat[(n // length) * length:]
    if len(tail) < length:
        pad = np.repeat(mat[-1:], length - len(tail), axis=0)
        tail = np.concatenate([tail, pad], axis=0) if len(tail) else pad
    out.append(tail)
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Perturbation models
# --------------------------------------------------------------------------- #

def perturb_rigid(movements, rng, magnitude: int = 1):
    """The published behaviour: ONE offset in {-magnitude..magnitude} for the session."""
    dx = rng.randint(-magnitude, magnitude)
    dy = rng.randint(-magnitude, magnitude)
    return [{**m, "x": m["x"] + dx, "y": m["y"] + dy} for m in movements]


def perturb_per_move(movements, rng, magnitude: int = 1):
    """An independent integer offset per movement. Breaks the score cache by design."""
    return [
        {**m,
         "x": m["x"] + rng.randint(-magnitude, magnitude),
         "y": m["y"] + rng.randint(-magnitude, magnitude)}
        for m in movements
    ]


def perturb_gaussian(movements, rng, sigma: float = 2.0):
    """Per-movement Gaussian jitter -- the augmentation the manuscript describes."""
    return [
        {**m, "x": m["x"] + rng.gauss(0, sigma), "y": m["y"] + rng.gauss(0, sigma)}
        for m in movements
    ]


PERTURBATIONS = {
    "none": lambda mv, rng, **kw: list(mv),
    "rigid": perturb_rigid,
    "per_move": perturb_per_move,
    "gaussian": perturb_gaussian,
}
