"""expkit -- the reviewer-response experiments.

Additive by construction: it imports `rlcaptcha` and never modifies it, so
`reproduce_table4.ipynb` and `reproduce_seeded_table_4.ipynb` keep producing exactly
what they produced before.

    partition      session-level three-way split (LSTM / RL / eval)
    features       input representations and perturbation models
    scorer_train   retrain the behavioural scorer under a clean protocol
    calibration    ECE, Brier, reliability, temperature scaling
    xscoring       adapters so a retrained scorer drives the simulation
    stochastic     probabilistic action model grounded in published measurements
    proxy          a reward built only from signals a deployment can observe
    simx           extended simulation loop (superset of rlcaptcha.simulate)
    trainer        DQN training on an arbitrary pool / reward source
    bootstrap      session-level uncertainty and data-sufficiency analysis
"""

from . import paths  # noqa: F401  (puts `training/` on sys.path as a side effect)

__all__ = [
    "paths", "partition", "features", "scorer_train", "calibration",
    "xscoring", "stochastic", "proxy", "simx", "trainer", "bootstrap",
]
