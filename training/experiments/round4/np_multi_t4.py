"""NP multi-threshold (equal bands above the NP threshold -> levels 1..10) on Table 4's protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import grounded_headline as G
import np_audit as NA


class NPMultiEqual(NA.NPStatic):
    def __init__(self, t):
        super().__init__(t, True)
        self.name = "Static NP multi-threshold"

    def select_action(self, user, n_active):
        q = user.bot_score
        if q <= self.t:
            return 0
        return int(min(10, 1 + np.floor((q - self.t) / (1 - self.t) * 10)))


if __name__ == "__main__":
    p = G.C.pools(final=True, smoke=False, reason="R6 NP multi-threshold, Table 4 protocol")
    t = NA.np_threshold(p, 0.1)
    d = G.evaluate(NPMultiEqual(t), "Static NP multi-threshold", p, G.EVAL_SEEDS,
                   {"family": "statics", "train_seed": -1, "t": t})
    d.to_csv(G.OUT / "np_multi_threshold_a0.10.csv", index=False)
    f = d[d.grid == "fixed"]
    print("t", t)
    print(f.groupby("level").DI.mean().round(1).to_string())
    print(f[["DI", "BOS", "SP_F1", "SI_F1", "human_survival", "friction_s"]].mean().round(3).to_string())
    print("varying", round(d[d.grid == "varying"].DI.mean(), 1))
