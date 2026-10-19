"""NP multi-threshold: below the NP threshold t nothing; above it (t, 1] is split into 10 equal
bands mapped to levels 1..10. Compared with the "special" NP multi-threshold (max(3, floor(10q)),
capped at 9) and the NP single threshold, at Table 5's points and on Table 4's protocol."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import np_audit as NA, grounded_headline as G
C, D = NA.C, NA.D


class NPMultiEqual(NA.NPStatic):
    def __init__(self, t):
        super().__init__(t, True)
        self.name = "NP multi-threshold (equal bands)"

    def select_action(self, user, n_active):
        q = user.bot_score
        if q <= self.t:
            return 0
        return int(min(10, 1 + np.floor((q - self.t) / (1 - self.t) * 10)))


p = D.pools(False, "R6 NP multi equal bands")
t = NA.np_threshold(p, 0.1)
pols = [NA.NPStatic(t, False), NPMultiEqual(t), NA.NPStatic(t, True)]
names = ["NP threshold", "NP multi-threshold", "special NP multi-threshold"]
pts = [("none", 0.0)] + [(d, r) for d in ("bots", "humans") for r in (0.5, 1.0)]
rows = []
for pol, nm in zip(pols, names):
    df = D._evaluate_points(pol, nm, p, pts, False, C.GROUNDED, C.GROUNDED_CFG, {})
    g = df.groupby(["eval_direction", "eval_rho"]).agg(DI=("DI", "mean"), fr=("friction_s", "mean"))
    rows.append(g.assign(policy=nm))
t5 = pd.concat(rows).reset_index()
print("t =", round(t, 3)); print(t5.pivot(index="policy", columns=["eval_direction", "eval_rho"], values="DI").round(1).to_string())
print(t5.pivot(index="policy", columns=["eval_direction", "eval_rho"], values="fr").round(1).to_string())
# Table 4 protocol
p4 = C.pools(final=True, smoke=False, reason="R6 NP multi equal bands, Table 4 protocol")
for pol, nm in zip(pols, names):
    d = G.evaluate(pol, nm, p4, G.EVAL_SEEDS, {})
    f = d[d.grid == "fixed"]
    print(f"T4 {nm:28s} fixed DI {f.DI.mean():.1f} humans kept {f.human_survival.mean():.3f} "
          f"bots through {f.bot_survival.mean():.3f} friction {f.friction_s.mean():.1f} varying DI {d[d.grid=='varying'].DI.mean():.1f}")
t5.to_csv(NA.OUT.parent / "np_multi_equal.csv", index=False)
