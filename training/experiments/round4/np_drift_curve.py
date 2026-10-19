"""Static NP threshold (alpha = 0.10) on Fig. 4's covert-bot grid (round-4 drift protocol, eval block)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pareto as P, np_audit as NP
C, D = P.C, P.D
p = D.pools(False, "R6 NP alpha 0.10 drift curve")
th = NP.np_threshold(p, 0.10)
pts = [("none", 0.0)] + [("bots", r) for r in D.LEVELS]
df = D._evaluate_points(P.Single(th, "Static NP threshold"), "Static NP threshold", p, pts, False,
                        C.GROUNDED, C.GROUNDED_CFG, {"panel": "offline", "t": th})
df.to_csv(P.OUT.parent / "np_drift_curve.csv", index=False)
print("t =", th); print(df.groupby("eval_rho").DI.mean().round(1).to_string())
