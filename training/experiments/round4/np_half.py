"""NP single threshold at alpha = 0.005 (adds to pareto.py's static sweep; same protocol)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pareto as P, np_audit as NP
G, C, D = P.G, P.C, P.D
p = D.pools(False, "R6 NP threshold alpha 0.005")
th = NP.np_threshold(p, 0.005)
df = D._evaluate_points(P.Single(th, "NP threshold"), "NP threshold", p, P.PTS, False,
                        C.GROUNDED, C.GROUNDED_CFG, {"knob": 0.005, "t": th})
df.to_csv(P.OUT.parent / "np_alpha_0005.csv", index=False)
print("t =", th); print(df.groupby(["eval_direction", "eval_rho"])[["DI", "human_survival", "bot_survival"]].mean().round(3))
