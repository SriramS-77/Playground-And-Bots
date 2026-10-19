"""Static NP single / multi threshold (alpha = 0.1, Table 4's) at Table 5's points, eval block."""
import os, sys; from pathlib import Path
os.environ["NP_ALPHA"] = "0.1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import np_audit as NA
C, D = NA.C, NA.D
p = D.pools(False, "R6 NP statics for Table 5 (alpha 0.1)")
t = NA.np_threshold(p, 0.1)
pts = [("none", 0.0)] + [(d, r) for d in ("bots", "humans") for r in (0.5, 1.0)]
import pandas as pd
df = pd.concat([D._evaluate_points(pol, pol.name, p, pts, False, C.GROUNDED, C.GROUNDED_CFG, {"t_np": t})
                for pol in (NA.NPStatic(t, False), NA.NPStatic(t, True))], ignore_index=True)
df.to_csv(NA.OUT.parent / "np_statics_table5.csv", index=False)
print("t =", t)
print(df.groupby(["policy", "eval_direction", "eval_rho"]).agg(DI=("DI", "mean"), fr=("friction_s", "mean")).round(1).to_string())
