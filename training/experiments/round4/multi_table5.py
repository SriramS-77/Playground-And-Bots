"""Normal static multi-threshold (level floor(10 q)) at Table 5's points, eval block."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import np_audit as NA
from rlcaptcha.policies.static import MultiThresholdPolicy, SingleThresholdPolicy
C, D = NA.C, NA.D
p = D.pools(False, "R6 static multi-threshold for Table 5")
pts = [("none", 0.0)] + [(d, r) for d in ("bots", "humans") for r in (0.5, 1.0)]
import os
POL = SingleThresholdPolicy() if os.environ.get("SINGLE") else MultiThresholdPolicy()
df = D._evaluate_points(POL, POL.name, p, pts, False, C.GROUNDED, C.GROUNDED_CFG, {})
df.to_csv(NA.OUT.parent / ("single_table5.csv" if os.environ.get("SINGLE") else "multi_table5.csv"), index=False)
print(df.groupby(["eval_direction", "eval_rho"]).agg(DI=("DI", "mean"), fr=("friction_s", "mean")).round(1).to_string())
