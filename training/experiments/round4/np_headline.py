"""NP single threshold (alpha = 0.10) on Table 4's protocol: held-out eval block, fixed traffic
(100 humans, 20..1000 bots) and varying traffic, 50 eval seeds. Threshold from scorer-block humans."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grounded_headline as G, np_audit as NP, pareto as P
p = G.C.pools(final=True, smoke=False, reason="R6 NP threshold alpha 0.10 headline")
th = NP.np_threshold(p, 0.10)
df = G.evaluate(P.Single(th, "NP threshold"), "Static NP threshold", p, G.EVAL_SEEDS,
                {"family": "statics", "train_seed": -1, "t": th})
df.to_csv(G.OUT / "np_threshold_a0.10.csv", index=False)
f = df[df.grid == "fixed"]
print("t =", th)
print(f.groupby("level").DI.mean().round(1).to_string())
print(f[["DI", "BOS", "SP_F1", "SI_F1", "human_survival", "bot_survival", "friction_s"]].mean().round(3).to_string())
print("varying DI", df[df.grid == "varying"].DI.mean().round(2))
