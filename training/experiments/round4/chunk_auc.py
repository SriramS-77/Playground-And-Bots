"""Chunk-level scorer AUC on the evaluation block as covert bots replace a fraction rho of bots."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import grounded_headline as G
D = G.D
p = D.pools(False, "R6 chunk AUC under covert bots")
def chunks(sessions):
    return [v for s in sessions for v in BY.get(s.key, [])]
BY = {}
for (k, i), v in p.cache._table.items():
    if v is not None and i < 12:
        BY.setdefault(k, []).append(v)
rows = []
for rho in (0.0, 0.5, 1.0):
    h, b, real = D.shifted(p, "bots", rho, held_out=True)
    hs, bs = chunks(h), chunks(b)
    rows.append({"rho": rho, "rho_realised": real, "n_h": len(hs), "n_b": len(bs),
                 "chunk_auc": roc_auc_score([0] * len(hs) + [1] * len(bs), hs + bs)})
df = pd.DataFrame(rows); print(df.round(3).to_string())
df.to_csv(D.R.OUT / "chunk_auc_covert.csv", index=False)
