# Round 4, fold 2 — verified findings

Every number below is copied from a CSV in this folder (file named per section). The cluster
agent's own write-up was checked against these files and **replaced**; its errors are listed in §6.
Not committed (reproducible / too large): the 1,286 per-task CSVs (81 MB; `drift_per_seed.csv` is
their concatenation), the Slurm logs (no tracebacks), the smoke outputs, the scorers.

**Provenance.** `ppo/prepare_log.json`: seed_used 0 (round 3's scorer). `drift/prepare_log.json`:
seed_used 1 (declared post-hoc choice, R1). `drift/eval_access.log`: 1,287 lines, as expected.

## 1. PPO architecture search (`ppo/search_ppo.csv`, `ppo/search_ppo_vs_dqn.csv`)

| candidate | DI | SE | friction s | human surv. | bot surv. | params |
|---|---|---|---|---|---|---|
| **ppo:128-64** | 81.46 | 1.06 | 4.22 | 0.869 | 0.054 | 9,739 |
| ppo:128-64-32-16 | 81.17 | 1.24 | 4.74 | 0.843 | 0.031 | 11,819 |
| ppo:64 | 80.42 | 1.25 | 4.46 | 0.851 | 0.047 | 1,099 |
| ppo:256-128-64 | 78.96 | 3.53 | 9.95 | 0.849 | 0.060 | 43,403 |
| ppo:128-64-32 | 77.16 | 2.40 | 8.38 | 0.823 | 0.051 | 11,467 |
| dqn:128-64 (round 3) | 83.60 | 0.79 | 5.28 | 0.870 | 0.034 | 9,739 |

`ppo:128-64` has the best mean DI and also the least friction within 1 SE, so D2 picks it
directly (`ppo/arch_choice.json`). PPO and DQN pick the same body; PPO is 2.1 DI below DQN
(about 1.6 pooled SE — not a demonstrated difference).

## 2. Calibration breaks, ranking does not (`drift/drift_metrics.csv`, pool = eval)

Bots look human: calibration error of bots 0.074 (ρ=0) → 0.339 (ρ=0.5) → 0.610 (ρ=1);
accuracy at 0.5 0.970 → 0.054; pool AUC 1.000 → 0.995.
Humans look bot-like: error of humans 0.128 → 0.300 → 0.451; accuracy 0.985 → 0.587; AUC 1.000 → 0.994.
The x-axis behaves as designed: monotone in ρ, while AUC barely moves.

## 3. Offline-trained, frozen (`drift/drift_curves.csv`, panel = offline)

| policy | ρ=0 | bots ρ=0.5 | bots ρ=1 | humans ρ=0.5 | humans ρ=1 |
|---|---|---|---|---|---|
| DQN | 97.6 | 53.9 | 10.2 | 75.2 | 57.9 |
| PPO | 99.6 | 52.3 | 3.8 | 76.0 | 56.4 |
| Thompson Sampling | 92.4 | 49.6 | 8.0 | 74.2 | 59.9 |
| LinUCB | 49.3 | 18.6 | −11.6 | 43.8 | 37.8 |
| DQN without H-Score | 16.4 | 15.8 | 16.2 | 16.4 | 16.2 |
| Static Single-Threshold | 90.3 | 59.2 | 28.4 | 54.2 | 22.5 |
| Static Multi-Threshold | 78.0 | 58.4 | 39.5 | 50.8 | 26.3 |

* Every score-using policy falls as calibration error grows; DQN without the score is flat (≈16),
  the expected negative control.
* **Bots look human:** the learned policies fall *further* than the static thresholds (ρ=1:
  bots through 0.874 DQN, 0.959 PPO, vs 0.617 / 0.392 static). The RL policies trust the score more.
* **Humans look bot-like:** the reverse — learned policies keep more humans (ρ=1 humans kept 0.579
  DQN, 0.564 PPO vs 0.225 / 0.268 static).
* Sanity check (HANDOFF §3): offline DQN at ρ=0 is 97.6, not ≈89. It misses *upward*, under a
  different scorer seed; no sign of breakage, but it is outside the stated expectation.

## 4. Online training vs the no-shift control (`drift/drift_contrasts.csv`, Holm over 24)

| | bots look human | | humans look bot-like | |
|---|---|---|---|---|
| arm | DQN Δ (p_holm) | PPO Δ (p_holm) | DQN Δ (p_holm) | PPO Δ (p_holm) |
| oracle | **+30.5** (7e-7) | **+33.2** (5e-8) | +5.2 (0.40) | **+9.8** (0.003) |
| score-only | **+27.6** (2e-4) | **+27.5** (0.003) | −8.9 (0.84) | **−8.7** (0.002) |
| posterior | +3.4 (1.0) | −5.7 (1.0) | −6.7 (1.0) | −11.3 (0.14) |
| posterior (floored) | −16.7 (0.14) | −6.6 (0.96) | **−13.2** (0.017) | **−19.1** (0.002) |
| posterior+labels | +2.0 (1.0) | +7.9 (0.074) | −20.1 (0.14) | **−11.0** (0.009) |
| posterior+labels (floored) | −6.9 (1.0) | +1.9 (1.0) | **−21.6** (0.011) | **−17.7** (0.015) |

Bold = p_holm < 0.05. Seven of 24 contrasts are significant gains or losses in each sign group
as shown; no posterior-family arm gains significantly anywhere.

**The means hide a collapse at high drift** (`drift/drift_curves.csv`, ρ = 1):

* Bots look human — every posterior arm lets essentially **all bots through** after online
  training (bot survival 1.00 for all four DQN posterior arms, 0.80–0.97 for PPO), DI −1.7 to 7.6,
  *below* its own control (12–47). Oracle and score-only hold (DI 78.1 / 80.6 oracle, 61.0 / 82.7
  score-only; controls 3.8–21.7).
* Humans look bot-like — every score-based arm drives humans out (humans kept 0.01–0.18,
  friction 47–66 s; controls 0.27–0.50). Only oracle survives (PPO 82.3 vs control 60.0).
* At low drift (ρ = 0.1) online training helps most arms (e.g. DQN posterior+labels 86.4 vs
  85.8 control, bots); the failure grows with calibration error.

## 5. What is new

1. **The posterior reward is not drift-robust, in either direction.** When the scorer is
   miscalibrated, a reward built from it teaches the policy the scorer's mistake: pass every
   bot that looks human, challenge every human that looks bot-like. Round 3 showed this for one
   point (advanced bots, seed-0 scorer); round 4 shows it as a dose-response over calibration error,
   for DQN and PPO, under the scorer that ranks advanced bots best.
2. **Score-only online training recovers the bot shift but not the human shift** — the only
   label-free arm with a significant gain. Why it adapts where the posterior does not is not
   explained by these files and should be checked before it is claimed.
3. **Frozen RL policies are more score-dependent than static thresholds**: worse than statics under
   covert bots, better under bot-like humans.
4. **PPO ≈ DQN.** Same architecture chosen; no PPO-vs-DQN contrast was run, so no robustness ranking.

## 6. Errors in the cluster's FINDINGS.md (replaced)

* §4 baseline (ρ=0) numbers for PPO (87.218), Thompson (91.479), LinUCB (24.291) and both statics
  (80.588, 82.630), with their survivals and frictions, appear in no CSV. The CSV values are
  99.6, 92.4, 49.3, 78.0 and 90.3. "PPO drops from 87.218" is therefore wrong.
* "Oracle adapts positively in both directions": DQN oracle, humans has p_holm 0.40.
* "PPO's stochastic policy gradients provide enhanced stability": no PPO-vs-DQN test exists.
* "Over-challenge … induce abandonment": the CSVs show humans kept collapsing and friction rising;
  abandonment is not separated from failure, so the mechanism is not shown.
* The posterior arms' collapse to ~0 DI at high drift (§4 above) was not reported.
* The ≈89 sanity check was not mentioned.
