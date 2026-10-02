# Slurm drivers

**Round 3 (current): fold 2 only — `r3_prepare.sh`, `r3_search.sh`, `r3_headline.sh`, and
`r3_e.sh <e2|e3|e4|e6|e7|e8>` for the experiment arrays in `round3/`.** The run order is
`HANDOFF_ROUND3.md` §2 and §5. The round-2 scripts below (`prepare_scorers.sh`,
`arch_search.sh`, `phase0.sh`) are kept for the record; do not run them in round 3 — they
fit scorer-only fold scorers into `results/fold*_scorer/`, the design that collapsed.

## Round 2

Every workload here is small and single-threaded — a 6,081-parameter LSTM, a ≤256-wide
MLP, and a pure-Python simulation loop. **A GPU buys almost nothing per run; throughput
does.** Round 1 ran `nb_03b`'s ten *independent* cross-fit splits serially for 357
minutes; as an array they finish in the time of the slowest one.

So: one array task per training unit, `--cpus-per-task=2`, and a dependent collect job.

Adjust `--partition`, `--time`, `--mem` and the conda path for your cluster — those are
the only lines that should need editing.

## Order

```bash
# 0. once: verify the environment and the wiring
python preflight.py
python exp_policy_arch_search.py --smoke
python exp_scorer_phase0.py --smoke

# 1. Phase 0 -- confirm the scorer config (cheap, no dependencies, launch first)
python exp_scorer_phase0.py --list                 # prints the --array line
sbatch --array=0-N%25 slurm/phase0.sh
python exp_scorer_phase0.py --audit                # P0.6, no training
python exp_scorer_phase0.py --collect

# 2. fit the three per-fold scorers ONCE (not per task -- see below)
sbatch slurm/prepare_scorers.sh

# 3. the policy + posterior search
python exp_policy_arch_search.py --list --seeds 5   # prints the --array line
sbatch --array=0-224%25 slurm/arch_search.sh
python exp_policy_arch_search.py --collect
```

Step 2 is **not** optional and must not be folded into the tasks. Every candidate has to
be ranked against the same scorer; if each task refits its own, TF/oneDNN nondeterminism
gives slightly different weights and the comparison is against different yardsticks.

Step 2 also **replaces round 2's cached scorers**. Those were fitted on `fold.scorer` alone
and fold 0's collapsed (eval AUC 0.70). They carry no provenance record, so an array task
that finds one fails with "is stale", and `--prepare` refits it. Check the prepare log for
three `preflight/scorer: val AUC ...` and three `preflight/canary: ...` lines before
launching step 3; a scorer "ready" in a few seconds means it was loaded, not fitted.

## Budget

Measured on one CPU core at reduced bot volumes `(0, 20, 100, 200)` — re-measure before
sizing, the full sampler reaching 1000 bots is several times slower:

| | per 200-episode training |
|---|---|
| Thompson | ~25 min |
| LinUCB | ~55 min |
| DQN | much faster (no per-arm matrix algebra) |

225 trainings ⇒ on the order of **150–250 core-hours**. At 25-way concurrency, most of a
day. If that is too much, in order of preference: raise `train_every` from 5; run 3 seeds
first and 5 only on the finalists; trim `BANDIT_ARCHS` to 3.

**Time one full training before you launch the array.**
