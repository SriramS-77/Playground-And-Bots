# Round 3 deliverables — fold 2

Outputs of the round-3 cluster run (Slurm, `nvidiaserver`, 2026-10-06), produced by
`exp_round3_fold2.py` and `round3/e*.py` (the bundled `expkit` is byte-identical to `c160cc0`). Fold 2 only; folds 0 and 1 were
never opened.

**Read [`FINDINGS.md`](FINDINGS.md)**: every number in it was checked against the CSVs here. It
replaces the cluster agent's own write-up, which misreported parts of E7–E9 (see its §7).

## Layout

```
FINDINGS.md                 verified findings (supersedes the cluster write-up)
results/
  prepare_log.json          scorer preparation: attempt 0, gates, AUCs
  arch_choice.json          selection rule + winners (dqn:128-64, linucb:h64_e32, thompson:h128_64_e64:mog3)
  search_runs.csv           all 225 search tasks, per eval cell
  search_{dqn,linucb,thompson}.csv   per-family ranking
  headline_runs.csv         all 21 headline tasks, per eval cell
  headline_{published,realistic}[_vs_static_multi].csv
  eval_access.log           446 eval-block openings
  scorer/                   5 cross-fitted part scorers + refit (model.keras, scorer.json)
  e2 .. e9/                 per-experiment summary tables
inference/                  standalone inference (refit scorer + DQN 128-64, seeds 0-4); fixed, see its README
```

## Not committed

Kept in the cluster zip (`deliverables_round3.zip`), not in git: the 748 Slurm logs, the
per-task CSVs (`search/`, `headline/`, `e*/tasks/`, `e9/curves/`, `e9/diag/`), and the headline
policy checkpoints (`results/agents/`, 6.8 MB). Every committed table is an aggregate of those.

This folder is deliberately **not** `results/round3/fold2/` (the runner's `OUT`): a partial copy
there would look like resumable state to `exp_round3_fold2.py`.

## Running inference

```bash
python training/experiments/round3/deliverables_fold2/inference/run_inference.py
```
