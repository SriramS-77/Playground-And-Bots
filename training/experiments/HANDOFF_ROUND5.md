# Round 5 — fold 2: the drift study in the v2 world with the new RL state

One script: `round5/r5_fold2.py`. **Read §0–§3 before running anything.** Round 3/4's rules hold
wherever this file is silent: fold 2 only, nothing on folds 0/1, the eval block opened only by
final-evaluation tasks and every opening logged.

## 0. What this round is, and why

The local pilot (`round4/sens_world.py`, seed 0, rl_val; results in
`round4/sens_pilot/`) showed the drift findings hold in a literature-grounded world. This
round produces the paper's numbers:

1. **v2 world** (`WORLD_V2.md`, now the default for all experiments): humans abandon when a
   visible challenge is shown (Searles et al., USENIX Security 2023 per-type and per-position
   rates), never after passing; an abandoning human costs −100 like a wrongly blocked one;
   level 10 is a hard block.
2. **New RL state** (`--state v2`, primary): 7 features — score, running score, challenges solved
   (levels ≥ 1), steps passed, last level / 10, highest level solved / 10, active users / 300.
   No feature depends on the scorer except the two scores. The round 1–4 state (`--state v1`)
   runs as a sensitivity arm.
3. **DI against calibration error**, offline (frozen) vs online training, with a **retrain
   ceiling**, on the **eval block**, 5 seeds, 10 shift levels.

## 1. Decisions — fixed before any run, do not change

| | decision | why |
|---|---|---|
| D1 | World `GROUNDED_V2` / `GROUNDED_V2_CFG` for every task, training and evaluation | user decision after the pilot (WORLD_V2.md) |
| D2 | Scorer: cross-fit seed 1, loaded from round 4's drift stage (`results/round4/fold2/drift/`) | round 4's R1, unchanged |
| D3 | Architectures fixed from round 4: DQN 128-64, PPO 128-64, Thompson h128_64_e64 **mog2**, LinUCB h64_e32 | mog2 chosen on rl_val by the pilot; all three Thompson posteriors still reported offline |
| D4 | Online rewards: oracle, score-only, posterior. Algorithms DQN, PPO, Thompson | user's scope after the pilot |
| D5 | State v2: levels 0.1 … 1.0 (10) + 0; state v1: levels {0.5, 1.0} + 0 | v1 is a sensitivity arm |
| D6 | Retrain ceiling: DQN and PPO only (oracle, shifted pool from episode 0) | Thompson's would add ~250 h of compute for a reference line |
| D7 | Primary contrast (state v2): per (algorithm, reward, direction), mean DI over the 10 levels, online − control, Welch over 5 seeds, Holm over 18 | as round 4 |
| D8 | Evaluation: eval block + external HELD-OUT halves, 20 eval seeds × bot volumes 20–1000 | as round 4 |

## 2. Run order

From `training/experiments/`, after `git pull` (main). **Prerequisite:** round 4's drift stage
directory `results/round4/fold2/drift/` (scorer, `ext_scores.pkl`, `arch_choice.json`) — it is
on the cluster from round 4. If missing, run `python round4/r4_fold2.py --stage drift --setup`.

```bash
python preflight.py
python round5/r5_fold2.py --state v2 --smoke      # toy budget on rl_val; must end "OK (eval block untouched)"
python round5/r5_fold2.py --state v1 --smoke
python round5/r5_fold2.py --state v2 --list       # 1172 tasks
python round5/r5_fold2.py --state v1 --list       # 251 tasks
sbatch --array=0-1171%25 slurm/r5.sh v2           # primary first (offset ids if the scheduler caps arrays)
sbatch --array=0-250%25  slurm/r5.sh v1
python round5/r5_fold2.py --state v2 --collect    # curves.csv, per_seed.csv, contrasts.csv, drift_figure.png
python round5/r5_fold2.py --state v1 --collect
```

Task 0 of state v2 is the `metrics` task (the x-axis); state v1's collect reads it, so collect
v1 after v2's task 0 has run. A task whose CSV exists is skipped, so resubmitting is safe.

**Compute** (from round 3/4 logs and the pilot; 2 CPUs per task):

| | tasks | per task | task-hours |
|---|---|---|---|
| state v2, DQN/PPO online + retrain | 800 | 5–10 min | ~100 |
| state v2, Thompson online + control | 315 | 1.5–3 h | ~700 |
| state v2, offline + controls (21 eval points) | 56 | 0.5–3 h | ~60 |
| state v1, all | 251 | as above | ~280 |

≈ 1,150 task-hours: **~45 h at 25 wide, ~28 h at 40 wide**. Thompson dominates; submit state v2
first. Thompson needs up to ~2 GB per task (`slurm/r5.sh` asks for 8 GB).

## 3. Gates and stop conditions

* **Smoke** must end with "OK (eval block untouched)" for both states.
* **Scorer**: tasks load round 4's seed-1 scorer; if any task tries to refit, stop and report.
* Each online/control/retrain task asserts it saw exactly 300 training episodes; a failure is a bug.
* `--collect` warns when task CSVs are missing: rerun **those ids**, never new seeds.
* `results/round4/fold2/drift/eval_access.log` gains one line per evaluating task (1,172 + 251);
  report the count.
* **Sanity** (state v2, from the pilot on rl_val, so approximate): at ρ = 0, offline DQN/PPO ≈ 100;
  offline DQN/PPO at covert bots ρ = 1 below 15; posterior online at covert bots ρ = 1 below 15.
  A large miss: stop and report.
* **Known issue to report, not fix**: in the pilot PPO with the score-only reward (and once with
  oracle) collapsed to a "challenge everyone" policy (DI exactly 39.9: all bots blocked, ~40 %
  of humans kept). Report how many PPO seeds do this (`per_seed.csv`).

## 4. Deliverables

Zip `results/round5/fold2/` (both `state_v2/` and `state_v1/`: `tasks/`, `per_seed.csv`,
`curves.csv`, `contrasts.csv`, `drift_metrics.csv`, `drift_figure.png`) plus the Slurm logs and
the eval-access line count.

**Findings rule:** every number in a write-up is copied from a CSV in the deliverable, with the
file named. Do not report a pre-declared check as passing without the column that shows it.
Rounds 3 and 4's write-ups broke this and were replaced.
