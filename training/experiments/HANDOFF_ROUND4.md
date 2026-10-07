# Round 4 — fold 2 only

Two stages, one script: `round4/r4_fold2.py`. **Read §0–§3 before running anything.**
Round 3's rules hold wherever this file is silent (`HANDOFF_ROUND3.md`): fold 2 only, nothing on
folds 0/1, the eval block opened only by final-evaluation tasks and every opening logged.

---

## 0. What this round is

1. **PPO architecture search** (stage `ppo`). PPO is new (`expkit/ppo.py`). Search its hidden
   layers exactly as round 3 searched DQN: the same five layer sets, 5 seeds × 3 rl_val
   rotations = **75 tasks**, the same sampler (D5), budget, deterministic world, oracle reward
   and selection rule (D2). It runs under **round 3's scorer** (cross-fit seed 0), so PPO is
   chosen under the same conditions as round 3's DQN and bandit winners.
2. **DI against scorer calibration error** (stage `drift`). How do offline-trained (frozen)
   and online-trained policies behave as the humanity scorer becomes miscalibrated on the
   users it meets? Runs under **our scorer** (cross-fit seed 1, `round4/scorer_seed1/`).
   **1,287 tasks.**

## 1. Decisions — fixed before any run, do not change

| | decision | why |
|---|---|---|
| R1 | Drift stage scorer: cross-fit **seed 1** (`round4/scorer_seed1/`) | **Declared, post-hoc**: the user chose it after seeing it rank advanced web-bots further from humans than round 3's seed-0 scorer (mean P(bot) 0.39 vs 0.30; on Balabit humans the two are equal). It passes every round-3 gate (held-out-part AUC 0.962–0.984, validation AUC ≥ 0.976, canary 70 % humans kept). **The findings' methods must state this choice and its reason**, and that seed 0 behaves differently on the same data. Its `prepare_log.json` records it as a declared choice, not a retry |
| R2 | PPO searched under round 3's seed-0 scorer; every architecture then **fixed** (`dqn:128-64`, `linucb:h64_e32:gaussian`, `thompson:h128_64_e64:mog3` from round 3; `ppo:<winner>` from this round) | the architecture choice must not depend on the drift experiment |
| R3 | PPO hyperparameters fixed, not searched: clip 0.2, GAE λ 0.95, γ 0.95, 10 epochs, minibatch 64, Adam 3e-4, entropy 0.01, value coef 0.5, grad-norm 0.5, separate actor/critic (`expkit/ppo.py`, `PPOConfig`) | only the body is searched, as for DQN |
| R4 | Drift design, x-axis, levels, panels and the primary contrast as declared in the docstring of `round4/r4_fold2.py` | pre-registration; summary in §4 |
| R5 | Full grid: 2 directions × 10 levels × 6 arms × {DQN, PPO} × 5 seeds | user's choice |

## 2. Run order

From `training/experiments/`, after `git pull` (main):

```bash
python preflight.py

# --- stage ppo ------------------------------------------------------------
python round4/r4_fold2.py --stage ppo --setup      # copies round 3's seed-0 scorers into
                                                   # results/round4/fold2/ppo/ and LOADS them
python round4/r4_fold2.py --stage ppo --smoke      # toy budget, rl_val only
python round4/r4_fold2.py --stage ppo --list       # 75 tasks
sbatch --array=0-74%25 slurm/r4.sh ppo
python round4/r4_fold2.py --stage ppo --collect    # -> results/round4/fold2/ppo/arch_choice.json

# --- stage drift (only after the ppo collect) ------------------------------
python round4/r4_fold2.py --stage drift --setup    # copies OUR seed-1 scorers + arch_choice.json,
                                                   # LOADS them, scores the external sessions once
python round4/r4_fold2.py --stage drift --smoke
python round4/r4_fold2.py --stage drift --list     # 1287 tasks
sbatch --array=0-1286%25 slurm/r4.sh drift
python round4/r4_fold2.py --stage drift --collect  # tables + the primary contrast
python round4/r4_fold2.py --stage drift --figure   # drift_vs_calibration.png
```

The external-data cache (`data/external_sessions.json.gz`, SHA-256 pinned in
`expkit/external.py`) is in git. Adjust only partition / time / mem / env in `slurm/r4.sh`.

**Time, from round 3's logs** (~3–6 min per E9 task at 2 CPUs): ppo ≈ 75 tasks × ~10 min / 25
wide ≈ 30 min; drift online/control ≈ 1,260 × ~5 min / 25 ≈ 4–5 h; the 26 offline tasks and the
60 controls evaluate at 21 points and take longer (~30–60 min each).

## 3. Gates and stop conditions

* **`--setup` must say "LOADED … none refitted"** for both stages. If it tries to fit (a
  provenance mismatch), **stop and report** — do not let it refit; a refitted scorer is not the
  declared one. Both scorer sets were verified to load through `fit_crossfit(refit_stale=False)`
  on the repository's `rotation.json` before this was pushed.
* Each task asserts it saw exactly 300 training episodes (online/control); a failure is a bug.
* `--collect` warns when task CSVs are missing: rerun **those task ids**, never new seeds.
* `eval_access.log` (per stage directory): one line per drift task that evaluates (all except
  the `metrics` task, which also opens it — 1,287 lines expected); none for the ppo stage.
* Smoke must end with "OK (eval block untouched)".
* Sanity: at ρ = 0 the offline DQN should be near round 3's headline (≈ 99 DI) — under a
  different scorer seed, so not identical — and the online oracle near E7's (≈ 89 DI).
  A large miss means something is broken: stop and report.

## 4. The drift experiment (summary of the pre-registration)

* **Shift.** Two directions, never mixed: *bots look human* — a fraction ρ of the bot pool
  is advanced web-bots (Iliou et al. 2021); *humans look bot-like* — a fraction ρ of the human
  pool is Balabit users (Fulop et al. 2016). ρ ∈ {0.1, …, 1.0} plus 0. Training uses the
  external INJECT halves, evaluation the disjoint HELD-OUT halves. The class ratio is the
  evaluation grid's own at every level (100 humans, 20–1000 bots), so the base rate never moves.
* **x-axis: calibration-in-the-large error of the shifted class** on the evaluation pool,
  from the scorer's chunk P(bot) as the policy sees it: bots `mean(1 − P(bot))`, humans
  `mean(P(bot))`. Pool AUC, Brier and accuracy at 0.5 are reported beside it
  (`drift_metrics.csv`). AUC is deliberately not the axis: same-source advanced bots keep
  AUC ≈ 0.998 while sitting at P(bot) 0.30–0.39 — ranking intact, calibration broken.
* **Offline panel.** Trained as the headline trains (deterministic world, oracle reward,
  200 episodes): DQN, PPO, Thompson, LinUCB, DQN without H-score (5 seeds each), Static
  Single/Multi — evaluated frozen at all 21 points.
* **Online panels.** E9's protocol in the grounded world: 300 episodes, the shift switched
  on at episode 100, evaluated at episode 300 at the same point (and at ρ = 0). Arms: oracle,
  posterior, posterior (floored), posterior+labels, posterior+labels (floored), score-only.
  **Control:** the same arm trained 300 episodes without the shift, evaluated at all points.
* **Primary contrast**, per (algorithm, arm, direction): mean DI over the 10 shifted levels,
  online − control, Welch over seeds, Holm over the 24 contrasts (`drift_contrasts.csv`).
  Per-level differences are descriptive.

## 5. Deliverables

Zip `results/round4/fold2/` **without** the `scorer/` folders and `ext_scores.pkl` (both
reproducible from git), plus the Slurm logs. In particular: `ppo/search_ppo.csv`,
`ppo/search_ppo_vs_dqn.csv`, `ppo/arch_choice.json`, `drift/drift_metrics.csv`,
`drift/drift_curves.csv`, `drift/drift_per_seed.csv`, `drift/drift_contrasts.csv`,
`drift/drift_vs_calibration.png`, both `eval_access.log`, `drift/drift/tasks/`.

**Findings rule (after round 3):** every number in a findings write-up must be copied from a
CSV in the deliverable, with the file named. Do not paraphrase a number from memory, and do
not report a pre-declared check as passing without the column that says so. Round 3's
write-up broke this and was replaced.
