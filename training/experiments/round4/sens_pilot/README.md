# Round-4 sensitivity pilot (local laptop, 2026-10-10 .. 10-13)

`round4/sens_world.py`, rotation 0 rl_fit -> rl_val + external held-out halves; the eval block was
never opened. Scorer: cross-fit seed 1.
- `*_offline.csv`: offline-trained policies in both worlds (grounded, v2), seeds 0-1.
- `*_online.csv`: v2 world, online + control, rewards oracle / score-only / posterior. State v1:
  seeds 0-1; state v2: seed 0. The "posterior (fixed share)" runs were invalid (the patch was
  applied after the reward hook binds the function, so they trained with the plain posterior)
  and are excluded; the bug is fixed in `sens_world.py` and `r4_fold2.py`.
- `state_*_table.csv` / `state_*_drift.png`: seed-0 summaries made by `round4/sens_report.py`.
Pilot only: one seed and the validation block, so it is not paper evidence (see HANDOFF_ROUND5.md).
