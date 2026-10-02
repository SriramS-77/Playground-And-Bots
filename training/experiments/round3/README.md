# Round 3 experiment scripts — fold 2 only

Reference implementations of E2–E9 for `HANDOFF_ROUND3.md` §5. **Run them; do not rewrite
them.** If one fails, report the error — round 2's hand-written experiment scripts are
where its results went wrong (all of E2–E8 ran on a collapsed scorer, E4's wrapper never
acted, and none of the scripts reached the repository).

| script | experiment | tasks | needs first |
|---|---|---|---|
| `e2_leakage.py` | policy-level leakage, role reversal inside fold 2's rl block — **never opens the eval block** | 50 | `arch_choice.json` |
| `e3_reward_sensitivity.py` | 9 reward settings retrained × 5 seeds; fixed-policy landscape, one task per reward setting (rl block); abandonment curves | 69 | `arch_choice.json`; tasks 46–68 also the headline's `agents/` |
| `e4_population.py` | the population confound: grid-trained DQN, frozen / true / random `n_active` | 51 | `arch_choice.json`; **tasks 0–4 (training) before 5–50** |
| `e6_data_sufficiency.py` | recording-level bootstrap + per-policy variance decomposition; learning curve | 70 | `arch_choice.json`; tasks 0–49 also `agents/` |
| `e7_proxy_reward.py` | deployable reward: oracle, proxy, penalty / immediate-gap sweeps, PBRS, learned reward model, normalisation + clipping, 1000 episodes, posterior reward (± labels, ± floor), score-only ablation, reward-bias diagnostic | 93 | `arch_choice.json` |
| `e8_stochastic.py` | α sweep of the headline checkpoints; DQN retrained in the grounded world | 26 | `arch_choice.json`; tasks 0–20 also `agents/` |
| `e9_adversary.py` | online training against unseen users: oracle, proxy, posterior variants × {no injection, adversary bots, adversary humans, both}; scorer report on the external data | 141 | `arch_choice.json`; `data/external_sessions.json.gz` (`python -m expkit.external --check`) |

Every script has the same interface:

```bash
python round3/e7_proxy_reward.py --smoke     # toy budget on rl_fit / rl_val, then --collect
python round3/e7_proxy_reward.py --list      # tasks + the sbatch line
sbatch --array=0-92%25 slurm/r3_e.sh e7      # from training/experiments/
python round3/e7_proxy_reward.py --collect   # tables -> results/round3/fold2/e7/
```

What they share, in `common.py`, so no experiment can drift from the others:

* the cross-fitted scorers, through `exp_round3_fold2.context()`;
* the policy — fold 2's **DQN winner** from `arch_choice.json`, 5 seeds × 200 episodes,
  the headline's training sampler;
* the evaluation function, which reports **DI with the quantities DI hides**: human
  survival, bot survival, friction seconds, false-positive and abandonment rates, and the
  share of decisions that were challenges;
* the eval-block discipline: `pools(final=True, reason=...)` is the only way in, it logs
  one line to `eval_access.log`, and `--smoke` asserts the log did not change.

Each script's docstring states its design, the reviewer point it answers, and what it
deliberately does not do (E7 does not implement off-policy evaluation, and says why).
`--collect` computes every conclusion from the tables; nothing is asserted in prose.
