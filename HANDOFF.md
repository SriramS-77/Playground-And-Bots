# Handoff — 2026-10-19

This file is the starting point for whoever continues this project on another account. It covers the paper's state, what code and results are in the repo, the experiment that is paused, and the open decisions.

## 1. State

- **Paper:** final. OJCS revision of *Balancing Security and Usability: Adaptive CAPTCHA Orchestration via RL*.
  - `latex_project/main.tex` and `main.pdf` (12 pages), plus `latex_project/response_to_reviewers.md`.
  - Commits `84e6751` and `51ea39a`.
  - **No further paper changes** unless the author asks.
  - **Value corrections (2026-10-19, after the final check against the CSVs).** Applied to `main.tex` and `response_to_reviewers.md`; the author mirrors them in Overleaf by hand.
    - §IV-C, frozen population input: unfrozen DQN is **97.0**, not 96.6 (`drift/extras_freeze.csv`, DQN unfrozen 96.95).
    - Fig. 5 text: the best static threshold that stops every bot without drift loses **4.3%** of humans, not 4.4% (`drift/pareto_points.csv`, raw threshold 0.7, human survival 0.9574).
    - §IV-D, ranking-destroying bots: "best static 3.8" was the *single* threshold. The best static policy is the multi-threshold at **18.1** (`drift/auc_drift/statics.csv`). It keeps that by losing 21% of humans and letting 61% of bots through. The sentence is reworded. "By challenging every user" for oracle DQN is replaced with "at the cost of 45% of humans" (human survival 0.551, challenge share 0.34).
- **Code:** every script behind the paper's numbers, plus the round-6 exploratory scripts, is committed under `training/experiments/` (this commit).
- **Paused experiment:** the no-abandonment rerun (§4). It is waiting for the author's go-ahead.

## 2. Ground rules the author has set

1. The paper reports **only the grounded stochastic world**:
   - stochastic bot solves (item-response model);
   - human pass rates and times from Bursztein 2010 and Searles 2023;
   - leave-after-pass with τ = 45 s.

   It uses **one scorer**: the cross-fit seed-1 refit (`round4/scorer_seed1/`). Choosing seed 1 was post hoc and is disclosed.
2. **Every number quoted must be checked against the CSVs.** Cluster write-ups have contained fabricated numbers before (round 3 and round 4 FINDINGS). Never trust a summary over its CSV.
3. The **embedding-state** (`emb_*`) and **audit** experiments beyond Tables 5–6 are exploratory. Keep them out of the paper unless asked.
4. **Fold 2 only.** The eval block is opened only by final-evaluation tasks, and every opening is logged to `eval_access.log`.
5. Commit and push only when the author asks.
6. **Terminology.** A human who passes a challenge and then leaves is "human abandonment during friction (a user capable of passing gives up)". The paper still uses other wordings in several places. These were listed for the author and deliberately left unchanged:
   - "abandonment from frustration";
   - "leave after a pass";
   - "no leaving".

## 3. Where things are

| What | Where |
|---|---|
| Experiment library | `training/experiments/expkit/`. `stochastic.py` holds the worlds, `simx.py` the simulator, `trainer.py` the RL learners and state, `emb_state.py` the opt-in embedding state. |
| Round-4 drift driver (all drift tasks) | `training/experiments/round4/r4_fold2.py`, documented in `HANDOFF_ROUND4.md` |
| Round-4 cluster outputs (Fig. 4 curves, drift metrics) | `training/experiments/round4/deliverables_fold2/` |
| Local round-6 outputs | `training/experiments/results/round4/fold2/drift/` |
| Paper figure scripts | `latex_project/figures_src/make_drift_figure.py`, `make_pareto_figure.py` |
| World v2 spec (used by round 5, **not** the paper) | `training/experiments/WORLD_V2.md` |
| Round 5 (v2 world, cluster) | `training/experiments/round5/r5_fold2.py`, `HANDOFF_ROUND5.md`. Only smoke runs exist locally; no full results. |

### Which script feeds which part of the paper

Outputs are under `results/round4/fold2/drift/` unless stated otherwise.

| Paper item | Script(s) in `training/experiments/round4/` | Output |
|---|---|---|
| Table 3 (web-bot benchmark) | `webbot_finetune.py` | `results/webbot/` |
| §IV-A scorer calibration | `scorer_eval.py`, `calib_study.py` | `scorer_eval.json`, `calib_*.csv` |
| Table 4 (held-out headline) | `grounded_headline.py`; NP static rows from `np_headline.py`, `np_multi_t4.py` | `headline_summary.csv`, `headline_*`, `headline/` (NP row: `headline/np_threshold_a0.10.csv`) |
| §IV-C simulator and population sensitivity | `grounded_extras.py` | `extras_alpha.csv`, `extras_tau.csv`, `extras_freeze.csv` |
| §IV-C reward constants | `grounded_reward_sweep.py` | `reward_sweep_summary.csv` |
| Fig. 4 (drift) | round-4 cluster run (`r4_fold2.py`); NP curve from `np_drift_curve.py`; audit markers from `np_audit.py` / `audit_curve.py` | `round4/deliverables_fold2/drift/drift_curves.csv`, `np_drift_curve.csv`, `np_audit/` |
| Fig. 5 (Pareto) | `pareto.py`, `np_half.py` | `pareto_points.csv`, `np_alpha_0005.csv` |
| Table 5 (online rewards; statics) | `np_audit.py`, `extra_seeds.py`; statics from `np_statics_table.py`, `multi_table5.py` | `np_audit/`, `np_statics_table5.csv`, `multi_table5.csv`, `single_table5.csv` |
| Table 6 (gated online training) | `gated.py`; detector thresholds from `detector_calib.py` | `gated/`, `gated_runs.csv`, `detector_thresholds.json` |
| Ranking-destroying covert bots | `auc_drift.py` | `auc_drift/` |
| TS posterior variants and LinUCB tuning (§III-E) | `ts_variants.py`, `linucb_tune.py` | `ts_variants/`, `linucb_tune.csv` |

**Exploratory, not in the paper:**
- `emb_prepare.py`, `emb_drift.py`, `emb_audit.py`
- `state_ablation.py`, `dr_oem.py`, `auc_sweep.py`, `guard_check.py`
- `posterior_statics.py`, `posterior_explain.py`, `figure_rl_vs_statics.py`

### What is committed and what is not

- **Committed:**
  - all code;
  - every summary CSV and JSON behind the paper;
  - the per-task CSVs for the paper's subdirectories;
  - the headline agents (`headline/agents/*.pt`, 7 MB);
  - `ext_scores.pkl`, the embedding tables and `eval_access.log`;
  - the round-4 cluster deliverables;
  - the paused rerun's complete state (§4).

  CSV, `.pt` and `.log` files are gitignored, so these were added with `git add -f`. Do the same for new results that must be kept.
- **Local only (not pushed, ~120 MB):**
  - the per-run CSVs of the exploratory studies (`emb_*`, `state_ablation`, `dr_oem`, `auc_sweep`, `posterior_statics*`);
  - the smoke directories;
  - the round-4 sensitivity pilots (`sens*`).

  They are still on the original machine under `training/experiments/results/round4/fold2/`.

## 4. Paused: the no-abandonment rerun

**Why.** Leave-after-pass is an assumption, not taken from the literature: no study measures people leaving *after* passing a CAPTCHA. With the default `abandonment_penalty=None`, an abandoning human earns the pass reward, while DI counts them as lost. This rerun turns leave-after-pass off to check whether the paper's conclusions depend on it.

**Not in the paper yet.** The author wants to review the results before deciding whether to drop leave-after-pass from the paper.

**Switch.** `RLC_ABANDON=none` (in `round3/common.py`) builds `GROUNDED_CFG` with `abandonment="none"`. The default is unchanged. Output goes to a separate root, so the paper's runs are untouched.

**Progress** (`training/experiments/results/round4_noabandon/fold2/`, committed):

- **Headline: done and collected** (`drift/headline_summary.csv`). The author has not reviewed it yet. Fixed-traffic DI:

  | Policy | DI |
  |---|---|
  | DQN | 99.3 |
  | PPO | 99.3 |
  | Static multi-threshold | 98.1 |
  | Static single threshold | 97.0 |
  | LinUCB | 93.3 |
  | Thompson Sampling | 87.6 |
  | DQN without H-score | 48.7 |

  This file has no static NP row. Before comparing it with Table 4's NP threshold (95.5), run `ROUND4_OUT=results/round4_noabandon/fold2 RLC_ABANDON=none python round4/np_headline.py`. It writes `headline/np_threshold_a0.10.csv`.
- **Drift: 42 of 417 tasks done.** It was stopped deliberately; the last log line, task 146, shows a kill exit code. Finished tasks have `drift/noabandon_logs/<i>.done` markers and their outputs in `drift/drift/tasks/`. The runner skips them.

**To resume**, from `training/experiments/`, with the repo's Python environment (`training/requirements.txt`):

```bash
python round4/noabandon_drift.py --list          # 417 tasks
python round4/noabandon_drift.py --run-all 10    # skips the 42 done; about 12-14 h at 10 jobs
```

**After it finishes**, collect with `ROUND4_OUT=results/round4_noabandon/fold2 RLC_ABANDON=none python round4/r4_fold2.py --stage drift --collect`. Then compare against `round4/deliverables_fold2/drift/` (the paper's runs).

**Fresh clone.** The scorer copy and `ppo/arch_choice.json` are already under `results/round4_noabandon/fold2/`, so no setup step is needed. If they are missing, run:

```bash
ROUND4_OUT=results/round4_noabandon/fold2 RLC_ABANDON=none python round4/r4_fold2.py --stage drift --setup
```

That copies the scorer from `round4/scorer_seed1/` and the PPO architecture choice from `<root>/ppo/arch_choice.json`. Copy the latter from `round4/deliverables_fold2/ppo/arch_choice.json` if needed.

**Scope.** These are the drift tasks the paper reports:
- metrics, statics, and the offline families (5 seeds);
- control and online training for the oracle, score-only and posterior rewards;
- covert bots at every ρ;
- bot-like humans at ρ 0.5 and 1.

The audit and gated arms are **not** included. Add them only if the author asks.

## 5. Open items waiting for the author

1. **Review the no-abandonment results** (headline now, drift when done), then decide whether the paper keeps leave-after-pass.
2. **Tier 1 screen-size rewording.** Offered, not decided.
3. **Ungated-baseline sentence in the gated paragraph (§IV-D).** Offered, not decided.
4. **Terminology** for abandonment during friction (§2.6). Listed, deliberately not changed.
5. **Paper text vs commented-out results.** The Limitations ("Rewards from observable outcomes alone fail") and the Conclusion ("no reward computed from observable outcomes alone adapted…") refer to the observable-outcome reward study. That subsection, "Rewards a Deployment Could Compute", is commented out in `main.tex`. The author said the paper is final, so this was left as is.
6. **Security: the git remote embeds a GitHub personal access token in plain text.** Revoke it, and use a credential helper or SSH on the new account.

## 6. Environment notes (Windows)

- **LaTeX:** build with `latexmk -g -pdf -interaction=nonstopmode main.tex` in `latex_project/`.
  - VS Code's LaTeX Workshop auto-build races with command-line builds and produces false errors. Wait a few seconds, then rebuild.
  - LaTeX Workshop 10.20 needs VS Code ≥ 1.114.
  - Citations are numbered by first appearance in uncommented text. The bibliography is kept in that order by hand.
- **Experiment runners** start one subprocess per task with `OMP_NUM_THREADS=1`. `--run-all J` sets the parallelism.
