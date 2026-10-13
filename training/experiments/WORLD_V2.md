# The v2 world — the default for all experiments from 2026-10-11

Decided by the user after the round-4 sensitivity check (`round4/sens_world.py`). Every new
experiment trains and evaluates in **`GROUNDED_V2` / `GROUNDED_V2_CFG`**
(`expkit/stochastic.py`). The round 3/4 world (`GROUNDED` / `GROUNDED_CFG`) is kept unchanged,
byte for byte, for reproducing earlier results and as a sensitivity analysis.

## What changed against the round 3/4 ("grounded") world

| | grounded (rounds 3–4) | v2 (default now) |
|---|---|---|
| Level 0 | no challenge | no challenge (unchanged) |
| Levels 1–9 | ladder in `LEVEL_MECHANISM`, `HUMAN_PASS`, `HUMAN_SECONDS`, IRT bots (α = 1.5) | unchanged |
| Level 10 | "audio / hard block", but humans passed 52 %, strong bots ~32 % | **hard block**: nobody passes; a human there is a false positive with no time spent |
| When humans abandon | **after passing**, P = 1 − exp(−seconds / 45 s) — no published source | **when a visible challenge is shown** (and again before the one retry), **never after passing** |
| Abandonment rate | function of seconds only | per challenge type × per position (below), from Searles et al. 2023 |
| Reward for an abandoning human | paid as a pass (+25 − friction) | **−100 − friction**, the false-positive penalty: DI counts both as a lost human |
| Bots | never abandon | never abandon (unchanged) |

## Abandonment parameters (Searles et al., USENIX Security 2023, Appendix A Tables 7–10)

**Per type** (`PRESENTATION_LEAVE`): abandons by captcha type summed over the study's four
groups ÷ ~456 exposures per type, × 1.34 (the contextualised setting's rate over the pooled rate,
120/2345 ÷ 174/4563 — a deployment embeds the CAPTCHA in a real task).

| Level | Mechanism | Searles type used | Rate |
|---|---|---|---|
| 0–2 | none / passive score / honeypot | invisible to a human | 0 |
| 3 | checkbox | reCAPTCHA easy (11/456) | 3.2 % |
| 4 | easy text | distorted-simple (9/456) | 2.6 % |
| 5 | standard text | distorted-masked (13/456) | 3.8 % |
| 6 | image grid | reCAPTCHA hard / hCAPTCHA (13/456) | 3.8 % |
| 7 | hard text | distorted-moving (16/456) | 4.7 % |
| 8 | multi-round | **assumption**: two rounds of level 7, 1 − (1 − 16/456)² | 9.2 % |
| 9 | interactive game | Arkose ×2 + Geetest, mean 28.3/456 | 8.3 % |
| 10 | hard block | — | 0 (blocked) |

**Per position** (`PRESENTATION_POSITION`): multiplier by how many visible challenges the user
has already been shown — ×3.25 (first), ×1.39, ×1.10, ×0.55 (fourth onward). From Searles'
contextualised per-position hazard (16.6 %, 7.1 %, 5.6 %, then 2–4 %) ÷ its mean (5.1 %),
smoothed. Abandonment is front-loaded; the data do **not** show it rising with repeated challenges.

## Assumptions to state in the paper

* Level 8 as two independent rounds of hard text (no multi-round type in Searles).
* Abandonment before a retry uses the same rate (Searles does not link abandonment to failures).
* The ×1.34 contextual scaling comes from one study; per-position cells are sparse (0–8 abandons
  per cell), hence the smoothing.
* The policy's state does not contain the number of challenges shown, on which the hazard
  depends (state v2's challenge counts are proxies).

## Verification

* The grounded world is unchanged: a fixed-seed episode gives identical outcomes and rewards
  before and after the change.
* `OutcomeModel` (the posterior reward's outcome model) mirrors v2 exactly, per level and per
  position: 40,000-draw simulation per (level, position), 0 mismatches.
* Sensitivity results (offline, 2 seeds): DQN/PPO DI within ~4 points between worlds at every
  drift point; the drift conclusions hold in both.
