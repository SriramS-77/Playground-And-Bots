# Adaptive CAPTCHA Orchestration — reproduction package

Everything needed to regenerate Table 4 and the ablation figure of *Balancing Security
and Usability: Adaptive CAPTCHA Orchestration via Reinforcement Learning*, from the
checkpoints that produced the published numbers.

```bash
pip install -r requirements.txt
jupyter lab reproduce_table4.ipynb     # the whole paper, ~2 minutes
python validate_against_paper.py       # reproduced vs. published, side by side
```

## Layout

```
rlcaptcha/            the library — architecture and shared logic
  config.py           paths, simulation constants, the two reward presets
  data.py             session loading, chunking, the simulated population
  rewards.py          reward function (Section III-F-3)
  scoring.py          LSTM behavioural scorer + the score cache
  simulate.py         the evaluation loop (one function, all six policies)
  metrics.py          DI, BOS, SP-F1, SI-F1
  policies/
    base.py           the Policy interface — read this first
    dqn.py            DQN and its no-Humanity-Score ablation
    bandits.py        LinUCB and Thompson Sampling
    static.py         single- and multi-threshold baselines
checkpoints/          the exact weights behind Table 4
data/                 73 recorded sessions (32 human, 41 bot)
results/              published eval JSONs + regenerated figures
reproduce_table4.ipynb
validate_against_paper.py
```

## Checkpoints

| Paper column | Checkpoint | Came from |
|---|---|---|
| LinUCB | `checkpoints/linucb/best_model_1.pt` | `offline/other_models/linucb/models/4/` |
| Thompson Sampling | `checkpoints/thompson_sampling/best_model.pt` | `offline/other_models/thompson_sampling/models/` |
| DQN | `checkpoints/dqn/offline_rl_model_1_200.pt` | `offline/models/5/` |
| DQN without H-Score | `checkpoints/dqn_ablation/offline_rl_model_0_1373.pt` | `offline/models/ablation/3/` |
| Humanity Scorer | `checkpoints/humanity_scorer/big_model.keras` | `offline/scoring_service/models/` |

The static baselines have no checkpoint — they are rules over the behavioural score.

The DQN checkpoints still carry their training replay buffers, which is why they are
~5 MB; only `policy_net_state_dict` is read at evaluation.

## The score is a bot score, not a humanity score

The LSTM was trained with `human=0`, `bot=1`, so its output is **P(bot)**: high means
bot-like. The manuscript's Humanity Score *H* is `1 - score`. The original code named the
variable `humanity_score` while using it with bot semantics; the arithmetic was correct
throughout, only the name was wrong. This package calls it `bot_score`.

## Known inconsistencies in the published runs

These are faithfully reproduced rather than fixed, because fixing any of them changes
Table 4. They matter if you revise the paper.

1. **Bots abandon the site in the bandit runs.** In the LinUCB and Thompson Sampling
   notebooks the "user gave up" check is written outside the human/bot branch, so a bot
   shown threat level 8 is discarded with probability 0.75 *regardless of its strength*,
   on top of being blocked when `T > B_s`. The DQN, ablation and static runs only ever
   apply this to humans. This is the largest single driver of the bandits' bot-kill
   numbers at high load — with it LinUCB leaves 7 bots alive out of 500, without it ~31.
   **The two bandits were therefore not evaluated in the same environment as the other
   four policies.** Controlled by `RewardConfig.abandonment_applies_to_bots`.

2. **The overkill penalty differs.** 2.5 in the DQN/ablation/static runs (the value
   printed in the paper), 2.0 in the bandit runs.

3. **The bandits' `last_threat_level` feature is dead.** It is initialised to `-1` and
   never written back, so context index 3 is a constant for the whole run. The DQN and
   static loops initialise it to `0` and do update it. Both the constant and the value
   matter: at 500 bots the `-1` alone shifts many bot decisions from threat 9 to 10.

4. **The bandits act on exhausted users.** A user whose session data has run out still
   receives one final action in the bandit loops but not in the others, which shifts the
   per-step reward averages.

5. **The training scripts don't match the evaluation.** `offline/offline_training.z.py`
   and `offline_training.z.ablation.py` were last saved with `LEAKAGE_PENALTY = -200`
   (and the ablation with `FRICTION_BASE = 2.5`), while every benchmark — and the paper —
   uses `-150` and `2.0`. The checkpoints here are the published ones; retraining from
   the old scripts as they stand will not reproduce them.

Items 1–4 are encoded as explicit flags on `RewardConfig` and `Policy` so they are
visible rather than buried in a duplicated loop. To evaluate all six policies on genuinely
equal footing, run every one with `DQN_REWARDS` and set `initial_last_threat = 0` and
`updates_last_threat = True` on the bandits — the numbers will not match the paper.

## How well it reproduces

`validate_against_paper.py` over 8 seeds: the static baselines and the ablation land on
the published values; DQN and LinUCB are within roughly one bot volume's noise. Thompson
Sampling's published row is erratic across bot volumes (13, 31, 41 bots surviving at
200/500/1000) in a way a single stochastic run explains — its posterior sampling makes it
the highest-variance policy, and the reproduced means are smoother than the one published
draw. The paper's headline conclusions all hold: DQN has the best mean DI with the
smallest spread, multi-threshold beats single-threshold, and the bandits win at low bot
volumes but collapse under heavy load.

## The humanity-score cache

Evaluating six policies over six bot volumes needs ~80k LSTM inferences, which dominated
the original runtime. Almost all of it is redundant:

- There are only 73 recordings, so with 1100 simulated users each one is reused ~15 times.
- Chunk boundaries depend only on timestamps, which copying does not change — two users
  from the same recording see byte-identical chunks.
- The only thing separating two such users is `perturb_mouse_data`, which applies a
  single random offset in {-1, 0, +1} to x and y for the *whole* session: a one-pixel
  rigid translation, not per-point noise.
- `HumanityScorer.handle_data` adapts a `Normalization` layer and then returns the
  *unadapted* array, so the network sees raw pixel coordinates in the hundreds. A
  one-pixel shift is far below its sensitivity.

So the score is a function of `(recording, chunk index)`, and 876 entries cover
everything. Measured cost of the approximation (`verify_cache`): mean absolute score
difference ~6e-4, max ~6e-3, and zero flips across the 0.5 decision boundary.

Pass `ScoreCache(scorer, exact=True)` to disable the cache and take the original path,
per-user jitter included. The scorer itself is bit-identical to
`offline/scoring_service/HumanityScorer.py` — verified to 0.0 absolute difference.
