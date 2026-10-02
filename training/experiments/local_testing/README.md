# Local testing — a single-seed preview of the GPU round

Self-contained. Nothing here feeds the main pipeline; it exists to see roughly what the
cluster run will say before a day of compute is spent on it.

```bash
python nb_local_01_train.py     # refit scorer, train every policy, evaluate  (~1-2 h CPU)
python nb_local_02_analysis.py  # tables + figures from the saved CSV          (seconds)
```

Notebook 01 appends to `results/local_runs.csv` after each policy and skips work already
recorded there, so it is restartable.

## What it does differently from the published design

**Traffic composition is realistic.** The published sweep fixes humans at 100 and varies
bots over `(0, 20, 100, 200, 500, 1000)` — bot fractions of 0–91 %, four of six cells at
or above 50 %. Published measurements put real traffic near 37 % bad bots
([Imperva 2025](https://www.imperva.com/resources/resource-library/reports/2025-bad-bot-report/))
and login endpoints near 43 % credential abuse
([Akamai](https://www.akamai.com/newsroom/press-release/state-of-the-internet-security-retail-attacks-and-api-traffic)).
So the grid here spans **5–75 % bots** with four of six cells at or below 43 %.

**Human population is drawn from a range** (50–250) rather than fixed, independently of
the bot fraction. `corr(total population, bot fraction)` drops from the published design's
**+1.000** to **+0.76**, and all six bot fractions occur at the same middle band of total
population — so population stops being a usable proxy for the attack level, by
construction rather than by ablation.

**No zero-bot cell**, so the `DI = 100 for any permissive policy` pathology cannot arise
and no exclusion rule is needed.

## Read it as a preview, not evidence

One training seed. Round 1's headline inversion came from exactly that. Read the pattern
across bot fractions — a policy that leads everywhere is saying something; a few points in
one cell is not.
