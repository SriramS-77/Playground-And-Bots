"""Reproduce Table 4 and compare against the published values."""
import sys, io, json, statistics as st
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from rlcaptcha import *

PAPER = {  # Table 4: policy -> {bots: (humans, bots)}
    "LinUCB":                 {0:(100,0),20:(96,2),100:(66,13),200:(35,3),500:(29,7),1000:(28,15)},
    "Thompson Sampling":      {0:(100,0),20:(100,1),100:(98,11),200:(22,13),500:(28,31),1000:(41,41)},
    "DQN":                    {0:(89,0),20:(94,4),100:(79,16),200:(81,19),500:(78,29),1000:(75,22)},
    "DQN without H-Score":    {0:(98,0),20:(98,13),100:(97,72),200:(96,116),500:(98,347),1000:(96,661)},
    "Static Single-Threshold":{0:(51,0),20:(58,1),100:(49,8),200:(53,15),500:(51,44),1000:(52,98)},
    "Static Multi-Threshold": {0:(69,0),20:(65,1),100:(69,15),200:(70,41),500:(68,79),1000:(65,170)},
}

humans, bots = load_sessions()
cache = ScoreCache(BotScorer()).precompute(humans, bots)

POLICIES = [
    (LinUCBPolicy(),           BANDIT_REWARDS),
    (ThompsonPolicy(),         BANDIT_REWARDS),
    (DQNPolicy(),              DQN_REWARDS),
    (DQNAblationPolicy(),      DQN_REWARDS),
    (SingleThresholdPolicy(),  DQN_REWARDS),
    (MultiThresholdPolicy(),   DQN_REWARDS),
]

N = 8
print(f"\n{'policy':26s} {'bots':>5}   {'paper':>9}   {'reproduced (mean of %d seeds)' % N}")
print("-" * 86)
for policy, rcfg in POLICIES:
    for nb in EVAL_BOT_COUNTS:
        runs = [run_simulation(policy, humans, bots, nb, cache=cache,
                               rewards=rcfg, seed=s) for s in range(N)]
        mh = st.mean(r.surviving_humans for r in runs)
        mb = st.mean(r.surviving_bots for r in runs)
        sh = st.pstdev([r.surviving_humans for r in runs])
        sb = st.pstdev([r.surviving_bots for r in runs])
        ph, pb = PAPER[policy.name][nb]
        ok = abs(mh-ph) <= max(2*sh,3) and abs(mb-pb) <= max(2*sb,3)
        print(f"{policy.name:26s} {nb:>5}   {ph:>4}/{pb:<4}   "
              f"{mh:5.1f}±{sh:<4.1f}/{mb:6.1f}±{sb:<5.1f}  {'ok' if ok else 'CHECK'}")
    print()
