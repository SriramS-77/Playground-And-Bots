"""An empirically grounded, probabilistic action model (Reviewer 1, point 1).

The published simulator blocks a bot iff ``T > B_s``. Level 10 therefore stops every bot
by construction and no advanced bot ever beats a hard challenge, so the experiment shows
that an agent can learn the simulator's rule rather than that it can orchestrate real
CAPTCHA services. This module replaces that rule with outcomes drawn from published
measurements.

HUMAN SIDE -- per-level pass probability and time cost
------------------------------------------------------
Anchored on two large studies:

* Bursztein et al., "How Good Are Humans at Solving CAPTCHAs? A Large Scale Evaluation",
  IEEE S&P 2010 -- 318k CAPTCHAs, 21 schemes. Image schemes: 87% mean solving accuracy
  (easiest authorize.net 98%, hardest mail.ru 70%); audio 52%. Mean solving times 6.8s
  (authorize.net) to 13.0s (Microsoft); audio 19-35s.
* Searles et al., "An Empirical Study & Evaluation of Modern CAPTCHAs", USENIX Security
  2023 -- 1,400 participants, 14,000 CAPTCHAs. reCAPTCHA checkbox median 3.7s;
  distorted text 9-15s; game-based (Arkose) 18-42s; abandonment 120% higher when the
  CAPTCHA is embedded in a realistic task rather than presented directly.

BOT SIDE -- a two-parameter logistic (IRT) solve model
------------------------------------------------------
``P(bot solves level T) = sigma(alpha * (theta(B_s) - d(T)))`` with ``theta(B_s) =
B_s + 0.5`` and ``d(T) = T``, for ``T >= 1``. This is the 2PL item-response model: bot
strength is latent ability, challenge level is item difficulty, ``alpha`` is
discrimination. Level 0 presents nothing, so every bot passes it with certainty.

The published deterministic rule is the ``alpha -> inf`` limit of exactly this model,
because ``theta > d`` iff ``B_s + 0.5 > T`` iff ``B_s >= T`` for integers. So
``SolveModel(deterministic=True)`` reproduces Table 4 and ``alpha`` sweeps continuously
away from it -- the comparison is a limit, not a different environment.

An optional ``solver_era`` shift reflects that contemporary solvers beat *legacy visual*
challenges far more easily than interactive or behavioural ones: YOLO-based solvers reach
100% on reCAPTCHAv2 image challenges (Plesner et al., 2024; previous SOTA 68-71%), while
a generalised agentic VLM solver manages 60.7% across 26 CAPTCHA types and 70.6% on
unseen challenges in the wild (Teoh et al., USENIX Security 2025). Default is off, so the
deterministic limit is preserved unless you ask for it.

BOT SIDE -- time spent per attempt
----------------------------------
Bots spend measured solve times (`BOT_SECONDS`, automated solvers; `SERVICE_BOT_SECONDS`
for human-powered solving services), lognormal around the median with log-sd 0.5. Until
round 3 they spent 0 s, so the observed time revealed the class perfectly: any challenge
answered in more than 0 s was a human's, and the proxy reward charged bots no friction --
paying a leaked bot MORE than a satisfied human. The oracle reward is unchanged: a bot's
time is not a usability cost, so it never enters `stochastic_reward`'s bot branch.

Humans still spend exactly the median per attempt (s_T, or 2 s_T after a retry). Against
continuous bot times an exact-value match on seconds would identify every human; nothing
in the pipeline does that (the proxy reward is linear in seconds, the learned reward model
is a linear logistic, the DQN state has no seconds), but a human-side spread would close it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

N_LEVELS = 11

#: Mechanism assumed at each threat level, for the paper's Table.
LEVEL_MECHANISM = (
    "no challenge",
    "passive risk score",
    "honeypot / JS challenge",
    "checkbox (reCAPTCHA v2 click)",
    "easy distorted text",
    "standard distorted text",
    "image grid selection",
    "hard distorted text",
    "very hard text / multi-round",
    "interactive game challenge",
    "audio fallback / hard block",
)

#: P(legitimate user completes the challenge), per level.
HUMAN_PASS = (1.000, 0.999, 0.995, 0.980, 0.980, 0.930, 0.860, 0.750, 0.700, 0.620, 0.520)

#: Median seconds of friction added, per level.
HUMAN_SECONDS = (0.0, 0.0, 0.5, 3.7, 6.8, 7.3, 9.7, 11.9, 13.0, 25.0, 30.0)

#: Median seconds a BOT spends on one attempt, per level, whether it then passes or fails
#: (Motoyama et al., USENIX Security 2010, found no time difference between correct and
#: incorrect answers). Attacker regime: AUTOMATED SOLVERS. Evidence per entry:
#:   L0   0     nothing is shown
#:   L1   0     passive score, nothing to interact with                          assumption
#:   L2   0.5   JS / honeypot runs in the bot's browser as in a human's          assumption
#:   L3   1.4   checkbox -- Sivakorn et al., EuroS&P 2016, ~2,500 solves/hour     low confidence:
#:              inverted throughput, not per-solve latency
#:   L4   1.0   easy text -- Motoyama et al. 2010 s3.1, Xrumer "a second or less"  measured
#:   L5   1.0   standard text -- modern-GPU assumption: Ye et al., CCS 2018, <50 ms
#:              inference only. The one end-to-end measurement, Gao et al., NDSS 2016
#:              (CPU era), is 2.8-13.2 s
#:   L6  14.9   image grid -- Hossen et al., RAID 2020 s5.3, reCAPTCHA v2, MEDIAN     measured
#:              (mean 19.9 s); HUMAN_SECONDS are medians too
#:   L7   7.0   hard text -- Motoyama s3.1, Xrumer on noisy phpBB, 6-7 s          measured
#:              (Gao et al.: 8.1 s on an older reCAPTCHA)
#:   L8  14.0   multi-round -- two rounds of L7                                  assumption
#:   L9  21.8   interactive -- Teoh et al., USENIX Security 2025, agentic VLM     measured, but
#:              solver, median over 26 challenge types incl. Arkose; sliders alone take
#:              2-5 s (Weng et al., Big Data Mining & Analytics 2019, Table 7)    not game-specific
#:   L10  5.4   audio -- Bock et al., WOOT 2017 s5.5, unCaptcha, MEAN (sd 1.25)    measured
#: Against these medians bots are much FASTER than humans at text and SLOWER at images,
#: so time is informative about class -- as it is in deployment -- but not decisive.
BOT_SECONDS = (0.0, 0.0, 0.5, 1.4, 1.0, 1.0, 14.9, 7.0, 14.0, 21.8, 5.4)

#: The alternative attacker regime, HUMAN-POWERED SOLVING SERVICES. Not used by default;
#: pass ``bot_seconds=SERVICE_BOT_SECONDS``. It flips the sign of the time signal on text.
#:   L4, L5, L7  14 s  -- Motoyama s5.6: median over 8 services (Q3 20 s, lognormal)  measured
#:   L6        73.1 s  -- Hossen et al. 2020 Table 9: 2Captcha mean on reCAPTCHA v2,  measured
#:                        the fastest of 5 services (73-131 s). Weng et al. 2019 report
#:                        4.3 s for the same service -- an unresolved conflict
#:   L8          28 s  -- two rounds of L7                                          assumption
#:   L9         9.7 s  -- Weng Table 7: GeeTest slider via hyocr (range 8.8-11.0 s)   measured
#:   L10         14 s  -- as text                                                   assumption
#:   L0-L3             -- as BOT_SECONDS: these need no human solver               assumption
SERVICE_BOT_SECONDS = (0.0, 0.0, 0.5, 1.4, 14.0, 14.0, 73.1, 14.0, 28.0, 9.7, 14.0)

#: Log-scale spread of a bot's time around the median. Motoyama's service latencies give
#: ln(Q3/median) / 0.674 = ln(20/14) / 0.674 = 0.53; Hossen's automated image solver gives
#: ln(p95/median) / 1.645 = ln(39.8/14.9) / 1.645 = 0.60; Bock's audio solver is narrower
#: (sd/mean 0.23). 0.5 sits between them.
BOT_SECONDS_SIGMA = 0.5

#: Optional per-level bonus to bot ability: legacy visual challenges are the ones
#: modern solvers have broken. Positive = easier for a bot than its level implies.
SOLVER_ERA_SHIFT = (0.0, 0.0, 0.2, 0.5, 1.5, 1.5, 1.2, 1.0, 0.6, 0.1, 0.0)


@dataclass(frozen=True)
class SolveModel:
    """Probabilistic outcomes for one challenge presentation."""

    alpha: float = 1.5                     # IRT discrimination; -> inf recovers the paper
    deterministic: bool = False            # True == the published T > B_s rule
    human_pass: tuple = HUMAN_PASS
    human_seconds: tuple = HUMAN_SECONDS
    solver_era: float = 0.0                # weight on SOLVER_ERA_SHIFT
    max_human_attempts: int = 2            # retries before a human is blocked
    human_skill_sd: float = 0.0            # per-user variation in pass probability (logit sd)
    bot_seconds: tuple = BOT_SECONDS       # median seconds per bot attempt
    bot_seconds_sigma: float = BOT_SECONDS_SIGMA   # lognormal spread; 0 = the median, no draw

    def bot_seconds_spent(self, threat: int, rng: random.Random) -> float:
        """Seconds a bot spends on one attempt at `threat`.

        Draws from `rng` only when there is something to draw: the median is positive and
        the spread is non-zero. The deterministic models set the spread to 0, so the
        published loop's RNG stream -- and the 144/144 reproduction -- is untouched.
        """
        median = self.bot_seconds[threat]
        if median <= 0.0 or self.bot_seconds_sigma <= 0.0:
            return median
        return median * rng.lognormvariate(0.0, self.bot_seconds_sigma)

    def bot_pass_probability(self, bot_strength: int, threat: int) -> float:
        """P(bot gets through when shown `threat`)."""
        if threat == 0:
            # Level 0 shows nothing, so nothing can be failed. The logistic alone gives
            # sigma(alpha * 0.5) = 0.68 for B_s = 0 at alpha 1.5: "no challenge" blocked
            # 32 % of the weakest bots per step and ~20 % of all bots over 12 steps of
            # never challenging. The deterministic rule already passes everyone here.
            return 1.0
        if self.deterministic:
            return 0.0 if threat > bot_strength else 1.0
        theta = bot_strength + 0.5 + self.solver_era * SOLVER_ERA_SHIFT[threat]
        return 1.0 / (1.0 + math.exp(-self.alpha * (theta - threat)))

    def human_pass_probability(self, threat: int, skill: float = 0.0) -> float:
        """P(legitimate user completes `threat`, single attempt).

        `skill` shifts the logit, giving a population that is not uniformly average.
        """
        p = self.human_pass[threat]
        if skill == 0.0 or p >= 1.0 or p <= 0.0:
            return p
        logit = math.log(p / (1 - p)) + skill
        return 1.0 / (1.0 + math.exp(-logit))

    def human_outcome(self, threat: int, rng: random.Random, skill: float = 0.0):
        """Returns (passed, attempts_used, seconds_spent)."""
        if threat == 0:
            return True, 0, 0.0
        p = self.human_pass_probability(threat, skill)
        if p >= 1.0:
            # A user who always passes needs no coin flip. Skipping the draw keeps the
            # RNG stream identical to the published loop, which never sampled here.
            return True, 1, self.human_seconds[threat]
        seconds = 0.0
        for attempt in range(1, self.max_human_attempts + 1):
            seconds += self.human_seconds[threat]
            if rng.random() < p:
                return True, attempt, seconds
        return False, self.max_human_attempts, seconds

    def describe(self):
        """A DataFrame of the level ladder, for the revised manuscript."""
        import pandas as pd

        return pd.DataFrame({
            "Level": range(N_LEVELS),
            "Mechanism": LEVEL_MECHANISM,
            "P(human passes)": self.human_pass,
            "Median seconds": self.human_seconds,
            "Bot median seconds": self.bot_seconds,
            "P(bot B_s=2 passes)": [round(self.bot_pass_probability(2, t), 3) for t in range(N_LEVELS)],
            "P(bot B_s=5 passes)": [round(self.bot_pass_probability(5, t), 3) for t in range(N_LEVELS)],
            "P(bot B_s=9 passes)": [round(self.bot_pass_probability(9, t), 3) for t in range(N_LEVELS)],
        })


#: The published environment: bots blocked iff T > B_s, and humans ALWAYS complete the
#: challenge -- the manuscript has no notion of a legitimate user failing one. Bot time is
#: the median with no spread: a draw would shift the published loop's RNG stream.
DETERMINISTIC = SolveModel(deterministic=True, human_pass=(1.0,) * N_LEVELS,
                           bot_seconds_sigma=0.0)

#: Deterministic bots, but humans fail at the measured rates. Isolates the effect of
#: false positives alone, without changing anything on the bot side.
DETERMINISTIC_FALLIBLE_HUMANS = SolveModel(deterministic=True, bot_seconds_sigma=0.0)

GROUNDED = SolveModel(alpha=1.5)
GROUNDED_MODERN = SolveModel(alpha=1.5, solver_era=1.0)


@dataclass(frozen=True)
class StochasticRewardConfig:
    """Reward parameters for the probabilistic environment.

    Keeps the manuscript's constants where they still apply, and adds the two terms the
    deterministic model had no way to express: the cost of blocking a legitimate user
    (a false positive), and friction priced in seconds rather than as 2^T.
    """

    max_reward: float = 50.0
    leakage_penalty: float = -150.0
    underestimation: float = 5.0
    overkill: float = 2.5
    false_positive_penalty: float = -100.0   # legitimate user failed the challenge
    friction_per_second: float = 2.0         # usability cost, priced in measured seconds
    friction_base: float | None = None       # set to 2.0 to keep the paper's 2^T instead
    abandonment: str = "empirical"           # "empirical" | "paper" | "none"
    abandonment_tau: float = 22.0            # seconds; P_leave = 1 - exp(-s/tau)
    abandonment_applies_to_bots: bool = False

    def friction(self, threat: int, seconds: float) -> float:
        if self.friction_base is not None:
            return 0.0 if threat == 0 else math.pow(self.friction_base, threat)
        return self.friction_per_second * seconds

    def leave_probability(self, threat: int, seconds: float) -> float:
        if self.abandonment == "none":
            return 0.0
        if self.abandonment == "paper":
            if threat == 10:
                return 1.0
            if threat <= 6:
                return 0.0
            return max(0.0, 1.0 - 0.5 / (threat - 6))
        return 1.0 - math.exp(-seconds / self.abandonment_tau)

    def sample_leave(self, threat: int, seconds: float, rng: random.Random) -> bool:
        """Draw an abandonment event.

        The "paper" branch reproduces the original expression verbatim --
        ``threat == 10 or random() * (threat - 6) > 0.5`` -- including its short circuit
        at level 10, where the published code never draws. Matching the draw pattern (not
        just the probability) is what makes PAPER_EQUIVALENT bit-identical to
        `rlcaptcha.simulate.run_simulation` rather than merely equal in distribution.
        """
        if self.abandonment == "none":
            return False
        if self.abandonment == "paper":
            return threat == 10 or rng.random() * (threat - 6) > 0.5
        return rng.random() < self.leave_probability(threat, seconds)


def stochastic_reward(user, threat: int, solve: SolveModel,
                      cfg: StochasticRewardConfig, rng: random.Random):
    """(reward, terminated, info) for presenting `threat` to `user`.

    `info` records what actually happened, so the notebooks can report false-positive
    rates and leak rates rather than only survivor counts.
    """
    info = {"threat": threat, "is_bot": user.is_bot, "outcome": None, "seconds": 0.0}

    if user.is_bot:
        p = solve.bot_pass_probability(user.bot_strength, threat)
        # A certain outcome needs no draw. Skipping it keeps the RNG stream aligned with
        # the published loop, which resolved `T > B_s` by comparison rather than sampling.
        passed = p >= 1.0 or (p > 0.0 and rng.random() < p)
        if passed:                                 # solved it -> leaked
            gap = max(0, user.bot_strength - threat)
            r = cfg.leakage_penalty - gap * cfg.underestimation
            user.captchas_solved += 1
            info["outcome"] = "bot_leaked"
            terminated = False
        else:                                      # blocked
            overkill = max(0, threat - user.bot_strength)
            r = cfg.max_reward - overkill * cfg.overkill
            user.end_session()
            info["outcome"] = "bot_blocked"
            terminated = True
        # Time spent on the attempt, passed or failed. Observable, so it reaches the proxy
        # reward; never part of this (oracle) reward -- a bot's time is not a usability cost.
        info["seconds"] = solve.bot_seconds_spent(threat, rng)

        if cfg.abandonment_applies_to_bots:
            # Drawn even when the bot was already blocked: the published bandit loops
            # evaluate this check unconditionally, so skipping the draw would desync the
            # RNG stream even though the outcome is unchanged.
            left = cfg.sample_leave(threat, solve.human_seconds[threat], rng)
            terminated = terminated or left
        return r, terminated, info

    # --- legitimate user -------------------------------------------------- #
    skill = getattr(user, "skill", 0.0)
    passed, attempts, seconds = solve.human_outcome(threat, rng, skill)
    info["seconds"] = seconds
    friction = cfg.friction(threat, seconds)

    if not passed:
        r = cfg.false_positive_penalty - friction
        user.end_session()
        info["outcome"] = "human_false_positive"
        return r, True, info

    r = cfg.max_reward if threat == 0 else (cfg.max_reward / 2.0 - friction)
    user.captchas_solved += 1
    info["outcome"] = "human_passed"

    terminated = cfg.sample_leave(threat, seconds, rng)
    if terminated:
        info["outcome"] = "human_abandoned"
    return r, terminated, info


#: Reproduces the published environment exactly when paired with DETERMINISTIC.
PAPER_EQUIVALENT = StochasticRewardConfig(
    friction_base=2.0,
    abandonment="paper",
    false_positive_penalty=0.0,   # cannot occur: humans always pass in the paper's model
)
