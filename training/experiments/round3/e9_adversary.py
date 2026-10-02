"""E9 -- online training against UNSEEN users: adversary bots, adversary humans, both.
(Reviewer 1, point 1 continued; HANDOFF_ROUND3 §4.8.)

Can a reward built without ground truth keep a policy working when the population changes to
users the humanity scorer was never trained or calibrated on? Policies train on fold 2's rl
block; at episode INJECT_EP unseen sessions join the playback; training continues to TOTAL.

    adversary bots     web-bot phase-2 ADVANCED bots (Iliou et al. 2021)
    adversary humans   Balabit remote-desktop users (Fulop et al. 2016) -- humans only
    both               the two together
    no injection       the CONTROL: the same arm trained to TOTAL on the original pool. The
                       200 episodes after the switch are also simply more training, so
                       adaptation = injected run minus this control, never episode 300 minus 100.

Moderate web-bots are NOT injected: the scorer is inverted on them (same-source AUC 0.05 in
the local run). They stay in the scorer report so the reason is visible. Data come from
`expkit.external` (cache pinned by SHA-256, re-chunked with rlcaptcha's own code) and are
scored at run time by THIS fold's cross-fit refit -- never retrained, never shipped scores.

PRE-DECLARED (fixed before any run; do not revise):
    TOTAL 300 episodes, INJECT_EP 100, 5 seeds; the DQN winner from arch_choice.json
    pools after injection (replication keeps each class roughly half seen, half unseen):
        bots    rl bots x BOT_COPIES (2)   + web-bot advanced INJECT (60)        ~55 / 45
        humans  rl humans x HUMAN_COPIES (10) + Balabit INJECT users 1-5 (150)    50 / 50
    curve checkpoints CHECKPOINTS, each on every evaluation condition, 100 humans, bots
    CURVE_VOLUMES, CURVE_SEEDS seeds; at episode 300 the full evaluation (SCORED_VOLUMES, 20
    seeds) on every evaluation condition:
        seen              eval-block humans + eval-block bots
        adversary bots    eval-block humans + advanced HELD-OUT bots
        adversary humans  Balabit HELD-OUT users 6-10 + eval-block bots
        adversary both    Balabit HELD-OUT + advanced HELD-OUT
        control humans    web-bot HELD-OUT humans + eval-block bots (same logger as the bots)
    adversary bots draw B_s ~ U{0..9} like our own -- strength is a simulator construct
    checks at episode 300, injected minus no-injection control of the same arm, seed means:
      K1  target condition improves: adversary bots -> bots left down >= 10 points on
          "adversary bots"; adversary humans -> humans kept up >= 10 points on "adversary
          humans"; both -> DI up >= 5 on "adversary both"
      K2  no harm on "seen": humans kept >= control - 5 points and bots left <= control + 5
    The oracle is the reference, with a caveat measured locally: after injection it let ~40 %
    of OUR bots through, and observable-outcome replay buckets did not fix it. Report every
    oracle-relative statement with that caveat.

Smoke mode evaluates on rotation 0's rl_val in place of the eval block (never opens it).
"""

from __future__ import annotations

import random
import time
from copy import copy

import numpy as np
import pandas as pd

import common as C
import e7_proxy_reward as E
import expkit.trainer as T
from exp_policy_arch_search import SCORED_VOLUMES

EXP = "e9"
TOTAL, INJECT_EP = 300, 100
CHECKPOINTS = (0, 50, 100, 105, 110, 125, 150, 200, 250)
CURVE_VOLUMES, CURVE_SEEDS = (20, 100, 500), 3
HUMAN_COPIES, BOT_COPIES = 10, 2
INJECTIONS = ("no injection", "adversary bots", "adversary humans", "adversary both")
EVALS = ("seen", "adversary bots", "adversary humans", "adversary both", "control humans")
ARMS = {"oracle": {"reward_source": "oracle"},
        **{k: E.CONFIGS[k] for k in ("proxy", "posterior", "posterior+labels", "score-only",
                                      "posterior (floored)", "posterior+labels (floored)")}}
SMOKE_TOTAL, SMOKE_INJECT, SMOKE_CHECKPOINTS = 4, 2, (0, 2)


def tasks():
    t = [{"kind": "scorer_report"}]
    return t + [{"kind": "train", "arm": a, "injection": c, "seed": s}
                for c in INJECTIONS for a in ARMS for s in range(C.SEEDS)]


def smoke_tasks():
    # Matched injected / no-injection pairs, so the adaptation-effect path is exercised too.
    pairs = (("oracle", "adversary both"), ("posterior (floored)", "adversary humans"),
             ("posterior+labels (floored)", "adversary bots"))
    return [{"kind": "scorer_report"}] + [
        {"kind": "train", "arm": a, "injection": c, "seed": 0}
        for arm, inj in pairs for a, c in ((arm, inj), (arm, "no injection"))]


# --------------------------------------------------------------------------- #

def pools(smoke, reason):
    """Training / evaluation pools plus the external sets, all in one score cache."""
    from expkit.crossfit import _score_sessions
    from expkit.external import load_sessions

    p = C.pools(final=not smoke, smoke=smoke, reason=reason)
    ext = load_sessions()
    used = [s for k, v in ext.items() for s in v]
    p.cache._table.update(_score_sessions(p.ctx.crossfit.refit, used))   # this fold's refit only
    p.ext = ext
    p.evals = {"seen": (p.ev_h, p.ev_b),
               "adversary bots": (p.ev_h, ext["webbot_advanced_heldout"]),
               "adversary humans": (ext["balabit_heldout"], p.ev_b),
               "adversary both": (ext["balabit_heldout"], ext["webbot_advanced_heldout"]),
               "control humans": (ext["webbot_humans_heldout"], p.ev_b)}
    return p


def injected(p, injection):
    h, b = list(p.train_h), list(p.train_b)
    if injection in ("adversary humans", "adversary both"):
        h = h * HUMAN_COPIES + p.ext["balabit_inject"]
    if injection in ("adversary bots", "adversary both"):
        b = b * BOT_COPIES + p.ext["webbot_advanced_inject"]
    return h, b


def curve_point(agent, p, episode, tags):
    eps, agent.epsilon = agent.epsilon, 0.0
    rows = []
    for name, (h, b) in p.evals.items():
        for nb in CURVE_VOLUMES:
            for s in range(CURVE_SEEDS):
                res, _ = C.run_x(agent, h, b, nb, cache=p.cache, n_humans=100, seed=s,
                                 solve=C.GROUNDED, cfg=C.GROUNDED_CFG)
                rows.append({**tags, "episode": episode, "eval_condition": name,
                             "human_survival": res.surviving_humans / 100,
                             "bot_survival": res.surviving_bots / nb,
                             "friction_s": res.mean_human_friction_seconds})
    agent.epsilon = eps
    return rows


def run_train(task, smoke):
    from expkit.posterior_reward import OutcomeModel, em_share

    arm, injection, seed = task["arm"], task["injection"], task["seed"]
    total, inject, cps = (SMOKE_TOTAL, SMOKE_INJECT, SMOKE_CHECKPOINTS) if smoke else (TOTAL, INJECT_EP, CHECKPOINTS)
    _, eval_seeds, volumes = C.budget(smoke)
    if not smoke:
        volumes = SCORED_VOLUMES
    p = pools(smoke, f"E9 {arm} / {injection} seed {seed}")
    after_h, after_b = injected(p, injection)
    spec = dict(ARMS[arm])
    hook = spec.pop("hook", None)
    spec.pop("episodes", None)
    om = calib = None
    if hook in E.POSTERIOR_HOOKS:
        om = OutcomeModel(C.GROUNDED, C.GROUNDED_CFG)
        calib = E.calibration_for(hook, E.fit_calibration(p) if hook != "score_only" else None)
    tags = {"arm": arm, "injection": injection, "train_seed": seed}
    st, curve, diag = {"ep": 0}, [], []
    real_run_x = T.run_x

    def run_x_hooked(policy, humans, bots, n_bots, **kw):
        if kw.get("collect") and humans is p.train_h:        # a training episode (checked first)
            if st["ep"] in cps:
                curve.extend(curve_point(policy, p, st["ep"], tags))
            if st["ep"] >= inject:
                humans, bots = after_h, after_b
            out = real_run_x(policy, humans, bots, n_bots, **kw)
            st["ep"] += 1
            return out
        return real_run_x(policy, humans, bots, n_bots, **kw)

    t0 = time.time()
    with E.hooked(hook, spec.get("proxy_cfg"), None, om=om, calib=calib):
        if hook in E.POSTERIOR_HOOKS and calib is not None:
            inner = T.relabel

            def relabel_diag(transitions, cfg, rng):           # diagnostic only, never the reward
                oracle_r = np.array([t.reward for t in transitions])
                out, n = inner(transitions, cfg, rng)
                first = {}
                for t in transitions:
                    first.setdefault(t.user_id, t)
                f = list(first.values())
                pi = em_share(calib.log_lr([t.state[0] for t in f], [t.state[1] for t in f],
                                           [t.step for t in f]), pi0=calib.pi_cal)
                diag.append({**tags, "episode": st["ep"] - 1, "em_share": pi,
                             "true_share": float(np.mean([t.is_bot for t in f])),
                             "reward_bias": float(np.mean(np.array([t.reward for t in out]) - oracle_r))})
                return out, n
            T.relabel = relabel_diag
        T.run_x = run_x_hooked
        try:
            agent, log = C.train_winner(p.train_h, p.train_b, p.cache, seed=seed, episodes=total,
                                        smoke=smoke, solve=C.GROUNDED, cfg=C.GROUNDED_CFG, **spec)
        finally:
            T.run_x = real_run_x
    agent.eval_mode()
    curve.extend(curve_point(agent, p, total, tags))
    frames = [C.evaluate(agent, arm, h, b, p.cache, eval_seeds, volumes, solve=C.GROUNDED,
                         cfg=C.GROUNDED_CFG, eval_condition=name, **tags)
              for name, (h, b) in p.evals.items()]
    name = f"{arm}__{injection}__s{seed}".replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    C.save(pd.concat(frames, ignore_index=True), EXP, name, smoke)
    d = C.out_dir(EXP, smoke)
    (d / "curves").mkdir(exist_ok=True)
    pd.DataFrame(curve).to_csv(d / "curves" / f"{name}.csv", index=False)
    if diag:
        (d / "diag").mkdir(exist_ok=True)
        pd.DataFrame(diag).to_csv(d / "diag" / f"{name}.csv", index=False)
    print(f"  {arm} / {injection} / s{seed}: {(time.time() - t0) / 60:.1f} min")


def run_scorer_report(smoke):
    """The scorer, never retrained, on unseen data: session-level mean chunk P(bot)."""
    from sklearn.metrics import roc_auc_score

    p = pools(smoke, "E9 scorer report")
    x = p.ext
    groups = {"eval humans": p.ev_h, "eval bots": p.ev_b,
              "Balabit humans": x["balabit_inject"] + x["balabit_heldout"],
              "web-bot humans (control)": x["webbot_humans_inject"] + x["webbot_humans_heldout"],
              "web-bot advanced bots": x["webbot_advanced_inject"] + x["webbot_advanced_heldout"],
              "web-bot moderate bots (not injected)": x["webbot_moderate_inject"] + x["webbot_moderate_heldout"]}

    def score(s):
        v = [p.cache._table.get((s.key, i)) for i in range(len(s.chunks))]
        v = [z for z in v if z is not None]
        return float(np.mean(v)) if v else np.nan
    sc = {g: np.array([score(s) for s in v]) for g, v in groups.items()}
    rng = np.random.default_rng(0)

    def ci(fn, *arrs, n=1000):
        vals = [fn(*[a[rng.integers(0, len(a), len(a))] for a in arrs]) for _ in range(n)]
        return tuple(np.percentile(vals, [2.5, 97.5]))
    rows = []
    for g, v in sc.items():
        bot = "bots" in g
        ok = (lambda a: float(np.mean(a >= 0.5))) if bot else (lambda a: float(np.mean(a < 0.5)))
        lo, hi = ci(ok, v)
        rows.append({"group": g, "sessions": len(v), "mean_score": float(np.nanmean(v)),
                     "classified_correctly": ok(v), "ci_lo": lo, "ci_hi": hi})
    pairs = [("eval humans", "eval bots"), ("web-bot humans (control)", "web-bot advanced bots"),
             ("web-bot humans (control)", "web-bot moderate bots (not injected)"),
             ("Balabit humans", "web-bot advanced bots"), ("Balabit humans", "eval bots"),
             ("eval humans", "web-bot advanced bots")]
    auc = lambda h, b: roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(b))], np.r_[h, b])
    aucs = [{"humans": h, "bots": b, "AUC": auc(sc[h], sc[b]),
             **dict(zip(("ci_lo", "ci_hi"), ci(auc, sc[h], sc[b], n=500)))} for h, b in pairs]
    d = C.out_dir(EXP, smoke)
    pd.DataFrame(rows).to_csv(d / "e9_scorer_external.csv", index=False)
    pd.DataFrame(aucs).to_csv(d / "e9_scorer_auc.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(aucs).round(3).to_string(index=False))


def run_task(task, smoke):
    if task["kind"] == "scorer_report":
        return run_scorer_report(smoke)
    return run_train(task, smoke)


# --------------------------------------------------------------------------- #

def collect(smoke):
    from scipy.stats import ttest_ind

    import preflight
    d = C.out_dir(EXP, smoke)
    runs = preflight.headline(C.load_tasks(EXP, smoke))           # bots > 0 cells
    cols = ["human_survival", "bot_survival", "friction_s", "DI"]
    per_seed = runs.groupby(["injection", "arm", "eval_condition", "train_seed"])[cols].mean()
    t = per_seed.groupby(["injection", "arm", "eval_condition"]).mean()
    t[[c + "_se" for c in cols]] = per_seed.groupby(["injection", "arm", "eval_condition"]).sem().to_numpy()
    t.to_csv(d / "e9_final.csv")
    pd.set_option("display.width", 240)
    print(t[cols].round(3).unstack("eval_condition").to_string(), "\n")

    rows = []
    for inj in INJECTIONS[1:]:
        for arm in ARMS:
            for ev in EVALS:
                try:
                    a = per_seed.loc[(inj, arm, ev)]
                    b = per_seed.loc[("no injection", arm, ev)]
                except KeyError:
                    continue
                r = {"injection": inj, "arm": arm, "eval_condition": ev}
                for c in cols:
                    r[f"d_{c}"] = a[c].mean() - b[c].mean()
                    r[f"p_{c}"] = ttest_ind(a[c], b[c], equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else np.nan
                rows.append(r)
    eff = pd.DataFrame(rows, columns=["injection", "arm", "eval_condition"]
                       + [f"{k}_{c}" for c in cols for k in ("d", "p")])
    eff.to_csv(d / "e9_adaptation_effect.csv", index=False)
    print("adaptation effect = injected - no-injection control, episode 300 (Welch p over seeds):")
    print(eff.round(3).to_string(index=False), "\n")

    print("pre-declared checks:")
    target = {"adversary bots": ("adversary bots", "d_bot_survival", lambda v: v <= -0.10),
              "adversary humans": ("adversary humans", "d_human_survival", lambda v: v >= 0.10),
              "adversary both": ("adversary both", "d_DI", lambda v: v >= 5)}
    for inj, (ev, col, ok) in target.items():
        for arm in ARMS:
            e = eff[(eff.injection == inj) & (eff.arm == arm)].set_index("eval_condition")
            if ev not in e.index or "seen" not in e.index:
                continue
            k1 = ok(e.loc[ev, col])
            k2 = e.loc["seen", "d_human_survival"] >= -0.05 and e.loc["seen", "d_bot_survival"] <= 0.05
            print(f"  {inj:17s} {arm:28s} K1 {'PASS' if k1 else 'FAIL'} ({col} {e.loc[ev, col]:+.3f})  "
                  f"K2 {'PASS' if k2 else 'FAIL'}")
    for name in ("e9_scorer_external.csv", "e9_scorer_auc.csv"):
        if (d / name).exists():
            print(f"\n{name}:\n" + pd.read_csv(d / name).round(3).to_string(index=False))

    cf = sorted((d / "curves").glob("*.csv"))
    if cf:
        cur = pd.concat([pd.read_csv(f, dtype={"arm": str, "injection": str, "eval_condition": str})
                         for f in cf], ignore_index=True)
        cur.groupby(["injection", "arm", "eval_condition", "episode"])[cols[:3]].mean().to_csv(d / "e9_curves.csv")
    df = sorted((d / "diag").glob("*.csv"))
    if df:
        dg = pd.concat([pd.read_csv(f, dtype={"arm": str, "injection": str}) for f in df], ignore_index=True)
        dg["phase"] = np.where(dg.episode >= (SMOKE_INJECT if smoke else INJECT_EP), "after", "before")
        dg["share_err"] = (dg.em_share - dg.true_share).abs()
        s = dg.groupby(["injection", "arm", "phase"])[["share_err", "reward_bias"]].mean()
        s.to_csv(d / "e9_reward_diagnostics.csv")
        print("\nreward diagnostics (posterior arms): |EM share - truth| and mean(r_hat - oracle r):")
        print(s.round(3).unstack("phase").to_string())


if __name__ == "__main__":
    C.cli(EXP, tasks, run_task, collect, smoke_tasks)
