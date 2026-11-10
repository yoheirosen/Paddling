"""Reading effort zones live in training, from heart rate and from GPS pace, with what each
arm decides about the day: the training-session check of the personal heart-rate model.

A paddler with a GPS speed coach and a heart-rate strap (no paddle sensor) trains through a
block (:mod:`.sim.training`).  Every 30 s a fast reader names the effort zone of the last minute
from two routes, following a **guide** that each arm sets:

* the **heart-rate route** reads the settled heart rate through the person's own model
  (:mod:`.heart_rate`, fitted on the calibration day), minus the guide's offset for the day;
  above the effort where heart rate stops telling efforts apart it says it cannot tell;
* the **pace route** reads the power reaching the boat (the boat computer, :class:`.race.RaceState`:
  GPS speed, its live current estimate and the hull model) as a share of the calibrated critical
  power times the guide's ``cp_scale`` for today;
* the guide says how to combine them, and whether to drop heart rates that cannot be real (below
  75 or locked to the stroke rate).

Arms: ``textbook`` (share of the maximum heart rate), ``hr_model`` (the personal model, no day
adjustment), ``pace`` (pace only), ``warmup_rule`` (the strongest fixed rule: the day's offset
from the warm-up, the plausibility check, and both routes must agree within a zone), and the
reasoning agent (:mod:`.hr_agent`).  Scored against the simulator's truth: the share of today's
critical power over the last 30 s, at checkpoints where the planned effort has been steady for
two minutes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from agent_lib.runtime import record
from agent_lib.runtime.clock import VirtualClock
from agent_lib.runtime.scheduler import Scheduler, fixed

from . import sessions as ss
from .heart_rate import EFFORT_ZONES, TEXTBOOK_ZONES, PersonalHeartRate, fit_personal_model, zone_of_share
from .race import Athlete, RaceState, calibrate
from .sim.training import SESSION_DISTANCE_M, SessionSpec, StrapFault, TrainingBlock, session_twin
from .sim.twin import PaddlerTwin, RaceSetup, STEP_TEST_SHARES

__all__ = ["Calibration", "calibrate_block", "Guide", "read_zone", "SessionCtx", "run_session", "run_block",
           "GuideArm", "TextbookArm", "HrModelArm", "PaceArm", "WarmupRuleArm", "score_session", "STEADY_S",
           "READ_EVERY_S"]

READ_EVERY_S = 30.0
STEADY_S = 120.0
ZONE_NAMES = [z[0] for z in EFFORT_ZONES]


# ── calibration day ─────────────────────────────────────────────────────────

@dataclass
class Calibration:
    model: PersonalHeartRate        # effort axis: power reaching the boat (W)
    athlete: Athlete                # critical power and reserve, power reaching the boat
    hr_max_seen: float
    separates_up_to: float
    summary: Dict[str, Any] = field(default_factory=dict)


def _test_twin(block: TrainingBlock, plan, seed: int) -> PaddlerTwin:
    tw = PaddlerTwin(block.traits, RaceSetup(current_mps=0.0, plan=plan), seed=seed)
    tw.enable_heart_rate(profiles=True)
    tw._hr_par = dict(block.hr_par)
    hp = tw._hr_par
    tw.hr = hp.get("rest", hp["low"]) if hp.get("shape") == "compressed" else hp["low"]
    return tw


def _max30(rows: List[Dict[str, Any]]) -> float:
    hr = np.array([r["heart_rate_bpm"] for r in rows if r.get("heart_rate_bpm") is not None], float)
    t = np.array([r["t"] for r in rows if r.get("heart_rate_bpm") is not None], float)
    best = float("nan")
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < 30.0:
            j += 1
        if j - i >= 5:
            m = float(hr[i:j].mean())
            best = m if not math.isfinite(best) else max(best, m)
    return best


def calibrate_block(block: TrainingBlock) -> Calibration:
    """The calibration day, on flat water: a heart-rate step test (5-minute steps at 55-100 %
    of critical power, out of order) and the 3- and 12-minute critical-power test, both read
    from GPS through the hull model (no paddle sensor)."""
    tr, h = block.traits, block.traits.habit_rate
    step_plan = [(300.0, 56.0, 0.5)] + [(300.0, h, x) for x in STEP_TEST_SHARES]
    tw = _test_twin(block, step_plan, block.seed * 100 + 90)
    rows = tw.run_alone(step_s=10.0)
    prep = ss.prepare_rows(rows, tw.craft, tw.total_kg)
    model = fit_personal_model(prep["all_rows"], effort_key="boat_power_w")
    p = tr.paddler
    e_s = 1.0 + 0.95 * p.w_prime_j / (p.critical_power_w * 180.0)
    e_l = 1.0 + 0.95 * p.w_prime_j / (p.critical_power_w * 720.0)
    cp_plan = [(300.0, 56.0, 0.6), (180.0, h + 4.0, e_s), (1800.0, 48.0, 0.4), (720.0, h + 1.0, e_l), (120.0, 50.0, 0.4)]
    tw2 = _test_twin(block, cp_plan, block.seed * 100 + 91)
    rows2 = tw2.run_alone(step_s=10.0)
    prep2 = ss.prepare_rows(rows2, tw2.craft, tw2.total_kg)
    best = ss.best_efforts(prep2["all_rows"], (180, 720))
    ath = calibrate(best, source="3- and 12-minute test on the calibration day, GPS through the hull model")
    cal = Calibration(model=model, athlete=ath, hr_max_seen=max(_max30(rows), _max30(rows2)),
                      separates_up_to=model.separates_up_to(ath.cp_w))
    cal.summary = {"critical_power_boat_w": round(ath.cp_w), "reserve_boat_kj": round(ath.w_prime_j / 1000.0, 1),
                   "highest_30s_heart_rate": round(cal.hr_max_seen),
                   "heart_rate_model": model.summary(cp_w=ath.cp_w)}
    return cal


# ── the reader ──────────────────────────────────────────────────────────────

@dataclass
class Guide:
    """How the live reader turns the last minute into a zone (what an arm decides)."""

    mode: str = "combined"                 # textbook | hr | pace | combined
    combine: str = "agree"                 # agree | hr_first | pace_first (mode combined)
    hr_offset_bpm: float = 0.0             # subtracted before the model reads the heart rate
    hr_trusted: bool = True                # False: ignore heart rate today
    hr_trust_up_to_share: Optional[float] = None   # above this the heart-rate route cannot tell
    plausibility_check: bool = False       # drop heart rates below 75 or within 3 of the stroke rate
    cp_scale: float = 1.0                  # today's critical power / the calibrated one (pace route)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    xs = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    return float(sum(xs) / len(xs)) if xs else None


def read_zone(guide: Guide, recent: List[Dict[str, Any]], cal: Calibration, minutes_in: float) -> Dict[str, Any]:
    """The zone of the last minute (``recent``: rows with ``heart_rate_bpm``, ``rate_spm``,
    ``power_w``): ``zone`` (an index into the effort zones) or None for "cannot tell", with
    each route's reading."""
    hr, rate, pw = _mean(recent, "heart_rate_bpm"), _mean(recent, "rate_spm"), _mean(recent, "power_w")
    out: Dict[str, Any] = {"hr": hr, "rate": rate, "power_w": pw, "share_hr": None, "share_pace": None,
                           "zone_hr": None, "zone_pace": None, "zone": None}
    cp = cal.athlete.cp_w
    if guide.mode == "textbook":
        if hr is not None and cal.hr_max_seen > 0:
            f = hr / cal.hr_max_seen
            z = next((j for j, (_, lo, hi) in enumerate(TEXTBOOK_ZONES) if lo <= f < hi),
                     0 if f < TEXTBOOK_ZONES[0][1] else len(TEXTBOOK_ZONES) - 1)
            out["zone"] = out["zone_hr"] = z
        return out
    hr_ok = guide.hr_trusted and hr is not None
    if hr_ok and guide.plausibility_check and (hr < 75.0 or (rate is not None and abs(hr - rate) < 3.0)):
        hr_ok = False
        out["hr_dropped"] = True
    if hr_ok and guide.mode in ("hr", "combined"):
        e = cal.model.effort_at(hr - guide.hr_offset_bpm, minutes_in=minutes_in)
        top = guide.hr_trust_up_to_share if guide.hr_trust_up_to_share is not None else 9.0
        if e is not None:
            s = e / cp
            out["share_hr"] = round(s, 3)
            if s <= top + 0.05:
                out["zone_hr"] = zone_of_share(s)
    if pw is not None and guide.mode in ("pace", "combined"):
        s = pw / (cp * max(0.3, guide.cp_scale))
        out["share_pace"] = round(s, 3)
        out["zone_pace"] = zone_of_share(s)
    zh, zp = out["zone_hr"], out["zone_pace"]
    if guide.mode == "hr":
        out["zone"] = zh
    elif guide.mode == "pace":
        out["zone"] = zp
    elif guide.combine == "hr_first":
        out["zone"] = zh if zh is not None else zp
    elif guide.combine == "pace_first":
        out["zone"] = zp if zp is not None else zh
    else:                                   # agree
        if zh is not None and zp is not None:
            out["zone"] = zone_of_share(0.5 * (out["share_hr"] + out["share_pace"])) if abs(zh - zp) <= 1 else None
        else:
            out["zone"] = zh if zh is not None else zp
    return out


# ── one session ─────────────────────────────────────────────────────────────

@dataclass
class SessionCtx:
    block: TrainingBlock
    cal: Calibration
    spec: SessionSpec
    twin: PaddlerTwin
    state: RaceState
    strap: StrapFault
    sched: Scheduler
    guide: Guide
    rows: List[Dict[str, Any]] = field(default_factory=list)       # the session as recorded (no truth)
    reads: List[Dict[str, Any]] = field(default_factory=list)      # every 30 s, with the truth for scoring
    history: List[Dict[str, Any]] = field(default_factory=list)    # earlier sessions, as the paddler's devices saw them
    answers: List[Dict[str, Any]] = field(default_factory=list)
    guide_log: List[Dict[str, Any]] = field(default_factory=list)

    def recent(self, seconds: float = 60.0) -> List[Dict[str, Any]]:
        t = self.twin.t
        return [r for r in self.rows if r["t"] > t - seconds]

    def set_guide(self, g: Guide, why: str = "") -> None:
        self.guide = g
        self.guide_log.append({"t": round(self.twin.t, 1), "why": why, **g.to_dict()})


class GuideArm:
    """An arm decides the guide; the reader applies it every 30 s."""

    name = "arm"
    setting: Dict[str, Any] = {}

    def start_block(self, block: TrainingBlock, cal: Calibration) -> None:
        self.block, self.cal = block, cal

    def initial_guide(self, ctx: SessionCtx) -> Guide:
        return Guide()

    def after_warmup(self, ctx: SessionCtx) -> None:
        pass

    def on_read(self, ctx: SessionCtx, read: Dict[str, Any]) -> None:
        pass

    def end_session(self, ctx: SessionCtx, score: Dict[str, Any]) -> None:
        pass

    def summary(self) -> Dict[str, Any]:
        return {}


class TextbookArm(GuideArm):
    name = "textbook"

    def initial_guide(self, ctx):
        return Guide(mode="textbook")


class HrModelArm(GuideArm):
    """The personal model as fitted on the calibration day; no day adjustment."""

    name = "hr_model"

    def initial_guide(self, ctx):
        return Guide(mode="hr", hr_trust_up_to_share=ctx.cal.separates_up_to)


class PaceArm(GuideArm):
    """GPS pace through the hull model and the boat computer's current estimate only."""

    name = "pace"

    def initial_guide(self, ctx):
        return Guide(mode="pace")


class WarmupRuleArm(GuideArm):
    """The strongest fixed rule: from the last four minutes of the warm-up, the day's offset
    is the median gap between the heart rate and what the model expects at the boat's power;
    heart rates that cannot be real are dropped; heart rate and pace must agree within a zone."""

    name = "warmup_rule"

    def initial_guide(self, ctx):
        return Guide(mode="combined", combine="agree", plausibility_check=True,
                     hr_trust_up_to_share=ctx.cal.separates_up_to)

    def after_warmup(self, ctx):
        t1 = ctx.twin.t
        W = [r for r in ctx.rows if t1 - 240.0 <= r["t"] <= t1 and r.get("power_w") is not None
             and r.get("heart_rate_bpm") is not None]
        gaps = [r["heart_rate_bpm"] - ctx.cal.model.hr_at(r["power_w"]) for r in W
                if r["heart_rate_bpm"] >= 75.0 and abs(r["heart_rate_bpm"] - (r.get("rate_spm") or 0.0)) >= 3.0]
        g = ctx.guide
        off = float(np.median(gaps)) if len(gaps) >= 20 else 0.0
        ctx.set_guide(Guide(mode="combined", combine="agree", plausibility_check=True,
                            hr_trust_up_to_share=g.hr_trust_up_to_share, hr_offset_bpm=round(off, 1)),
                      why="warm-up offset")


def _segment_at(plan, t: float) -> Tuple[int, float, float]:
    """(segment index, seconds into it, its planned effort) at session time ``t``."""
    acc = 0.0
    for i, (dur, _, eff) in enumerate(plan):
        if t < acc + dur:
            return i, t - acc, eff
        acc += dur
    return len(plan) - 1, 0.0, 0.0


def run_session(block: TrainingBlock, cal: Calibration, spec: SessionSpec, arm: GuideArm, *,
                history: Optional[List[Dict[str, Any]]] = None, latency_s: float = 5.0) -> SessionCtx:
    """One training session with ``arm`` deciding the guide (simulated time; a model call
    takes ``latency_s`` of session time)."""
    tw = session_twin(block, spec)
    st = RaceState(tw.craft, tw.total_kg, cal.athlete, distance_m=None, start_t=0.0)
    sched = Scheduler(VirtualClock(0.0), latency=fixed({}, default=latency_s))
    ctx = SessionCtx(block=block, cal=cal, spec=spec, twin=tw, state=st, strap=StrapFault(spec.day.strap, block.seed * 100 + spec.index),
                     sched=sched, guide=Guide(), history=list(history or []))
    ctx.set_guide(arm.initial_guide(ctx), why="start")
    end_t = sum(p[0] for p in spec.plan)
    nxt = {"read": READ_EVERY_S, "warm": False}

    def tick():
        now = sched.now()
        for r in tw.advance(now):
            hr_true = r.get("heart_rate_bpm")
            if hr_true is not None:
                r["heart_rate_bpm"] = ctx.strap.read(r["t"], hr_true, r.get("rate_spm") or 0.0)
            snap = st.update(r)
            ctx.rows.append({"t": r["t"], "heading_deg": r.get("heading_deg"), "speed_gps": r.get("v_mean"),
                             "rate_spm": r.get("rate_spm"), "heart_rate_bpm": r.get("heart_rate_bpm"),
                             "power_w": snap.get("power_w") if snap.get("power_w") and math.isfinite(snap["power_w"]) else None,
                             "speed_water": snap.get("speed_water"), "push_along_mps": snap.get("push_along_mps")})
        if not nxt["warm"] and now >= spec.warmup_s:
            nxt["warm"] = True
            arm.after_warmup(ctx)
        if now >= nxt["read"]:
            rd = read_zone(ctx.guide, ctx.recent(60.0), cal, minutes_in=now / 60.0)
            L = tw.log
            w = [p for p, tt in zip(L["p_shaft"][-200:], L["t"][-200:]) if tt > now - READ_EVERY_S]
            seg, into, eff = _segment_at(spec.plan, now)
            share_true = (sum(w) / len(w)) / tw._cp_eff() if w else float("nan")
            rd.update({"t": round(now, 1), "share_true": round(share_true, 3),
                       "zone_true": zone_of_share(share_true) if math.isfinite(share_true) else None,
                       "steady": bool(into >= STEADY_S and now >= spec.warmup_s), "segment": seg,
                       "strap_fault": ctx.strap.active(now)})
            ctx.reads.append(rd)
            record.emit("observation", kind="zone_read", t=rd["t"], zone=rd["zone"], zone_true=rd["zone_true"])
            arm.on_read(ctx, rd)
            nxt["read"] += READ_EVERY_S
        if now < end_t - 1.0:
            sched.after(1.0, tick)

    sched.at(1.0, tick)
    sched.run()
    return ctx


def score_session(ctx: SessionCtx) -> Dict[str, Any]:
    R = [r for r in ctx.reads if r["steady"] and r["zone_true"] is not None]
    n = len(R)
    right = sum(1 for r in R if r["zone"] is not None and r["zone"] == r["zone_true"])
    wrong = sum(1 for r in R if r["zone"] is not None and r["zone"] != r["zone_true"])
    return {"session": ctx.spec.index, "kind": ctx.spec.kind, "day": ctx.spec.day.kind, "n": n,
            "right": right, "wrong": wrong, "cannot_tell": n - right - wrong,
            "right_frac": round(right / n, 3) if n else None, "wrong_frac": round(wrong / n, 3) if n else None}


def session_summary(ctx: SessionCtx) -> Dict[str, Any]:
    """What the paddler's devices recorded, per steady stretch (no truth): for later sessions."""
    segs = []
    for i, (dur, rate, eff) in enumerate(ctx.spec.plan):
        t0 = sum(p[0] for p in ctx.spec.plan[:i])
        W = [r for r in ctx.rows if t0 + STEADY_S <= r["t"] < t0 + dur]
        if len(W) < 30:
            continue
        pw, hr = _mean(W, "power_w"), _mean(W, "heart_rate_bpm")
        segs.append({"minutes": [round(t0 / 60.0, 1), round((t0 + dur) / 60.0, 1)],
                     "boat_power_w": None if pw is None else round(pw), "rate_spm": round(_mean(W, "rate_spm") or 0, 1),
                     "heart_rate_bpm": None if hr is None else round(hr, 1),
                     "model_heart_rate_at_this_power": None if pw is None else round(ctx.cal.model.hr_at(pw), 1)})
    return {"session": ctx.spec.index, "description": ctx.spec.description, "air_c": round(ctx.spec.day.air_c),
            "stretches": segs, "paddler_said": [a["says"] for a in ctx.answers],
            "guide_used": ctx.guide.to_dict()}


def run_block(block: TrainingBlock, arm: GuideArm, *, cal: Optional[Calibration] = None,
              latency_s: float = 5.0) -> Dict[str, Any]:
    """Calibration day, then every session in order with ``arm``; scored per session."""
    cal = cal or calibrate_block(block)
    arm.start_block(block, cal)
    hist: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {"seed": block.seed, "pool": block.pool, "arm": arm.name, "setting": dict(arm.setting),
                           "shape": block.shape, "fitness_gain": round(block.fitness_gain, 3),
                           "calibration": cal.summary, "sessions": []}
    for spec in block.sessions:
        ctx = run_session(block, cal, spec, arm, history=hist, latency_s=latency_s)
        sc = score_session(ctx)
        sc["guide_log"] = ctx.guide_log
        sc["answers"] = ctx.answers
        arm.end_session(ctx, sc)
        out["sessions"].append(sc)
        hist.append(session_summary(ctx))
    tot = {k: sum(s[k] for s in out["sessions"]) for k in ("n", "right", "wrong", "cannot_tell")}
    tot["right_frac"] = round(tot["right"] / tot["n"], 3) if tot["n"] else None
    tot["wrong_frac"] = round(tot["wrong"] / tot["n"], 3) if tot["n"] else None
    out["total"] = tot
    out.update(arm.summary())
    return out
