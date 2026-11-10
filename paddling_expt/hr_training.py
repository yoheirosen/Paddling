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
adjustment), ``refit`` (the personal model refitted after every session on everything so far),
``pace`` (pace only), ``pace_forecast`` (pace, with the tide table and wind forecast used for the
current until the boat computer has its own estimate), ``warmup_rule`` (the strongest fixed rule: the day's offset
from the warm-up, the plausibility check, and both routes must agree within a zone), and the
reasoning agent (:mod:`.hr_agent`).  Scored against the simulator's truth: the share of today's
critical power over the last 30 s, at checkpoints where the planned effort has been steady for
two minutes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from agent_lib.runtime import record
from agent_lib.runtime.clock import VirtualClock
from agent_lib.runtime.scheduler import Scheduler, fixed

from . import sessions as ss
from .heart_rate import EFFORT_ZONES, TEXTBOOK_ZONES, PersonalHeartRate, fit_personal_model, zone_of_share
from .boatspeed import power_at_speed
from .race import Athlete, RaceState, calibrate
from .sim.training import (KT_PER_MPS_PUSH, SESSION_DISTANCE_M, SessionSpec, StrapFault, TrainingBlock,
                           session_forecast, session_twin)
from .sim.twin import PaddlerTwin, RaceSetup, STEP_TEST_SHARES

__all__ = ["Calibration", "calibrate_block", "Guide", "read_zone", "SessionCtx", "run_session", "run_block",
           "pace_paddle_w", "GuideArm", "TextbookArm", "HrModelArm", "PaceArm", "PaceForecastArm", "WarmupRuleArm", "RefitArm", "OracleDayArm", "score_session", "STEADY_S",
           "READ_EVERY_S"]

READ_EVERY_S = 30.0
STEADY_S = 120.0
ZONE_NAMES = [z[0] for z in EFFORT_ZONES]


# ── calibration day ─────────────────────────────────────────────────────────

@dataclass
class Calibration:
    model: PersonalHeartRate        # effort axis: paddle power (W), from the borrowed power paddle
    athlete: Athlete                # critical power and reserve, paddle power
    athlete_boat: Athlete           # the same test as power reaching the boat (the boat computer's reserve)
    eta_coef: Tuple[float, float, float]   # log(boat / paddle power) = a + b r + c r^2 (r: stroke rate)
    rate_range: Tuple[float, float]
    hr_max_seen: float
    separates_up_to: float
    summary: Dict[str, Any] = field(default_factory=dict)
    rows: List[Dict[str, Any]] = field(default_factory=list, repr=False)   # the step test: t, paddle_w, heart rate
    eff_points: List[Dict[str, float]] = field(default_factory=list)      # measured blade efficiency per test stretch

    def efficiency(self, rate: Optional[float]) -> float:
        """The share of paddle power reaching the boat at this stroke rate (calibration day)."""
        a, b, c = self.eta_coef
        lo, hi = self.rate_range
        r = 0.5 * (lo + hi) if rate is None or not math.isfinite(rate) else min(max(rate, lo), hi)
        return float(math.exp(a + b * r + c * r * r))

    def paddle_w(self, boat_w: Optional[float], rate: Optional[float]) -> Optional[float]:
        """Paddle power from the power reaching the boat, through this paddler's efficiency
        by stroke rate."""
        if boat_w is None or not math.isfinite(boat_w):
            return None
        return boat_w / max(0.2, self.efficiency(rate))


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


def _stretches(rows: List[Dict[str, Any]], plan) -> List[Tuple[float, float, float]]:
    """(boat power, paddle power, stroke rate) over the settled part of each stretch of a test."""
    t = np.array([r["t"] for r in rows], float)
    out, acc = [], 0.0
    for dur, _, _ in plan:
        m = (t > acc + min(90.0, dur / 3.0)) & (t < acc + dur)
        acc += dur
        R = [rows[i] for i in np.flatnonzero(m)
             if rows[i].get("boat_power_w") is not None and rows[i].get("shaft_power_w") is not None]
        if len(R) >= 20:
            out.append(tuple(float(np.mean([r[k] for r in R])) for k in ("boat_power_w", "shaft_power_w", "rate_spm")))
    return out


def calibrate_block(block: TrainingBlock) -> Calibration:
    """The calibration day, on flat water, with a borrowed power paddle: a heart-rate step test
    (5-minute steps at 55-100 % of critical power, out of order) and the 3- and 12-minute
    critical-power test.  The paddle measures paddle power; GPS through the hull model gives
    the power reaching the boat; their ratio by stroke rate (blade efficiency) is what the pace
    route uses on later days, when the paddler trains without the power paddle."""
    tr, h = block.traits, block.traits.habit_rate
    step_plan = [(300.0, 56.0, 0.5)] + [(300.0, h, x) for x in STEP_TEST_SHARES]
    tw = _test_twin(block, step_plan, block.seed * 100 + 90)
    rows = tw.run_alone(step_s=10.0)
    prep = ss.prepare_rows(rows, tw.craft, tw.total_kg)
    model = fit_personal_model(prep["all_rows"], effort_key="shaft_power_w")
    p = tr.paddler
    e_s = 1.0 + 0.95 * p.w_prime_j / (p.critical_power_w * 180.0)
    e_l = 1.0 + 0.95 * p.w_prime_j / (p.critical_power_w * 720.0)
    cp_plan = [(300.0, 56.0, 0.6), (180.0, h + 4.0, e_s), (1800.0, 48.0, 0.4), (720.0, h + 1.0, e_l), (120.0, 50.0, 0.4)]
    tw2 = _test_twin(block, cp_plan, block.seed * 100 + 91)
    rows2 = tw2.run_alone(step_s=10.0)
    prep2 = ss.prepare_rows(rows2, tw2.craft, tw2.total_kg)
    best_boat = ss.best_efforts(prep2["all_rows"], (180, 720))
    best = ss.best_efforts([{**r, "boat_power_w": r.get("shaft_power_w")} for r in prep2["all_rows"]], (180, 720))
    ath = calibrate(best, source="3- and 12-minute test on the calibration day, power paddle")
    ath_boat = calibrate(best_boat, source="3- and 12-minute test on the calibration day, GPS through the hull model")
    pts = _stretches(prep["all_rows"], step_plan) + _stretches(prep2["all_rows"], cp_plan)
    r = np.array([x[2] for x in pts])
    le = np.log(np.array([x[0] / x[1] for x in pts]))
    c2, c1, c0 = np.polyfit(r, le, 2)
    if c2 >= 0.0:                          # no hill seen: a straight line in the stroke rate
        c1, c0 = np.polyfit(r, le, 1)
        c2 = 0.0
    cal = Calibration(model=model, athlete=ath, athlete_boat=ath_boat, eta_coef=(float(c0), float(c1), float(c2)),
                      rate_range=(float(r.min()) - 2.0, float(r.max()) + 2.0),
                      hr_max_seen=max(_max30(rows), _max30(rows2)), separates_up_to=model.separates_up_to(ath.cp_w))
    cal.eff_points = [{"stroke_rate": round(x[2], 1), "efficiency": round(x[0] / x[1], 3)} for x in sorted(pts, key=lambda x: x[2])]
    cal.rows = [{"t": q["t"], "paddle_w": q.get("shaft_power_w"), "heart_rate_bpm": q.get("heart_rate_bpm"),
                 "rate_spm": q.get("rate_spm"), "session": 0} for q in prep["all_rows"]]
    rates = sorted({int(round(x)) for x in r})
    cal.summary = {"critical_power_paddle_w": round(ath.cp_w), "reserve_kj": round(ath.w_prime_j / 1000.0, 1),
                   "highest_30s_heart_rate": round(cal.hr_max_seen),
                   "blade_efficiency_by_stroke_rate": {str(q): round(cal.efficiency(q), 3) for q in rates},
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
    hr_extra_creep_per_10min: float = 0.0  # added to the model's slow creep today (heart-rate route)
    current_before_first_turn_mps: Optional[float] = None   # pace route, until the computer knows the current
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    xs = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
    return float(sum(xs) / len(xs)) if xs else None


def pace_paddle_w(rows: List[Dict[str, Any]], guide: Guide, cal: Calibration,
                  boat_power: Optional[Callable[[float], float]] = None) -> Optional[float]:
    """The pace route's mean paddle power over ``rows``: the boat computer's, except that
    while the computer has no estimate of the current the guide's assumed current is used."""
    c0 = guide.current_before_first_turn_mps
    if c0 is None or boat_power is None or all(r.get("push_known") for r in rows):
        return _mean(rows, "paddle_w")
    P = []
    for r in rows:
        if not r.get("push_known") and r.get("speed_gps") is not None and r.get("heading_deg") is not None:
            vw = r["speed_gps"] - float(c0) * math.cos(math.radians(r["heading_deg"]))
            P.append(cal.paddle_w(boat_power(vw), r.get("rate_spm")))
        elif r.get("paddle_w") is not None:
            P.append(r["paddle_w"])
    return float(sum(P) / len(P)) if P else None


def _plausible_hr(hr: Optional[float], rate: Optional[float]) -> bool:
    return hr is not None and hr >= 75.0 and abs(hr - (rate or 0.0)) >= 3.0


def read_zone(guide: Guide, recent: List[Dict[str, Any]], cal: Calibration, minutes_in: float,
              boat_power: Optional[Callable[[float], float]] = None) -> Dict[str, Any]:
    """The zone of the last minute (``recent``: rows with ``heart_rate_bpm``, ``rate_spm``,
    ``paddle_w``: paddle power estimated from GPS, the hull model and the paddler's blade
    efficiency by stroke rate): ``zone`` (an index into the effort zones) or None for "cannot
    tell", with each route's reading.  Both routes are always read (``share_hr``,
    ``share_pace``, ``zone_hr``, ``zone_pace``), whatever the guide uses: the anomaly checks
    watch them.  With the plausibility check, heart-rate readings below 75 or within 3 of the
    stroke rate are dropped one by one; if fewer than half of the minute's readings are left,
    the heart-rate route cannot tell."""
    hr_raw, rate = _mean(recent, "heart_rate_bpm"), _mean(recent, "rate_spm")
    pw = pace_paddle_w(recent, guide, cal, boat_power)
    H = [r for r in recent if r.get("heart_rate_bpm") is not None]
    ok = [r for r in H if _plausible_hr(r["heart_rate_bpm"], r.get("rate_spm"))]
    implaus = 1.0 - len(ok) / len(H) if H else 0.0
    out: Dict[str, Any] = {"hr": hr_raw, "rate": rate, "paddle_w": pw, "hr_implausible_frac": round(implaus, 2),
                           "share_hr": None, "share_pace": None, "zone_hr": None, "zone_pace": None, "zone": None}
    cp = cal.athlete.cp_w
    if guide.mode == "textbook":
        if hr_raw is not None and cal.hr_max_seen > 0:
            f = hr_raw / cal.hr_max_seen
            z = next((j for j, (_, lo, hi) in enumerate(TEXTBOOK_ZONES) if lo <= f < hi),
                     0 if f < TEXTBOOK_ZONES[0][1] else len(TEXTBOOK_ZONES) - 1)
            out["zone"] = out["zone_hr"] = z
        return out
    hr = hr_raw
    if guide.plausibility_check and H:
        hr = _mean(ok, "heart_rate_bpm") if len(ok) >= 0.5 * len(H) else None
        out["hr_dropped"] = hr is None
    if hr is not None:
        e = cal.model.effort_at(hr - guide.hr_offset_bpm - guide.hr_extra_creep_per_10min * minutes_in / 10.0,
                                minutes_in=minutes_in)
        top = guide.hr_trust_up_to_share if guide.hr_trust_up_to_share is not None else 9.0
        if e is not None:
            s = e / cp
            out["share_hr"] = round(s, 3)
            if s <= top:
                out["zone_hr"] = zone_of_share(s)
    if pw is not None:
        s = pw / (cp * max(0.3, guide.cp_scale))
        out["share_pace"] = round(s, 3)
        out["zone_pace"] = zone_of_share(s)
    zh = out["zone_hr"] if guide.hr_trusted and guide.mode in ("hr", "combined") else None
    zp = out["zone_pace"] if guide.mode in ("pace", "combined") else None
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
    turns: List[float] = field(default_factory=list)               # session times of the course's turns
    boat_power: Optional[Callable[[float], float]] = None          # hull model: speed through the water -> W

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

    def on_turn(self, ctx: SessionCtx) -> None:
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


class PaceForecastArm(GuideArm):
    """Pace only, with the forecast applied mechanically: until the boat computer knows the
    current, the pace route assumes the tide table's stream plus the forecast wind's push at
    the start of the session (5 kt along the course ~ 0.1 m/s)."""

    name = "pace_forecast"

    def initial_guide(self, ctx):
        f = session_forecast(ctx.block, ctx.spec)
        w = f["wind_forecast"]
        push = 0.0
        if w.get("direction"):
            sgn = 1.0 if w["direction"].startswith("behind") else -1.0
            push = sgn * w["kt_at_start"] / KT_PER_MPS_PUSH
        return Guide(mode="pace", current_before_first_turn_mps=round(f["tide_table"]["stream_at_start_mps"] + push, 3))


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
        W = [r for r in ctx.rows if t1 - 240.0 <= r["t"] <= t1 and r.get("paddle_w") is not None
             and r.get("heart_rate_bpm") is not None]
        gaps = [r["heart_rate_bpm"] - ctx.cal.model.hr_at(r["paddle_w"]) for r in W
                if r["heart_rate_bpm"] >= 75.0 and abs(r["heart_rate_bpm"] - (r.get("rate_spm") or 0.0)) >= 3.0]
        g = ctx.guide
        off = float(np.median(gaps)) if len(gaps) >= 20 else 0.0
        ctx.set_guide(Guide(mode="combined", combine="agree", plausibility_check=True,
                            hr_trust_up_to_share=g.hr_trust_up_to_share, hr_offset_bpm=round(off, 1)),
                      why="warm-up offset")


def _plausible(r: Dict[str, Any]) -> bool:
    hr = r.get("heart_rate_bpm")
    return hr is not None and hr >= 75.0 and abs(hr - (r.get("rate_spm") or 0.0)) >= 3.0


class RefitArm(GuideArm):
    """The personal model refitted after every session on everything so far: the calibration
    step test and every session (paddle power from pace; heart rates that cannot be real
    dropped), each session with its own clock.  Read as the warm-up rule reads, without a
    day offset: both routes must agree within a zone."""

    name = "refit"

    def start_block(self, block, cal):
        super().start_block(block, cal)
        self.pool = [dict(r) for r in cal.rows if _plausible(r)]
        self.cal_now = cal
        self.refits: List[Dict[str, Any]] = []

    def initial_guide(self, ctx):
        ctx.cal = self.cal_now
        return Guide(mode="combined", combine="agree", plausibility_check=True,
                     hr_trust_up_to_share=self.cal_now.separates_up_to)

    def end_session(self, ctx, score):
        k = ctx.spec.index
        self.pool += [{"t": r["t"], "paddle_w": r["paddle_w"], "heart_rate_bpm": r["heart_rate_bpm"], "session": k}
                      for r in ctx.rows if r.get("paddle_w") is not None and _plausible(r)]
        m = fit_personal_model(self.pool, effort_key="paddle_w", segment_key="session")
        if m is not None:
            self.cal_now = replace(self.cal, model=m, separates_up_to=m.separates_up_to(self.cal.athlete.cp_w))
            self.refits.append({"after_session": k, "rest": round(m.rest, 1), "rise": round(m.rise, 1),
                                "separates_up_to": round(self.cal_now.separates_up_to, 2)})

    def summary(self):
        return {"refits": self.refits}


class OracleDayArm(GuideArm):
    """A reference, not a competitor: told the planted day (its heart-rate offset, today's
    critical power, a strap fault and when it ends), with the warm-up rule's combination.
    It shows how much a guide can gain from knowing the day; it does not know how blade
    efficiency bends the pace route."""

    name = "oracle_day"

    def initial_guide(self, ctx):
        d = ctx.spec.day
        return Guide(mode="combined", combine="agree", plausibility_check=True,
                     hr_trust_up_to_share=ctx.cal.separates_up_to, hr_offset_bpm=d.hr_offset_bpm,
                     cp_scale=d.cp_scale)

    def on_read(self, ctx, rd):
        on = ctx.strap.active(rd["t"])
        if on == ctx.guide.hr_trusted:
            ctx.set_guide(Guide(**{**ctx.guide.to_dict(), "hr_trusted": not on}), why="strap")


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
    st = RaceState(tw.craft, tw.total_kg, cal.athlete_boat, distance_m=None, start_t=0.0)
    sched = Scheduler(VirtualClock(0.0), latency=fixed({}, default=latency_s))
    ctx = SessionCtx(block=block, cal=cal, spec=spec, twin=tw, state=st, strap=StrapFault(spec.day.strap, block.seed * 100 + spec.index),
                     sched=sched, guide=Guide(), history=list(history or []))
    ctx.boat_power = lambda v: float(power_at_speed(tw.craft, tw.total_kg, max(0.0, v), st.ama_kg)) if v > 0.3 else 0.0
    ctx.set_guide(arm.initial_guide(ctx), why="start")
    end_t = sum(p[0] for p in spec.plan)
    nxt = {"read": READ_EVERY_S, "warm": False, "leg0": tw._wu_leg_start}

    def tick():
        now = sched.now()
        for r in tw.advance(now):
            hr_true = r.get("heart_rate_bpm")
            if hr_true is not None:
                r["heart_rate_bpm"] = ctx.strap.read(r["t"], hr_true, r.get("rate_spm") or 0.0)
            snap = st.update(r)
            bp = snap.get("power_w") if snap.get("power_w") and math.isfinite(snap["power_w"]) else None
            ctx.rows.append({"t": r["t"], "heading_deg": r.get("heading_deg"), "speed_gps": r.get("v_mean"),
                             "rate_spm": r.get("rate_spm"), "heart_rate_bpm": r.get("heart_rate_bpm"),
                             "boat_power_w": bp, "paddle_w": cal.paddle_w(bp, r.get("rate_spm")),
                             "speed_water": snap.get("speed_water"), "push_along_mps": snap.get("push_along_mps"),
                             "push_known": bool(snap.get("push_known"))})
        if tw._wu_leg_start != nxt["leg0"]:           # the course turned
            nxt["leg0"] = tw._wu_leg_start
            ctx.turns.append(round(now, 1))
            arm.on_turn(ctx)
        if not nxt["warm"] and now >= spec.warmup_s:
            nxt["warm"] = True
            arm.after_warmup(ctx)
        if now >= nxt["read"]:
            rd = read_zone(ctx.guide, ctx.recent(60.0), ctx.cal, minutes_in=now / 60.0, boat_power=ctx.boat_power)
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


def _share_hr(cal: Calibration, hr: Optional[float], minutes_in: float) -> Optional[float]:
    """The heart-rate route's share with no day adjustment (None where heart rate cannot tell)."""
    if hr is None:
        return None
    e = cal.model.effort_at(hr, minutes_in=minutes_in)
    return None if e is None else round(e / cal.athlete.cp_w, 3)


def session_summary(ctx: SessionCtx) -> Dict[str, Any]:
    """What the paddler's devices recorded, per steady stretch (no truth): for later sessions."""
    segs = []
    for i, (dur, rate, eff) in enumerate(ctx.spec.plan):
        t0 = sum(p[0] for p in ctx.spec.plan[:i])
        W = [r for r in ctx.rows if t0 + STEADY_S <= r["t"] < t0 + dur]
        if len(W) < 30:
            continue
        pw = _mean(W, "paddle_w")
        Hp = [r for r in W if _plausible_hr(r.get("heart_rate_bpm"), r.get("rate_spm"))]
        hr = _mean(Hp, "heart_rate_bpm") if len(Hp) >= 0.5 * len(W) else None
        mid = (t0 + STEADY_S + t0 + dur) / 2.0 / 60.0
        segs.append({"minutes": [round(t0 / 60.0, 1), round((t0 + dur) / 60.0, 1)],
                     "paddle_power_est_w": None if pw is None else round(pw), "rate_spm": round(_mean(W, "rate_spm") or 0, 1),
                     "heart_rate_bpm": None if hr is None else round(hr, 1),
                     "model_heart_rate_at_this_power_and_minute": None if pw is None else
                     round(ctx.cal.model.hr_at(pw) + ctx.cal.model.creep_per_10min * mid / 10.0, 1),
                     "share_by_pace_uncorrected": None if pw is None else round(pw / ctx.cal.athlete.cp_w, 3),
                     "share_by_heart_rate_uncorrected": _share_hr(ctx.cal, hr, mid)})
    return {"session": ctx.spec.index, "air_c": round(ctx.spec.day.air_c),
            "forecast": {k: session_forecast(ctx.block, ctx.spec)[k] for k in ("tide_table", "wind_forecast")},
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
