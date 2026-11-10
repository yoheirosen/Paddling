"""A block of training sessions for one simulated paddler, with planted day-to-day effects
on heart rate, so that reading effort from heart rate has known right answers.

The paddler (:func:`.twin.draw_traits`, heart-rate shape from :func:`.twin.heart_rate_profile`)
has a calibration day (a heart-rate step test and the 3- and 12-minute critical-power test, on
flat water), then ``n_sessions`` training sessions on an out-and-back course with a tidal
stream that changes through the session (it can turn; the boat computer's estimate of it lags).
Every session carries a planted day state:

* fitness changes over the block (critical power ends 2 % lower to 7 % higher);
* development kinds: ``normal``; ``breezy`` (a breeze that builds along the course, with gusts:
  GPS cannot tell wind from current, and gusts change faster than the boat computer's estimate
  of the current); ``hot`` (heart rate 5-10 beats higher at every effort, twice
  the slow creep; 30-34 °C); ``tired`` (critical power up to 5 % lower, heart rate a few beats
  higher at easy effort and held down at hard effort, maximum lower); ``cadence_lock`` (for
  6-12 minutes the strap reads the stroke rate instead of the heart rate);
* held-out kinds (kept out of development): ``poor_contact`` (the strap reads far too low and
  erratically for the first 8-14 minutes); ``coming_down`` (heart rate about 8 beats higher,
  critical power 8 % lower); ``humid_drift`` (four times the slow creep, 26-29 °C).

Before each session the paddler checks the tide table and the wind forecast
(:func:`session_forecast`, with planted errors).  The paddler answers questions about how they feel, the conditions and the strap
(:func:`answer`), and can wet or move the strap when asked (which ends a strap fault a minute
later).  All of this is planted: it says what the simulator does, not how real bodies or straps
behave.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .twin import CoastalWater, PaddlerTwin, RaceSetup, Traits, draw_traits, heart_rate_profile

__all__ = ["DEVELOPMENT_DAY_KINDS", "HELD_OUT_DAY_KINDS", "SESSION_KINDS", "DayState", "SessionSpec",
           "TrainingBlock", "draw_block", "executed_plan", "session_twin", "session_forecast", "StrapFault", "answer"]

DEVELOPMENT_DAY_KINDS = ("normal", "hot", "tired", "cadence_lock", "breezy")
HELD_OUT_DAY_KINDS = ("poor_contact", "coming_down", "humid_drift")
SESSION_KINDS = ("steady", "intervals", "tempo", "easy_long", "mixed")
SESSION_DISTANCE_M = 1e6            # a session ends with its plan, not at a finish line
LEG_M = 1500.0


@dataclass
class DayState:
    kind: str = "normal"
    cp_scale: float = 1.0           # today's critical power / the calibration day's
    hr_offset_bpm: float = 0.0      # added to the settled heart rate at every effort
    fatigue_suppress_bpm: float = 0.0   # held down at full effort (and off the maximum)
    drift_scale: float = 1.0        # x the slow creep
    strap: Optional[Dict[str, float]] = None   # {"kind", "start_s", "end_s", "factor"}
    air_c: float = 20.0
    current_mps: float = 0.0        # the tidal stream at the start, + flows north (with the first leg)
    current_change_mps: float = 0.0  # how much it changes by the end of the session (it can turn)
    wind: Optional[Dict[str, float]] = None   # {"start", "end", "gust_sd", "gust_tau_s"}: along the course, as water speed


@dataclass
class SessionSpec:
    index: int
    kind: str
    plan: List[Tuple[float, float, float]]   # (seconds, stroke rate, effort as a share of today's CP)
    day: DayState
    warmup_s: float
    description: str


@dataclass
class TrainingBlock:
    seed: int
    pool: str
    traits: Traits
    hr_par: Dict[str, Any]          # the person's heart-rate physiology (calibration day)
    shape: str
    fitness_gain: float
    sessions: List[SessionSpec] = field(default_factory=list)


def _plan(kind: str, rng: np.random.Generator, h: float) -> Tuple[List[Tuple[float, float, float]], float, str]:
    u = lambda a, b: float(rng.uniform(a, b))      # noqa: E731
    if kind == "steady":
        e = u(0.62, 0.70)
        return ([(360.0, 52.0, 0.50), (2400.0, h, e), (300.0, 50.0, 0.45)], 360.0,
                "steady endurance: 40 minutes at a conversational effort")
    if kind == "intervals":
        e = u(0.95, 1.02)
        p = [(480.0, 52.0, 0.55)]
        for _ in range(5):
            p += [(240.0, h + 4.0, e), (120.0, 50.0, 0.45)]
        return p + [(300.0, 50.0, 0.45)], 480.0, "intervals: 5 x 4 minutes hard, 2 minutes easy between"
    if kind == "tempo":
        e = u(0.80, 0.88)
        return ([(480.0, 52.0, 0.55), (720.0, h + 2.0, e), (180.0, 50.0, 0.50), (720.0, h + 2.0, e),
                 (300.0, 50.0, 0.45)], 480.0, "tempo: 2 x 12 minutes comfortably hard")
    if kind == "easy_long":
        e = u(0.58, 0.64)
        return ([(360.0, 52.0, 0.50), (3000.0, h - 2.0, e), (300.0, 50.0, 0.45)], 360.0,
                "long easy: 50 minutes easy")
    # mixed: steps through the zones
    return ([(480.0, 52.0, 0.55), (900.0, h, u(0.64, 0.70)), (600.0, h + 2.0, u(0.82, 0.88)),
             (300.0, h + 4.0, u(0.96, 1.02)), (600.0, h, u(0.58, 0.64)), (300.0, 50.0, 0.45)], 480.0,
            "mixed: 15 minutes steady, 10 tempo, 5 hard, 10 easy")


def _day(kind: str, rng: np.random.Generator, cp_scale: float, plan_s: float) -> DayState:
    u = lambda a, b: float(rng.uniform(a, b))      # noqa: E731
    d = DayState(kind=kind, cp_scale=cp_scale, air_c=u(16.0, 24.0), current_mps=u(-0.30, 0.30),
                 current_change_mps=u(-0.40, 0.40))
    if kind == "hot":
        d.hr_offset_bpm, d.drift_scale, d.air_c = u(5.0, 10.0), 2.0, u(30.0, 34.0)
    elif kind == "tired":
        f = u(0.5, 0.9)
        d.cp_scale *= 1.0 - 0.06 * f
        d.hr_offset_bpm, d.fatigue_suppress_bpm = 3.0 * f, 10.0 * f
    elif kind == "cadence_lock":
        a = u(600.0, 1500.0)
        d.strap = {"kind": "cadence_lock", "start_s": a, "end_s": a + u(360.0, 720.0)}
    elif kind == "poor_contact":
        d.strap = {"kind": "poor_contact", "start_s": 0.0, "end_s": u(480.0, 840.0), "factor": u(0.55, 0.70)}
    elif kind == "coming_down":
        d.cp_scale *= 0.92
        d.hr_offset_bpm, d.fatigue_suppress_bpm = u(6.0, 10.0), 4.0
    elif kind == "breezy":                 # a breeze that builds, with gusts (along the course)
        sgn = 1.0 if float(rng.random()) < 0.5 else -1.0
        d.wind = {"start": sgn * u(0.0, 0.08), "end": sgn * u(0.15, 0.30), "gust_sd": u(0.08, 0.14),
                  "gust_tau_s": u(60.0, 150.0)}
    elif kind == "humid_drift":
        d.hr_offset_bpm, d.drift_scale, d.air_c = 3.0, 4.0, u(26.0, 29.0)
    return d


def draw_block(seed: int, pool: str = "development", n_sessions: int = 5) -> TrainingBlock:
    """A paddler and their block (its own random stream; the paddler's traits and heart-rate
    shape come from the twin's draws for this seed)."""
    tr = draw_traits(seed)
    ref = PaddlerTwin(tr, RaceSetup(current_mps=0.0, plan=[(60.0, 50.0, 0.5)]), seed=seed)
    ref.enable_heart_rate(profiles=True)
    hr_par = dict(ref._hr_par)
    rng = np.random.default_rng([seed, 41])
    gain = float(rng.uniform(-0.02, 0.07))
    kinds = list(SESSION_KINDS)[:n_sessions] if n_sessions <= len(SESSION_KINDS) else list(SESSION_KINDS)
    rng.shuffle(kinds)
    while len(kinds) < n_sessions:
        kinds.append(str(rng.choice(SESSION_KINDS)))
    pool_kinds = DEVELOPMENT_DAY_KINDS if pool == "development" else DEVELOPMENT_DAY_KINDS[1:] + HELD_OUT_DAY_KINDS
    block = TrainingBlock(seed=seed, pool=pool, traits=tr, hr_par=hr_par, shape=hr_par.get("shape", "typical"),
                          fitness_gain=gain)
    for k, kind in enumerate(kinds):
        plan, wu, desc = _plan(kind, rng, tr.habit_rate)
        day_kind = "normal" if float(rng.random()) < 0.3 else str(rng.choice(pool_kinds))
        scale = 1.0 + gain * (k + 1) / n_sessions
        block.sessions.append(SessionSpec(index=k + 1, kind=kind, plan=plan,
                                          day=_day(day_kind, rng, scale, sum(p[0] for p in plan)),
                                          warmup_s=wu, description=desc))
    return block


class StrapFault:
    """What the strap reports, given the true heart rate (cadence lock, poor contact); a
    fault the paddler fixes ends a minute after they are asked."""

    def __init__(self, strap: Optional[Dict[str, float]], seed: int) -> None:
        self.s = dict(strap) if strap else None
        self.rng = np.random.default_rng([seed, 43])

    def fix(self, t: float) -> bool:
        if self.s is None or t >= self.s["end_s"]:
            return False
        self.s["end_s"] = min(self.s["end_s"], max(t + 60.0, self.s["start_s"] + 60.0))
        return True

    def active(self, t: float) -> bool:
        return self.s is not None and self.s["start_s"] <= t < self.s["end_s"]

    def read(self, t: float, hr: float, rate: float) -> float:
        if not self.active(t):
            return hr
        if self.s["kind"] == "cadence_lock" and rate > 45.0:
            return rate + float(self.rng.normal(0.0, 1.0))
        if self.s["kind"] == "poor_contact":
            return hr * self.s["factor"] + float(self.rng.normal(0.0, 6.0))
        return hr


class TrainingWater(CoastalWater):
    """The tidal stream plus, on a breezy day, wind along the course as an equivalent water
    speed: a mean that builds through the session and gusts (an Ornstein-Uhlenbeck process
    with its own random stream).  GPS cannot tell it from current; gusts change faster than
    the boat computer's estimate of the current."""

    _wind: Optional[Dict[str, float]] = None
    _gust: Optional[np.ndarray] = None

    def set_wind(self, wind: Dict[str, float], seed: int, dt: float = 5.0) -> None:
        self._wind = dict(wind)
        rng = np.random.default_rng([seed, 59])
        n = int(self.trend_s / dt) + 2
        a = math.exp(-dt / wind["gust_tau_s"])
        g = np.zeros(n)
        for i in range(1, n):
            g[i] = a * g[i - 1] + wind["gust_sd"] * math.sqrt(1.0 - a * a) * float(rng.normal())
        self._gust, self._dt = g, dt

    def wind_at(self, t: float) -> float:
        if not self._wind:
            return 0.0
        f = min(1.0, max(0.0, t / self.trend_s))
        i = min(len(self._gust) - 1, max(0, int(t / self._dt)))
        return self._wind["start"] + (self._wind["end"] - self._wind["start"]) * f + float(self._gust[i])

    def at(self, y: float, s: float, t: float) -> float:
        return super().at(y, s, t) + self.wind_at(t)


KT_PER_MPS_PUSH = 50.0      # planted rule of thumb: 5 kt of wind along the course pushes like 0.1 m/s of current


def session_forecast(block: TrainingBlock, spec: SessionSpec) -> Dict[str, Any]:
    """The tide table and the wind forecast the paddler checks before the session (planted,
    with errors; its own random stream).  The tide table has the timing right but the strength
    at this spot off by about 0.1 m/s; the wind forecast misses some breezes (about 1 in 5) and
    forecasts one that never comes on about 1 day in 7, and is off in strength and timing."""
    d = spec.day
    rng = np.random.default_rng([block.seed, spec.index, 61])
    dur = sum(p[0] for p in spec.plan)
    c0 = d.current_mps + float(rng.normal(0.0, 0.08))
    c1 = d.current_change_mps + float(rng.normal(0.0, 0.08))
    tide: Dict[str, Any] = {"stream_at_start_mps": round(c0, 2), "stream_at_end_mps": round(c0 + c1, 2),
                            "changes_steadily_over_min": round(dur / 60.0)}
    if c0 * (c0 + c1) < 0 and abs(c1) > 1e-6:
        tide["stream_turns_at_about_min"] = round(dur / 60.0 * (-c0 / c1))
    kt = lambda w: int(round(abs(w) * KT_PER_MPS_PUSH))          # noqa: E731
    side = lambda w: "behind you on the way out" if w > 0 else "on the nose on the way out"   # noqa: E731
    r = float(rng.random())
    if d.wind and r < 0.8:
        f0 = d.wind["start"] * float(rng.uniform(0.6, 1.4))
        f1 = d.wind["end"] * float(rng.uniform(0.7, 1.3))
        wind = {"summary": f"breeze building from about {kt(f0)} to {kt(f1)} kt over the session, {side(f1)}, gusty",
                "kt_at_start": kt(f0), "kt_at_end": kt(f1), "direction": side(f1), "gusty": True}
    elif not d.wind and r < 0.14:
        sgn = 1.0 if float(rng.random()) < 0.5 else -1.0
        f1 = sgn * float(rng.uniform(0.15, 0.28))
        wind = {"summary": f"sea breeze filling in to about {kt(f1)} kt during the session, {side(f1)}",
                "kt_at_start": 3, "kt_at_end": kt(f1), "direction": side(f1), "gusty": False}
    else:
        wind = {"summary": "light and variable, under 6 kt", "kt_at_start": 3, "kt_at_end": 4, "direction": None,
                "gusty": False}
    return {"tide_table": tide, "wind_forecast": wind,
            "convention": "stream + flows with you on the way out (the first leg)",
            "rule_of_thumb": f"in this boat, wind along the course pushes about like current: 5 kt ~ 0.1 m/s "
                             f"(GPS cannot tell them apart)",
            "reliability": "tide tables get the timing right but the strength at this spot is often 0.1 m/s off; "
                           "wind forecasts miss some breezes, forecast some that never come, and are often off "
                           "in strength and timing"}


def executed_plan(block: TrainingBlock, spec: SessionSpec) -> List[Tuple[float, float, float]]:
    """The plan as the paddler paddles it by feel: each stretch of work off its planned
    effort by a few percent (planted; its own random stream)."""
    rng = np.random.default_rng([block.seed, spec.index, 53])
    return [(dur, rate, eff * (1.0 + float(rng.normal(0.0, 0.04))) if eff >= 0.5 else eff)
            for dur, rate, eff in spec.plan]


def session_twin(block: TrainingBlock, spec: SessionSpec) -> PaddlerTwin:
    """The twin for one session: today's critical power, the person's heart rate with the
    day's effects, the day's current (strap faults are applied by :class:`StrapFault`)."""
    d = spec.day
    p = block.traits.paddler
    tr = replace(block.traits, paddler=replace(p, critical_power_w=p.critical_power_w * d.cp_scale))
    plan = executed_plan(block, spec)
    water = TrainingWater(c0=d.current_mps, c1=d.current_change_mps, trend_s=sum(p[0] for p in plan),
                          inshore_floor=1.0, eddies=[])            # a tidal stream that changes, even across the course
    if d.wind:
        water.set_wind(d.wind, seed=block.seed * 100 + spec.index)
    su = RaceSetup(distance_m=SESSION_DISTANCE_M, leg_m=LEG_M, current_mps=d.current_mps, plan=plan, water=water)
    tw = PaddlerTwin(tr, su, seed=block.seed * 100 + spec.index)
    tw.enable_heart_rate(profiles=True)
    hp = dict(block.hr_par)
    hp["day_offset"] = d.hr_offset_bpm
    hp["fatigue_suppress"] = d.fatigue_suppress_bpm
    hp["drift_per_10min"] = hp["drift_per_10min"] * d.drift_scale
    tw._hr_par = hp
    tw.hr = hp.get("rest", hp["low"]) if hp.get("shape") == "compressed" else hp["low"]
    return tw


def answer(block: TrainingBlock, spec: SessionSpec, ask: str, t: float, asked_before: int = 0) -> Dict[str, Any]:
    """The paddler's answer.  ``ask``: ``feel`` (how they feel), ``conditions`` (the water and
    the weather), ``strap`` (does the strap seem right), ``fix_strap`` (wet and re-seat it)."""
    d = spec.day
    rng = np.random.default_rng([block.seed, spec.index, int(t), asked_before, 47])
    r = float(rng.random())
    if ask == "feel":
        says = {"hot": "Feeling the heat today." if r < 0.8 else "Fine.",
                "tired": "Legs heavy, didn't sleep well." if r < 0.7 else "Okay I guess.",
                "coming_down": "A bit off today, scratchy throat." if r < 0.6 else "Alright.",
                "humid_drift": "It's muggy." if r < 0.6 else "Fine."}.get(d.kind, "Good." if r < 0.5 else "Fine.")
    elif ask == "conditions":
        frac = min(1.0, t / max(1.0, sum(p[0] for p in spec.plan)))
        now = d.current_mps + d.current_change_mps * frac
        flow = "with" if now > 0.08 else "against" if now < -0.08 else "hardly any"
        says = (f"About {d.air_c:.0f} degrees. The water is moving {flow} us on the way out." if flow != "hardly any"
                else f"About {d.air_c:.0f} degrees. Not much current right now.")
        if d.wind:
            w_now = d.wind["start"] + (d.wind["end"] - d.wind["start"]) * frac
            says += (f" The breeze is {'up' if abs(w_now) > 0.12 else 'coming up'}, gusty, "
                     f"{'behind' if w_now > 0 else 'on the nose'} on the way out.")
        c0, end = d.current_mps, d.current_mps + d.current_change_mps
        if r < 0.75:                       # the tide table, as the paddler remembers it
            if c0 * end < 0 and min(abs(c0), abs(end)) > 0.05:
                says += " The tide table has it turning during the session."
            elif abs(end) > abs(c0) + 0.12:
                says += " The tide table has it building through the session."
            elif abs(end) < abs(c0) - 0.12:
                says += " The tide table has it easing through the session."
    elif ask == "strap":
        if d.strap and d.strap["kind"] == "poor_contact" and t < d.strap["end_s"]:
            says = "It felt dry when I put it on." if r < 0.5 else "Seems fine?"
        else:
            says = "Seems fine."
    elif ask == "fix_strap":
        says = "Done, wet it and moved it a bit."
    else:
        says = "Sorry?"
    return {"says": says}
