"""The in-race coach: a cheap screen on every stroke, a reasoning coach on demand,
and spoken cues behind a gate.

Three tiers, the same shape as the agentic_experiments platform's other loops:

1. **Race state** (:class:`~.race.RaceState`), updated per stroke: speed through
   the water, power, the reserve above critical power, the pace sustainable to the
   finish, fade, the push of current and wind.
2. **Screen** (:class:`RaceScreen`), deterministic, on every stroke: raises an
   event when something may be worth saying, and only then (or every few minutes)
   wakes the coach.
   * ``over_pace``: early in the race (after the start sprint), the reserve runs
     out well before the finish at the power being paddled (the reserve model is
     trustworthy while fresh);
   * ``fading``: speed through the water at the stroke rate being paddled has
     fallen against the same rate earlier in the race;
   * ``reserve_conflict``: the reserve reads empty but the speed is holding (the
     calibration is probably low; the coach should not tell the paddler to ease);
   * ``finish_window``: reserve left near the finish while the speed is known to
     hold, and no fade in the last five minutes (a refilled reserve late in a race
     can be a tiring paddler, not a fresh one).

   A reading that is unknown (the fade is held through turns) freezes an event's
   timers rather than clearing it.
3. **Coach** (:class:`RaceCoach`), a language model through
   ``agent_lib.runtime.tool_loop.run_tool_loop``: it sees the compact race state
   and the events, may look at recent history, and gives at most one short cue
   through the ``cue`` tool.  Only this tier can speak.

Every cue goes through an ``agent_lib.runtime.gate.ActuatorGate``: a word limit,
quiet periods (the start, the last metres), a cooldown and a cap on cues per race.
:func:`replay_race` drives the whole loop on agentic_experiments' scheduler with a
virtual clock, where each model call takes a fixed latency (5–10 s is real time
for a race), so a replay shows what a live coach could have said, and when.
:func:`stand_in_coach` is a scripted stand-in for the model (no network, for
tests and demos): it is not evidence of what a model would say.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional

from agent_lib.providers.fake_provider import FakeProvider
from agent_lib.providers.types import Message
from agent_lib.runtime import record
from agent_lib.runtime.clock import VirtualClock
from agent_lib.runtime.gate import ActuatorGate, Cooldown, MaxCount, Validate
from agent_lib.runtime.record import RunRecord
from agent_lib.runtime.scheduler import Scheduler, fixed
from agent_lib.runtime.tool_loop import run_tool_loop
from agent_lib.tool_registry import ToolPolicy, ToolRegistry

from .race import RaceState

_ASKS = ("ease", "push", "hold", "rate_up", "rate_down", "finish", "line_in", "line_out", "none")

__all__ = ["RaceScreen", "CueSink", "make_cue_gate", "RaceCoach", "COACH_SYSTEM", "COACH_PANEL_SYSTEM",
           "stand_in_coach", "make_coach_provider", "replay_race", "format_pace"]


def format_pace(s500: Optional[float]) -> str:
    """Seconds per 500 m as m:ss."""
    if s500 is None or not math.isfinite(s500):
        return "?"
    return f"{int(s500 // 60)}:{int(round(s500 % 60)):02d}"


class RaceScreen:
    """Deterministic per-stroke screen (see the module docstring).  Each event is
    raised once when its condition has held for its hold time, then re-armed after
    the condition has been false for ``rearm_s``."""

    def __init__(self, *, overpace_margin_m: float = 300.0, trust_reserve_frac: float = 0.35,
                 start_s: float = 90.0, fade_pct: float = -4.0, fade_hold_s: float = 60.0,
                 conflict_hold_s: float = 120.0, finish_m: float = 800.0,
                 rearm_s: float = 60.0, fade_memory_s: float = 300.0) -> None:
        self.margin, self.trust, self.start_s = float(overpace_margin_m), float(trust_reserve_frac), float(start_s)
        self.fade_pct, self.fade_hold = float(fade_pct), float(fade_hold_s)
        self.conflict_hold, self.finish_m, self.rearm = float(conflict_hold_s), float(finish_m), float(rearm_s)
        self.fade_memory = float(fade_memory_s)
        self._last_fading: Optional[float] = None
        self._since: Dict[str, Optional[float]] = {}
        self._off_since: Dict[str, Optional[float]] = {}
        self._armed: Dict[str, bool] = {}
        self.log: List[Dict[str, Any]] = []

    def _conditions(self, s: Dict[str, Any], distance_m: Optional[float]):
        rem, emp, fade = s.get("remaining_m"), s.get("empties_in_m"), s.get("fade_pct")
        wb = s.get("w_bal_frac", 1.0)
        early = (distance_m is not None and s.get("distance_m", 0.0) < self.trust * distance_m
                 and s.get("elapsed_s", 0.0) >= self.start_s)
        yield "over_pace", 20.0, bool(early and rem is not None and emp is not None
                                       and emp < rem - max(self.margin, 0.1 * rem))
        yield "fading", self.fade_hold, None if fade is None else fade <= self.fade_pct
        yield "reserve_conflict", self.conflict_hold, None if fade is None else (wb < 0.05 and fade >= -1.0)
        recent_fade = self._last_fading is not None and s["t"] - self._last_fading < self.fade_memory
        yield "finish_window", 10.0, (None if fade is None else
                                      bool(rem is not None and 0 < rem < self.finish_m and wb > 0.4
                                           and fade >= -2.0 and not recent_fade))

    def check(self, s: Dict[str, Any], distance_m: Optional[float] = None) -> List[Dict[str, Any]]:
        if not s:
            return []
        t, out = s["t"], []
        for kind, hold, on in self._conditions(s, distance_m):
            armed = self._armed.setdefault(kind, True)
            if kind == "fading" and on:
                self._last_fading = t
            if on is None:
                continue
            if on:
                self._off_since[kind] = None
                since = self._since.get(kind)
                if since is None:
                    self._since[kind] = since = t
                if armed and t - since >= hold:
                    ev = {"kind": kind, "t": t, "elapsed_s": s.get("elapsed_s"),
                          "distance_m": s.get("distance_m"), "detail": self._detail(kind, s)}
                    self._armed[kind] = False
                    self.log.append(ev)
                    record.emit("screen", **ev)
                    out.append(ev)
            else:
                self._since[kind] = None
                off = self._off_since.get(kind)
                if off is None:
                    self._off_since[kind] = off = t
                if not armed and t - off >= self.rearm:
                    self._armed[kind] = True
        return out

    @staticmethod
    def _detail(kind: str, s: Dict[str, Any]) -> str:
        if kind == "over_pace":
            return (f"at {s['power_w']:.0f} W the reserve runs out in {s['empties_in_m']:.0f} m with "
                    f"{s['remaining_m']:.0f} m to go; sustainable pace {format_pace(s.get('sustainable_pace_s500'))}"
                    f" vs {format_pace(s.get('pace_water_s500'))} now")
        if kind == "fading":
            return f"speed at {s['rate_spm']:.0f} spm is {s['fade_pct']:+.1f}% vs the same rate early in the race"
        if kind == "reserve_conflict":
            return (f"reserve reads {100 * s['w_bal_frac']:.0f}% but speed at this rate holds "
                    f"({s['fade_pct']:+.1f}%): critical power is probably set too low")
        if kind == "finish_window":
            return f"{s['remaining_m']:.0f} m to go with {100 * s['w_bal_frac']:.0f}% of the reserve left"
        return ""


class CueSink:
    """Where delivered cues go: a log here; text-to-speech on a phone; ``act`` (the
    paddler twin) receives the cue's ask and pace."""

    def __init__(self, state: RaceState, speak: Optional[Callable[[str], Any]] = None,
                 act: Optional[Callable[[Dict[str, Any]], Any]] = None) -> None:
        self.state, self.speak, self.act = state, speak, act
        self.log: List[Dict[str, Any]] = []

    def __call__(self, action: Dict[str, Any]) -> str:
        s = self.state.last
        entry = {"t": s.get("t"), "elapsed_s": s.get("elapsed_s"), "distance_m": s.get("distance_m"),
                 "text": action["text"], "ask": action.get("ask", "none"),
                 "pace_s500": action.get("pace_s500"), "line_m": action.get("line_m"),
                 "source_events": action.get("events", []),
                 "source": action.get("source")}
        self.log.append(entry)
        record.emit("cue", **entry)
        if self.speak is not None:
            self.speak(action["text"])
        if self.act is not None:
            self.act(action)
        return "spoken"


def make_cue_gate(sink: CueSink, state: RaceState, *, max_words: int = 12,
                  quiet_start_s: float = 60.0, quiet_finish_m: float = 100.0,
                  cooldown_s: float = 120.0, max_cues: int = 15) -> ActuatorGate:
    """The gate every cue passes: word limit, quiet start and finish, cooldown, cap."""

    def words(a):
        n = len(str(a.get("text", "")).split())
        if n == 0:
            return "empty cue"
        return None if n <= max_words else f"cue has {n} words; the limit is {max_words}"

    def menu(a):
        ask = a.get("ask", "none")
        if ask not in _ASKS:
            return f"unknown ask {ask!r}; one of {', '.join(_ASKS)}"
        pace = a.get("pace_s500")
        if pace is not None and not (60.0 <= float(pace) <= 400.0):
            return f"pace {pace} s/500 m is not a racing pace"
        line = a.get("line_m")
        if line is not None and not (40.0 <= float(line) <= 600.0):
            return f"line_m {line} is outside 40-600 m from shore"
        return None

    def quiet(a):
        s = state.last
        if s.get("elapsed_s", 0.0) < quiet_start_s:
            return "quiet during the start"
        rem = s.get("remaining_m")
        if rem is not None and rem < quiet_finish_m:
            return "quiet in the last metres"
        return None

    return ActuatorGate(sink, [Validate(words, "words"), Validate(menu, "menu"), Validate(quiet, "quiet"),
                               Cooldown(cooldown_s), MaxCount(max_cues)], name="cue")


COACH_SYSTEM = """You are an outrigger canoe (OC1) race coach speaking into the paddler's ear \
during a race. You receive the live race state from the boat's sensors and the events a fast \
screen raised. Your only aim is the fastest time to the finish.

Rules:
- Give at most one cue per update, only if acting on it would make the boat faster to the \
finish. Otherwise give no cue. Silence is often right.
- A cue is at most 12 words, plain, something the paddler can do now (pace, rate, length, \
line, when to go). Paces are per 500 m through the water.
- The reserve figure comes from a model calibrated on a test. Trust it early in the race. \
Later, if it disagrees with the speed at the same stroke rate (no fade), trust the speed.
- Fading speed at the same rate late in a race usually means tiring blade work: a slightly \
lower rate with a longer stroke can be faster than forcing the rate.
- To speak, call the cue tool once. Then reply in one line with the reason, or reply \
"no cue: <reason>" without calling it."""


COACH_PANEL_SYSTEM = """You are an outrigger canoe (OC1) race coach speaking into the paddler's ear \
during a race. Your only aim is the fastest time to the finish.

What you have:
- A live panel: the boat's numbers stream every second as named metrics (list_available_metrics, \
get_metric_series, characterize_window_metric, metric_rise_fall).
- Standing watches: conditions over the panel checked every second with no model call; when \
one holds for its sustain time it wakes you. You start with four: over_pace, fading, \
reserve_conflict, finish_window. You may add, replace or remove watches (add_watch, \
remove_watch, list_watches, get_alerts) and define your own metrics for them or for yourself \
(define_derived_metric: one formula; define_program_metric: several steps, or a short program \
admitted after a speed check; define_streaming_estimator: bands, change points). Anything you \
define keeps computing every second for the rest of the race. In formulas, max and min reduce \
a series (numpy style): for the larger or smaller of two numbers use maximum(a, b) / minimum(a, b).
- You are also woken every few minutes to check in.

Speaking: call cue once, with text (at most 12 words, plain, something to do now) and ask, one \
of: ease (a little less effort), push (a little more), hold (keep this effort), rate_up, \
rate_down (about two strokes a minute), finish (start the final sprint now), none (no change). \
With ease, push or hold you may give pace_s500, a target pace per 500 m AS THE PADDLER'S DISPLAY \
SHOWS IT (over the ground: through-the-water speed plus the push along the heading); they will \
chase it by feel. The paddler may not follow, and a change takes several seconds. Cues are \
limited: no cue in the first minute or the last 100 m, at least 60 s apart, and a cap per race.

Know:
- The reserve figure comes from a model calibrated on a short test; it can be off. Trust it \
early; later, if it disagrees with the speed at the same stroke rate, trust the speed.
- Running the reserve empty forces a long slowdown well below critical power. Even pacing to the \
line, with a finishing sprint, is usually fastest.
- As paddlers tire their best stroke rate drops; fading speed at the same rate can mean the rate \
is now too high.
- Each of your model calls takes several seconds of race time. Silence is often right.

End with one line: the reason for your cue, or "no cue: <reason>"."""


COACH_PANEL_SYSTEM_V2 = """You are an outrigger canoe (OC1) race coach speaking into the paddler's ear \
during a race. Your only aim is the fastest time to the finish. You are judged on the finish \
time, not on how much you say: a wrong cue costs time, and a paddler who is pacing well is best \
left alone.

Each update gives you the live state, the last five minutes as one-minute means, and any alert. \
A live panel streams every number each second; you can read series, define your own metrics \
(define_derived_metric; define_program_metric for several steps or a short program) and your own \
watch rules (add_watch, remove_watch). You start with four watches: over_pace, fading, \
reserve_conflict, finish_window. They are crude; rewrite or remove ones that fire wrongly for this \
paddler. In formulas max/min reduce a series; for two numbers use maximum(a, b) / minimum(a, b).

What the numbers mean, and how far to trust them:
- shaft_power_w is the paddler's own effort, measured by the paddle pod. w_bal_pod_frac is the \
reserve above critical power computed from it; it is the best effort signal you have.
- w_bal_frac, empties_in_m and sustainable_pace_s500 come from boat speed through the hull model. \
They mistake tired, less efficient blade work for easing off (late in a race they refill while the \
paddler is spent) and depend on a short pre-race test. Use them only with care.
- Both reserves rest on a 3- and 12-minute test, which can be off. Check it against the race: \
cp_implied_w is the highest steady 10-minute power held while a reserve read nearly empty, with no \
fade; if it is well above the critical power in use, the calibration is low. You can correct it \
with set_critical_power (a logged revision); then the reserves recompute from now on.
- fade_pct (speed through the water at this stroke rate vs early in the race) needs no \
calibration: a fade of several percent that lasts means real fatigue or a rate that is now too \
high, since the best rate drops as a paddler tires.
- push_along_mps is the current and wind along the heading (+ helps); the display shows speed over \
the ground.

Coaching:
- Running the reserve empty forces a long slowdown well below critical power: prevent it early, \
with one clear cue, then let the paddler settle. Even pacing to the line plus a finishing sprint \
spent from what is left is usually fastest.
- Do not ease a paddler whose effort is at or below a sustainable level and whose speed holds.
- Prefer one decisive cue with a target over several small ones. Repeat only if the evidence says \
the paddler did not respond (the effect takes 10-20 s and may be partial).
- Each model call costs seconds of race time; when nothing needs saying, say "no cue" at once.

Speaking: call cue once, with text (at most 12 words, plain, something to do now) and ask, one \
of: ease (a little less effort), push (a little more), hold (keep this effort), rate_up, \
rate_down (about two strokes a minute), finish (start the final sprint now), none. With ease, \
push or hold you may give pace_s500, a target pace per 500 m as the display shows it (over the \
ground); they will chase it by feel. No cue in the first minute or last 100 m, at least 60 s \
apart, and a cap per race.

End with one line: the reason for your cue, or "no cue: <reason>"."""


WATER_SECTION = """

The water (this race): the course runs along a shore, one long leg out and one back. The tidal \
stream is weaker close to shore and, behind points, eddies run against it inshore; the stream can \
strengthen, weaken or turn during the race. The race brief has a forecast (also current_forecast); \
it is local knowledge and can be wrong. line_m is the paddler's distance from shore, along_m the \
position along the course from the start, heading_deg the heading (0 = north, the first leg). \
push_local_mps estimates the current along the heading right now from GPS speed against what the \
pod's shaft power should give (20-s means; it drifts slowly as the blade work tires); compare it \
with push_along_mps (the race-long fit), over time and across lines, to see where the water is \
better and whether the forecast holds. Changing line is slow: crossing 100 m takes about two \
minutes and costs about 12 m of distance, so it pays only if the water differs for long enough. \
Against the stream the inshore water is usually better; with it, the channel. Cue line_in or \
line_out (100 m), or give line_m (a target distance from shore, 40-600 m)."""


WATER_SECTION_V4 = """

How to use the water: at the start of each leg (you are woken then) make a plan for the leg: \
compare lines with stream_by_line (the forecast's stream along your heading at a line and \
position) and choose where to paddle. Against the stream, the inshore water is often worth \
several seconds per minute; with it, the channel. When a line is clearly better for most of the \
leg, go there early and in one move: cue line_in or line_out WITH line_m (e.g. 50-80 m from shore \
against the stream, 300-450 m with it); 100-m steps arrive too late. Eddies (forecast, uncertain) \
can reverse the inshore water. Once there, check the forecast against push_local_mps; if the \
water is not what the forecast said (the stream has turned, an eddy is elsewhere), change the \
plan. Pacing still matters: keep the reserve from running empty."""


_CUE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "what the paddler hears, at most 12 words"},
        "ask": {"type": "string", "enum": ["ease", "push", "hold", "rate_up", "rate_down", "finish",
                                           "line_in", "line_out", "none"],
                "description": "the change asked for"},
        "pace_s500": {"type": "number", "description": "optional target pace, s per 500 m, as the display shows it"},
        "line_m": {"type": "number", "description": "with line_in / line_out: optional target distance from shore (m, 40-600)"},
    },
    "required": ["text"],
}


class RaceCoach:
    """The reasoning tier: one consultation per wake-up, through ``run_tool_loop``.

    With ``panel`` (a :class:`~.race_panel.RacePanel`) the coach also gets the
    panel's tools (series, its own metrics and watch rules) and the panel system
    prompt."""

    def __init__(self, provider: Any, state: RaceState, gate: ActuatorGate, *,
                 system: Optional[str] = None, caller: str = "race_coach", max_tokens: int = 400,
                 panel: Any = None, max_iterations: Optional[int] = None,
                 cache_system: Optional[bool] = None, version: int = 1) -> None:
        self.provider, self.state, self.gate, self.panel = provider, state, gate, panel
        self.version = int(version)
        if system is None and panel is not None and self.version >= 2:
            system = (COACH_PANEL_SYSTEM_V2 + (WATER_SECTION if self.version >= 3 else "")
                      + (WATER_SECTION_V4 if self.version >= 4 else ""))
        self.forecast: Optional[Dict[str, Any]] = None
        self.cache_system = (panel is not None) if cache_system is None else bool(cache_system)
        self.system = system or (COACH_PANEL_SYSTEM if panel is not None else COACH_SYSTEM)
        self.caller, self.max_tokens = caller, max_tokens
        self.consultations: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        reg = ToolRegistry()
        reg.register(self.race_history, name="race_history",
                     description="Recent race history: 30-second means of speed through the water, "
                                 "power, reserve, stroke rate, fade and the push of current and wind.")
        reg.register(self.cue, name="cue", writes=True, input_schema=_CUE_SCHEMA,
                     description="Speak one short cue (at most 12 words) to the paddler now.")
        allowed = ["race_history", "cue"]
        if panel is not None:
            for name, tool in panel.tools().items():
                reg.register(tool.func, name=name, description=tool.description,
                             input_schema=tool.input_schema, writes=tool.writes)
                allowed.append(name)
        self.recalibrations: List[Dict[str, Any]] = []
        if self.version >= 3:
            reg.register(self.current_forecast, name="current_forecast",
                         description="The pre-race forecast of the tidal stream, its change during the race, "
                                     "the inshore weakening and the eddies (local knowledge; can be wrong).")
            allowed.append("current_forecast")
        if self.version >= 4:
            reg.register(self.stream_by_line, name="stream_by_line",
                         description="The forecast's stream along the current heading (m/s, + helps) at each "
                                     "given distance from shore, here or at another position along the course.")
            allowed.append("stream_by_line")
        if self.version >= 2:
            reg.register(self.set_critical_power, name="set_critical_power", writes=True,
                         description="Correct the critical power the reserve models use from now on "
                                     "(boat power cp_w and/or shaft power cp_shaft_w, in W), with the reason.")
            allowed.append("set_critical_power")
        self.policy = ToolPolicy(reg, allowed, max_iterations=max_iterations or (5 if panel is not None else 2))

    def race_history(self, minutes: float = 5.0) -> Dict[str, Any]:
        H = [h for h in self.state.history
             if h.get("elapsed_s") is not None
             and h["elapsed_s"] >= (self.state.last.get("elapsed_s", 0.0) - 60.0 * float(minutes))]
        rows, bucket = [], []
        for h in H:
            bucket.append(h)
            if len(bucket) == 6:
                rows.append(_mean_row(bucket))
                bucket = []
        if bucket:
            rows.append(_mean_row(bucket))
        return {"every_30s": rows}

    def cue(self, text: str, ask: str = "none", pace_s500: Optional[float] = None,
            line_m: Optional[float] = None) -> Dict[str, Any]:
        res = self.gate.deliver({"text": text, "ask": ask, "pace_s500": pace_s500, "line_m": line_m,
                                 "events": [e["kind"] for e in self._events], "source": self.caller},
                                source=self.caller)
        return {"spoken": bool(res.get("delivered")), "blocked": bool(res.get("blocked")),
                "reason": res.get("reason")}

    def set_critical_power(self, cp_w: Optional[float] = None, cp_shaft_w: Optional[float] = None,
                           reason: str = "") -> Dict[str, Any]:
        st = self.state
        base_b = st.athlete.cp_w
        base_s = st.athlete_shaft.cp_w if st.athlete_shaft is not None else None
        for v, base, nm in ((cp_w, base_b, "cp_w"), (cp_shaft_w, base_s, "cp_shaft_w")):
            if v is not None and (base is None or not 0.8 * base <= float(v) <= 1.6 * base):
                return {"ok": False, "error": f"{nm}={v} is outside 0.8-1.6 x the test value ({base})"}
        if len(self.recalibrations) >= 3:
            return {"ok": False, "error": "at most 3 recalibrations per race"}
        out = st.recalibrate(cp_w=cp_w, cp_shaft_w=cp_shaft_w)
        entry = {"t": st.last.get("t"), "elapsed_s": st.last.get("elapsed_s"), "reason": reason, **out}
        self.recalibrations.append(entry)
        record.emit("revision", target="critical_power", **entry)
        return {"ok": True, **out}

    def stream_by_line(self, lines_m: Optional[List[float]] = None, along_m: Optional[float] = None) -> Dict[str, Any]:
        from .sim.twin import forecast_stream
        if not self.forecast:
            return {"ok": False, "error": "no forecast for this race"}
        s = self.state.last
        hd = s.get("heading_deg", 0.0) or 0.0
        sign = 1.0 if (hd < 90.0 or hd > 270.0) else -1.0
        pos = float(along_m) if along_m is not None else float(s.get("along_m", 0.0) or 0.0)
        t = float(s.get("elapsed_s", 0.0) or 0.0)
        lines = lines_m or [50, 80, 120, 200, 300, 450]
        return {"ok": True, "heading_deg": round(hd), "along_m": round(pos), "race_s": round(t),
                "along_heading_mps": {str(int(y)): round(sign * forecast_stream(self.forecast, float(y), pos, t), 3)
                                      for y in lines}}

    def current_forecast(self) -> Dict[str, Any]:
        return dict(self.forecast or {"note": "no forecast for this race"})

    def _recent_minutes(self, n: int = 5) -> List[Dict[str, Any]]:
        H = [h for h in self.state.history if h.get("elapsed_s") is not None]
        if not H:
            return []
        end = H[-1]["elapsed_s"]
        out = []
        for k in range(n, 0, -1):
            b = [h for h in H if end - 60.0 * k < h["elapsed_s"] <= end - 60.0 * (k - 1)]
            if not b:
                continue
            def m(key, d=1):
                xs = [x[key] for x in b if x.get(key) is not None and math.isfinite(x[key])]
                return round(sum(xs) / len(xs), d) if xs else None
            row = {"min_ago": k, "speed_water": m("speed_water", 2), "speed_gps": m("speed_gps", 2),
                   "rate_spm": m("rate_spm", 0), "boat_power_w": m("power_w", 0),
                   "shaft_power_w": m("shaft_power_w", 0), "fade_pct": m("fade_pct", 1),
                   "reserve_pod_pct": None if m("w_bal_pod_frac", 3) is None else round(100 * m("w_bal_pod_frac", 3)),
                   "reserve_boat_pct": None if m("w_bal_frac", 3) is None else round(100 * m("w_bal_frac", 3)),
                   "push_along_mps": m("push_along_mps", 2), "push_local_mps": m("push_local_mps", 2),
                   "line_m": m("line_m", 0), "along_m": m("along_m", 0)}
            out.append({k2: v for k2, v in row.items() if v is not None})
        return out

    def consult(self, events: List[Dict[str, Any]], *, why: str = "events") -> Dict[str, Any]:
        self._events = list(events)
        s = self.state
        msg = {"update": why, "events": [{"kind": e["kind"], "detail": e["detail"]} for e in events],
               "state": s.compact()}
        if self.version >= 2:
            last = s.last
            extra = {k: last.get(k) for k in ("shaft_power_w", "cp_shaft_w", "w_bal_pod_frac",
                                              "empties_in_m_pod", "sustainable_shaft_power_w",
                                              "line_m", "along_m", "heading_deg", "push_local_mps")}
            msg["state"].update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in extra.items() if v is not None})
            if self.panel is not None:
                ev = self.panel.service.buffer
                for k in ("cp_implied_w", "steady_power_10min_w", "reserve_low_s"):
                    ser = ev.series(k, 3.0)
                    if ser:
                        v = ser[-1][1] if isinstance(ser[-1], (tuple, list)) else ser[-1]
                        msg["state"][k] = round(float(v), 1)
            msg["last_5_min"] = self._recent_minutes()
            if not self.consultations:
                msg["race"] = {"distance_m": s.distance_m, "course": "out-and-back legs; turns cost a few seconds",
                               "calibration": ("critical power and reserve from an all-out 3- and 12-minute test: "
                                               f"boat {s.athlete.cp_w:.0f} W / {s.athlete.w_prime_j / 1000:.1f} kJ" +
                                               (f"; shaft (pod) {s.athlete_shaft.cp_w:.0f} W / "
                                                f"{s.athlete_shaft.w_prime_j / 1000:.1f} kJ" if s.athlete_shaft else ""))}
                if self.version >= 3 and self.forecast:
                    msg["race"]["forecast"] = self.forecast
                    msg["race"]["course"] = (f"along a shore: one leg of {s.distance_m / 2:.0f} m north, "
                                             "then back south to the start")
        t0 = s.last.get("t")
        res = run_tool_loop(self.provider, system=self.system,
                            messages=[Message(role="user", text=json.dumps(msg))],
                            policy=self.policy, max_tokens=self.max_tokens, caller=self.caller,
                            cache_system=self.cache_system, catch_errors=True)
        out = {"t_asked": t0, "t_answered": s.last.get("t"), "why": why,
               "events": [e["kind"] for e in events], "text": res.text,
               "cued": any(tc.name == "cue" for tc, _ in res.tool_log),
               "error": None if res.error is None else str(res.error)}
        self.consultations.append(out)
        record.emit("decision", **out)
        return out


def _mean_row(bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    def m(k, n=2):
        xs = [b[k] for b in bucket if b.get(k) is not None and math.isfinite(b[k])]
        return round(sum(xs) / len(xs), n) if xs else None
    e = m("elapsed_s", 0)
    return {"elapsed_min": None if e is None else round(e / 60.0, 1),
            "speed_water": m("speed_water"), "power_w": m("power_w", 0),
            "reserve_pct": None if m("w_bal_frac") is None else round(100 * m("w_bal_frac"), 0),
            "rate_spm": m("rate_spm", 0), "fade_pct": m("fade_pct", 1),
            "push_along_mps": m("push_along_mps")}


def make_coach_provider(name: str, model: Optional[str] = None, *,
                        api_key_env: str = "ANTHROPIC_API_KEY",
                        api_base_url: str = "https://api.anthropic.com") -> Any:
    """The coach's model: ``stand-in`` (scripted, no network) or an agentic_experiments
    provider.  For ``anthropic`` the key is read from ``api_key_env`` (Claude Code cloud
    sessions withhold ``ANTHROPIC_API_KEY``) and requests go only to ``api_base_url``
    (inside a Claude Code session ``ANTHROPIC_BASE_URL`` points at its own connection)."""
    if name == "stand-in":
        return stand_in_coach()
    import os
    from agent_lib.providers.factory import make_provider
    p = make_provider(name, model)
    if name != "anthropic":
        return p
    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(f"--provider anthropic needs the API key in ${api_key_env}")
    os.environ["ANTHROPIC_BASE_URL"] = api_base_url
    from agent_lib.providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider(model=p.model, api_key=key)


def stand_in_coach() -> FakeProvider:
    """A scripted stand-in for the coach model: maps each event to a fixed cue.
    For tests and demos without a model; not evidence of what a model would say."""

    def answer(system, messages, tools):
        last = messages[-1]
        if last.tool_results:
            return "cued: event-driven stand-in"
        msg = json.loads(last.text)
        kinds = [e["kind"] for e in msg.get("events", [])]
        st = msg.get("state", {})
        if "over_pace" in kinds:
            sus, push = st.get("sustainable_pace_s500"), st.get("push_along_mps") or 0.0
            pace = None if not sus else round(500.0 / (500.0 / sus + push), 1)
            args = {"text": f"Too hot for the distance. Settle to {format_pace(pace or sus)}.",
                    "ask": "ease"}
            if pace:
                args["pace_s500"] = pace
        elif "fading" in kinds:
            args = {"text": "Speed fading at this rate. Drop two beats, lengthen.", "ask": "rate_down"}
        elif "finish_window" in kinds:
            args = {"text": "You have reserve left. Go now, lift the rate.", "ask": "finish"}
        else:
            return "no cue: nothing to change"
        return {"tool_calls": [{"name": "cue", "arguments": args}]}

    return FakeProvider(answer, model="stand-in-coach")


def replay_race(rows: List[Dict[str, Any]], state: RaceState, coach: RaceCoach,
                screen: Optional[RaceScreen] = None, *, latency_s: float = 6.0,
                periodic_s: Optional[float] = 300.0, record_path: Optional[str] = None
                ) -> Dict[str, Any]:
    """Feed stroke rows at their own times through the race state and the screen on
    a virtual clock; wake the coach on screen events (and every ``periodic_s``),
    one consultation at a time, each model call taking ``latency_s``."""
    screen = screen or RaceScreen()
    rows = [r for r in rows if r.get("t") is not None]
    if not rows:
        return {}
    sched = Scheduler(VirtualClock(float(rows[0]["t"])), latency=fixed({}, default=latency_s))
    busy = {"on": False, "pending": [], "last_wake": None}

    def wake(events, why):
        if busy["on"]:
            busy["pending"].extend(events)
            return
        busy["on"], busy["last_wake"] = True, sched.now()

        def run():
            try:
                coach.consult(events, why=why)
            finally:
                busy["on"] = False
                if busy["pending"]:
                    ev, busy["pending"] = busy["pending"], []
                    sched.post(wake, ev, "events")
        sched.spawn(run, name="race_coach")

    def on_row(row):
        s = state.update(row)
        if not s or s.get("elapsed_s") is None:
            return
        ev = screen.check(s, state.distance_m)
        if ev:
            wake(ev, "events")
        elif periodic_s and (busy["last_wake"] is None or sched.now() - busy["last_wake"] >= periodic_s) \
                and s["elapsed_s"] > 60.0:
            wake([], "periodic")

    for r in rows:
        sched.at(float(r["t"]), on_row, r)
    if record_path:
        with record.recording(RunRecord(record_path)):
            sched.run()
    else:
        sched.run()
    return {"screen": screen.log, "consultations": coach.consultations,
            "gate": coach.gate.status(), "history": state.history}
