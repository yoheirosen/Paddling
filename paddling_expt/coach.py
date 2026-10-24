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

__all__ = ["RaceScreen", "CueSink", "make_cue_gate", "RaceCoach", "COACH_SYSTEM",
           "stand_in_coach", "replay_race", "format_pace"]


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
    """Where delivered cues go: a log here; text-to-speech on a phone."""

    def __init__(self, state: RaceState, speak: Optional[Callable[[str], Any]] = None) -> None:
        self.state, self.speak = state, speak
        self.log: List[Dict[str, Any]] = []

    def __call__(self, action: Dict[str, Any]) -> str:
        s = self.state.last
        entry = {"t": s.get("t"), "elapsed_s": s.get("elapsed_s"), "distance_m": s.get("distance_m"),
                 "text": action["text"], "source_events": action.get("events", [])}
        self.log.append(entry)
        record.emit("cue", **entry)
        if self.speak is not None:
            self.speak(action["text"])
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

    def quiet(a):
        s = state.last
        if s.get("elapsed_s", 0.0) < quiet_start_s:
            return "quiet during the start"
        rem = s.get("remaining_m")
        if rem is not None and rem < quiet_finish_m:
            return "quiet in the last metres"
        return None

    return ActuatorGate(sink, [Validate(words, "words"), Validate(quiet, "quiet"),
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


class RaceCoach:
    """The reasoning tier: one consultation per wake-up, through ``run_tool_loop``."""

    def __init__(self, provider: Any, state: RaceState, gate: ActuatorGate, *,
                 system: str = COACH_SYSTEM, caller: str = "race_coach", max_tokens: int = 400) -> None:
        self.provider, self.state, self.gate = provider, state, gate
        self.system, self.caller, self.max_tokens = system, caller, max_tokens
        self.consultations: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        reg = ToolRegistry()
        reg.register(self.race_history, name="race_history",
                     description="Recent race history: 30-second means of speed through the water, "
                                 "power, reserve, stroke rate, fade and the push of current and wind.")
        reg.register(self.cue, name="cue", writes=True,
                     description="Speak one short cue (at most 12 words) to the paddler now.")
        self.policy = ToolPolicy(reg, ["race_history", "cue"], max_iterations=2)

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

    def cue(self, text: str) -> Dict[str, Any]:
        res = self.gate.deliver({"text": text, "events": [e["kind"] for e in self._events]},
                                source=self.caller)
        return {"spoken": bool(res.get("delivered")), "blocked": bool(res.get("blocked")),
                "reason": res.get("reason")}

    def consult(self, events: List[Dict[str, Any]], *, why: str = "events") -> Dict[str, Any]:
        self._events = list(events)
        s = self.state
        msg = {"update": why, "events": [{"kind": e["kind"], "detail": e["detail"]} for e in events],
               "state": s.compact()}
        t0 = s.last.get("t")
        res = run_tool_loop(self.provider, system=self.system,
                            messages=[Message(role="user", text=json.dumps(msg))],
                            policy=self.policy, max_tokens=self.max_tokens, caller=self.caller,
                            catch_errors=True)
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
            text = f"Too hot for the distance. Settle to {format_pace(st.get('sustainable_pace_s500'))}."
        elif "fading" in kinds:
            text = "Speed fading at this rate. Drop two beats, lengthen."
        elif "finish_window" in kinds:
            text = "You have reserve left. Go now, lift the rate."
        else:
            return "no cue: nothing to change"
        return {"tool_calls": [{"name": "cue", "arguments": {"text": text}}]}

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
