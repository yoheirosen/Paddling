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

import numpy as np
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

_ASKS = ("ease", "push", "hold", "rate_up", "rate_down", "finish", "line_in", "line_out",
         "hang_on", "let_go", "clear_rudder", "reassure", "tell", "piece", "none")

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
                 "distance_left_m": action.get("distance_left_m"),
                 "segments": action.get("segments"),
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
        if ask == "piece":
            from .sim.twin import check_piece
            return check_piece(a.get("segments"))
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


WATER_SECTION_V6 = """

How to use the water: before moving the paddler across, price the move with line_change_value \
(the seconds a move to a line saves over the rest of the leg by the forecast, net of the crossing). \
Against the stream the inshore water usually pays; moving out with the stream rarely pays back the \
crossing unless the stream is strong and much of the leg is left. When a move clearly pays, make \
it early and in one go, with line_m. Eddies (forecast, uncertain) can reverse the inshore water. \
Once there, check the forecast against push_local_mps; if the water is not what the forecast said, \
re-plan. Pacing still matters: keep the reserve from running empty."""


DRAFT_SECTION = """

Other boats (this race): the paddler starts with a pack slightly faster than their even pace. \
Sitting on a wash, a few metres behind a boat, cuts drag by roughly a tenth, so the pack's speed \
costs less effort there; falling off the wash costs that saving, and a gap of more than a few \
boat lengths is hard to close. You cannot see the boats: you see a wash in the numbers, as speed \
through the water higher than the shaft power alone would give (push_local_mps rising well above \
push_along_mps) at the same effort. Hanging on pays if the effort it takes (shaft power while on \
the wash) can be held to the finish on the pod's reserve, plus the sprint; if it cannot, letting \
go early, before the reserve runs out, beats blowing up and being dropped anyway. Paddlers tend to \
hang on by instinct until it is too late, or give up a wash that was worth keeping. Cue hang_on \
(work to stay on the wash) or let_go (drop off and settle to your own pace)."""


TIER_SECTION = """

A rule layer coaches by default, and you supervise it. Its rules: pacing alarms on the pod's \
reserve (over-pace -> ease; fading -> rate down; reserve conflict -> hold; finish window -> \
finish) and "inside against the stream": at the start of a leg, if the FORECAST stream is \
against the heading, it sends the paddler to 60 m from shore. It is right most of the time. You \
are the exception handler: step in only when the live data shows a rule's assumption is wrong for \
this paddler or this water, for example the stream is not what the forecast said (compare \
push_local_mps with stream_by_line), it has turned, an eddy is elsewhere, the calibration is off \
(cp_implied_w), an ease would slow a paddler whose speed and effort are fine, or the finish needs \
judgement. To overrule, call mute_rule for that lever ("line" or "pace") for some minutes and give \
your own cue; unmute_rule hands it back. Each update lists what the rule did recently. When the \
rule is doing fine, say "no cue" at once."""


_CUE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "what the paddler hears, at most 12 words"},
        "ask": {"type": "string", "enum": ["ease", "push", "hold", "rate_up", "rate_down", "finish",
                                           "line_in", "line_out", "hang_on", "let_go", "none"],
                "description": "the change asked for"},
        "pace_s500": {"type": "number", "description": "optional target pace, s per 500 m, as the display shows it"},
        "line_m": {"type": "number", "description": "with line_in / line_out: optional target distance from shore (m, 40-600)"},
    },
    "required": ["text"],
}


OPEN_COACH_SYSTEM = """You are coaching one paddler through an outrigger canoe (OC1) race, \
speaking into their ear over a radio link. What you are there for: their fastest finish.

What you have:
- Instruments, live every second: the paddle pod (the paddler's effort, as shaft power), GPS \
(speed and heading over the ground), and what the boat's computer derives from them through a \
hull model. Each wake-up shows the live state, the last five minutes, your notes and your \
recent cues. You can read any series further back, compute your own metrics over the live \
streams, and set your own alerts (watches) so that you are woken when something you care \
about happens. Otherwise you are woken every few minutes, whenever the paddler says \
something, and when a radio message comes in.
- The paddler. You can ask how they feel (ask_paddler) and speak to them (cue). They hear your \
words but act on the request attached to them: ease, push, hold, rate_up, rate_down, finish \
(start the final sprint), clear_rudder (a back-stroke that knocks anything caught off the \
rudder; it costs the boat's momentum), reassure (no change asked: you are telling them they are \
doing fine), tell (information; distance_left_m sets where they think the finish is), or none. \
With ease, push or hold you may add a target pace over the ground; they chase it by feel, \
through turns too, until the change fades. A cue takes 10-20 s to act; paddlers do not always \
follow one, and a followed change fades over minutes.
- Notes: you remember nothing between wake-ups. What you write with take_note is shown to you \
at every later wake-up: a plan, a suspicion to check, what you have tried and what it did.
- The boat computer's assumptions: set_critical_power (at most 3 times per race) and \
set_race_distance correct them.
- Names: the live state's names are for reading; watches and your own metrics use the stream \
names that list_available_metrics gives.

Facts about the instruments (facts, not instructions):
- Any instrument can fail or drift.
- The numbers derived from boat speed (power_w, reserve_left_pct, reserve_runs_out_in_m, \
sustainable_pace_s500, speed_vs_early_race_same_rate_pct) treat everything that slows the boat \
as the paddler: current, wind, drag, a tired blade. push_local_mps is the GPS speed minus the \
speed the pod's power would give in still water; it moves with current and wind and with \
anything that changes the boat's drag. push_along_mps is the current along the heading, \
estimated from the turns of the out-and-back course; it lags a change.
- Critical power and the reserve (w_bal_pod_frac from the pod, reserve_left_pct from boat speed) \
come from a 3- and 12-minute test before the race and can be 10-25 % off.
- The start is hard (a minute or two above critical power) and then settles.
- Paddlers misjudge themselves: what they say is evidence, not fact.

How you coach is up to you: what to watch, what to plan, when to speak and what to say, when to \
change your mind. Each model call costs a few seconds of race time; when nothing needs doing, \
stop. The cue gate: at most 12 words, at least 60 s between cues, none in the first minute or \
the last 100 m, and at most 20 cues per race.

End each wake-up with one line: what you did and why, or "no cue: <reason>"."""

OPEN_COACH_SYSTEM_V9 = """You are coaching one paddler through an outrigger canoe (OC1) race, \
speaking into their ear over a radio link. What you are there for: their fastest finish.

What you have:
- Instruments, live every second: the paddle pod (the paddler's effort, as shaft power), GPS \
(speed and heading over the ground), and what the boat's computer derives from them through a \
hull model. Each wake-up shows the live state, the last five minutes, your notes and your \
recent cues. You can read any series further back, compute your own metrics over the live \
streams, and set your own alerts (watches) so that you are woken when something you care \
about happens. Otherwise you are woken every few minutes, whenever the paddler says \
something, and when a radio message comes in.
- The paddler. You can ask how they feel (ask_paddler) and speak to them (cue). They hear your \
words but act on the request attached to them: ease, push, hold, rate_up, rate_down, finish \
(start the final sprint), clear_rudder (a back-stroke that knocks anything caught off the \
rudder; it costs the boat's momentum), reassure (no change asked: you are telling them they are \
doing fine), tell (information; distance_left_m sets where they think the finish is), or none. \
With ease, push or hold you may add a target pace over the ground; they chase it by feel, \
through turns too, until the change fades. A cue takes 10-20 s to act; paddlers do not always \
follow one, and a followed change fades over minutes.
- What the instruments saw after each cue: your recent cues show the pod's power, the boat-model \
power, stroke rate and GPS speed in the minute before and the minute after, and for clear_rudder \
whether the boat's speed dipped as a back-stroke does. A turn changes these numbers too.
- Memory: you remember nothing between wake-ups. Your plan (set_plan) is shown at the top of \
every wake-up; your notes (take_note) show the last 15.
- Timing: wake_me wakes you after a time or at a distance to go, for example when the next cue \
is allowed (wake_me with clear=true cancels the pending ones).
- The course: legs() gives one row per leg (direction, time, GPS speed, pod power, boat-model \
power, stroke rate, local push; turns left out), so legs in the same direction can be compared. \
get_metric_series returns a summary of about 30 points in race seconds; list_available_metrics \
gives the stream names.
- The boat computer's assumptions: set_critical_power (critical power and reserve size, at most \
3 times per race; both reserves are then recomputed over the race so far) and set_race_distance.
- Names: the live state's names are for reading; watches and your own metrics use the stream \
names that list_available_metrics gives.

Facts about the instruments (facts, not instructions):
- Any instrument can fail or drift.
- The numbers derived from boat speed (power_w, reserve_left_pct, reserve_runs_out_in_m, \
sustainable_pace_s500, speed_vs_early_race_same_rate_pct) treat everything that slows the boat \
as the paddler: current, wind, drag, a tired blade. push_local_mps is the GPS speed minus the \
speed the pod's power would give in still water; it moves with current and wind and with \
anything that changes the boat's drag. push_along_mps is the current along the heading, \
estimated from the turns of the out-and-back course; it lags a change.
- Critical power and the reserve (w_bal_pod_frac from the pod, reserve_left_pct from boat speed) \
come from a 3- and 12-minute test before the race and can be 10-25 % off.
- The start is hard (a minute or two above critical power) and then settles.
- A paddler who runs the reserve empty is forced well below critical power until part of it \
comes back.
- Paddlers misjudge themselves: what they say is evidence, not fact.

How you coach is up to you: what to watch, what to plan, when to speak and what to say, when to \
change your mind. Each model call costs a few seconds of race time; when nothing needs doing, \
stop. The cue gate: at most 12 words, at least 60 s between cues, none in the first minute or \
the last 100 m, and at most 20 cues per race.

End each wake-up with one line: what you did and why, or "no cue: <reason>"."""

_CUE_SCHEMA_OPEN = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "what the paddler hears, at most 12 words"},
        "ask": {"type": "string", "enum": ["ease", "push", "hold", "rate_up", "rate_down", "finish",
                                           "clear_rudder", "reassure", "tell", "none"],
                "description": "what the paddler should do"},
        "pace_s500": {"type": "number", "description": "with ease / push / hold: optional target pace, s per 500 m over the ground"},
        "distance_left_m": {"type": "number", "description": "with tell: how far the finish is from here (m)"},
    },
    "required": ["text"],
}


OPEN_COACH_SYSTEM_V10 = """You are coaching one paddler through an outrigger canoe (OC1) race, \
speaking into their ear over a radio link. What you are there for: their fastest finish.

What you have:
- Instruments, live every second: the paddle pod (the paddler's effort, as shaft power), GPS \
(speed and heading over the ground), a heart-rate chest strap, and what the boat's computer derives \
from them through a hull model. Each wake-up shows the live state, the last five minutes, your \
beliefs, plan and notes, and your recent cues. You can read any series further back, compute your \
own metrics over the live streams, and set your own alerts (watches) so that you are woken when \
something you care about happens. Otherwise you are woken every few minutes, whenever the paddler \
says something, and when a radio message comes in.
- The paddler. You can ask how they feel (ask_paddler) and speak to them (cue). They hear your \
words but act on the request attached to them: ease, push, hold, rate_up, rate_down, finish \
(start the final sprint), clear_rudder (a back-stroke that knocks anything caught off the \
rudder; it costs the boat's momentum), reassure (no change asked: you are telling them they are \
doing fine), tell (information; distance_left_m sets where they think the finish is), piece (below), \
or none. With ease, push or hold you may add a target pace over the ground; they chase it by feel, \
through turns too, until the change fades. A cue takes 4-15 s to act; paddlers do not always \
follow one, and a followed change fades over minutes.
- Experiments. A cue with ask 'piece' asks for a short test piece: up to 3 segments, each a change \
of effort (effort_pct, against what they are doing when it starts) and/or of stroke rate (rate_spm), \
held by the clock for its seconds; for example 150 s at +6 % then 120 s at 0. effort_pct is against \
the effort they hold when the piece starts (a piece already running counts, and the new piece \
replaces it); a pace target pauses while a piece runs. test_cost prices a piece by the boat \
computer's numbers. measure() reads any stretches of the race you name in race \
seconds: means with their uncertainty, the heart-rate trend, and a fit through the hull model that \
splits speed into the share of pod power that reaches the boat (scaled by drag, the pod's gain and \
blade efficiency) and a push of current and wind (which flips with direction on this course). \
Whether a test is worth its cost, and what its result means, is your call.
- What the instruments saw after each cue: your recent cues show the pod's power, the boat-model \
power, stroke rate, heart rate and GPS speed in the minute before and the minute after, and for \
clear_rudder whether the boat's speed dipped as a back-stroke does. A turn changes these numbers too.
- Beliefs: set_belief records what you believe, with a range and the evidence, shown at every \
wake-up. The beliefs critical_power_pod_w, reserve_size_pod_kj, critical_power_boat_w, \
reserve_size_boat_kj and pod_gain (the pod's reading divided by the true effort, from a race second \
you give) set the boat computer, which then recomputes both reserves over the race so far. Any \
other name (how honest the paddler is, their best stroke rate, how they take cues, ...) is for you.
- Memory: you remember nothing between wake-ups. Your plan (set_plan) is shown at the top of every \
wake-up; your notes (take_note) show the last 15.
- Timing: wake_me wakes you after a time or at a distance to go (wake_me with clear=true cancels \
the pending ones).
- The course: legs() gives one row per leg (direction, time, GPS speed, pod power, boat-model \
power, stroke rate, heart rate, local push; turns left out). get_metric_series returns a summary of \
about 30 points in race seconds; list_available_metrics gives the stream names. set_race_distance \
corrects the course length the boat's computer uses.
- Names: the live state's names are for reading; watches and your own metrics use the stream \
names that list_available_metrics gives. A watch condition or a derived metric is one expression \
over the current values (arithmetic, comparisons, and / or, maximum(a, b), minimum(a, b)); a \
program metric can also read past values with window(name, seconds).
- After a piece, your recent cues show each segment's readings against the minute before. With a \
critical_power_pod_w belief that has a range, the live state shows the pod's reserve at both ends.

Facts (facts, not instructions):
- Any instrument can fail or drift.
- The numbers derived from boat speed (power_w, reserve_left_pct, reserve_runs_out_in_m, \
sustainable_pace_s500, speed_vs_early_race_same_rate_pct) treat everything that slows the boat \
as the paddler: current, wind, drag, a tired blade. push_local_mps is the GPS speed minus the \
speed the pod's power would give in still water; it moves with current and wind and with \
anything that changes the boat's drag. push_along_mps is the current along the heading, \
estimated from the turns of the out-and-back course; it lags a change.
- Current and wind add to speed over the ground; drag, a drifting pod and blade efficiency scale \
how much speed a pod watt gives. A back-stroke clears weed; it does not change a drifting pod.
- Critical power and the reserve (w_bal_pod_frac from the pod, reserve_left_pct from boat speed) \
come from a 3- and 12-minute test before the race and can be 10-25 % off.
- Heart rate follows a change of effort over about a minute. At a steady effort below critical \
power it levels off; above critical power it keeps climbing as the reserve drains, and it falls \
back as the reserve refills. Its level at a given effort differs between paddlers and creeps up a \
few beats over a race.
- The start is hard (a minute or two above critical power) and then settles.
- A paddler who runs the reserve empty is forced well below critical power until part of it \
comes back.
- Paddlers misjudge themselves: what they say is evidence, not fact.
- A pace target is chased by feel within about -15 % to +12 % of the effort when it was given; \
ease or push adds to it; hold or a new target replaces it.
- On this course current and wind both push along the course, so each flips sign with direction.

How you coach is up to you: what to watch, what to test, what to plan, when to speak and what to \
say, when to change your mind. Each model call costs a few seconds of race time; when nothing \
needs doing, stop. The cue gate: at most 12 words, at least 60 s between cues, none in the first \
minute or the last 100 m, and at most 20 cues per race (a piece counts as one cue).

End each wake-up with one line: what you did and why, or "no cue: <reason>"."""

_CUE_SCHEMA_V10 = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "what the paddler hears, at most 12 words"},
        "ask": {"type": "string", "enum": ["ease", "push", "hold", "rate_up", "rate_down", "finish",
                                           "clear_rudder", "reassure", "tell", "piece", "none"],
                "description": "what the paddler should do"},
        "pace_s500": {"type": "number", "description": "with ease / push / hold: optional target pace, s per 500 m over the ground"},
        "distance_left_m": {"type": "number", "description": "with tell: how far the finish is from here (m)"},
        "segments": {"type": "array", "description": "with piece: 1-3 segments [{seconds 20-300, effort_pct -25..25, rate_spm -10..10}], at most 480 s in all",
                     "items": {"type": "object", "properties": {"seconds": {"type": "number"},
                                                                "effort_pct": {"type": "number"},
                                                                "rate_spm": {"type": "number"}}}},
    },
    "required": ["text"],
}

_MEASURE_SCHEMA = {
    "type": "object",
    "properties": {"windows": {"type": "array", "description": "1-4 stretches of the race, each {start_race_s, end_race_s}",
                               "items": {"type": "object", "properties": {"start_race_s": {"type": "number"},
                                                                          "end_race_s": {"type": "number"}}}},
                   "share": {"type": "number", "description": "optional: hold the share of pod power reaching the "
                                                              "boat at this value (e.g. from a clean earlier fit) "
                                                              "and give each window's GPS speed against it"}},
    "required": ["windows"],
}

_BELIEF_COMPUTER = ("critical_power_pod_w", "reserve_size_pod_kj", "critical_power_boat_w",
                    "reserve_size_boat_kj", "pod_gain")


# ── measuring stretches of a race (the experimenting coach; the scheduled-test rules) ──

def _speed_tab(state):
    from .race import _SPEED_TABLES, _speed_for
    st = state
    _speed_for(st.craft, st.total_kg, 100.0, st.ama_kg)          # builds the cached table
    return _SPEED_TABLES[(id(st.craft), round(st.total_kg, 3), round(st.ama_kg, 2))]

def measure_windows(state: RaceState, leg: Optional[float], windows: List[Dict[str, Any]],
                    share: Optional[float] = None) -> Dict[str, Any]:
    """Read stretches of the race (race seconds) from the 5-second record: means with
    standard errors (effective n = seconds / 30, the record being 30-s means), the
    heart-rate trend, and a hull-model fit of speed over the ground = still-water speed at
    (share x pod power) + push, the push flipping sign between the out and back legs."""
    st = state
    if not isinstance(windows, list) or not 1 <= len(windows) <= 4:
        return {"ok": False, "error": "give 1-4 windows, each {start_race_s, end_race_s}"}
    H = [h for h in st.history if h.get("elapsed_s") is not None]
    now = H[-1]["elapsed_s"] if H else 0.0
    out, fit_rows = [], []
    for w in windows:
        try:
            a, b = float(w["start_race_s"]), float(w["end_race_s"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "each window needs start_race_s and end_race_s (numbers)"}
        if b - a < 30.0 or a < 0 or a > now:
            out.append({"window": [a, b], "error": "a window must be at least 30 s long and already raced"})
            continue
        rows = [h for h in H if a <= h["elapsed_s"] <= b]
        turn = False
        if leg:
            legs_in = {int(h["distance_m"] // leg) for h in rows if h.get("distance_m") is not None}
            turn = len(legs_in) > 1
            rows = [h for h in rows if h.get("distance_m") is None or (h["distance_m"] % leg) >= 130.0]
        if len(rows) < 4:
            out.append({"window": [a, b], "error": "too little clean data (turns and the 30 s after them are left out)"})
            continue
        secs = rows[-1]["elapsed_s"] - rows[0]["elapsed_s"] + 5.0
        n_eff = max(1.0, secs / 30.0)

        def ms(k, nd=1):
            xs = np.array([h[k] for h in rows if h.get(k) is not None and math.isfinite(h[k])], float)
            if xs.size < 3:
                return None
            return [round(float(xs.mean()), nd), round(float(xs.std(ddof=1) / math.sqrt(n_eff)), nd + 1)]
        res: Dict[str, Any] = {"window": [round(a), round(b)], "clean_seconds": round(secs)}
        if leg:
            dirs = sorted({"out" if int(h["distance_m"] // leg) % 2 == 0 else "back" for h in rows
                           if h.get("distance_m") is not None})
            res["direction"] = dirs[0] if len(dirs) == 1 else "both"
            if turn:
                res["turn_inside"] = True
        for k, nm, nd in (("shaft_power_w", "pod_power_w", 0), ("shaft_power_raw_w", "pod_reading_raw_w", 0),
                          ("power_w", "boat_power_w", 0), ("speed_gps", "speed_gps", 3),
                          ("speed_water", "speed_water", 3), ("rate_spm", "rate_spm", 1),
                          ("heart_rate_bpm", "heart_rate_bpm", 1)):
            v = ms(k, nd)
            if v is not None:
                res[nm] = v
        if "pod_reading_raw_w" in res and res["pod_reading_raw_w"] == res.get("pod_power_w"):
            res.pop("pod_reading_raw_w")
        hr = [(h["elapsed_s"], h["heart_rate_bpm"]) for h in rows if h.get("heart_rate_bpm") is not None]
        if len(hr) >= 6:
            x = np.array([p[0] for p in hr]) / 60.0
            y = np.array([p[1] for p in hr])
            A = np.vstack([x - x.mean(), np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            sxx = float(((x - x.mean()) ** 2).sum())
            se = float(np.sqrt((resid ** 2).sum() / max(1, len(y) - 2) / max(1e-9, sxx)) * math.sqrt(len(y) / n_eff))
            res["heart_rate_trend_bpm_per_min"] = [round(float(coef[0]), 2), round(se, 2)]
        res["mean_values_are"] = "[mean, standard error]"
        out.append(res)
        if "pod_power_w" in res and "speed_gps" in res and res.get("direction") in ("out", "back"):
            fit_rows.append((res["pod_power_w"][0], res["pod_power_w"][1], res["speed_gps"][0],
                             res["speed_gps"][1], 1.0 if res["direction"] == "out" else -1.0))
    ans: Dict[str, Any] = {"ok": True, "windows": out}
    if fit_rows:
        ans["hull_fit"] = hull_fit(st, fit_rows)
    if share is not None:
        try:
            k = float(share)
        except (TypeError, ValueError):
            return {**ans, "fixed_share_error": "share must be a number"}
        if not 0.3 <= k <= 1.1:
            return {**ans, "fixed_share_error": "share must be 0.3-1.1"}
        Ptab, Vtab = _speed_tab(st)
        rows = []
        for w in out:
            if "pod_power_w" in w and "speed_gps" in w:
                v_still = float(np.interp(k * w["pod_power_w"][0], Ptab, Vtab))
                rows.append({"window": w["window"], "direction": w.get("direction"),
                             "still_water_speed_at_this_share": round(v_still, 3),
                             "gps_minus_that_mps": round(w["speed_gps"][0] - v_still, 3)})
        ans["at_fixed_share"] = {"share": k, "windows": rows,
                                 "note": "GPS speed minus the still-water speed this share of pod power gives: "
                                         "the push of current and wind if the share is right; drag or a pod "
                                         "change shows as a drop in both directions"}
    return ans

def hull_fit(state: RaceState, rows: List[tuple]) -> Dict[str, Any]:
    """Speed over the ground = still-water speed at (k x pod power) + s x push, s = +1 out, -1 back."""
    Ptab, Vtab = _speed_tab(state)
    P = np.array([r[0] for r in rows]); Pse = np.array([r[1] for r in rows])
    V = np.array([r[2] for r in rows]); Vse = np.array([r[3] for r in rows])
    S = np.array([r[4] for r in rows])
    ks = np.linspace(0.30, 1.10, 401)

    def solve(Pm, Vm):                                # (B, W) -> best k, push per draw
        Vk = np.interp(ks[None, :, None] * Pm[:, None, :], Ptab, Vtab)          # (B, K, W)
        if len(set(S.tolist())) > 1 or len(S) > 1:
            u = ((Vm[:, None, :] - Vk) * S).mean(axis=2)                           # (B, K)
            sse = ((Vm[:, None, :] - Vk - u[:, :, None] * S) ** 2).sum(axis=2)
        else:
            u = np.zeros(Vk.shape[:2]); sse = np.zeros(Vk.shape[:2])
        j = sse.argmin(axis=1)
        return ks[j], u[np.arange(len(j)), j], sse
    kd, ud, sse0 = solve(P[None, :], V[None, :])
    k_ref = state.eta_est
    out: Dict[str, Any] = {"windows_used": len(rows)}
    one_dir = len(set(S.tolist())) == 1
    if len(rows) == 1:
        if k_ref:
            u1 = float(V[0] - np.interp(k_ref * P[0], Ptab, Vtab)) * S[0]
            out.update({"note": "one stretch: the split needs a second stretch (the other direction, or another effort)",
                        "push_if_share_is_the_computers_mps": round(u1, 3), "computers_share": round(k_ref, 3)})
        return out
    rng = np.random.default_rng(int(1000 * float(P.sum() + V.sum())) % (2 ** 32))
    B = 300
    Pb = P[None, :] + rng.normal(size=(B, len(P))) * Pse[None, :]
    Vb = V[None, :] + rng.normal(size=(B, len(V))) * Vse[None, :]
    kb, ub, _ = solve(Pb, Vb)
    lo, hi = np.percentile(kb, [10, 90]); ulo, uhi = np.percentile(ub, [10, 90])
    out.update({"share_of_pod_power_reaching_boat": [round(float(kd[0]), 3), round(float(lo), 3), round(float(hi), 3)],
                "push_along_out_direction_mps": [round(float(ud[0]), 3), round(float(ulo), 3), round(float(uhi), 3)],
                "values_are": "[best, 10th percentile, 90th percentile]",
                "computers_running_share": None if k_ref is None else round(k_ref, 3)})
    if one_dir:
        out["note"] = ("all stretches in one direction: the split rests on the speed change between efforts "
                       "(weak unless the efforts differ a lot)")
    if kd[0] <= ks[0] + 1e-9 or kd[0] >= ks[-1] - 1e-9:
        out["warning"] = "the fit hit the edge of its range: these stretches do not fit one share and one push"
    return out


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
        if system is None and self.version == 8:
            system = OPEN_COACH_SYSTEM
        if system is None and self.version == 9:
            system = OPEN_COACH_SYSTEM_V9
        if system is None and self.version >= 10:
            system = OPEN_COACH_SYSTEM_V10
        if system is None and panel is not None and 2 <= self.version < 8:
            system = (COACH_PANEL_SYSTEM_V2 + (DRAFT_SECTION if self.version == 7 else "")
                      + (WATER_SECTION if 3 <= self.version < 7 else "")
                      + (WATER_SECTION_V6 if self.version == 6 else WATER_SECTION_V4 if 4 <= self.version < 7 else "")
                      + (TIER_SECTION if self.version in (5, 6) else ""))
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
        reg.register(self.cue, name="cue", writes=True,
                     input_schema=(_CUE_SCHEMA_V10 if self.version >= 10 else
                                   _CUE_SCHEMA_OPEN if self.version >= 8 else _CUE_SCHEMA),
                     description="Speak one short cue (at most 12 words) to the paddler now.")
        allowed = ["race_history", "cue"]
        self.notes: List[Dict[str, Any]] = []
        self.cue_attempts: List[Dict[str, Any]] = []
        self.ask_fn: Optional[Callable[[], Dict[str, Any]]] = None
        self._asked_t: Optional[float] = None
        self.distance_changes: List[Dict[str, Any]] = []
        self.plan: Optional[Dict[str, Any]] = None
        self.timers: List[Dict[str, Any]] = []
        if self.version >= 8:
            reg.register(self.ask_paddler, name="ask_paddler",
                         description="Ask the paddler how they feel; they answer with an effort out of 10 and a few words.")
            reg.register(self.take_note, name="take_note", writes=True,
                         description="Write a note to yourself (up to 800 characters); every later wake-up "
                                     "shows your last 15 notes.")
            reg.register(self.set_race_distance, name="set_race_distance", writes=True,
                         description="Correct the race distance the boat's computer uses (m, from the start), with the reason.")
            allowed += ["ask_paddler", "take_note", "set_race_distance"]
        if panel is not None:
            for name, tool in panel.tools().items():
                if self.version >= 9 and name in ("get_metric_series", "get_alerts", "list_available_metrics"):
                    continue                      # summarised, race-time versions below
                reg.register(tool.func, name=name, description=tool.description,
                             input_schema=tool.input_schema, writes=tool.writes)
                allowed.append(name)
            if self.version >= 9:
                self._panel_series = panel.tools()["get_metric_series"].func
                self._panel_alerts = panel.tools()["get_alerts"].func
                reg.register(self.get_metric_series, name="get_metric_series",
                             description="A metric over a time window as a summary (mean, min, max, first, last) "
                                         "and about 30 averaged points, all in race seconds.")
                reg.register(self.get_alerts, name="get_alerts",
                             description="Alerts raised by your watches, with the race minute of each.")
                reg.register(self.wake_me, name="wake_me", writes=True,
                             description=("Wake me after in_s seconds or when at_remaining_m metres are left, "
                                          "with a reason (at most 3 pending); clear=true cancels the pending ones"
                                          + ("; cancel='text' cancels those whose reason contains the text."
                                             if self.version >= 10 else ".")))
                self._panel_list = panel.tools()["list_available_metrics"].func
                reg.register(self.list_streams, name="list_available_metrics",
                             description="The stream names watches and your own metrics can use, with one line each.")
                reg.register(self.set_plan, name="set_plan", writes=True,
                             description="Set the race plan shown at the top of every wake-up (replaces the old one).")
                reg.register(self.legs, name="legs",
                             description="One row per leg of the course so far: direction, race minutes, GPS speed, "
                                         "pod power, boat-model power, stroke rate and local push (turns left out).")
                allowed += ["get_metric_series", "get_alerts", "wake_me", "set_plan", "legs", "list_available_metrics"]
        self.recalibrations: List[Dict[str, Any]] = []
        if 3 <= self.version < 7:
            reg.register(self.current_forecast, name="current_forecast",
                         description="The pre-race forecast of the tidal stream, its change during the race, "
                                     "the inshore weakening and the eddies (local knowledge; can be wrong).")
            allowed.append("current_forecast")
        self.rule_control: Optional[Dict[str, Callable]] = None
        if self.version in (5, 6):
            reg.register(self.mute_rule, name="mute_rule", writes=True,
                         description="Stop the rule layer's cues on one lever ('line' or 'pace') for some "
                                     "minutes, with the reason (you take over that lever).")
            reg.register(self.unmute_rule, name="unmute_rule", writes=True,
                         description="Hand a lever ('line' or 'pace') back to the rule layer.")
            allowed += ["mute_rule", "unmute_rule"]
        if self.version == 6:
            reg.register(self.line_change_value, name="line_change_value",
                         description="Seconds saved over the rest of this leg by moving to each given distance "
                                     "from shore, by the forecast, net of the crossing (negative = costs time).")
            allowed.append("line_change_value")
        if 4 <= self.version < 7:
            reg.register(self.stream_by_line, name="stream_by_line",
                         description="The forecast's stream along the current heading (m/s, + helps) at each "
                                     "given distance from shore, here or at another position along the course.")
            allowed.append("stream_by_line")
        self.beliefs: Dict[str, Dict[str, Any]] = {}
        self.belief_log: List[Dict[str, Any]] = []
        self._first_cal: Dict[str, Optional[float]] = {}
        if self.version >= 10:
            reg.register(self.measure, name="measure", input_schema=_MEASURE_SCHEMA,
                         description="Read 1-4 stretches of the race (race seconds): means with uncertainty of pod "
                                     "power, boat-model power, GPS speed, stroke rate and heart rate, the heart-rate "
                                     "trend, and a hull-model fit splitting speed into the share of pod power reaching "
                                     "the boat and the push of current and wind.")
            reg.register(self.test_cost, name="test_cost",
                         description="Price a piece by the boat computer's numbers: metres gained or lost while it "
                                     "lasts, reserve spent, and the net seconds at the finish once the reserve is "
                                     "repaid (with critical power 10 % lower or higher too).")
            reg.register(self.set_belief, name="set_belief", writes=True,
                         description="Record a belief: name, value, low, high, evidence. critical_power_pod_w, "
                                     "reserve_size_pod_kj, critical_power_boat_w, reserve_size_boat_kj and pod_gain "
                                     "(with since_race_s) also set the boat computer; other names are for you.")
            allowed += ["measure", "test_cost", "set_belief"]
        elif self.version >= 9:
            reg.register(self.set_calibration, name="set_critical_power", writes=True,
                         description="Correct critical power (boat cp_w and/or pod cp_shaft_w, W) and/or the "
                                     "reserve size (w_prime_kj / w_prime_shaft_kj), with the reason; both reserves "
                                     "are recomputed over the race so far.")
            allowed.append("set_critical_power")
        elif 2 <= self.version < 9:
            reg.register(self.set_critical_power, name="set_critical_power", writes=True,
                         description="Correct the critical power the reserve models use from now on "
                                     "(boat power cp_w and/or shaft power cp_shaft_w, in W), with the reason.")
            allowed.append("set_critical_power")
        self.policy = ToolPolicy(reg, allowed, max_iterations=max_iterations or (5 if panel is not None else 2))

    # ── the open coach's later tools (version 9) ───────────────────────────
    def _race_s(self, t_data: Optional[float]) -> Optional[float]:
        st = self.state.start_t if self.state.start_t is not None else 0.0
        return None if t_data is None else float(t_data) - float(st)

    def get_metric_series(self, metric: str, window_s: float = 120.0, ago_s: float = 0.0,
                          start_race_s: float = 0.0, end_race_s: float = 0.0, points: int = 30) -> Dict[str, Any]:
        st = float(self.state.start_t or 0.0)
        raw = self._panel_series(metric=metric, window_s=window_s, ago_s=ago_s,
                                 start_s=(start_race_s + st) if start_race_s else 0.0,
                                 end_s=(end_race_s + st) if end_race_s else 0.0)
        vals = [float(v) for v in (raw.get("values") or []) if v is not None and math.isfinite(float(v))]
        out = {k: (round(raw[k], 3) if isinstance(raw.get(k), float) else raw.get(k))
               for k in ("metric", "available", "n", "mean", "min", "max", "first", "last") if k in raw}
        t0, t1 = self._race_s(raw.get("t_start")), self._race_s(raw.get("t_end"))
        if t0 is not None and t1 is not None:
            out["race_s"] = [round(t0), round(t1)]
        if vals and t0 is not None and t1 is not None:
            k = max(1, math.ceil(len(vals) / max(5, min(60, int(points or 30)))))
            dt = (t1 - t0) / max(1, len(vals) - 1)
            out["points"] = [{"race_s": round(t0 + (i + (len(vals[i:i + k]) - 1) / 2.0) * dt),
                              "mean": round(sum(vals[i:i + k]) / len(vals[i:i + k]), 3)}
                             for i in range(0, len(vals), k)]
        return out

    def get_alerts(self, since_race_s: Optional[float] = None, watch_id: str = "", limit: int = 10) -> Dict[str, Any]:
        st = float(self.state.start_t or 0.0)
        raw = self._panel_alerts(since_s=(since_race_s + st) if since_race_s is not None else None,
                                 watch_id=watch_id, limit=limit)
        rows = []
        for a in raw.get("alerts", []):
            rows.append({"watch_id": a.get("watch_id"), "race_min": round((a.get("t_data_s", st) - st) / 60.0, 2),
                         "message": a.get("message"),
                         "values": {k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in (a.get("values") or {}).items()}})
        return {"ok": True, "n_total": raw.get("n_total"), "alerts": rows}

    def wake_me(self, in_s: Optional[float] = None, at_remaining_m: Optional[float] = None,
                reason: str = "", clear: bool = False, cancel: str = "") -> Dict[str, Any]:
        s = self.state.last
        if cancel and self.version >= 10:
            keep = [w for w in self.timers if cancel.lower() not in str(w.get("reason", "")).lower()]
            n, self.timers = len(self.timers) - len(keep), keep
            if in_s is None and at_remaining_m is None:
                return {"ok": True, "cancelled": n, "pending": self.timers}
        if clear:
            n, self.timers = len(self.timers), []
            if in_s is None and at_remaining_m is None:
                return {"ok": True, "cleared": n, "pending": []}
        if (in_s is None) == (at_remaining_m is None):
            return {"ok": False, "error": "give exactly one of in_s or at_remaining_m"}
        if len(self.timers) >= 3:
            return {"ok": False, "error": "3 wake-ups already pending", "pending": self.timers}
        if in_s is not None:
            if not 10.0 <= float(in_s) <= 1800.0:
                return {"ok": False, "error": "in_s must be 10-1800"}
            due = {"at_race_s": round((s.get("elapsed_s") or 0.0) + float(in_s), 1)}
        else:
            rem = s.get("remaining_m")
            if self.version >= 10 and self.state.distance_m:      # after a course change in the same turn
                rem = max(0.0, float(self.state.distance_m) - float(self.state.travelled))
            if rem is None or not 50.0 <= float(at_remaining_m) < float(rem):
                return {"ok": False, "error": f"at_remaining_m must be 50 m or more and less than the {rem} m left"}
            due = {"at_remaining_m": round(float(at_remaining_m))}
        w = {**due, "reason": str(reason)[:200]}
        if self.version >= 10:
            w["set_at_race_s"] = round(s.get("elapsed_s") or 0.0)
        self.timers.append(w)
        return {"ok": True, "pending": self.timers}

    def list_streams(self) -> Dict[str, Any]:
        raw = self._panel_list()
        rows = raw.get("metrics") if isinstance(raw, dict) else raw
        out = []
        for m in rows or []:
            if not isinstance(m, dict):
                continue
            nm = m.get("name") or m.get("metric")
            desc = str(m.get("description") or m.get("expression") or "")
            row = {"name": nm, "about": desc[:110]}
            if m.get("kind") not in (None, "catalog"):
                row["yours"] = m.get("kind")
            if m.get("streaming") is False:
                row["streaming_now"] = False
            out.append(row)
        return {"ok": True, "streams": out}

    def due_timers(self) -> List[Dict[str, Any]]:
        """Pop the wake-ups now due (called by the race loop each second)."""
        s = self.state.last
        el, rem = s.get("elapsed_s"), s.get("remaining_m")
        due, keep = [], []
        for w in self.timers:
            if (("at_race_s" in w and el is not None and el >= w["at_race_s"])
                    or ("at_remaining_m" in w and rem is not None and rem <= w["at_remaining_m"])):
                due.append(w)
            else:
                keep.append(w)
        self.timers = keep
        return due

    def set_plan(self, text: str) -> Dict[str, Any]:
        s = self.state.last
        self.plan = {"race_min": round((s.get("elapsed_s") or 0.0) / 60.0, 1), "plan": str(text)[:1200]}
        return {"ok": True}

    def legs(self) -> Dict[str, Any]:
        leg = (getattr(self, "race_info", {}) or {}).get("leg_m")
        if not leg:
            return {"ok": False, "error": "the course's leg length is not known"}
        rows: Dict[int, List[Dict[str, Any]]] = {}
        for h in self.state.history:
            d, el = h.get("distance_m"), h.get("elapsed_s")
            if d is None or el is None or el < 90.0 or (d % leg) < 40.0:
                continue                                  # the start and the 40 m after each turn left out
            rows.setdefault(int(d // leg), []).append(h)
        def m(b, k, n=2):
            xs = [x[k] for x in b if x.get(k) is not None and math.isfinite(x[k])]
            return round(sum(xs) / len(xs), n) if xs else None
        out = []
        for i in sorted(rows)[-12:]:
            b = rows[i]
            out.append({"leg": i + 1, "direction": "out" if i % 2 == 0 else "back",
                        "race_min": [round(b[0]["elapsed_s"] / 60.0, 1), round(b[-1]["elapsed_s"] / 60.0, 1)],
                        "speed_gps": m(b, "speed_gps"), "pod_power_w": m(b, "shaft_power_w", 0),
                        "boat_power_w": m(b, "power_w", 0), "rate_spm": m(b, "rate_spm", 1),
                        "push_local_mps": m(b, "push_local_mps")})
            if self.version >= 10:
                out[-1]["heart_rate_bpm"] = m(b, "heart_rate_bpm", 0)
        return {"ok": True, "leg_m": leg, "legs": out}

    def set_calibration(self, cp_w: Optional[float] = None, cp_shaft_w: Optional[float] = None,
                        w_prime_kj: Optional[float] = None, w_prime_shaft_kj: Optional[float] = None,
                        reason: str = "") -> Dict[str, Any]:
        st = self.state
        a, sh = st.athlete, st.athlete_shaft
        checks = [(cp_w, a.cp_w, "cp_w", 0.8, 1.6), (cp_shaft_w, sh.cp_w if sh else None, "cp_shaft_w", 0.8, 1.6),
                  (None if w_prime_kj is None else 1000.0 * float(w_prime_kj), a.w_prime_j, "w_prime_kj", 0.5, 2.0),
                  (None if w_prime_shaft_kj is None else 1000.0 * float(w_prime_shaft_kj),
                   sh.w_prime_j if sh else None, "w_prime_shaft_kj", 0.5, 2.0)]
        for v, base, nm, lo, hi in checks:
            if v is not None and (base is None or not lo * base <= float(v) <= hi * base):
                return {"ok": False, "error": f"{nm} is outside {lo}-{hi} x the test value"}
        if all(v is None for v, *_ in checks):
            return {"ok": False, "error": "nothing to change"}
        if len(self.recalibrations) >= 3:
            return {"ok": False, "error": "at most 3 recalibrations per race"}
        out = st.recalibrate(cp_w=cp_w, cp_shaft_w=cp_shaft_w,
                             w_prime_j=None if w_prime_kj is None else 1000.0 * float(w_prime_kj),
                             w_prime_shaft_j=None if w_prime_shaft_kj is None else 1000.0 * float(w_prime_shaft_kj),
                             recompute=True)
        entry = {"t": st.last.get("t"), "elapsed_s": st.last.get("elapsed_s"), "reason": reason, **out}
        self.recalibrations.append(entry)
        record.emit("revision", target="calibration", **entry)
        return {"ok": True, **out}

    # ── the experimenting coach's tools (version 10) ───────────────────────
    def measure(self, windows: List[Dict[str, Any]], share: Optional[float] = None) -> Dict[str, Any]:
        return measure_windows(self.state, (getattr(self, "race_info", {}) or {}).get("leg_m"), windows,
                               share=share)

    def test_cost(self, effort_pct: float, seconds: float, rate_spm: float = 0.0) -> Dict[str, Any]:
        from .race import _speed_for, w_balance_step
        st = self.state
        last = st.last
        Ps, k = last.get("shaft_power_w"), st.eta_est
        if Ps is None or k is None or st.athlete_shaft is None:
            return {"ok": False, "error": "not enough race yet to price a piece"}
        secs = float(seconds)
        if not 10.0 <= secs <= 480.0 or not -25.0 <= float(effort_pct) <= 25.0:
            return {"ok": False, "error": "effort_pct -25..25, seconds 10-480"}
        P1 = Ps * (1.0 + float(effort_pct) / 100.0)
        v0 = _speed_for(st.craft, st.total_kg, k * Ps, st.ama_kg)
        v1 = _speed_for(st.craft, st.total_kg, k * P1, st.ama_kg)
        rem = max(0.0, float(st.distance_m or 0.0) - float(st.travelled))
        out: Dict[str, Any] = {"ok": True, "metres_gained_during": round((v1 - v0) * secs, 1)}
        for nm, f in (("as_calibrated", 1.0), ("cp_10pct_lower", 0.9), ("cp_10pct_higher", 1.1)):
            cp = st.cp_shaft_w * f
            w0 = st.w_bal_shaft if f == 1.0 else min(st.athlete_shaft.w_prime_j, max(0.0, st.w_bal_shaft))
            wa, wb = w0, w0
            for _ in range(int(secs)):
                wa = w_balance_step(wa, Ps, 1.0, cp, st.athlete_shaft.w_prime_j)
                wb = w_balance_step(wb, P1, 1.0, cp, st.athlete_shaft.w_prime_j)
            dW = wa - wb                                   # extra reserve spent by the piece (J)
            T_rem = rem / max(0.5, v0) if rem > 0 else 0.0
            if T_rem > 0:
                dP = dW / T_rem                            # repaid as lower power over the rest of the race
                v_low = _speed_for(st.craft, st.total_kg, k * max(1.0, Ps - dP), st.ama_kg)
                lost_s = T_rem * (v0 / max(0.3, v_low) - 1.0)
            else:
                lost_s = 0.0
            gained_s = (v1 - v0) * secs / max(0.3, v0)
            out[nm] = {"extra_reserve_spent_kj": round(dW / 1000.0, 2), "net_seconds_at_finish": round(lost_s - gained_s, 1)}
        out["note"] = ("net_seconds_at_finish > 0 costs time. By the boat computer's numbers; a rate change's cost "
                       "depends on blade efficiency, which only a measurement shows" if rate_spm else
                       "net_seconds_at_finish > 0 costs time. By the boat computer's numbers.")
        return out

    def set_belief(self, name: str, value: Any, low: Any = None, high: Any = None, evidence: str = "",
                   since_race_s: Optional[float] = None) -> Dict[str, Any]:
        st = self.state
        name = str(name).strip()[:40]
        if not name:
            return {"ok": False, "error": "a belief needs a name"}
        if name not in self.beliefs and len(self.beliefs) >= 12:
            return {"ok": False, "error": "at most 12 beliefs; reuse a name"}
        el = st.last.get("elapsed_s") or 0.0
        res: Dict[str, Any] = {"ok": True}
        if name in _BELIEF_COMPUTER:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return {"ok": False, "error": f"{name} needs a number"}
            sh = st.athlete_shaft
            if name == "pod_gain":
                if not 0.5 <= v <= 2.0:
                    return {"ok": False, "error": "pod_gain must be 0.5-2.0"}
                since = None if since_race_s is None else max(0.0, float(since_race_s))
                res["computer"] = st.set_pod_gain(v, since)
            else:
                base = {"critical_power_pod_w": sh.cp_w if sh else None, "critical_power_boat_w": st.athlete.cp_w,
                        "reserve_size_pod_kj": sh.w_prime_j / 1000.0 if sh else None,
                        "reserve_size_boat_kj": st.athlete.w_prime_j / 1000.0}[name]
                first = self._first_cal.setdefault(name, base)
                if first is None or not 0.6 * first <= v <= 1.6 * first:
                    return {"ok": False, "error": f"{name} must be 0.6-1.6 x the pre-race test ({first})"}
                kw = {"critical_power_pod_w": {"cp_shaft_w": v}, "critical_power_boat_w": {"cp_w": v},
                      "reserve_size_pod_kj": {"w_prime_shaft_j": 1000.0 * v},
                      "reserve_size_boat_kj": {"w_prime_j": 1000.0 * v}}[name]
                res["computer"] = st.recalibrate(recompute=True, **kw)
        b = {"value": value, "low": low, "high": high, "evidence": str(evidence)[:300],
             "set_at_race_s": round(el)}
        if since_race_s is not None:
            b["since_race_s"] = since_race_s
        self.beliefs[name] = b
        self.belief_log.append({"name": name, **b})
        record.emit("revision", target="belief", name=name, value=value, race_s=round(el))
        return res

    def _seen_after(self, c: Dict[str, Any]) -> Any:
        """What the instruments showed after a spoken cue (version 9)."""
        if "_seen" in c:
            return c["_seen"]
        if not c.get("spoken"):
            return None
        st = self.state
        now_el = st.last.get("elapsed_s") or 0.0
        el0 = c.get("_el")
        if el0 is None:
            return None
        if c["ask"] == "clear_rudder":
            t0 = c.get("_t")
            if now_el < el0 + 25.0:
                return f"too early to tell ({round(el0 + 25.0 - now_el)} s)"
            before = sorted(v for t, v in st.speed_trace if t0 - 15.0 <= t <= t0)
            after = [v for t, v in st.speed_trace if t0 + 3.0 <= t <= t0 + 40.0]
            if not before or not after:
                res = "not measurable"
            else:
                base = before[len(before) // 2]
                dip = 1.0 - min(after) / max(1e-6, base)
                res = {"speed_dip_pct": round(100.0 * dip), "back_stroke_seen": bool(dip >= 0.25)}
            c["_seen"] = res
            return res
        if self.version >= 10 and c["ask"] == "piece":
            return self._piece_seen(c)
        if now_el < el0 + 30.0:
            return f"too early to tell ({round(el0 + 30.0 - now_el)} s)"
        def m(lo, hi, k, n=0):
            xs = [h[k] for h in st.history if h.get("elapsed_s") is not None and lo <= h["elapsed_s"] <= hi
                  and h.get(k) is not None and math.isfinite(h[k])]
            return round(sum(xs) / len(xs), n) if xs else None
        b0, b1, a0, a1 = el0 - 60.0, el0, el0 + 20.0, min(now_el, el0 + 80.0)
        res = {"window_after_s": [20, round(a1 - el0)]}
        leg = (getattr(self, "race_info", {}) or {}).get("leg_m")
        if leg:
            ds = [h["distance_m"] for h in st.history if h.get("elapsed_s") is not None and b0 <= h["elapsed_s"] <= a1
                  and h.get("distance_m") is not None]
            if ds and int(min(ds) // leg) != int(max(ds) // leg):
                res["turn_inside"] = True
        keys = (("shaft_power_w", "pod_power_w", 0), ("power_w", "boat_power_w", 0),
                ("rate_spm", "rate_spm", 1), ("speed_gps", "speed_gps", 2))
        if self.version >= 10:
            keys = keys + (("heart_rate_bpm", "heart_rate_bpm", 0),)
        for k, nm, n in keys:
            x, y = m(b0, b1, k, n), m(a0, a1, k, n)
            if x is not None and y is not None:
                res[nm] = [x, y]
        if now_el >= el0 + 80.0:
            c["_seen"] = res
        return res

    def _piece_seen(self, c: Dict[str, Any]) -> Any:
        """A test piece: the instruments in each segment against the minute before the cue
        (segments counted from 15 s after the cue: paddlers take 4-15 s to act)."""
        st = self.state
        now_el = st.last.get("elapsed_s") or 0.0
        el0 = c.get("_el") or 0.0
        segs = c.get("segments") or []
        ends, acc = [], el0 + 15.0
        for sg in segs:
            try:
                acc += float(sg.get("seconds", 0.0))
            except (TypeError, ValueError):
                return None
            ends.append(acc)
        if not ends or now_el < ends[-1] + 5.0:
            return f"piece running: segments end at race s {[round(e) for e in ends]}"

        def m(lo, hi, k, n=0):
            xs = [h[k] for h in st.history if h.get("elapsed_s") is not None and lo <= h["elapsed_s"] <= hi
                  and h.get(k) is not None and math.isfinite(h[k])]
            return round(sum(xs) / len(xs), n) if xs else None
        keys = (("shaft_power_w", "pod_power_w", 0), ("speed_gps", "speed_gps", 2),
                ("rate_spm", "rate_spm", 1), ("heart_rate_bpm", "heart_rate_bpm", 0))
        res: Dict[str, Any] = {"before": {nm: m(el0 - 60.0, el0, k, n) for k, nm, n in keys}, "segments": []}
        start = el0 + 15.0
        for sg, end in zip(segs, ends):
            lo = start + min(30.0, 0.25 * (end - start))         # the 30-s means need time to move
            row = {"race_s": [round(start), round(end)], "asked": {k: sg.get(k) for k in ("effort_pct", "rate_spm") if sg.get(k)}}
            row.update({nm: m(lo, end, k, n) for k, nm, n in keys})
            res["segments"].append(row)
            start = end
        leg = (getattr(self, "race_info", {}) or {}).get("leg_m")
        if leg:
            ds = [h["distance_m"] for h in st.history if h.get("elapsed_s") is not None
                  and el0 - 60.0 <= h["elapsed_s"] <= ends[-1] and h.get("distance_m") is not None]
            if ds and int(min(ds) // leg) != int(max(ds) // leg):
                res["turn_inside"] = True
        c["_seen"] = res
        return res

    def ask_paddler(self) -> Dict[str, Any]:
        if self.ask_fn is None:
            return {"ok": False, "error": "no paddler on the link"}
        t = self.state.last.get("t") or 0.0
        if self._asked_t is not None and t - self._asked_t < 30.0:
            return {"ok": False, "error": "asked less than 30 s ago; let them paddle"}
        self._asked_t = t
        return {"ok": True, **self.ask_fn()}

    def take_note(self, text: str) -> Dict[str, Any]:
        s = self.state.last
        self.notes.append({"race_min": round((s.get("elapsed_s") or 0.0) / 60.0, 1),
                           "at_m": round(s.get("distance_m") or 0.0), "note": str(text)[:800]})
        return {"ok": True, "n_notes": len(self.notes)}

    def set_race_distance(self, distance_m: float, reason: str = "") -> Dict[str, Any]:
        st = self.state
        d = float(distance_m)
        done = float(st.last.get("distance_m") or 0.0)
        if not (done + 100.0 <= d <= 2.0 * (st.distance_m or d)):
            return {"ok": False, "error": f"{d:.0f} m is not ahead of the boat ({done:.0f} m done) or is implausible"}
        old, st.distance_m = st.distance_m, d
        entry = {"elapsed_s": st.last.get("elapsed_s"), "old_m": old, "new_m": d, "reason": reason}
        self.distance_changes.append(entry)
        record.emit("revision", target="race_distance", **entry)
        return {"ok": True, "distance_m": d, "remaining_m": round(d - done)}

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
        if self.version >= 8:
            for r, b in zip(rows, _buckets(H)):
                r.update(_pod_row(b))
        return {"every_30s": rows}

    def cue(self, text: str, ask: str = "none", pace_s500: Optional[float] = None,
            line_m: Optional[float] = None, distance_left_m: Optional[float] = None,
            segments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        act = {"text": text, "ask": ask, "pace_s500": pace_s500, "line_m": line_m,
               "events": [e["kind"] for e in self._events], "source": self.caller}
        if distance_left_m is not None:
            act["distance_left_m"] = distance_left_m
        if segments is not None:
            act["segments"] = segments
        res = self.gate.deliver(act, source=self.caller)
        out = {"spoken": bool(res.get("delivered")), "blocked": bool(res.get("blocked")),
               "reason": res.get("reason")}
        s = self.state.last
        self.cue_attempts.append({"race_min": round((s.get("elapsed_s") or 0.0) / 60.0, 1), "text": text,
                                  "ask": ask, "pace_s500": pace_s500, "spoken": out["spoken"],
                                  **({"race_s": round(s.get("elapsed_s") or 0.0), "segments": segments}
                                     if self.version >= 10 and segments else
                                     {"race_s": round(s.get("elapsed_s") or 0.0)} if self.version >= 10 else {}),
                                  **({"blocked_because": out["reason"]} if out["blocked"] else {}),
                                  "_t": s.get("t"), "_el": s.get("elapsed_s")})
        return out

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

    def mute_rule(self, lever: str, minutes: float = 5.0, reason: str = "") -> Dict[str, Any]:
        if not self.rule_control:
            return {"ok": False, "error": "no rule layer in this race"}
        if lever not in ("line", "pace"):
            return {"ok": False, "error": "lever must be 'line' or 'pace'"}
        minutes = min(30.0, max(0.5, float(minutes)))
        out = self.rule_control["mute"](lever, minutes * 60.0)
        record.emit("revision", target="rule_mute", lever=lever, minutes=minutes, reason=reason,
                    t=self.state.last.get("t"))
        return {"ok": True, **out}

    def unmute_rule(self, lever: str) -> Dict[str, Any]:
        if not self.rule_control:
            return {"ok": False, "error": "no rule layer in this race"}
        out = self.rule_control["unmute"](lever)
        record.emit("revision", target="rule_unmute", lever=lever, t=self.state.last.get("t"))
        return {"ok": True, **out}

    def line_change_value(self, lines_m: Optional[List[float]] = None) -> Dict[str, Any]:
        from .sim.twin import forecast_stream
        if not self.forecast:
            return {"ok": False, "error": "no forecast for this race"}
        s = self.state.last
        hd = s.get("heading_deg", 0.0) or 0.0
        north = hd < 90.0 or hd > 270.0
        sign = 1.0 if north else -1.0
        pos = float(s.get("along_m", 0.0) or 0.0)
        leg = (self.state.distance_m or 0.0) / 2.0
        end = leg if north else 0.0
        if abs(end - pos) < 50.0:
            return {"ok": True, "note": "the leg is nearly over"}
        here = float(s.get("line_m", 250.0) or 250.0)
        vw = float(s.get("speed_water", 3.0) or 3.0)
        t0 = float(s.get("elapsed_s", 0.0) or 0.0)

        def leg_time(line: float, cross: float) -> float:
            n, total, t = 40, 0.0, t0
            step = (end - pos) / n
            for k in range(n):
                sk = pos + (k + 0.5) * step
                y = line if cross <= 0 else here + (line - here) * min(1.0, (t - t0) / max(cross, 1e-6))
                vg = max(0.3, vw + sign * forecast_stream(self.forecast, y, sk, t))
                dt = abs(step) / vg
                total += dt
                t += dt
            return total
        base = leg_time(here, 0.0)
        out = {}
        for y in (lines_m or [50, 80, 120, 200, 300, 450]):
            dy = abs(float(y) - here)
            cross = dy / 0.8
            lost = cross * (vw - math.sqrt(max(0.0, vw * vw - 0.64)))       # distance given up while crossing
            t_y = leg_time(float(y), cross) + lost / max(0.5, vw)
            out[str(int(y))] = round(base - t_y, 1)
        return {"ok": True, "from_line_m": round(here), "metres_left_in_leg": round(abs(end - pos)),
                "seconds_saved_by_moving_to": out}

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
            if self.version >= 10:
                row["heart_rate_bpm"] = m("heart_rate_bpm", 0)
            out.append({k2: v for k2, v in row.items() if v is not None})
        return out

    def _open_packet(self, events: List[Dict[str, Any]], why: str) -> Dict[str, Any]:
        s = self.state
        last = s.last
        state = s.compact()
        for k in ("shaft_power_w", "cp_shaft_w", "w_bal_pod_frac", "empties_in_m_pod",
                  "sustainable_shaft_power_w", "heading_deg", "push_local_mps"):
            v = last.get(k)
            if v is not None:
                state[k] = round(v, 3) if isinstance(v, float) else v
        leg = self.race_info.get("leg_m") if hasattr(self, "race_info") else None
        if leg and last.get("distance_m") is not None:
            done = float(last["distance_m"])
            nt = leg - (done % leg)
            rem = last.get("remaining_m")
            if self.version >= 9 and rem is not None and nt >= rem - 1.0:
                state["next_turn_in_m"] = "no turn before the finish"
            else:
                state["next_turn_in_m"] = round(nt)
            state["leg_number"] = int(done // leg) + 1
        msg: Dict[str, Any] = {"update": why,
                               "events": [{"kind": e["kind"], "detail": e["detail"]} for e in events],
                               "state": state, "last_5_min": self._recent_minutes()}
        t = last.get("t") or 0.0
        if self.version >= 10:
            state["race_s"] = round(last.get("elapsed_s") or 0.0)
            for k, nm in (("heart_rate_bpm", "heart_rate_bpm"), ("pod_gain", "pod_gain_set"),
                          ("shaft_power_raw_w", "pod_reading_raw_w")):
                v = last.get(k)
                if v is not None:
                    state[nm] = round(v, 3 if k == "pod_gain" else 0)
            if self.beliefs:
                cpb = self.beliefs.get("critical_power_pod_w") or {}
                try:
                    lo_hi = [float(cpb["low"]), float(cpb["high"])]
                except (KeyError, TypeError, ValueError):
                    lo_hi = None
                if lo_hi:
                    r_lo, r_hi = s.pod_reserve_at(lo_hi[0]), s.pod_reserve_at(lo_hi[1])
                    if r_lo is not None and r_hi is not None:
                        state["reserve_pod_pct_across_your_cp_range"] = {
                            f"at_{lo_hi[0]:.0f}_w": round(100 * r_lo), f"at_{lo_hi[1]:.0f}_w": round(100 * r_hi)}
                msg = {"your_beliefs": self.beliefs, **msg}
        if self.version >= 9:
            if self.plan:
                msg = {"your_plan": self.plan, **msg}
            recent = []
            for c in self.cue_attempts[-5:]:
                row = {k: v for k, v in c.items() if not k.startswith("_")}
                seen = self._seen_after(c)
                if seen is not None:
                    row["instruments_after"] = seen
                recent.append(row)
            if self.timers:
                msg["your_wake_ups"] = self.timers
        else:
            recent = [{k: v for k, v in c.items() if not k.startswith("_")} for c in self.cue_attempts[-5:]]
        spoken = [c for c in self.cue_attempts if c["spoken"]]
        msg["your_recent_cues"] = recent
        if spoken and self.version >= 10:
            nxt = (spoken[-1]["_el"] or 0.0) + 60.0
            el_now = last.get("elapsed_s") or 0.0
            msg["next_cue_allowed_at_race_s"] = round(nxt + 0.5) if nxt > el_now else "now"
        elif spoken:
            msg["next_cue_allowed_in_s"] = max(0, round(60.0 - (t - (spoken[-1]["_t"] or t))))
        msg["cues_left_this_race"] = max(0, 20 - len(spoken))
        rem = last.get("remaining_m")
        for k in ("reserve_runs_out_in_m", "empties_in_m_pod"):     # a reserve that lasts past the finish
            v = state.get(k)
            if v is not None and rem is not None and v > rem:
                state[k] = "lasts past the finish"
        msg["your_notes"] = self.notes[-15:]
        if not self.consultations:
            info = getattr(self, "race_info", {}) or {}
            msg["race"] = {"distance_m": s.distance_m,
                           "course": (f"out-and-back legs of {info['leg_m']:.0f} m (turn at the end of each leg); "
                                      "turns cost a few seconds") if info.get("leg_m") else "out-and-back legs",
                           "calibration": ("critical power and reserve from an all-out 3- and 12-minute test: "
                                           f"boat {s.athlete.cp_w:.0f} W / {s.athlete.w_prime_j / 1000:.1f} kJ" +
                                           (f"; shaft (pod) {s.athlete_shaft.cp_w:.0f} W / "
                                            f"{s.athlete_shaft.w_prime_j / 1000:.1f} kJ" if s.athlete_shaft else ""))}
        return msg

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
            if self.version >= 5 and self.rule_control:
                msg["rule_layer"] = self.rule_control["status"]()
            if not self.consultations:
                msg["race"] = {"distance_m": s.distance_m, "course": "out-and-back legs; turns cost a few seconds",
                               "calibration": ("critical power and reserve from an all-out 3- and 12-minute test: "
                                               f"boat {s.athlete.cp_w:.0f} W / {s.athlete.w_prime_j / 1000:.1f} kJ" +
                                               (f"; shaft (pod) {s.athlete_shaft.cp_w:.0f} W / "
                                                f"{s.athlete_shaft.w_prime_j / 1000:.1f} kJ" if s.athlete_shaft else ""))}
                if self.version == 7:
                    msg["race"]["others"] = ("a pack of boats starts with you, a little faster than your even "
                                             "pace; their wash cuts drag by roughly a tenth")
                if 3 <= self.version < 7 and self.forecast:
                    msg["race"]["forecast"] = self.forecast
                    msg["race"]["course"] = (f"along a shore: one leg of {s.distance_m / 2:.0f} m north, "
                                             "then back south to the start")
        if self.version >= 8:
            msg = self._open_packet(events, why)
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


def _buckets(H: List[Dict[str, Any]], n: int = 6) -> List[List[Dict[str, Any]]]:
    return [H[i:i + n] for i in range(0, len(H), n)]


def _pod_row(bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    def m(k, n=2):
        xs = [b[k] for b in bucket if b.get(k) is not None and math.isfinite(b[k])]
        return round(sum(xs) / len(xs), n) if xs else None
    w = m("w_bal_pod_frac", 3)
    return {k: v for k, v in {"shaft_power_w": m("shaft_power_w", 0),
                              "reserve_pod_pct": None if w is None else round(100 * w),
                              "speed_gps": m("speed_gps"), "push_local_mps": m("push_local_mps")}.items()
            if v is not None}


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
