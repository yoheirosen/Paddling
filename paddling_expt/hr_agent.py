"""A reasoning agent that keeps a paddler's zone display honest through a training block.

It does not read the zone itself: every 30 s a fast reader does that (:func:`.hr_training.read_zone`)
from the guide the agent sets.  The agent is woken when the warm-up ends and when an anomaly
check fires (heart rate and pace disagree for two minutes; a heart rate that cannot be real),
looks at the data, asks the paddler, can ask them to wet or move the strap, and sets the guide:
which route to trust, the day's heart-rate offset, today's critical power against the calibrated
one, how to combine.  Its notes and earlier sessions carry across the block.  Model calls go
through the platform's tool loop and take simulated time.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from agent_lib.providers.types import Message
from agent_lib.runtime import record
from agent_lib.runtime.tool_loop import run_tool_loop
from agent_lib.tool_registry import ToolPolicy, ToolRegistry

from .heart_rate import EFFORT_ZONES
from .hr_training import (Calibration, GuideArm, Guide, READ_EVERY_S, SessionCtx, read_zone)

__all__ = ["TRAINING_ANALYST_SYSTEM", "TrainingAnalyst", "AgentGuideArm"]

TRAINING_ANALYST_SYSTEM = """\
You look after one paddler's heart-rate and effort display through a block of training sessions in an \
outrigger canoe (OC1). The paddler has a GPS speed coach (speed over the ground, heading, stroke rate) and \
a heart-rate strap; they train without a power meter. What you are there for: that the display tells them the \
effort zone they are really in, so they can train in the zone they mean to.

The display: every 30 seconds a fast reader names the zone of the last minute, as a share of the \
paddler's critical power today (the highest power they can hold for a long time): recovery below 55 %, \
endurance 55-75 %, tempo 75-90 %, threshold 90-105 %, above threshold over 105 %. It reads through a \
guide that you set; you do not name zones yourself. It is scored on stretches of steady effort: the right \
zone is useful, "cannot tell" is safe but leaves the paddler without a reading, a wrong zone misleads.

The calibration day was paddled with a borrowed power paddle (it measures paddle power); the paddler now \
trains without it. The reader has two routes:
- Heart rate, through this paddler's own model, fitted on the calibration day from a step test: the heart \
rate this person settles at for each power, the slow upward creep over a session, and the effort above \
which their heart rate no longer tells efforts apart (for some people that is well below critical power). \
The guide's hr_offset_bpm is subtracted first.
- Pace: GPS speed with the boat computer's live estimate of the current removed, turned into power \
reaching the boat by the hull model, then into paddle power by this paddler's blade efficiency at the \
stroke rate (measured on the calibration day), as a share of the calibrated critical power times the \
guide's cp_scale (today's critical power against the calibration day's).

What is known to go wrong (none of it is announced; you have to find it):
- Heart rate lags effort by about a minute and creeps up slowly over a session. Heat and humidity raise \
it at every effort, and the creep grows. Tiredness or illness can raise it at easy effort, hold it down \
at hard effort and lower the maximum, and lowers what the paddler can sustain that day.
- Straps fail: one can lock onto the stroke rate (it then reads the stroke rate, not the heart rate), or \
read far too low and erratically while dry. Asking the paddler to wet and re-seat the strap \
(ask_paddler with ask='fix_strap') usually ends it within a minute.
- Pace: the current changes and the computer's estimate of it lags, especially after a change and \
before the first turn; turns cut the reading for a few seconds. Blade efficiency also falls as the paddler tires within a \
session, so the same speed later costs more effort. Fitness changes over weeks.
- The calibration itself is one day's measurement.

The guide (set_guide): mode (hr, pace, combined); combine, for combined: agree (if the two routes' zones \
are within one zone of each other, the zone of the average of the two shares; further apart, "cannot \
tell"; if one route cannot tell, the other route's zone), hr_first or pace_first (that route's zone; the \
other route's only when the first cannot tell); hr_offset_bpm; hr_trusted (False ignores heart rate); \
hr_trust_up_to_share (above this share the heart-rate route cannot tell); plausibility_check (heart-rate \
readings below 75 or within 3 of the stroke rate are dropped one by one; if fewer than half of the \
minute's readings are left, the heart-rate route cannot tell); cp_scale; and a note to the paddler. \
The guide stays until you change it, also into the next session. Both routes are always computed and \
shown to you, whatever the guide uses.

Scoring: checkpoints every 30 seconds, from two minutes into each stretch of steady planned effort \
after the warm-up (so heart rate's lag after a change of effort is not scored). Nobody sees the right \
answers, you included; you judge from the data, the paddler and the calibration.

When you are woken: at the end of each warm-up (you see the session's timing, not what it asks for); \
when an anomaly check fires (the gap between the two routes moves by more than 0.15 of critical power \
from where it settled in the same stretch, for three minutes; most heart-rate readings implausible for a \
minute; the strap reading plausibly again for two minutes after that; at most 3 a session, 5 minutes \
apart); and when you asked to be (wake_me). Each wake-up allows a few rounds of tool calls; a reply of \
plain text ends it. Your notes (take_note) are shown at every wake-up and carry across the block; history shows earlier \
sessions as the devices recorded them; calibration shows the calibration day.
"""

_GUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["hr", "pace", "combined"]},
        "combine": {"type": "string", "enum": ["agree", "hr_first", "pace_first"]},
        "hr_offset_bpm": {"type": "number", "description": "subtracted from the heart rate before the model reads it"},
        "hr_trusted": {"type": "boolean"},
        "hr_trust_up_to_share": {"type": "number", "description": "share of critical power above which heart rate cannot tell"},
        "plausibility_check": {"type": "boolean"},
        "cp_scale": {"type": "number", "description": "today's critical power / the calibration day's (pace route)"},
        "note": {"type": "string", "description": "a short note to the paddler shown with the display"},
    },
}
_ASK_SCHEMA = {
    "type": "object",
    "properties": {"ask": {"type": "string", "enum": ["feel", "conditions", "strap", "fix_strap"],
                           "description": "feel: how they feel; conditions: weather and water; strap: does "
                                          "the strap seem right; fix_strap: ask them to wet and re-seat it"}},
    "required": ["ask"],
}
_WAKE_SCHEMA = {
    "type": "object",
    "properties": {"minutes": {"type": "number"}, "why": {"type": "string"}},
    "required": ["minutes"],
}
_DATA_SCHEMA = {
    "type": "object",
    "properties": {"from_min": {"type": "number"}, "to_min": {"type": "number"}},
}


def _r(x: Any, n: int = 1) -> Any:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


class TrainingAnalyst:
    """The agent's tools and one consultation (``consult``) over a session context."""

    def __init__(self, provider: Any, *, max_iterations: int = 5, max_tokens: int = 3000,
                 caller: str = "training_analyst") -> None:
        self.provider, self.max_tokens, self.caller = provider, max_tokens, caller
        self.system = TRAINING_ANALYST_SYSTEM
        self.notes: List[Dict[str, Any]] = []
        self.ctx: Optional[SessionCtx] = None
        self.consultations: List[Dict[str, Any]] = []
        self._asked: Optional[tuple] = None                   # (session, t) of the last question
        self.request_wake = None                               # set by the arm: (minutes, why) -> result
        reg = ToolRegistry()
        reg.register(self.session_data, name="session_data", input_schema=_DATA_SCHEMA,
                     description="Today's session minute by minute (default: the last 8 minutes; at most 20): "
                                 "leg (out/back), GPS speed, speed through the water and the current the computer "
                                 "estimates (null until the course first turns: before that it cannot tell current "
                                 "from speed), stroke rate, paddle power estimated from pace (the pace route's "
                                 "power), the heart rate the strap reads, the heart rate the model expects at that "
                                 "power and minute (with the slow creep) and the gap, the share of implausible "
                                 "heart-rate readings, and what the reader said by each route.")
        reg.register(self.history, name="history",
                     description="Earlier sessions of the block as the devices recorded them: each steady stretch "
                                 "from two minutes in: paddle power estimated from pace, stroke rate, heart rate "
                                 "(plausible readings only) and the model's heart rate at that power and minute (with "
                                 "the slow creep); what the paddler said; the guide used at the end.")
        reg.register(self.calibration, name="calibration",
                     description="The calibration day (with a borrowed power paddle): critical power and reserve "
                                 "(paddle power), the highest heart rate seen, blade efficiency by stroke rate (the "
                                 "fitted curve and the stretches it was fitted on), and the paddler's heart-rate model "
                                 "with the range of paddle power it was fitted over.")
        reg.register(self.ask_paddler, name="ask_paddler", writes=True, input_schema=_ASK_SCHEMA,
                     description="Ask the paddler (at most once a minute).")
        reg.register(self.set_guide, name="set_guide", writes=True, input_schema=_GUIDE_SCHEMA,
                     description="Set the reader's guide (fields left out keep their value). Returns what the "
                                 "reader says now with the old and the new guide.")
        reg.register(self.wake_me, name="wake_me", writes=True, input_schema=_WAKE_SCHEMA,
                     description="Be woken again in this session after this many minutes (1-30; one pending "
                                 "request at a time, a new one replaces it; at most 4 a session).")
        reg.register(self.take_note, name="take_note", writes=True,
                     description="A note to yourself (up to 600 characters); every wake-up shows your last 8 notes, "
                                 "across sessions.")
        self.policy = ToolPolicy(reg, ["session_data", "history", "calibration", "ask_paddler", "set_guide",
                                       "wake_me", "take_note"], max_iterations=max_iterations)

    # ── tools ───────────────────────────────────────────────────────────
    def session_data(self, from_min: Optional[float] = None, to_min: Optional[float] = None) -> Dict[str, Any]:
        c = self.ctx
        t_now = c.twin.t
        a = (t_now - 480.0) if from_min is None else float(from_min) * 60.0
        b = t_now if to_min is None else min(t_now, float(to_min) * 60.0)
        if b - a > 1200.0:
            a = b - 1200.0
        out = []
        m0 = int(max(0.0, a) // 60)
        for m in range(m0, int(b // 60) + 1):
            W = [r for r in c.rows if m * 60.0 <= r["t"] < (m + 1) * 60.0 and r["t"] <= b]
            if len(W) < 10:
                continue
            def mean(k):
                xs = [r[k] for r in W if r.get(k) is not None and math.isfinite(r[k])]
                return sum(xs) / len(xs) if xs else None
            hds = [r["heading_deg"] for r in W if r.get("heading_deg") is not None]
            leg = None
            if hds:
                s = sum(math.cos(math.radians(h)) for h in hds) / len(hds)
                leg = "out" if s > 0.5 else "back" if s < -0.5 else "turning"
            pw, hr = mean("paddle_w"), mean("heart_rate_bpm")
            known = all(r.get("push_known") for r in W)
            bad = [r for r in W if r.get("heart_rate_bpm") is not None]
            implaus = (sum(1 for r in bad if r["heart_rate_bpm"] < 75.0 or abs(r["heart_rate_bpm"] - (r.get("rate_spm") or 0.0)) < 3.0)
                       / len(bad)) if bad else None
            mhr = c.cal.model.hr_at(pw) + c.cal.model.creep_per_10min * (m + 0.5) / 10.0 if pw else None
            R = [x for x in c.reads if m * 60.0 < x["t"] <= (m + 1) * 60.0]
            last = R[-1] if R else {}
            nm = lambda z: None if z is None else EFFORT_ZONES[z][0]       # noqa: E731
            out.append({"minute": m, "leg": leg, "speed_gps": _r(mean("speed_gps"), 2),
                        "speed_water": _r(mean("speed_water"), 2),
                        "current_est": _r(mean("push_along_mps"), 2) if known else None,
                        "rate_spm": _r(mean("rate_spm")), "paddle_power_est_w": _r(pw, 0), "heart_rate": _r(hr),
                        "model_heart_rate_at_power_and_minute": _r(mhr),
                        "gap_bpm": _r(None if hr is None or mhr is None else hr - mhr),
                        "implausible_hr_share": _r(implaus, 2),
                        "reader": {"zone": nm(last.get("zone")), "by_heart_rate": nm(last.get("zone_hr")),
                                   "by_pace": nm(last.get("zone_pace")), "share_hr": last.get("share_hr"),
                                   "share_pace": last.get("share_pace")} if last else None})
        return {"ok": True, "now_min": round(t_now / 60.0, 1), "minutes": out[-20:]}

    def history(self) -> Dict[str, Any]:
        return {"ok": True, "sessions": self.ctx.history}

    def calibration(self) -> Dict[str, Any]:
        c = self.ctx.cal
        lo, hi = c.model.effort_range_w
        return {"ok": True, **c.summary, "heart_rate_separates_efforts_up_to_share": c.separates_up_to,
                "blade_efficiency_measured": c.eff_points,
                "heart_rate_model_fitted_over_paddle_w": [round(lo), round(hi)],
                "note": f"outside {round(lo)}-{round(hi)} W the heart-rate model is extrapolated"}

    def ask_paddler(self, ask: str) -> Dict[str, Any]:
        from .sim.training import answer
        c = self.ctx
        t = c.twin.t
        if self._asked is not None and self._asked[0] == c.spec.index and t - self._asked[1] < 60.0:
            return {"ok": False, "error": "asked less than a minute ago"}
        self._asked = (c.spec.index, t)
        a = answer(c.block, c.spec, str(ask), t, asked_before=len(c.answers))
        if ask == "fix_strap":
            c.strap.fix(t)
        c.answers.append({"t": round(t, 1), "ask": ask, **a})
        return {"ok": True, **a}

    def set_guide(self, **kw: Any) -> Dict[str, Any]:
        c = self.ctx
        g = c.guide.to_dict()
        for k in ("mode", "combine", "hr_offset_bpm", "hr_trusted", "hr_trust_up_to_share", "plausibility_check",
                  "cp_scale", "note"):
            if k in kw and kw[k] is not None:
                g[k] = kw[k]
        try:
            new = Guide(**g)
            new.hr_offset_bpm = float(new.hr_offset_bpm)
            new.cp_scale = float(new.cp_scale)
        except (TypeError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        if not -40.0 <= new.hr_offset_bpm <= 40.0 or not 0.6 <= new.cp_scale <= 1.5:
            return {"ok": False, "error": "hr_offset_bpm must be -40..40 and cp_scale 0.6..1.5"}
        if new.mode not in ("hr", "pace", "combined") or new.combine not in ("agree", "hr_first", "pace_first"):
            return {"ok": False, "error": "unknown mode or combine"}
        recent = c.recent(60.0)
        mins = c.twin.t / 60.0
        old_r, new_r = read_zone(c.guide, recent, c.cal, mins), read_zone(new, recent, c.cal, mins)
        c.set_guide(new, why="analyst")
        nm = lambda z: None if z is None else EFFORT_ZONES[z][0]       # noqa: E731
        return {"ok": True, "guide": new.to_dict(),
                "reader_now": {"old_guide": nm(old_r["zone"]), "new_guide": nm(new_r["zone"]),
                               "share_hr": new_r["share_hr"], "share_pace": new_r["share_pace"]}}

    def wake_me(self, minutes: float, why: str = "") -> Dict[str, Any]:
        if self.request_wake is None:
            return {"ok": False, "error": "not available"}
        return self.request_wake(float(minutes), str(why)[:200])

    def take_note(self, text: str) -> Dict[str, Any]:
        c = self.ctx
        self.notes.append({"session": c.spec.index, "min": round(c.twin.t / 60.0, 1), "note": str(text)[:600]})
        return {"ok": True, "n_notes": len(self.notes)}

    # ── a wake-up ───────────────────────────────────────────────────────
    def packet(self, why: str, events: List[str]) -> Dict[str, Any]:
        c = self.ctx
        spec = c.spec
        cuts, acc = [], 0.0
        for dur, _, _ in spec.plan[:-1]:
            acc += dur
            cuts.append(round(acc / 60.0, 1))
        return {"wake": why, "events": events,
                "session": {"number": spec.index, "of": len(c.block.sessions),
                            "minutes": round(sum(p[0] for p in spec.plan) / 60.0),
                            "planned_changes_of_effort_at_min": cuts, "air_temperature_c": round(spec.day.air_c)},
                "now_min": round(c.twin.t / 60.0, 1), "guide": c.guide.to_dict(),
                "your_notes": self.notes[-8:], "older_notes_not_shown": max(0, len(self.notes) - 8),
                "last_minutes": self.session_data()["minutes"][-5:]}

    def consult(self, ctx: SessionCtx, why: str, events: List[str]) -> Dict[str, Any]:
        self.ctx = ctx
        msg = self.packet(why, events)
        res = run_tool_loop(self.provider, system=self.system, messages=[Message(role="user", text=json.dumps(msg))],
                            policy=self.policy, max_tokens=self.max_tokens, caller=self.caller, catch_errors=True)
        out = {"session": ctx.spec.index, "t": round(ctx.twin.t, 1), "why": why, "events": events,
               "text": res.text, "tools": [tc.name for tc, _ in res.tool_log],
               "error": None if res.error is None else str(res.error)}
        self.consultations.append(out)
        record.emit("decision", **out)
        return out


class AgentGuideArm(GuideArm):
    """The reasoning agent as an arm: woken at the end of each warm-up, when it asked to be
    (``wake_me``: at most 4 a session) and by the anomaly checks (at most 3 a session, 5 minutes
    apart): the gap between the heart-rate and pace routes moving more than 0.15 of critical
    power from where it settled early in the same stretch of planned effort (or after the last
    look), for 3 minutes; most heart-rate readings implausible (below 75 or on the stroke rate)
    for a minute; and, after that, the strap reading plausibly again for two minutes."""

    name = "agent"

    def __init__(self, provider: Any, *, label: str = "agent", max_anomaly_wakes: int = 3,
                 max_requested_wakes: int = 4) -> None:
        self.analyst = TrainingAnalyst(provider)
        self.analyst.request_wake = self._request_wake
        self.name = label
        self.setting = {"model": getattr(provider, "model", "?"), "max_anomaly_wakes": max_anomaly_wakes,
                        "max_requested_wakes": max_requested_wakes}
        self.max_wakes = max_anomaly_wakes
        self.max_requested = max_requested_wakes
        self._last_guide: Optional[Guide] = None
        self._ctx: Optional[SessionCtx] = None

    def start_block(self, block, cal):
        super().start_block(block, cal)
        self._last_guide = Guide(mode="combined", combine="agree", plausibility_check=True,
                                 hr_trust_up_to_share=cal.separates_up_to)

    def initial_guide(self, ctx):
        self._ctx = ctx
        self._busy, self._pending, self._wakes, self._last_wake_t = False, [], 0, -1e9
        self._dis, self._implaus, self._good, self._strap_was_bad = 0, 0, 0, False
        self._gaps, self._gap_ref, self._seg = [], None, None
        self._requested, self._wake_req = 0, None
        return Guide(**self._last_guide.to_dict())

    def _request_wake(self, minutes: float, why: str) -> Dict[str, Any]:
        ctx = self._ctx
        if not 1.0 <= minutes <= 30.0:
            return {"ok": False, "error": "minutes must be 1-30"}
        if self._requested >= self.max_requested:
            return {"ok": False, "error": f"at most {self.max_requested} requested wake-ups a session"}
        end = sum(p[0] for p in ctx.spec.plan)
        at = ctx.twin.t + 60.0 * minutes
        if at >= end - 30.0:
            return {"ok": False, "error": f"the session ends at minute {end / 60.0:.1f}"}
        self._requested += 1
        token = object()
        self._wake_req = token

        def fire():
            if self._wake_req is token and self._ctx is ctx:
                self._wake_req = None
                self._wake(ctx, "you asked to be woken", [f"your wake-up: {why}" if why else "your wake-up"])
        ctx.sched.at(at, fire)
        return {"ok": True, "wake_at_min": round(at / 60.0, 1),
                "requests_left_this_session": self.max_requested - self._requested}

    def _wake(self, ctx: SessionCtx, why: str, events: List[str]) -> None:
        if self._busy:
            self._pending.extend(events)
            return
        self._busy = True

        def run():
            try:
                self.analyst.consult(ctx, why, events)
            finally:
                self._busy = False
                if self._pending:
                    ev, self._pending = self._pending, []
                    ctx.sched.post(self._wake, ctx, "events while you were busy", ev)
        ctx.sched.spawn(run, name="training_analyst")

    def on_read(self, ctx, rd):
        if not rd.get("t") or rd["t"] < ctx.spec.warmup_s:
            return
        sh, sp, hr, rate = rd.get("share_hr"), rd.get("share_pace"), rd.get("hr"), rd.get("rate")
        bad = (rd.get("hr_implausible_frac") or 0.0) >= 0.5
        if rd.get("segment") != self._seg:         # a planned change of effort: the gap settles afresh
            self._seg, self._gaps, self._gap_ref, self._dis = rd.get("segment"), [], None, 0
        if sh is not None and sp is not None and not bad:   # the gap between the routes, against where
            self._gaps.append(sh - sp)                       # it settled in this stretch
            if self._gap_ref is None and len(self._gaps) >= 4:
                self._gap_ref = sorted(self._gaps[:4])[2]
            self._dis = self._dis + 1 if (self._gap_ref is not None and abs(sh - sp - self._gap_ref) > 0.15) else 0
        self._implaus = self._implaus + 1 if bad else 0
        self._good = 0 if bad else self._good + 1
        events = []
        if self._dis >= 6:
            events.append(f"for three minutes the gap between the routes has moved: heart rate reads {sh:.2f} of "
                          f"critical power and pace {sp:.2f}, against a gap of {self._gap_ref:+.2f} early in this stretch")
        if self._implaus >= 2:
            self._strap_was_bad = True
            events.append(f"for a minute most heart-rate readings have been implausible: the strap's minute "
                          f"average {hr:.0f} with the stroke rate at {rate:.0f}")
        recovered = self._strap_was_bad and self._good >= 4
        if recovered:
            events.append(f"the strap has read plausibly for two minutes again ({hr:.0f} with the stroke rate at {rate:.0f})")
        if events and self._wakes < self.max_wakes and rd["t"] - self._last_wake_t >= 300.0 and not self._busy:
            self._wakes += 1
            self._last_wake_t = rd["t"]
            if recovered:
                self._strap_was_bad = False
            self._dis = self._implaus = 0
            self._gaps, self._gap_ref = [], None
            self._wake(ctx, "anomaly", events)

    def after_warmup(self, ctx):
        self._gaps, self._gap_ref = [], None
        self._wake(ctx, "warm-up done", ["the warm-up has ended; the session's main work begins"])

    def end_session(self, ctx, score):
        self._last_guide = Guide(**ctx.guide.to_dict())
        self._wake_req = None

    def summary(self):
        a = self.analyst
        errs = [c["error"] for c in a.consultations if c.get("error")]
        return {"consultations": len(a.consultations), "consult_errors": len(errs),
                "notes": a.notes, "tools_used": sum(len(c["tools"]) for c in a.consultations)}
