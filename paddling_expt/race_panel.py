"""The race state as a live panel on agentic_experiments' streaming-metric service.

Every second the race state's numbers (speed through the water, power, the reserve
above critical power, the pace sustainable to the finish, fade, the push of current
and wind, ...) are streamed as named metrics into the platform's own
:class:`agent_lib.streaming_metrics.service.MetricService`, run in-process on an
in-memory stream (no server; a phone can run it).  That gives the race coach the
same machinery as the platform's other loops, unchanged:

* every metric has a history on the data clock (``get_metric_series``);
* the coach can **define its own metrics** while the race runs: a single formula
  (``define_derived_metric``), a multi-step calculation or a short program with
  loops (``define_program_metric``, admitted to the per-second path only after the
  platform's latency check), or a streaming estimator such as a baseline band or a
  change-point detector (``define_streaming_estimator``);
* **standing watch rules** (``add_watch``): a condition over the panel that must
  hold for a time, checked every second with no model call; an alert wakes the
  coach.  The race starts with :data:`SCREEN_WATCHES` (the four screen events of
  :class:`~.coach.RaceScreen`, rewritten as watch rules); the coach may add,
  replace or remove watches;
* every definition is a versioned revision in the service's journal.

:data:`COACH_PANEL_TOOLS` is the subset of the service's tools the race coach gets
(the rest serve the platform's neural recordings).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent_lib.streaming_metrics.memory_streams import MemoryStreams
from agent_lib.streaming_metrics.metric_control import MetricInfo
from agent_lib.streaming_metrics.service import MetricService, ServiceKeys

from .race import RaceState

__all__ = ["RACE_CATALOG", "SCREEN_WATCHES", "COACH_PANEL_TOOLS", "RacePanel", "panel_values"]


def _m(name: str, window_s: float, description: str, use: str = "pacing") -> MetricInfo:
    return MetricInfo(name, name, window_s, description, core=True, source="race", use_cases=use)


RACE_CATALOG = (
    _m("elapsed_s", 1.0, "Race time since the start (s)."),
    _m("distance_m", 1.0, "Distance raced since the start (m)."),
    _m("remaining_m", 1.0, "Distance to the finish (m)."),
    _m("rate_spm", 30.0, "Stroke rate, strokes per minute (30-s median).", "rate, technique"),
    _m("speed_gps", 30.0, "Speed over the ground from GPS (m/s, 30-s mean): what the paddler's display shows."),
    _m("pace_gps_s500", 30.0, "Pace over the ground, s per 500 m (30-s mean): what the paddler's display shows."),
    _m("speed_water", 30.0, "Speed through the water (m/s, 30-s mean): GPS speed minus the fitted push of current and wind."),
    _m("pace_water_s500", 30.0, "Pace through the water, s per 500 m (30-s mean)."),
    _m("power_w", 30.0, "Power reaching the boat (W, 30-s mean), from speed through the water and the hull model."),
    _m("cp_w", 1.0, "Critical power reaching the boat from the pre-race test (W): sustainable for a long time."),
    _m("w_bal_frac", 1.0, "Reserve above critical power left, 0-1 (W' balance; drains above critical power, refills below). Calibrated on a test: trust early, check against speed later."),
    _m("w_bal_j", 1.0, "Reserve above critical power left (J)."),
    _m("empties_in_m", 1.0, "Distance until the reserve runs out at the current power (m); absent when power is at or below critical power."),
    _m("sustainable_power_w", 1.0, "The constant power that spends the reserve exactly at the finish (W)."),
    _m("sustainable_pace_s500", 1.0, "Pace through the water at that power, s per 500 m."),
    _m("pace_gap_s500", 1.0, "Current pace minus sustainable pace through the water (s/500 m); negative = faster than can be held to the finish."),
    _m("eta_s", 1.0, "Projected race time at the current GPS speed (s)."),
    _m("fade_pct", 30.0, "Speed through the water at this stroke rate vs the same rate early in the race (%); held through turns.", "fatigue, technique"),
    _m("push_along_mps", 30.0, "Push of current and wind along the boat's heading (m/s; + helps)."),
    _m("push_mps", 30.0, "Strength of the fitted push of current and wind (m/s)."),
    _m("push_known", 1.0, "1 once headings cover enough directions to fit the push, else 0."),
    _m("pct_best_at_rate", 30.0, "Power as % of the paddler's best at this rate (needs a rate envelope from past sessions)."),
    _m("shaft_power_w", 30.0, "The paddler's own power at the paddle shaft from the paddle pod (W, 30-s mean): effort, independent of how efficiently the blade moves the boat.", "effort, fatigue"),
    _m("cp_shaft_w", 1.0, "Critical power at the shaft from the same pre-race test, measured by the pod (W)."),
    _m("w_bal_pod_frac", 1.0, "Reserve above critical power left, 0-1, from the pod's shaft power (does not drift when the blade work gets less efficient, unlike w_bal_frac).", "pacing"),
    _m("empties_in_m_pod", 1.0, "Distance until the pod's reserve runs out at the current shaft power (m); absent at or below critical power.", "pacing"),
    _m("sustainable_shaft_power_w", 1.0, "Constant shaft power that spends the pod's reserve exactly at the finish at the current speed (W).", "pacing"),
    _m("power_5min_w", 300.0, "Mean power reaching the boat over the last 5 minutes (W).", "calibration"),
    _m("steady_power_10min_w", 600.0, "Mean power over the last 10 minutes, present only if the speed at the same rate did not fade (fade_pct >= -2) in that time (W).", "calibration"),
    _m("cp_implied_w", 600.0, "Evidence about critical power from this race: the highest 10-minute steady power held while the reserve model read below 15 % (a paddler cannot hold power above critical power on an empty reserve without fading, so critical power is at least about this). Absent until such a stretch occurs.", "calibration"),
    _m("reserve_low_s", 1.0, "Seconds the reserve model has read below 5 % without a break.", "calibration"),
)

# The four screen events of coach.RaceScreen as standing watches on the panel.
SCREEN_WATCHES: List[Dict[str, Any]] = [
    {"watch_id": "over_pace", "sustain_s": 20.0, "cooldown_s": 180.0,
     "condition": ("elapsed_s >= 90 and distance_m < 0.35 * (distance_m + remaining_m) and "
                   "empties_in_m < remaining_m - maximum(300, 0.1 * remaining_m)"),
     "message": "the reserve runs out well before the finish at this power (early in the race)"},
    {"watch_id": "fading", "sustain_s": 60.0, "cooldown_s": 180.0,
     "condition": "fade_pct <= -4",
     "message": "speed through the water at this rate has fallen against early in the race"},
    {"watch_id": "reserve_conflict", "sustain_s": 120.0, "cooldown_s": 300.0,
     "condition": "w_bal_frac < 0.05 and fade_pct >= -1",
     "message": "the reserve reads empty but speed at this rate holds: critical power is probably set low"},
    {"watch_id": "finish_window", "sustain_s": 10.0, "cooldown_s": 600.0,
     "condition": ("remaining_m > 0 and remaining_m < 800 and w_bal_frac > 0.4 and fade_pct >= -2 "
                   "and fade_min_5min > -4"),
     "message": "reserve left near the finish and the speed is holding"},
]
# metrics the screen watches need, defined at the start like any agent-defined metric
SCREEN_PROGRAMS: List[Dict[str, Any]] = [
    {"name": "fade_min_5min", "tier": "let",
     "source": "w = window('fade_pct', 300)\nmin(w) if len(w) > 0 else 0.0",
     "description": "lowest fade_pct in the last 5 minutes"},
]

HELD_THROUGH_TURNS = ("fade_pct",)

COACH_PANEL_TOOLS = (
    "list_available_metrics", "get_metric_series", "characterize_window_metric", "metric_rise_fall",
    "define_derived_metric", "define_program_metric", "list_program_metrics", "remove_streaming_metric",
    "define_streaming_estimator", "list_streaming_estimators",
    "add_watch", "remove_watch", "list_watches", "get_alerts",
)


def panel_values(snap: Mapping[str, Any]) -> Dict[str, float]:
    """A race-state snapshot → ``{metric: float}`` (finite values only)."""
    out: Dict[str, float] = {}
    for m in RACE_CATALOG:
        v = snap.get(m.name)
        if m.name == "pace_gps_s500":
            g = snap.get("speed_gps")
            v = 500.0 / g if g and g > 0.3 else None
        elif m.name == "push_known":
            v = 1.0 if snap.get("push_known") else 0.0
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[m.name] = f
    return out


class RacePanel:
    """The race state streamed into an in-process metric service, one tick a second.

    ``watches`` / ``programs``: the watch rules and metrics defined at the start
    (default: the screen's four events and what they need).  ``store_path``: the
    service's definition journal (every metric and watch definition, versioned)."""

    def __init__(self, state: RaceState, *, watches: Optional[Sequence[Mapping[str, Any]]] = None,
                 programs: Optional[Sequence[Mapping[str, Any]]] = None,
                 store_path: Optional[str] = None) -> None:
        self.state = state
        self.service = MetricService(MemoryStreams(), ServiceKeys.for_panel("race"),
                                     catalog=RACE_CATALOG, store_path=store_path,
                                     forward_to_producer=False, warm_s=0.0)
        self.alerts: List[Dict[str, Any]] = []
        self.definitions: List[Dict[str, Any]] = []
        self._held: Dict[str, float] = {}
        from collections import deque
        self._hist = deque()                 # (t, power, fade or None, w_bal_frac)
        self._cp_implied: Optional[float] = None
        self._low_since: Optional[float] = None
        for p in (SCREEN_PROGRAMS if programs is None else programs):
            self._define(p)
        for w in (SCREEN_WATCHES if watches is None else watches):
            self.add_watch(**w)

    def _define(self, p: Mapping[str, Any]) -> Dict[str, Any]:
        res = self.service.call_tool("define_program_metric", dict(p))
        self.definitions.append({"kind": "program", "name": p.get("name"), "ok": bool(res.get("ok"))})
        return res

    def add_watch(self, watch_id: str, condition: str, sustain_s: float = 0.0,
                  cooldown_s: float = 60.0, message: str = "", **kw: Any) -> Dict[str, Any]:
        res = self.service.add_watch(watch_id, condition, sustain_s, cooldown_s, message,
                                     kw.get("kind", "watch"), kw.get("repeat_s", 0.0))
        self.definitions.append({"kind": "watch", "name": watch_id, "ok": bool(res.get("ok"))})
        return res

    def step(self, t: float) -> List[Dict[str, Any]]:
        """Stream the race state's latest numbers at data time ``t``; return new alerts."""
        snap = self.state.last
        if not snap or snap.get("elapsed_s") is None:
            return []
        vals = panel_values(snap)
        for k in HELD_THROUGH_TURNS:          # blanked by the race state during a turn
            if k in vals:
                self._held[k] = vals[k]
            elif k in self._held:
                vals[k] = self._held[k]
        vals.update(self._calibration_evidence(float(t), vals))
        out = self.service.step(vals, float(t), publish=False)
        self.alerts.extend(out["alerts"])
        return out["alerts"]

    def _calibration_evidence(self, t: float, vals: Dict[str, float]) -> Dict[str, float]:
        P, fade, wb = vals.get("power_w"), vals.get("fade_pct"), vals.get("w_bal_frac")
        out: Dict[str, float] = {}
        if P is None or wb is None:
            return out
        h = self._hist
        h.append((t, P, fade, wb))
        while h and h[0][0] < t - 600.0:
            h.popleft()
        five = [x[1] for x in h if x[0] >= t - 300.0]
        out["power_5min_w"] = sum(five) / len(five)
        span = h[-1][0] - h[0][0]
        fades = [x[2] for x in h if x[2] is not None]
        if span >= 590.0 and fades and min(fades) >= -2.0:
            steady = sum(x[1] for x in h) / len(h)
            out["steady_power_10min_w"] = steady
            if max(x[3] for x in h) < 0.15:
                self._cp_implied = steady if self._cp_implied is None else max(self._cp_implied, steady)
        if self._cp_implied is not None:
            out["cp_implied_w"] = self._cp_implied
        if wb < 0.05:
            self._low_since = t if self._low_since is None else self._low_since
            out["reserve_low_s"] = t - self._low_since
        else:
            self._low_since = None
            out["reserve_low_s"] = 0.0
        return out

    def tools(self) -> Dict[str, Any]:
        """The service's tools the coach may use (name → registry Tool)."""
        reg = self.service.registry
        return {n: reg.get(n) for n in COACH_PANEL_TOOLS if reg.get(n) is not None}

    def authored(self) -> List[Dict[str, Any]]:
        """Metrics and watches defined during the race (beyond the starting set)."""
        listing = self.service.list_definitions()
        return listing.get("definitions", listing) if isinstance(listing, dict) else listing
