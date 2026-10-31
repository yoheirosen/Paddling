"""Closed-loop coaching on the paddler twin: does a reasoning coach in the loop make
the boat faster to the line than no coach, a fixed pacing plan, or a rule autopilot?

Every arm races the same paddler in the same race for a given seed (the
:class:`Preparation`: hidden physiology, own pacing, cue response, course and
current, and the critical-power calibration from a test on the same twin).  An
episode steps the twin second by second on agentic_experiments' scheduler with a
virtual clock; each second the race state's numbers stream into the race panel (the
platform's metric service, :mod:`.race_panel`), whose watch rules raise alerts.
Every cue, from any arm, passes the same gate (:func:`.coach.make_cue_gate`) and
reaches the twin, which may follow it.

Arms:

* ``no_coach`` — the paddler's own pacing.
* ``fixed_plan`` — the usual race plan: hold a target pace on the display, computed
  before the race from the calibration (the pace that spends the reserve at the line
  in flat water, times ``mult``), cued after the start and every ``every_s``.
* ``autopilot`` — the screen's four watch rules, each mapped to a fixed cue
  (over-pace → ease to the sustainable pace; fading → rate down; reserve conflict →
  hold; finish window → finish), no model.
* ``agent`` — the reasoning coach (:class:`.coach.RaceCoach` with the panel): woken by
  watch alerts and every ``periodic_s``; may read series, define its own metrics and
  watches, and cue.  Each model call takes its latency in race time.

The perfect-knowledge even pacer (:func:`.sim.twin.oracle_effort`) is a reference,
not an arm.  :data:`PREREGISTERED` fixes the design, the outcomes and the tests.
The scripted stand-in coach (:func:`.coach.stand_in_coach`) validates the harness
only: it is not evidence for the claim.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agent_lib.runtime import record
from agent_lib.runtime.clock import VirtualClock
from agent_lib.runtime.record import RunRecord
from agent_lib.runtime.scheduler import Scheduler, fixed

from . import sessions as ss
from .boatspeed import speed_at_power
from .coach import CueSink, RaceCoach, format_pace, make_cue_gate
from .race import Athlete, RaceState, calibrate, sustainable_power
from .race_panel import SCREEN_PROGRAMS, SCREEN_WATCHES, RacePanel
from .sim.paddler import PaddlerParams
from .sim.twin import (PaddlerTwin, RaceSetup, Traits, draw_setup, draw_traits, even_effort,
                       oracle_effort, twin_cp_test)

__all__ = ["PREREGISTERED", "Preparation", "draw_preparation", "Arm", "NoCoach", "FixedPlan",
           "Autopilot", "AgentArm", "run_episode", "baseline_arms", "paired", "holm",
           "CountingProvider", "HONESTY_STAND_IN"]

HONESTY_STAND_IN = ("provider=stand-in: a scripted coach that maps each alert to a fixed cue, "
                    "after a model latency. HARNESS VALIDATION ONLY, not evidence for the claim.")

PREREGISTERED: Dict[str, Any] = {
    "question": ("On the paddler twin, does a reasoning coach in the loop, on the platform's "
                 "streaming panel, make the boat faster to the line than no coach, a fixed pacing "
                 "plan, or a rule autopilot?"),
    "seeds": {"development": list(range(0, 20)), "evaluation": list(range(1000, 1030)),
              "rule": ("design choices may be tuned on development seeds only; the baselines' best "
                       "settings are selected on the evaluation seeds themselves (optimistic for "
                       "the baselines)")},
    "twin": {"draws": "sim.twin.draw_traits / draw_setup (ranges in the code)",
             "measurement_noise": {"gps_speed_sd_mps": 0.06, "rate_sd_spm": 0.4, "heading_sd_deg": 4.0},
             "calibration": "3- and 12-minute all-out test on the same twin, flat water"},
    "loop": {"tick_s": 1.0, "panel": "race_panel.RacePanel (agent_lib metric service, in process)",
             "starting_watches": [w["watch_id"] for w in SCREEN_WATCHES]},
    "gate": {"max_words": 12, "quiet_start_s": 60.0, "quiet_finish_m": 100.0, "cooldown_s": 60.0,
             "max_cues": 20},
    "agent": {"periodic_s": 300.0, "max_tool_rounds": 5, "max_tokens": 4000,
              "latency": ("each model call takes its measured wall-clock time in race time "
                          "(the stand-in: a fixed 6 s per call)"),
              "model": "chosen by the owner before the evaluation run; recorded with the results"},
    "fixed_plan_sweep": {"mult": [0.97, 1.0, 1.03], "every_s": [120.0, 300.0]},
    "autopilot_sweep": {"overpace_margin_m": [150.0, 300.0, 600.0], "fade_pct": [-3.0, -5.0]},
    "metrics": {
        "race_time_s": "start to the line, from the twin (primary)",
        "blown_s": "race time spent blown up (reserve spent, held below critical power)",
        "reserve_left": "true reserve left at the line (share of W')",
        "cues": "cues spoken (passed the gate), blocked, and followed (twin truth)",
        "gap_closed": ("(no_coach - arm) / (no_coach - reference) per seed, only where the "
                       "no-coach race is at least 10 s slower than the reference"),
    },
    "selection": ("for each baseline family, settings ranked by mean race_time_s over the "
                  "evaluation seeds: best = rank 1, typical = the median-ranked setting"),
    "statistics": ("paired per seed: d = agent - comparator (s); Wilcoxon signed-rank (two-sided); "
                   "bootstrap 95% CI of the mean d (10 000 resamples, seed 0); Holm over the primary "
                   "family {race_time_s} x {no_coach, fixed_plan best, autopilot best}"),
    "revisions": [{
        "date": "2026-10-31",
        "before_any_real_model_run": True,
        "kind": "owner decision (the model choice the registration left open)",
        "changes": [
            "primary agent arm 'agent': claude-opus-5-5 (default effort); the Holm family is "
            "computed for this arm",
            "secondary agent arm 'agent_haiku': claude-haiku-5-5 (default effort); reported with "
            "the same tests, outside the primary family",
            "the coach's system prompt and tool descriptions are prompt-cached (cost only; the "
            "model sees the same input)",
            "budget cap: about USD 45 for both arms; stop and ask the owner if it would be exceeded",
            "a smoke test on development seeds precedes the evaluation run (plumbing only; any "
            "change it forces is logged here before the evaluation run)",
            "smoke test, development seed 1 (6 km): both models ran the whole loop with no errors "
            "(Haiku 41 calls, median 1.4 s, ~USD 0.01; Opus 36 calls, median 3.7 s, ~USD 0.36); both "
            "wrote watch rules of their own; no change was forced",
        ],
    }, {
        "date": "2026-10-31",
        "kind": "data-handling rule (after the API account ran out of credit mid-run)",
        "during_evaluation": True,
        "changes": [
            "a race in which any of the coach's model calls failed is not a coached race: it is kept "
            "out of the results (results/coaching_trial/failed_<tag>.jsonl) and rerun on the same seed",
            "at the time: 3 clean primary-arm races (seeds 1000-1002) and 9 with refused calls "
            "(seeds 1003-1011, moved out); no arm comparison had been computed",
        ],
    }],
}


# ── preparations ────────────────────────────────────────────────────────────

@dataclass
class Preparation:
    """Everything shared by every arm for one seed."""

    seed: int
    traits: Traits
    setup: RaceSetup
    even: float
    reference: Dict[str, float]
    athlete: Athlete
    calibration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"seed": self.seed, "traits": asdict(self.traits), "setup": asdict(self.setup),
             "even": self.even, "reference": self.reference,
             "athlete": {"cp_w": self.athlete.cp_w, "w_prime_j": self.athlete.w_prime_j,
                         "source": self.athlete.source},
             "calibration": self.calibration}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Preparation":
        tr = dict(d["traits"])
        tr["paddler"] = PaddlerParams(**{k: (tuple(v) if isinstance(v, list) else v)
                                         for k, v in tr["paddler"].items()})
        return cls(seed=int(d["seed"]), traits=Traits(**tr), setup=RaceSetup(**d["setup"]),
                   even=float(d["even"]), reference=dict(d["reference"]),
                   athlete=Athlete(cp_w=d["athlete"]["cp_w"], w_prime_j=d["athlete"]["w_prime_j"],
                                   source=d["athlete"].get("source", "")),
                   calibration=dict(d.get("calibration", {})))


def draw_preparation(seed: int) -> Preparation:
    tr, su = draw_traits(seed), draw_setup(seed)
    rows, tw = twin_cp_test(tr, seed=seed)
    prep = ss.prepare_rows(rows, tw.craft, tw.total_kg)
    best = ss.best_efforts(prep["all_rows"], (180, 720))
    ath = calibrate(best, source="3- and 12-minute test on the twin")
    true_boat_cp = tr.paddler.critical_power_w * tr.paddler.eta_max
    return Preparation(seed=seed, traits=tr, setup=su, even=even_effort(tr, su),
                       reference=oracle_effort(tr, su), athlete=ath,
                       calibration={"best_180_w": best[180]["power"], "best_720_w": best[720]["power"],
                                    "cp_w": ath.cp_w, "w_prime_j": ath.w_prime_j,
                                    "true_shaft_cp_w": tr.paddler.critical_power_w,
                                    "true_boat_cp_at_best_w": true_boat_cp})


# ── arms ────────────────────────────────────────────────────────────────────

class Arm:
    name = "arm"
    setting: Dict[str, Any] = {}

    def watches(self) -> List[Dict[str, Any]]:
        return []

    def setup(self, ep: "Episode") -> None:
        pass

    def on_tick(self, ep: "Episode", t: float, alerts: List[Dict[str, Any]]) -> None:
        pass

    def summary(self) -> Dict[str, Any]:
        return {}


class NoCoach(Arm):
    name = "no_coach"


class FixedPlan(Arm):
    """Hold a pace on the display planned before the race; reminders every ``every_s``."""

    name = "fixed_plan"

    def __init__(self, mult: float = 1.0, every_s: float = 300.0) -> None:
        self.mult, self.every_s = float(mult), float(every_s)
        self.setting = {"mult": self.mult, "every_s": self.every_s}
        self._next = None
        self.pace = None

    def setup(self, ep: "Episode") -> None:
        a, su = ep.prep.athlete, ep.prep.setup
        sp = sustainable_power(ep.twin.craft, ep.twin.total_kg, a.cp_w, a.w_prime_j, su.distance_m,
                               ama_kg=5.0 if ep.twin.craft.outrigger else 0.0)
        self.pace = 500.0 / (self.mult * sp["speed_mps"])
        self._next = 90.0

    def on_tick(self, ep: "Episode", t: float, alerts: List[Dict[str, Any]]) -> None:
        el = ep.state.last.get("elapsed_s")
        if el is None or el < self._next:
            return
        ep.gate.deliver({"text": f"Hold {format_pace(self.pace)}.", "ask": "hold",
                         "pace_s500": round(self.pace, 1), "source": self.name}, source=self.name)
        self._next = el + self.every_s


def _screen_watches(margin_m: float = 300.0, fade_pct: float = -4.0) -> List[Dict[str, Any]]:
    out = []
    for w in SCREEN_WATCHES:
        w = dict(w)
        if w["watch_id"] == "over_pace":
            w["condition"] = w["condition"].replace("maximum(300,", f"maximum({margin_m:g},")
        if w["watch_id"] == "fading":
            w["condition"] = f"fade_pct <= {fade_pct:g}"
        out.append(w)
    return out


def _display_pace(state: RaceState, water_pace_s500: Optional[float]) -> Optional[float]:
    if not water_pace_s500:
        return None
    s = state.last
    push = s.get("push_along_mps") if s.get("push_known") else 0.0
    v = 500.0 / water_pace_s500 + (push or 0.0)
    return round(500.0 / v, 1) if v > 0.5 else None


class Autopilot(Arm):
    """Each screen alert mapped to a fixed cue (no model)."""

    name = "autopilot"

    def __init__(self, overpace_margin_m: float = 300.0, fade_pct: float = -4.0) -> None:
        self.margin, self.fade = float(overpace_margin_m), float(fade_pct)
        self.setting = {"overpace_margin_m": self.margin, "fade_pct": self.fade}

    def watches(self) -> List[Dict[str, Any]]:
        return _screen_watches(self.margin, self.fade)

    def on_tick(self, ep: "Episode", t: float, alerts: List[Dict[str, Any]]) -> None:
        for a in alerts:
            wid, s = a["watch_id"], ep.state.last
            if wid == "over_pace":
                pace = _display_pace(ep.state, s.get("sustainable_pace_s500"))
                act = {"text": f"Too hot. Settle to {format_pace(pace)}.", "ask": "ease", "pace_s500": pace}
            elif wid == "fading":
                act = {"text": "Fading at this rate. Drop two beats, lengthen.", "ask": "rate_down"}
            elif wid == "reserve_conflict":
                act = {"text": "Speed is holding. Keep this.", "ask": "hold"}
            elif wid == "finish_window":
                act = {"text": "Reserve left. Go now.", "ask": "finish"}
            else:
                continue
            act["source"] = self.name
            ep.gate.deliver(act, source=self.name)


class AgentArm(Arm):
    """The reasoning coach on the panel (``provider``: a model, or the stand-in)."""

    name = "agent"

    def __init__(self, provider: Any, *, periodic_s: float = 300.0, max_tool_rounds: int = 5,
                 max_tokens: int = 4000, label: str = "agent") -> None:
        self.provider, self.periodic_s = provider, float(periodic_s)
        self.max_tool_rounds, self.max_tokens = int(max_tool_rounds), int(max_tokens)
        self.name = label
        self.setting = {"periodic_s": self.periodic_s, "model": getattr(provider, "model", "?")}
        self.coach: Optional[RaceCoach] = None
        self._busy, self._pending, self._last_wake = False, [], None

    def watches(self) -> List[Dict[str, Any]]:
        return _screen_watches()

    def setup(self, ep: "Episode") -> None:
        self.coach = RaceCoach(self.provider, ep.state, ep.gate, panel=ep.panel,
                               max_iterations=self.max_tool_rounds, max_tokens=self.max_tokens,
                               caller="race_coach")

    def _wake(self, ep: "Episode", events: List[Dict[str, Any]], why: str) -> None:
        if self._busy:
            self._pending.extend(events)
            return
        self._busy, self._last_wake = True, ep.sched.now()

        def run():
            try:
                self.coach.consult(events, why=why)
            finally:
                self._busy = False
                if self._pending:
                    ev, self._pending = self._pending, []
                    ep.sched.post(self._wake, ep, ev, "events")
        ep.sched.spawn(run, name="race_coach")

    def on_tick(self, ep: "Episode", t: float, alerts: List[Dict[str, Any]]) -> None:
        s = ep.state.last
        if alerts:
            self._wake(ep, [{"kind": a["watch_id"], "detail": _alert_detail(a)} for a in alerts], "alerts")
        elif (self.periodic_s and s.get("elapsed_s", 0.0) > 60.0
              and (self._last_wake is None or t - self._last_wake >= self.periodic_s)):
            self._wake(ep, [], "periodic")

    def summary(self) -> Dict[str, Any]:
        c = self.coach
        if c is None:
            return {}
        errs = [x["error"] for x in c.consultations if x.get("error")]
        return {"consultations": len(c.consultations),
                "consult_errors": len(errs), "consult_error_example": errs[0][:200] if errs else "",
                "cue_calls": sum(1 for x in c.consultations if x.get("cued"))}


def _alert_detail(a: Dict[str, Any]) -> str:
    vals = ", ".join(f"{k}={v:.3g}" for k, v in (a.get("values") or {}).items() if v is not None)
    return f"{a.get('message', '')} ({vals})" if vals else str(a.get("message", ""))


def baseline_arms() -> List[Arm]:
    pre = PREREGISTERED
    arms: List[Arm] = [NoCoach()]
    arms += [FixedPlan(m, e) for m in pre["fixed_plan_sweep"]["mult"] for e in pre["fixed_plan_sweep"]["every_s"]]
    arms += [Autopilot(m, f) for m in pre["autopilot_sweep"]["overpace_margin_m"]
             for f in pre["autopilot_sweep"]["fade_pct"]]
    return arms


# ── one episode ─────────────────────────────────────────────────────────────

@dataclass
class Episode:
    prep: Preparation
    twin: PaddlerTwin
    state: RaceState
    panel: RacePanel
    sink: CueSink
    gate: Any
    sched: Scheduler


def run_episode(prep: Preparation, arm: Arm, *, latency_s: Optional[float] = 6.0,
                record_path: Optional[str] = None, series_every_s: float = 15.0,
                max_race_s: float = 3 * 3600.0) -> Dict[str, Any]:
    """Race ``arm`` on this preparation; returns the scored episode.  ``latency_s``:
    race time per model call (``None`` = its measured wall-clock time)."""
    g = PREREGISTERED["gate"]
    twin = PaddlerTwin(prep.traits, prep.setup, seed=prep.seed, even=prep.even)
    state = RaceState(twin.craft, twin.total_kg, prep.athlete, distance_m=prep.setup.distance_m,
                      start_t=twin.start_t)
    panel = RacePanel(state, watches=arm.watches(), programs=SCREEN_PROGRAMS)
    sink = CueSink(state, act=lambda a: twin.cue(a.get("ask", "none"), a.get("pace_s500")))
    gate = make_cue_gate(sink, state, max_words=g["max_words"], quiet_start_s=g["quiet_start_s"],
                         quiet_finish_m=g["quiet_finish_m"], cooldown_s=g["cooldown_s"],
                         max_cues=g["max_cues"])
    sched = Scheduler(VirtualClock(0.0), latency=fixed({}, default=latency_s))
    ep = Episode(prep, twin, state, panel, sink, gate, sched)
    arm.setup(ep)
    series: List[Dict[str, Any]] = []
    nxt = {"series": 0.0}
    W0 = prep.traits.paddler.w_prime_j

    def tick():
        now = sched.now()
        for r in twin.advance(now):
            state.update(r)
        alerts = panel.step(now) if twin.t >= twin.start_t else []
        if twin.finish_t is None:
            arm.on_tick(ep, now, alerts)
        s = state.last
        if s.get("elapsed_s") is not None and now >= nxt["series"] and twin.log["t"]:
            L = twin.log
            series.append({"race_s": round(now - twin.start_t, 1), "race_m": round(twin.race_m, 1),
                           "reserve_true": round(twin.w / W0, 3),
                           "reserve_est": round(s.get("w_bal_frac", float("nan")), 3),
                           "effort": round(L["effort"][-1], 3), "rate": round(L["rate"][-1], 1),
                           "rate_best": round(L["rate_best"][-1], 1),
                           "v_water": round(L["v_water"][-1], 3), "blown": bool(twin.blown)})
            nxt["series"] = now + series_every_s
        if twin.finish_t is None and now < twin.start_t + max_race_s:
            sched.after(1.0, tick)

    sched.at(1.0, tick)
    if record_path:
        with record.recording(RunRecord(record_path)):
            sched.run()
    else:
        sched.run()
    cues = list(twin.cues)
    status = gate.status()
    out = {"seed": prep.seed, "arm": arm.name, "setting": dict(arm.setting),
           "race_time_s": twin.race_time_s, "blown_s": twin.blown_s,
           "reserve_left": twin.w / W0 if twin.finish_t is not None else None,
           "reference_s": prep.reference["race_time_s"],
           "cues_spoken": len(sink.log), "cues_followed": sum(1 for c in cues if c["followed"]),
           "cues_blocked": int(status.get("n_blocked", 0)),
           "asks": _count(c["ask"] for c in sink.log),
           "cue_log": [{"race_s": round((c["t"] or 0) - twin.start_t, 1), "text": c["text"],
                        "ask": c["ask"], "pace_s500": c["pace_s500"]} for c in sink.log],
           "alerts": _count(a["watch_id"] for a in panel.alerts),
           "definitions": panel.service.list_definitions(),
           "series": series}
    out.update(arm.summary())
    return out


def _count(xs) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


class CountingProvider:
    """Wraps a provider; logs each call's wall time and tokens (for latency and cost)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: List[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        import time
        t0 = time.perf_counter()
        err, res = None, None
        try:
            res = self._inner.ask(*args, **kwargs)
            return res
        except Exception as exc:          # noqa: BLE001
            err = repr(exc)
            raise
        finally:
            u = getattr(res, "usage", None)
            self.calls.append({"wall_s": time.perf_counter() - t0, "error": err,
                               **{k: int(getattr(u, k, 0) or 0) for k in (
                                   "input_tokens", "output_tokens", "cache_create_tokens",
                                   "cache_read_tokens")}})

    def totals(self) -> Dict[str, Any]:
        c = self.calls
        out = {"calls": len(c), "errors": sum(1 for x in c if x["error"])}
        for k in ("input_tokens", "output_tokens", "cache_create_tokens", "cache_read_tokens"):
            out[k] = sum(x[k] for x in c)
        if c:
            w = sorted(x["wall_s"] for x in c)
            out["latency_median_s"] = w[len(w) // 2]
            out["latency_p90_s"] = w[min(len(w) - 1, int(0.9 * len(w)))]
        return out


# ── statistics ──────────────────────────────────────────────────────────────

def paired(a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """d = a - b per seed: mean, median, Wilcoxon signed-rank p (two-sided), bootstrap
    95% CI of the mean, and how many seeds favour each side."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    out = {"n": int(d.size), "mean_d": float(d.mean()) if d.size else None,
           "median_d": float(np.median(d)) if d.size else None,
           "n_a_faster": int((d < 0).sum()), "n_b_faster": int((d > 0).sum())}
    if d.size >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        means = d[idx].mean(axis=1)
        out["ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        try:
            from scipy.stats import wilcoxon
            out["wilcoxon_p"] = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) \
                if np.any(d != 0) else 1.0
        except Exception as exc:          # noqa: BLE001
            out["wilcoxon_p"] = None
            out["wilcoxon_error"] = str(exc)
    return out


def holm(pvals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Holm-adjusted p values."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    m, out, running = len(items), {k: None for k in pvals}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
