"""Ready-made simulated sessions, each with its planted answers.

* :func:`training_session` — a full-effort **rate ladder** (short pieces at
  several fixed stroke rates, in interleaved order so rising fatigue is not
  confused with rising rate), a long steady piece, and the ladder again at the
  end (shifted lower, to bracket the tired best rate), on out-and-back legs in a
  tidal current.  It has every answer the analyses
  need: the best rate fresh and tired, full-effort speed at each rate, the change
  in the pull-force profile with fatigue, and the current along each leg.
* :func:`steady_race` — a flat-water race at a held rate and effort.
* :func:`downwind_race` — the same with swell from behind (surf runs).
* :func:`cp_test` — the critical-power test: an all-out 3-minute and an all-out
  12-minute piece, well rested between (calibrates :class:`~..race.Athlete`).
* :func:`paced_race` — a race on out-and-back legs in a current, with a planted
  pacing: by default too fast a start and a body just above critical power, so
  the reserve runs out before the finish (the coach should see it coming).

Every session takes ``kind="oc1"`` (several strokes a side, ama, paddle pod) or
``kind="surfski"`` (alternating double-paddle strokes, no ama, more wobble, no pod —
a single-bridge pod on a kayak paddle is not modelled).

Each returns a :class:`Session`: the truth, the records both devices would send,
the sensor truth, and the labelled segments in both true and head-unit time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..boats import CRAFT
from ..hydro import Craft
from .environment import Course, CurrentField, Swell, WaveTrain
from .hull import Truth, simulate_truth
from .paddler import BoatParams, PaddlerParams, Plan, Segment
from .sensors import HeadUnitConfig, PodConfig, SensorTruth, make_records

__all__ = ["Session", "training_session", "steady_race", "downwind_race", "cp_test",
           "paced_race"]


@dataclass
class Session:
    truth: Truth
    records: List[Dict[str, Any]]
    sensors: SensorTruth
    segments: List[Tuple[float, float, str]]          # true time
    plan: Plan
    craft: Craft = None
    total_kg: float = 0.0

    def head_segments(self, kind: Optional[str] = None) -> List[Tuple[float, float, str]]:
        out = []
        for a, b, lab in self.segments:
            if kind is None or lab.startswith(kind):
                out.append((float(self.sensors.head_time(a)), float(self.sensors.head_time(b)), lab))
        return out


def _run(plan: Plan, *, seed: int, course: Course, current: CurrentField = None,
         swell: Swell = None, paddler: PaddlerParams = None, head: HeadUnitConfig = None,
         pod: PodConfig = None, kind: str = "oc1", craft: Craft = None,
         boat: BoatParams = None, head_wind: float = 0.0) -> Session:
    craft = craft or CRAFT[kind]
    if craft.outrigger is None:
        boat = boat or BoatParams(wobble_dps=3.0, roll_wobble_deg=1.5)
        pod = pod or PodConfig(present=False)
    tr = simulate_truth(plan, paddler=paddler, boat=boat, current=current, swell=swell,
                        course=course, craft=craft, head_wind=head_wind, seed=seed)
    recs, st = make_records(tr, head=head, pod=pod, seed=seed)
    segs = [(a, b, s.label) for a, b, s in plan.boundaries()]
    return Session(tr, recs, st, segs, plan, craft, float(tr.params["m_tot"]))


def _out_and_back(n_legs: int = 40, leg_m: float = 1000.0) -> Course:
    return Course([(0.0, leg_m if i % 2 == 0 else 0.0) for i in range(n_legs)])


def training_session(*, seed: int = 0, rates=(62, 54, 70, 58, 66),
                     tired_rates=(58, 50, 66, 54, 62), piece_s: float = 60.0,
                     rest_s: float = 60.0, steady_s: float = 900.0, steady_rate: float = 62.0,
                     steady_effort: float = 1.0, full_effort: float = 1.15,
                     current_mps: float = 0.25, paddler: PaddlerParams = None,
                     pod: PodConfig = None, head: HeadUnitConfig = None,
                     kind: str = "oc1", **kw) -> Session:
    segs = [Segment(5, 0, 0, "rest"), Segment(120, 56, 0.6, "warmup")]
    for tag in ("fresh", "tired"):
        if tag == "tired":
            segs.append(Segment(steady_s, steady_rate, steady_effort, "steady"))
        for r in (rates if tag == "fresh" else tired_rates):
            segs.append(Segment(piece_s, r, full_effort, f"test_{tag}_{r}"))
            segs.append(Segment(rest_s, 50, 0.45, "easy"))
    p = paddler or PaddlerParams(work_capacity_j=6.0e5, rate_width=12.0)
    return _run(Plan(segs), seed=seed, course=_out_and_back(),
                current=CurrentField(peak_mps=current_mps, set_deg=0.0,
                                     t_max_flood_s=1_792_857_600.0),
                paddler=p, head=head, pod=pod, kind=kind, **kw)


def steady_race(*, seed: int = 0, duration_s: float = 1200.0, rate: float = 62.0,
                effort: float = 0.95, current_mps: float = 0.0, **kw) -> Session:
    plan = Plan([Segment(5, 0, 0, "rest"), Segment(30, 68, 1.2, "start"),
                 Segment(duration_s, rate, effort, "race")])
    return _run(plan, seed=seed, course=Course([(0.0, 20000.0)]),
                current=CurrentField(peak_mps=current_mps, set_deg=0.0), **kw)


def downwind_race(*, seed: int = 0, duration_s: float = 900.0, rate: float = 62.0,
                  effort: float = 1.0, **kw) -> Session:
    sw = Swell([WaveTrain(0.12, 3.0, 0, 0.0), WaveTrain(0.18, 3.3, 5, 1.0),
                WaveTrain(0.12, 3.6, -5, 2.0)])
    plan = Plan([Segment(5, 0, 0, "rest"), Segment(duration_s, rate, effort, "race")])
    head = kw.pop("head", None) or HeadUnitConfig(slap_per_min=3.0)
    return _run(plan, seed=seed, course=Course([(0.0, 20000.0)]), swell=sw, head=head, **kw)


def cp_test(*, seed: int = 0, short_s: float = 180.0, long_s: float = 720.0,
            rest_s: float = 1800.0, drain: float = 0.95, paddler: PaddlerParams = None,
            current_mps: float = 0.0, **kw) -> Session:
    """An all-out ``short_s`` and ``long_s`` piece (3 and 12 minutes), each paced to
    spend ``drain`` of the planted reserve, with ``rest_s`` easy paddling between."""
    p = paddler or PaddlerParams()
    e_short = 1.0 + drain * p.w_prime_j / (p.critical_power_w * short_s)
    e_long = 1.0 + drain * p.w_prime_j / (p.critical_power_w * long_s)
    segs = [Segment(5, 0, 0, "rest"), Segment(300, 56, 0.6, "warmup"),
            Segment(short_s, 66, e_short, "test_short"), Segment(rest_s, 48, 0.4, "recover"),
            Segment(long_s, 63, e_long, "test_long"), Segment(120, 50, 0.4, "cooldown")]
    return _run(Plan(segs), seed=seed, course=_out_and_back(leg_m=1500.0),
                current=CurrentField(peak_mps=current_mps, set_deg=0.0,
                                     t_max_flood_s=1_792_857_600.0), paddler=p, **kw)


def paced_race(*, seed: int = 0, warmup_s: float = 300.0, start_s: float = 90.0,
               start_effort: float = 1.35, body_s: float = 1500.0, body_effort: float = 1.04,
               finish_s: float = 120.0, finish_effort: float = 1.3, rate: float = 62.0,
               leg_m: float = 1250.0, warmup_leg_m: float = 250.0, current_mps: float = 0.2,
               paddler: PaddlerParams = None, **kw) -> Session:
    """A race at a planted pacing (effort = shaft power / critical power) on
    out-and-back legs in a current, after an easy warm-up up and down a short
    stretch (which lets the push of the current be measured before the start).
    The default start and body over-spend the reserve: it runs out before the
    finish and the paddler is held to critical power."""
    p = paddler or PaddlerParams()
    segs = [Segment(5, 0, 0, "rest")]
    if warmup_s > 0:
        segs.append(Segment(warmup_s, 52, 0.55, "warmup"))
    segs += [Segment(start_s, rate + 6, start_effort, "start"),
             Segment(body_s, rate, body_effort, "body"),
             Segment(finish_s, rate + 6, finish_effort, "finish")]
    wps = []
    if warmup_s > 0:
        wps = [(0.0, warmup_leg_m), (0.0, 0.0)] * 2
    course = Course(wps + list(_out_and_back(leg_m=leg_m).waypoints))
    return _run(Plan(segs), seed=seed, course=course,
                current=CurrentField(peak_mps=current_mps, set_deg=0.0,
                                     t_max_flood_s=1_792_857_600.0), paddler=p, **kw)
