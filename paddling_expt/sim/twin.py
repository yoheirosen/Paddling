"""A paddler twin that hears the coach: one race stepped stroke by stroke, whose
paddler changes what they do when a cue reaches them.

The full simulator (:mod:`.hull`) integrates the boat at 100 Hz through a plan fixed
in advance, so nothing said during a race can change it.  This twin keeps the same
planted physiology (:mod:`.paddler`: the reserve above critical power and its refill,
slow fatigue lowering critical power, the efficiency hill in stroke rate whose best
rate falls as the paddler tires, effort held to critical power once the reserve is
spent) and the same hull model for the speed a power holds, but steps once per
stroke on the caller's clock, so a race takes milliseconds and a cue changes the
rest of it.

Planted here, beyond :mod:`.paddler` (the honesty ledger in ``PADDLING_PLAN.md``
lists them):

* **own pacing**, with no coach: a start sprint; a body effort that misjudges, by
  ``body_bias``, the constant effort that would spend the reserve exactly at the line
  at the paddler's habitual rate; a habitual stroke rate that drifts down only a
  little as the paddler tires (the best rate drops more); slow wander of effort by
  feel; a finishing kick at a fixed distance to go.  The paddler ignores the current.
* **blowing up**: once the reserve is spent the paddler can hold only
  ``blowup_effort`` of critical power until ``recover_frac`` of the reserve has come
  back (the full simulator only holds them to critical power, which makes an empty
  reserve nearly free and pacing nearly irrelevant); after a blow-up their own effort
  settles to ``settle_frac`` of their even pace.
* **cue response**: a cue is followed with probability ``compliance``, after
  ``response_s``.  ``ease`` / ``push`` move effort by ``effort_step`` (of critical
  power); ``rate_up`` / ``rate_down`` move the rate by ``rate_step``; a pace (per
  500 m, as the paddler's display shows it, over the ground) is chased by feel,
  within -15 % / +12 % of the effort of the moment, and given up on blowing up;
  ``hold`` keeps the effort of the moment; ``finish`` starts the kick.  A followed
  change fades back to the paddler's own pacing with time constant ``adherence_s``
  unless it is repeated.
* **measurement**: rows carry the noise of a per-stroke GPS speed, stroke rate and
  heading, as a SpeedCoach or the head unit reports them, and (OC1) the paddle pod's
  shaft power with 5 % noise per stroke.

* **incidents** (open races only; :mod:`.incidents` draws them): weed caught on the
  rudder (extra drag until the paddler clears it, which costs a back-stroke), a wind
  shift nobody forecast, the paddle pod's reading drifting, the race committee
  changing the course by radio.  The paddler **talks**: an honest paddler says when
  the legs are going; a pessimist says they are cooked while they are not and eases
  off by themselves unless reassured; an optimist says they feel great while the
  reserve is low and pushes unless held.  Messages go to ``outbox``; ``answer()`` is
  the paddler's reply when asked how they feel.

So the answers a coach could find are planted: what over- or under-pacing costs, how
much a lower rate helps a tired paddler, how far a paddler follows a cue.  A result
on this twin says what coaching is worth *if paddlers respond as modelled*; it is
not evidence that coaching helps real paddlers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..boats import CRAFT
from ..boatspeed import power_at_speed
from ..hydro import Craft
from .paddler import PaddlerParams, best_rate, efficiency

__all__ = ["Traits", "RaceSetup", "PaddlerTwin", "ASKS", "Incident", "CoastalWater", "water_forecast", "forecast_stream", "Pack", "draw_traits", "draw_setup",
           "even_effort", "oracle_effort", "twin_cp_test"]

ASKS = ("ease", "push", "hold", "rate_up", "rate_down", "finish", "line_in", "line_out",
        "hang_on", "let_go", "clear_rudder", "reassure", "tell", "none")


@dataclass
class Incident:
    """Something that happens during a race (planted; :func:`.incidents.draw_incidents`).

    ``kind``: ``weed`` (extra drag ``size``, a fraction, until cleared), ``wind_shift``
    (a push of ``size`` m/s along the first leg's heading, building over a minute),
    ``pod_drift`` (the pod's reading drifts to ``size`` x the truth over two minutes),
    ``course_change`` (the race committee adds ``size`` metres, or removes them if
    negative).  ``remark_after_s``: the paddler says something this long after the onset
    (weed), or the escort radios (wind); None = nobody says anything.  ``self_fix_s``:
    the paddler clears the weed on their own this long after the onset (None = never)."""

    kind: str
    t_race_s: float
    size: float
    remark_after_s: Optional[float] = None
    self_fix_s: Optional[float] = None


@dataclass
class Pack:
    """A group of boats racing the same course at a steady speed through the water (planted):
    ``speed_frac`` x the paddler's even-pace speed; sitting on their wash (0 to ``wash_m``
    behind them) cuts the paddler's drag by ``draft``."""

    speed_frac: float = 1.03
    draft: float = 0.12
    wash_m: float = 8.0


@dataclass
class CoastalWater:
    """A tidal stream along a shore that varies across and along the course (planted).

    The stream in the channel is ``c0`` at the start and changes by ``c1`` over
    ``trend_s`` (it can turn).  Inshore it is weaker: the share of the channel stream at
    distance ``y`` from shore is ``inshore_floor + (1 - inshore_floor)(1 - exp(-y /
    inshore_scale_m))``.  Behind each point (``eddies``: along-shore position, half
    length, width from shore, strength) the inshore water runs against the stream."""

    c0: float = 0.3
    c1: float = 0.0
    trend_s: float = 3000.0
    inshore_floor: float = 0.25
    inshore_scale_m: float = 150.0
    eddies: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def stream(self, t: float) -> float:
        return self.c0 + self.c1 * min(1.0, max(0.0, t / self.trend_s))

    def at(self, y: float, s: float, t: float) -> float:
        """Along-shore water velocity (m/s, + north) at ``y`` m from shore, ``s`` m along."""
        c = self.stream(t)
        g = self.inshore_floor + (1.0 - self.inshore_floor) * (1.0 - math.exp(-max(0.0, y) / self.inshore_scale_m))
        v = c * g
        for s0, half, width, k in self.eddies:
            if abs(s - s0) <= half and y <= width:
                v -= 2.0 * k * c * (1.0 - abs(s - s0) / half) * (1.0 - y / width)
        return v


def water_forecast(water: CoastalWater, seed: int) -> Dict[str, Any]:
    """The pre-race forecast a paddler would have (tide table plus local knowledge), with
    errors: the stream and its trend misjudged, eddies misplaced, one missing, one false."""
    rng = np.random.default_rng([seed, 37])
    eddies = []
    for s0, half, width, k in water.eddies:
        if rng.random() < 0.7:
            eddies.append({"along_m": round(float(s0 + rng.normal(0.0, 120.0))), "length_m": round(2 * half),
                           "within_m_of_shore": round(width)})
    if rng.random() < 0.3:
        eddies.append({"along_m": round(float(rng.uniform(0.1, 0.9) * max(1.0, max([e[0] for e in water.eddies] + [1000.0])))),
                       "length_m": 300, "within_m_of_shore": 120})
    out = {"stream_at_start_mps": round(float(water.c0 + rng.normal(0.0, 0.1)), 2),
           "stream_change_mps": round(float(water.c1 + rng.normal(0.0, 0.15)), 2),
           "over_s": round(water.trend_s),
           "inshore": f"the stream is weaker close to shore (roughly half strength within ~{round(water.inshore_scale_m)} m)",
           "eddies": sorted(eddies, key=lambda e: e["along_m"]),
           "convention": "stream + flows north (with the first leg); along_m is measured from the start line along the course"}
    # the inshore profile as local knowledge has it (shape known, its width uncertain)
    out["inshore_profile"] = {"share_at_shore": water.inshore_floor,
                              "scale_m": round(float(water.inshore_scale_m * rng.uniform(0.75, 1.25)))}
    return out


def forecast_stream(fc: Dict[str, Any], line_m: float, along_m: float, race_s: float) -> float:
    """Along-shore stream (m/s, + north) the forecast expects at a line and position."""
    f = min(1.0, max(0.0, race_s / max(1.0, float(fc.get("over_s", 3000.0)))))
    c = float(fc.get("stream_at_start_mps", 0.0)) + f * float(fc.get("stream_change_mps", 0.0))
    prof = fc.get("inshore_profile") or {"share_at_shore": 0.25, "scale_m": 150.0}
    a, sc = float(prof["share_at_shore"]), float(prof["scale_m"])
    v = c * (a + (1.0 - a) * (1.0 - math.exp(-max(0.0, line_m) / sc)))
    for e in fc.get("eddies", []):
        half, width = 0.5 * float(e.get("length_m", 300)), float(e.get("within_m_of_shore", 120))
        if abs(along_m - float(e["along_m"])) <= half and line_m <= width:
            v -= 1.5 * c * (1.0 - abs(along_m - float(e["along_m"])) / half) * (1.0 - line_m / width)
    return v


@dataclass
class Traits:
    """One paddler: physiology (:class:`PaddlerParams`), own pacing, cue response."""

    paddler: PaddlerParams = field(default_factory=lambda: PaddlerParams(rate_width=12.0))
    habit_rate: float = 62.0            # the rate they hold in the body of a race (spm)
    rate_fatigue_drop: float = 2.0      # spm the held rate drifts down at fatigue 1
    start_effort: float = 1.3           # start sprint (x critical power)
    start_s: float = 90.0
    body_bias: float = 1.0              # own body effort / the even effort at habit rate
    kick_m: float = 400.0               # finishing kick this far from the line
    kick_effort: float = 1.3
    wander_sd: float = 0.02             # effort wander by feel (x critical power, s.d.)
    wander_tau_s: float = 60.0
    compliance: float = 0.8             # chance a cue is followed
    response_s: float = 8.0             # time from cue to the change
    adherence_s: float = 240.0          # a followed change fades back with this time constant
    effort_step: float = 0.04           # ease / push (x critical power)
    rate_step: float = 2.0              # rate up / down (spm)
    pace_gain: float = 0.5              # effort per unit pace error per 10 s, chasing a pace
    max_kick_m: float = 1500.0          # 'finish' further out than this counts as a push
    blowup_effort: float = 0.85         # once the reserve is spent: held to this x critical power ...
    recover_frac: float = 0.15          # ... until this share of the reserve has come back
    settle_frac: float = 0.98           # after a blow-up their own effort settles to this x even pace
    habit_line_m: float = 250.0         # distance from shore they hold unless cued (water that varies across the course)
    lateral_mps: float = 0.8            # how fast they move across when changing line
    line_step_m: float = 100.0          # line in / out moves the line by this
    hang_on: float = 0.5                # 0-1: how hard they will work to stay on a pack's wash
    talk: str = "quiet"                 # quiet | honest | pessimist | optimist (open races)


@dataclass
class RaceSetup:
    """The race: distance, out-and-back legs, a current along the course."""

    distance_m: float = 8000.0
    leg_m: float = 1000.0
    current_mps: float = 0.2            # + flows north (with the first leg)
    warmup_s: float = 300.0
    warmup_leg_m: float = 250.0
    turn_s: float = 6.0
    craft: str = "oc1"
    plan: Optional[List[Tuple[float, float, float]]] = None   # (seconds, rate, effort): a session, not a race
    water: Optional[CoastalWater] = None   # water that varies across and along the course (else: current_mps)
    pack: Optional[Pack] = None            # other boats to draft (else: alone)
    incidents: List[Incident] = field(default_factory=list)   # open races: things that happen


def draw_traits(seed: int) -> Traits:
    """A random paddler (the pre-registered ranges; ``PADDLING_PLAN.md``)."""
    rng = np.random.default_rng([seed, 17])
    u = lambda a, b: float(rng.uniform(a, b))           # noqa: E731
    p = PaddlerParams(critical_power_w=u(150, 220), w_prime_j=u(11_000, 20_000),
                      work_capacity_j=u(2.0e6, 3.2e6), rate_opt_fresh=u(60, 68),
                      rate_opt_shift=u(3, 9), rate_width=u(11, 15), eta_max=u(0.70, 0.78))
    return Traits(paddler=p, habit_rate=round(p.rate_opt_fresh + u(-3, 3)),
                  rate_fatigue_drop=u(0, 4), start_effort=u(1.15, 1.5), start_s=u(60, 120),
                  body_bias=u(0.94, 1.08), kick_m=u(250, 700), kick_effort=u(1.15, 1.4),
                  wander_sd=u(0.01, 0.03), compliance=u(0.55, 0.95), response_s=u(4, 15),
                  adherence_s=u(120, 420), effort_step=u(0.03, 0.06),
                  blowup_effort=u(0.75, 0.9), recover_frac=u(0.1, 0.2),
                  settle_frac=u(0.95, 1.0), habit_line_m=u(120.0, 380.0), hang_on=u(0.0, 1.0))


def draw_setup(seed: int, coastal: bool = False, pack: bool = False) -> RaceSetup:
    rng = np.random.default_rng([seed, 29])
    su = RaceSetup(distance_m=float(rng.choice([6000.0, 8000.0, 10000.0])),
                   leg_m=float(rng.choice([1000.0, 1250.0])),
                   current_mps=float(rng.uniform(-0.35, 0.35)))
    if pack:
        rp = np.random.default_rng([seed, 41])
        su.pack = Pack(speed_frac=float(rp.uniform(1.0, 1.07)), draft=float(rp.uniform(0.08, 0.16)),
                       wash_m=float(rp.uniform(6.0, 10.0)))
    if not coastal:
        return su
    rw = np.random.default_rng([seed, 31])
    su.leg_m = leg = su.distance_m / 2.0          # coastal: one long leg out, one back
    eddies = [(float(rw.uniform(0.1, 0.9) * leg), float(rw.uniform(100.0, 250.0)),
               float(rw.uniform(80.0, 160.0)), float(rw.uniform(0.5, 1.0)))
              for _ in range(int(rw.integers(1, 4)))]
    su.water = CoastalWater(c0=float(rw.uniform(-0.5, 0.5)), c1=float(rw.uniform(-0.5, 0.5)),
                            trend_s=su.distance_m / 3.0, eddies=eddies)
    su.current_mps = 0.0
    return su


class PaddlerTwin:
    """One race on the twin.  ``advance(t)`` paddles strokes up to time ``t`` (s since
    the warm-up began) and returns their rows; ``cue(ask, pace_s500)`` is a cue
    reaching the paddler now.  ``body_effort`` sets the own body effort directly
    (default: ``body_bias`` x :func:`even_effort`); ``rate_policy="best"`` makes the
    paddler hold the best rate for their fatigue (the perfect-knowledge reference);
    ``noise=False`` drops wander and measurement noise."""

    def __init__(self, traits: Traits, setup: RaceSetup, *, seed: int = 0,
                 body_effort: Optional[float] = None, rate_policy: str = "own",
                 noise: bool = True, even: Optional[float] = None) -> None:
        self.tr, self.su, self.p = traits, replace(setup), traits.paddler   # own copy: a course change stays here
        self.craft: Craft = CRAFT[setup.craft]
        c = self.craft
        self.total_kg = self.p.body_kg + c.hull_mass_kg + (c.outrigger.mass_kg if c.outrigger else 0.0)
        self._table = _speed_table(c, self.total_kg)
        self.noise, self.rate_policy = bool(noise), rate_policy
        self.rng_meas = np.random.default_rng([seed, 1])
        self.rng_feel = np.random.default_rng([seed, 2])
        self.rng_cue = np.random.default_rng([seed, 3])
        self.rng_pod = np.random.default_rng([seed, 4])    # its own stream: adding the pod moves no other draw
        self.rng_pos = np.random.default_rng([seed, 5])    # GPS line / course position (varying water only)
        self.y = float(traits.habit_line_m)
        self.y_target = self.y
        # the pack: its distance raced, its speed through the water, whether the paddler hangs on
        self.pack_m = 0.0
        self.hanging = setup.pack is not None
        self._hang_bonus = (0.0, 0.0)
        self.on_wash = False
        self.wash_s = 0.0
        self._v_pack_w = None
        self._pack_leg = 0
        self._pack_turn_until = -1.0
        if body_effort is None and setup.plan is None:
            even = even if even is not None else even_effort(traits, setup)
            body_effort = traits.body_bias * even
        self.even = even if even is not None else body_effort
        self.body_effort = body_effort
        self.t = 0.0
        self.start_t = 0.0 if setup.plan is not None else float(setup.warmup_s)
        self.w = self.p.w_prime_j
        self.work = 0.0
        self.v = 0.0
        self.total_m = 0.0
        self.race_m = 0.0
        self.heading = 0.0
        self._leg_start = 0.0          # race metres at the start of this leg
        self._wu_leg_start = 0.0
        self._turn_until = -1.0
        self.wander = 0.0
        self.v_disp = None
        self.kicking = False
        self.blown = False
        self.blown_s = 0.0
        self.finish_t: Optional[float] = None
        self.started_kick_t: Optional[float] = None
        # coached state: (value, time set)
        self._effort_off = (0.0, 0.0)
        self._rate_off = (0.0, 0.0)
        self._pace: Optional[Tuple[float, float, float, float]] = None   # (target v, effort, t set, effort then)
        self._hold: Optional[Tuple[float, float]] = None          # (effort, t set)
        self._pending: List[Tuple[float, str, Optional[float]]] = []
        self.cues: List[Dict[str, Any]] = []
        # open races: incidents, talk, what the paddler believes about the course
        self.believed_m = float(self.su.distance_m)
        self.rng_talk = np.random.default_rng([seed, 6])
        self.outbox: List[Dict[str, Any]] = []
        self.incident_log: List[Dict[str, Any]] = []
        self._inc_started = [False] * len(self.su.incidents)
        self._scheduled: List[Tuple[float, str, Any]] = []     # (t, what, payload)
        self._weed = 0.0
        self._wind: Optional[Tuple[float, float]] = None       # (push m/s along the first leg, t onset)
        self._pod: Optional[Tuple[float, float]] = None        # (gain at the end, t onset)
        self._self_eff = (0.0, 0.0)                            # self-ease / self-push (offset, t set)
        self._talk_pending: Optional[Tuple[float, float]] = None   # (t start, offset)
        self._talk_next = self.start_t + float(self.rng_talk.uniform(240.0, 480.0))
        self._talk_n = 0
        self._talk_good_said = False
        self.log: Dict[str, List[float]] = {k: [] for k in (
            "t", "race_s", "race_m", "effort", "p_shaft", "p_boat", "eta", "rate", "rate_best",
            "fatigue", "w_bal", "v_water", "v_ground", "current_along", "coached")}

    # ── the paddler's choices ────────────────────────────────────────────
    @property
    def racing(self) -> bool:
        return self.t >= self.start_t and self.finish_t is None

    @property
    def remaining_m(self) -> float:
        return max(0.0, self.su.distance_m - self.race_m)

    def _fatigue(self) -> float:
        return min(1.0, max(0.0, 0.7 * (1.0 - self.w / self.p.w_prime_j)
                            + self.work / self.p.work_capacity_j))

    def _cp_eff(self) -> float:
        return self.p.critical_power_w * (1.0 - self.p.slow_cp_loss * min(1.0, self.work / self.p.work_capacity_j))

    def _decay(self, t_set: float) -> float:
        return math.exp(-max(0.0, self.t - t_set) / self.tr.adherence_s)

    def _plan_now(self) -> Tuple[float, float]:
        acc = 0.0
        for dur, rate, eff in self.su.plan:
            acc += dur
            if self.t < acc:
                return rate, eff
        return 0.0, 0.0

    def _own(self, fatigue: float) -> Tuple[float, float]:
        """(effort, rate) the paddler would choose on their own now."""
        tr = self.tr
        if self.su.plan is not None:
            rate, eff = self._plan_now()
            return eff, rate
        if self.t < self.start_t:
            return 0.55, 52.0
        race_s = self.t - self.start_t
        if not self.kicking and self.believed_m - self.race_m <= tr.kick_m:
            self.kicking = True
            self.started_kick_t = self.t
            if self.su.incidents:
                self._note("kick_started", true_left_m=round(self.su.distance_m - self.race_m),
                           believed_left_m=round(self.believed_m - self.race_m), by="own")
        if race_s < tr.start_s:
            e, r = tr.start_effort, tr.habit_rate + 6.0
        elif self.kicking:
            e, r = tr.kick_effort, tr.habit_rate + 4.0
        else:
            e, r = self.body_effort, tr.habit_rate
        r -= tr.rate_fatigue_drop * fatigue
        if self.rate_policy == "best" and race_s >= tr.start_s:
            r = best_rate(self.p, fatigue) + (4.0 if self.kicking else 0.0)
        return e, r

    def _pack_effort(self, fatigue: float, rate: float) -> Optional[float]:
        """Effort that holds the pack's speed on their wash now (None when not near them)."""
        pk = self.su.pack
        if pk is None or not self.racing or self._v_pack_w is None:
            return None
        gap = self.pack_m - self.race_m
        if gap < -2.0 or gap > 25.0:
            return None
        eta = efficiency(self.p, rate, fatigue)
        ama = (self.p.ama_load_kg + self.p.ama_load_fatigue_kg * fatigue) if self.craft.outrigger else 0.0
        P_tab, v_tab = _table_at(self._table, ama)
        on = 0.0 <= gap <= pk.wash_m
        # on the wash: the pack's speed with the drag saving; off it: without, a little faster to close
        v_need = self._v_pack_w * (1.0 if on else 1.03)
        p_boat = float(np.interp(v_need, v_tab, P_tab)) * ((1.0 - pk.draft) if on else 1.0)
        return p_boat / max(1e-6, eta) / self._cp_eff()

    def _choose(self, fatigue: float, T: float, commit: bool = False) -> Tuple[float, float, bool]:
        e_own, r = self._own(fatigue)
        if r <= 0:
            return 0.0, 0.0, False
        coached = False
        e = e_own
        if self.hanging and not self.kicking and not self.blown:
            e_need = self._pack_effort(fatigue, r)
            if e_need is not None:
                b, tb = self._hang_bonus
                limit = self.body_effort * (1.0 + 0.15 * self.tr.hang_on) + b * self._decay(tb)
                if self.t - self.start_t < self.tr.start_s:
                    limit = max(limit, self.tr.start_effort * 1.05)
                    e_need = max(e_need, 0.0)
                if e_need <= limit:
                    return max(0.3, e_need), min(85.0, max(40.0, r)), b * self._decay(tb) > 0.005
                if commit:
                    self.hanging = False            # too hard: they let the pack go
        if self.racing and not self.kicking:
            if self._hold is not None:
                k = self._decay(self._hold[1])
                e = k * self._hold[0] + (1.0 - k) * e
                coached |= k > 0.2
            if self._pace is not None and self.blown:
                self._pace = None             # a pace they cannot hold is given up
            if self._pace is not None:
                v_t, e_p, t_set, e0 = self._pace
                if commit and self.v_disp is not None:
                    e_p += self.tr.pace_gain * (v_t - self.v_disp) / v_t * T / 10.0
                    e_p = min(1.12 * e0, max(0.85 * e0, e_p))   # chased by feel, within reason
                    self._pace = (v_t, e_p, t_set, e0)
                k = self._decay(t_set)
                e = k * e_p + (1.0 - k) * e
                coached |= k > 0.2
            off, t_set = self._effort_off
            e += off * self._decay(t_set)
            coached |= abs(off * self._decay(t_set)) > 0.005
            so, st = self._self_eff
            if so:
                e += so * math.exp(-max(0.0, self.t - st) / 240.0)
        roff, rt = self._rate_off
        r += roff * self._decay(rt)
        coached |= abs(roff * self._decay(rt)) > 0.2
        holding = self._hold is not None and self._decay(self._hold[1]) >= 0.5
        if self.noise and not holding:
            e += self.wander
        return max(0.3, e), min(85.0, max(40.0, r)), coached

    # ── cues ─────────────────────────────────────────────────────────────
    def cue(self, ask: str, pace_s500: Optional[float] = None, line_m: Optional[float] = None,
            distance_left_m: Optional[float] = None) -> Dict[str, Any]:
        """A cue reaches the paddler now.  Returns whether they will follow it (the
        twin's truth, never shown to a coach).  ``distance_left_m`` goes with ``tell``:
        how far the paddler should now believe the finish is."""
        ask = str(ask)
        follow = bool(self.rng_cue.random() < self.tr.compliance)
        entry = {"t": self.t, "ask": ask, "pace_s500": pace_s500, "followed": follow}
        if line_m is not None:
            entry["line_m"] = line_m
        if distance_left_m is not None:
            entry["distance_left_m"] = distance_left_m
        self.cues.append(entry)
        if follow and ask in ASKS and ask != "none":
            self._pending.append((self.t + self.tr.response_s, ask, pace_s500, line_m, distance_left_m))
        return entry

    def _note(self, event: str, **kw: Any) -> None:
        self.incident_log.append({"t": round(self.t, 2), "race_s": round(self.t - self.start_t, 2),
                                  "race_m": round(self.race_m, 1), "event": event, **kw})

    def _say(self, who: str, text: str, **kw: Any) -> None:
        self.outbox.append({"t": round(self.t, 2), "race_s": round(self.t - self.start_t, 1),
                            "from": who, "text": text})
        self._note("said", who=who, text=text, **kw)

    def answer(self) -> Dict[str, Any]:
        """The paddler's reply to "how do you feel?" (a 1-10 effort rating and a remark),
        coloured by how they talk."""
        wf = self.w / self.p.w_prime_j
        bias = {"pessimist": 1.5, "optimist": -1.5}.get(self.tr.talk, 0.0)
        rpe = int(round(min(10.0, max(1.0, 4.0 + 6.0 * (1.0 - wf) + bias + float(self.rng_talk.normal(0.0, 0.7))))))
        if self._weed > 0 and self.rng_talk.random() < 0.6:
            says = "The boat feels heavy, like something is dragging."
        elif self.tr.talk == "pessimist":
            says = "It is really hurting."
        elif self.tr.talk == "optimist":
            says = "All good, I have plenty."
        else:
            says = "Close to my limit." if wf < 0.25 else "Hard but holding." if wf < 0.6 else "Steady, feels fine."
        self._note("asked", rpe=rpe, says=says)
        return {"effort_out_of_10": rpe, "says": says}

    def _apply(self, ask: str, pace_s500: Optional[float], e_now: float, line_m: Optional[float] = None,
               distance_left_m: Optional[float] = None) -> None:
        tr, t = self.tr, self.t
        if ask == "clear_rudder":                   # a back-stroke: the boat checks, the weed comes off
            self.v *= 0.55
            self._note("cleared_by_cue" if self._weed > 0 else "clear_without_weed")
            self._weed = 0.0
            return
        if ask == "reassure":
            if self._talk_pending is not None and self._talk_pending[1] < 0:
                self._talk_pending = None
                self._note("reassured_in_time")
            elif self._self_eff[0] < 0:
                self._self_eff = (0.0, t)
                self._note("reassured_late")
            return
        if ask == "tell":
            if distance_left_m is not None and distance_left_m > 0:
                self.believed_m = self.race_m + float(distance_left_m)
                self._note("told_distance", believed_m=round(self.believed_m), true_m=self.su.distance_m)
            return
        if ask in ("hold", "ease"):                 # holding back a paddler who wants to go
            if self._talk_pending is not None and self._talk_pending[1] > 0:
                self._talk_pending = None
                self._note("held_in_time")
            elif self._self_eff[0] > 0:
                self._self_eff = (0.0, t)
                self._note("held_late")
        if ask == "hang_on":
            self.hanging = True
            self._hang_bonus = (0.08, t)
            return
        if ask == "let_go":
            self.hanging = False
            return
        if ask in ("line_in", "line_out"):
            if line_m is not None:
                self.y_target = min(600.0, max(40.0, float(line_m)))
            else:
                step = -tr.line_step_m if ask == "line_in" else tr.line_step_m
                self.y_target = min(600.0, max(40.0, self.y_target + step))
            return
        if pace_s500 is not None and pace_s500 > 0 and ask in ("ease", "push", "hold"):
            self._pace = (500.0 / float(pace_s500), e_now, t, e_now)
            self._hold = None
            return
        if ask in ("ease", "push"):
            cur = self._effort_off[0] * self._decay(self._effort_off[1])
            self._effort_off = (cur + (tr.effort_step if ask == "push" else -tr.effort_step), t)
        elif ask in ("rate_up", "rate_down"):
            cur = self._rate_off[0] * self._decay(self._rate_off[1])
            self._rate_off = (cur + (tr.rate_step if ask == "rate_up" else -tr.rate_step), t)
        elif ask == "hold":
            self._hold, self._pace = (e_now, t), None
        elif ask == "finish":
            if self.believed_m - self.race_m <= tr.max_kick_m:
                self.kicking = True
                self.started_kick_t = self.started_kick_t or t
                if self.su.incidents:
                    self._note("kick_started", true_left_m=round(self.su.distance_m - self.race_m),
                               believed_left_m=round(self.believed_m - self.race_m), by="cue")
            else:
                self._apply("push", None, e_now)

    # ── stepping ─────────────────────────────────────────────────────────
    def advance(self, until_t: float) -> List[Dict[str, Any]]:
        """Paddle every stroke that ends by ``until_t``; return their rows."""
        rows: List[Dict[str, Any]] = []
        while self.finish_t is None or self.t < self.finish_t:
            if self.su.plan is not None and self.t >= sum(d for d, _, _ in self.su.plan):
                break
            fat = self._fatigue()
            e_guess, r_guess, _ = self._choose(fat, 1.0)
            if r_guess <= 0:                     # resting in a session plan
                if self.t + 1.0 > until_t:
                    break
                self._step_rest(1.0)
                continue
            T = 60.0 / r_guess
            if self.noise:
                T *= 1.0 + 0.012 * float(self.rng_feel.normal())
            if self.t + T > until_t:
                break
            self._open_race_step()
            for when, ask, pace, line, dist in [x for x in self._pending if x[0] <= self.t]:
                self._apply(ask, pace, e_guess, line, dist)
            self._pending = [x for x in self._pending if x[0] > self.t]
            e, r, coached = self._choose(fat, T, commit=True)
            rows.append(self._stroke(e, r, T, fat, coached))
            if self.finish_t is not None:
                break
        return rows

    def _open_race_step(self) -> None:
        """Incidents starting, scheduled remarks, the paddler's own talk (open races)."""
        if not self.racing:
            return
        race_s = self.t - self.start_t
        for i, inc in enumerate(self.su.incidents):
            if self._inc_started[i] or race_s < inc.t_race_s:
                continue
            self._inc_started[i] = True
            if inc.kind == "weed":
                self._weed = float(inc.size)
                self._note("onset", kind="weed", size=inc.size)
                if inc.remark_after_s is not None:
                    self._scheduled.append((self.t + inc.remark_after_s, "weed_remark", None))
                if inc.self_fix_s is not None:
                    self._scheduled.append((self.t + inc.self_fix_s, "self_clear", None))
            elif inc.kind == "wind_shift":
                self._wind = (float(inc.size), self.t)
                self._note("onset", kind="wind_shift", size=inc.size)
                if inc.remark_after_s is not None:
                    self._scheduled.append((self.t + inc.remark_after_s, "wind_radio", inc.size))
            elif inc.kind == "pod_drift":
                self._pod = (float(inc.size), self.t)
                self._note("onset", kind="pod_drift", size=inc.size)
            elif inc.kind == "course_change":
                old = self.su.distance_m
                self.su.distance_m = new = float(old + inc.size)
                self._note("onset", kind="course_change", old_m=old, new_m=new)
                how = (f"course shortened by one leg, the finish is now at {new:.0f} m" if new < old else
                       f"course extended by one leg, the finish is now at {new:.0f} m")
                self._say("race committee (radio)", f"All boats: {how}.")
        for item in [x for x in self._scheduled if x[0] <= self.t]:
            self._scheduled.remove(item)
            _, what, payload = item
            if what == "weed_remark" and self._weed > 0:
                self._say("paddler", "Boat feels heavy all of a sudden. Something on the rudder?")
            elif what == "self_clear" and self._weed > 0:
                self.v *= 0.55
                self._weed = 0.0
                self._note("self_cleared")
            elif what == "wind_radio":
                frm = "south" if payload > 0 else "north"
                self._say("escort boat (radio)", f"Wind is filling in from the {frm}, maybe 8 to 12 knots.")
        # the paddler's own talk
        if self.tr.talk == "quiet" or self.kicking or self.finish_t is not None:
            return
        if self._talk_pending is not None and self.t >= self._talk_pending[0]:
            self._self_eff = (self._talk_pending[1], self.t)
            self._note("self_ease" if self._talk_pending[1] < 0 else "self_push")
            self._talk_pending = None
        wf = self.w / self.p.w_prime_j
        frac = self.race_m / max(1.0, self.su.distance_m)
        if self.t < self._talk_next or self.blown:
            return
        rt = self.rng_talk
        if self.tr.talk == "pessimist" and self._talk_n < 2 and 0.4 <= wf <= 0.85 and 0.15 <= frac <= 0.8:
            self._say("paddler", ["I'm cooked. I can't hold this.", "This hurts. I don't think I can keep this up."][self._talk_n])
            self._talk_pending = (self.t + 30.0, -0.07)
        elif self.tr.talk == "optimist" and self._talk_n < 2 and wf < 0.35 and 0.15 <= frac <= 0.85:
            self._say("paddler", ["Feeling great! I've got more, I'm going.", "I feel strong, let me go harder."][self._talk_n])
            self._talk_pending = (self.t + 30.0, 0.07)
        elif self.tr.talk == "honest" and self._talk_n < 3 and wf < 0.2:
            self._say("paddler", "Legs are going. I'm close to the limit.")
        elif self.tr.talk == "honest" and not self._talk_good_said and wf > 0.6 and 0.3 <= frac <= 0.6:
            self._say("paddler", "Feeling good. This is steady.")
            self._talk_good_said = True
            self._talk_next = self.t + float(rt.uniform(240.0, 480.0))
            return
        else:
            return
        self._talk_n += 1
        self._talk_next = self.t + float(rt.uniform(360.0, 720.0))

    def _step_rest(self, dt: float) -> None:
        self._refill(0.0, dt)
        self.v *= math.exp(-dt / 6.0)
        self.total_m += self.v * dt
        self.t += dt

    def _refill(self, P: float, dt: float) -> None:
        cp = self._cp_eff()
        if P > cp:
            self.w -= (P - cp) * dt
        else:
            tau_r = 546.0 * math.exp(-0.01 * (cp - P)) + 316.0
            self.w += (self.p.w_prime_j - self.w) / tau_r * dt
        self.w = max(0.0, self.w)
        self.work += P * dt

    def _along_pos(self) -> float:
        """Position along the shore (m from the start line)."""
        if self.su.water is None or self.t < self.start_t:
            return 0.0
        into = self.race_m - self._leg_start
        return into if self.heading < 90.0 or self.heading > 270.0 else self.su.leg_m - into

    def _current_along(self) -> float:
        wind = 0.0
        if self._wind is not None:
            wind = self._wind[0] * min(1.0, max(0.0, (self.t - self._wind[1]) / 60.0))
        if self.su.water is not None:
            v = self.su.water.at(self.y, self._along_pos(), self.t - self.start_t)
            return (v + wind) * math.cos(math.radians(self.heading))
        return (self.su.current_mps + wind) * math.cos(math.radians(self.heading))

    def _stroke(self, e: float, rate: float, T: float, fat: float, coached: bool) -> Dict[str, Any]:
        p, cp = self.p, self._cp_eff()
        P = e * cp
        if self.w <= 0.01 * p.w_prime_j:
            if not self.blown and self.racing and self.even is not None:
                self.body_effort = min(self.body_effort, self.tr.settle_frac * self.even)
            self.blown = True
        elif self.blown and self.w >= self.tr.recover_frac * p.w_prime_j:
            self.blown = False
        if self.blown:
            P = min(P, self.tr.blowup_effort * cp)
            if self.racing:
                self.blown_s += T
        turning = self.t < self._turn_until
        eta = efficiency(p, rate, fat) * (0.6 if turning else 1.0)
        ama = (p.ama_load_kg + p.ama_load_fatigue_kg * fat) if self.craft.outrigger else 0.0
        pk = self.su.pack
        if pk is not None and self.racing:
            if self._v_pack_w is None and self.even is not None:
                P_tab, v_tab = _table_at(self._table, ama)
                p_even = self.even * cp * efficiency(p, self.tr.habit_rate, 0.3)
                self._v_pack_w = pk.speed_frac * float(np.interp(p_even, P_tab, v_tab))
            gap = self.pack_m - self.race_m
            self.on_wash = 0.0 <= gap <= pk.wash_m
        else:
            self.on_wash = False
        p_eff = eta * P / (1.0 - pk.draft) if self.on_wash else eta * P
        if self._weed > 0:
            p_eff /= 1.0 + self._weed                 # weed on the rudder: more drag at every speed
        v_target = float(np.interp(p_eff, *_table_at(self._table, ama)))
        self.v += (v_target - self.v) * (1.0 - math.exp(-T / 6.0))
        along = self._current_along()
        u_lat = 0.0
        if self.su.water is not None and self.racing and abs(self.y_target - self.y) > 1.0:
            dy = self.y_target - self.y
            u_lat = math.copysign(min(self.tr.lateral_mps, abs(dy) / T), dy)
            self.y += u_lat * T
        v_fwd = math.sqrt(max(0.0, self.v * self.v - u_lat * u_lat)) if u_lat else self.v
        vg = max(0.0, v_fwd + along)
        d = vg * T
        self._refill(P, T)
        if self.noise:
            self.wander += (-self.wander * T / self.tr.wander_tau_s
                            + self.tr.wander_sd * math.sqrt(2.0 * T / self.tr.wander_tau_s)
                            * float(self.rng_feel.normal()))
        t_end = self.t + T
        if pk is not None and self.racing and self._v_pack_w is not None:
            leg_k = int(self.pack_m // self.su.leg_m)
            if leg_k != self._pack_leg:                 # the pack turns too
                self._pack_leg = leg_k
                self._pack_turn_until = self.t + self.su.turn_s
            pack_heading = 0.0 if leg_k % 2 == 0 else 180.0
            if self.su.water is not None:
                c_pack = self.su.water.at(self.y, self._along_pos(), self.t - self.start_t) * math.cos(math.radians(pack_heading))
            else:
                c_pack = self.su.current_mps * math.cos(math.radians(pack_heading))
            v_pk = self._v_pack_w * (0.85 if self.t < self._pack_turn_until else 1.0)
            self.pack_m += max(0.0, v_pk + c_pack) * T
            if self.on_wash:
                self.wash_s += T
        self.total_m += d
        if self.t >= self.start_t and self.su.plan is None:
            before = self.race_m
            self.race_m += d
            if self.race_m >= self.su.distance_m and self.finish_t is None:
                frac = (self.su.distance_m - before) / max(d, 1e-9)
                self.finish_t = self.t + frac * T
                self.race_m = self.su.distance_m
            if self.race_m - self._leg_start >= self.su.leg_m:
                self._leg_start += self.su.leg_m
                self._turn(t_end)
            if self.finish_t is None and self.race_m >= self.believed_m:    # no finish line where they thought
                self.believed_m = self.su.distance_m
                self.kicking = False
                self._note("overran_believed_finish")
                self._say("paddler", "Where's the finish? I thought that was it!")
        else:
            leg = self.su.warmup_leg_m if self.su.plan is None else 1500.0
            if self.total_m - self._wu_leg_start >= leg:
                self._wu_leg_start = self.total_m
                self._turn(t_end)
            if self.su.plan is None and t_end >= self.start_t and self.t < self.start_t:
                self.heading = 0.0            # race start: first leg north
        a = 0.15
        self.v_disp = vg if self.v_disp is None else (1 - a) * self.v_disp + a * vg
        lg = self.log
        lg.setdefault("line_m", []).append(float(self.y))
        lg.setdefault("on_wash", []).append(float(self.on_wash))
        for k, v in (("t", t_end), ("race_s", t_end - self.start_t), ("race_m", self.race_m),
                     ("effort", e), ("p_shaft", P), ("p_boat", eta * P), ("eta", eta), ("rate", rate),
                     ("rate_best", best_rate(p, fat)), ("fatigue", fat), ("w_bal", self.w),
                     ("v_water", self.v), ("v_ground", vg), ("current_along", along),
                     ("coached", float(coached))):
            lg[k].append(float(v))
        self.t = t_end
        nm = self.rng_meas
        n = (lambda s: float(nm.normal(0.0, s))) if self.noise else (lambda s: 0.0)
        row = {"t": round(t_end, 3), "v_mean": max(0.0, vg + n(0.06)), "rate_spm": rate + n(0.4),
               "heading_deg": (self.heading + n(4.0)) % 360.0, "distance_m": self.total_m}
        if self.craft.outrigger is not None:          # the paddle pod (single-blade paddle)
            g = 1.0
            if self._pod is not None:            # the pod's reading drifting (the paddler's effort does not change)
                g = 1.0 + (self._pod[0] - 1.0) * min(1.0, max(0.0, (t_end - self._pod[1]) / 120.0))
            row["shaft_power_w"] = max(0.0, g * P * (1.0 + (float(self.rng_pod.normal(0.0, 0.05)) if self.noise else 0.0)))
        if self.su.water is not None:                  # GPS position against the course map
            q = (lambda sd: float(self.rng_pos.normal(0.0, sd))) if self.noise else (lambda sd: 0.0)
            row["line_m"] = max(0.0, self.y + q(3.0))
            row["along_m"] = self._along_pos() + q(3.0)
        return row

    def _turn(self, t: float) -> None:
        self.heading = (self.heading + 180.0) % 360.0
        self._turn_until = t + self.su.turn_s

    # ── scoring helpers ──────────────────────────────────────────────────
    def run_alone(self, step_s: float = 60.0, max_s: float = 6 * 3600.0) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        t = 0.0
        while t < max_s:
            t += step_s
            rows += self.advance(t)
            if self.finish_t is not None:
                break
            if self.su.plan is not None and self.t >= sum(d for d, _, _ in self.su.plan) - 1e-6:
                break
        return rows

    @property
    def race_time_s(self) -> Optional[float]:
        return None if self.finish_t is None else self.finish_t - self.start_t

    def truth(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, float) for k, v in self.log.items()}


def _speed_table(craft: Craft, total_kg: float) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    v = np.linspace(0.2, 7.0, 400)
    loads = (0.0,) if craft.outrigger is None else (0.0, 5.0, 10.0, 15.0, 20.0)
    return {L: (np.asarray(power_at_speed(craft, total_kg, v, L), float), v) for L in loads}


def _table_at(table, ama: float):
    loads = sorted(table)
    if len(loads) == 1:
        return table[loads[0]]
    ama = min(max(ama, loads[0]), loads[-1])
    j = max(0, min(len(loads) - 2, int(np.searchsorted(loads, ama)) - 1))
    w = (ama - loads[j]) / (loads[j + 1] - loads[j])
    P0, v = table[loads[j]]
    P1, _ = table[loads[j + 1]]
    return (1 - w) * P0 + w * P1, v


def _race(traits: Traits, setup: RaceSetup, effort: float, rate_policy: str) -> "PaddlerTwin":
    tw = PaddlerTwin(replace(traits, start_effort=effort, kick_m=0.0, talk="quiet"), setup,
                     body_effort=effort, rate_policy=rate_policy, noise=False)
    tw.run_alone()
    return tw


def _highest_unblown(traits: Traits, setup: RaceSetup, rate_policy: str, iters: int = 22) -> float:
    lo, hi = 0.7, 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _race(traits, setup, mid, rate_policy).blown_s > 0:
            hi = mid
        else:
            lo = mid
    return lo


def even_effort(traits: Traits, setup: RaceSetup) -> float:
    """The paddler's even pace: the highest constant effort (x critical power, no start
    sprint, no kick) that reaches the line without blowing up, at their own rate and
    ignoring the current (which they cannot judge)."""
    return _highest_unblown(traits, replace(setup, current_mps=0.0, water=None, incidents=[]), "own")


def oracle_effort(traits: Traits, setup: RaceSetup) -> Dict[str, float]:
    """The perfect-knowledge even pacer: the highest constant effort that reaches the
    line without blowing up in this race (current included), holding the best rate for
    its fatigue.  A reference for how much better a well-paced race could go, not a
    bound on what a coach can do (pacing with the current, a kick, can beat it)."""
    setup = replace(setup, incidents=[])
    e = _highest_unblown(traits, setup, "best")
    tw = _race(traits, setup, e, "best")
    return {"effort": e, "race_time_s": tw.race_time_s, "reserve_left": tw.w / traits.paddler.w_prime_j}


def twin_cp_test(traits: Traits, *, seed: int = 0, short_s: float = 180.0, long_s: float = 720.0,
                 rest_s: float = 1800.0, drain: float = 0.95, craft: str = "oc1") -> Tuple[List[Dict[str, Any]], "PaddlerTwin"]:
    """The critical-power test on this paddler, flat water: an all-out 3- and 12-minute
    piece each paced to spend ``drain`` of the reserve (as :func:`.scenarios.cp_test`)."""
    p, h = traits.paddler, traits.habit_rate
    e_s = 1.0 + drain * p.w_prime_j / (p.critical_power_w * short_s)
    e_l = 1.0 + drain * p.w_prime_j / (p.critical_power_w * long_s)
    plan = [(300.0, 56.0, 0.6), (short_s, h + 4.0, e_s), (rest_s, 48.0, 0.4),
            (long_s, h + 1.0, e_l), (120.0, 50.0, 0.4)]
    tw = PaddlerTwin(traits, RaceSetup(current_mps=0.0, plan=plan, craft=craft), seed=seed)
    return tw.run_alone(), tw
