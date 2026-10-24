"""The simulated paddler: what they plan to do, how they tire, how each stroke looks.

This is a **phenomenological** model chosen so that the coaching questions have a
known right answer, not a physiological claim.  The assumptions planted here (and
listed in the plan's honesty ledger) are:

* **Fatigue** has a fast part and a slow part.  The fast part is the W' balance of
  the critical-power model from endurance sport: work done above the paddler's
  critical power drains a reserve W', which refills below it.  The slow part is the
  fraction of a long-session work capacity already spent.  ``fatigue`` in [0, 1]
  combines them, and the slow part also lowers the sustainable critical power.
* **Stroke-rate efficiency**: the fraction of the paddler's shaft power that ends
  up moving the boat falls off on either side of a best stroke rate, as a Gaussian
  in strokes per minute.  The best rate moves **lower as the paddler tires**
  (``rate_opt_fresh - rate_opt_shift * fatigue``).  This is the planted answer to
  "is the stroke rate right for how tired they are?".
* **Each stroke** applies a propulsive force over a drive phase whose share of
  the cycle shrinks with rate and fatigue.  Fresh, the force is a half sine with
  its peak at mid-drive; as the paddler tires the peak moves **earlier** and the
  pull fades towards the exit (``profile_peak_shift``) — the planted change in
  the pull-force profile.  Strokes get less consistent with fatigue; one side is
  stronger than the other by ``side_asymmetry``; the first stroke after a side
  switch is weaker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = ["PaddlerParams", "BoatParams", "Segment", "Plan", "efficiency",
           "best_rate", "drive_duty", "profile_exponent", "profile_area"]


@dataclass
class PaddlerParams:
    body_kg: float = 78.0
    critical_power_w: float = 180.0      # shaft power sustainable when fresh
    w_prime_j: float = 15_000.0          # anaerobic reserve above critical power
    work_capacity_j: float = 2.5e6       # long-session capacity (slow fatigue)
    slow_cp_loss: float = 0.30           # CP lost at a fully spent capacity
    eta_max: float = 0.75                # best blade efficiency when fresh
    eta_fatigue_loss: float = 0.10       # efficiency lost at fatigue 1
    rate_opt_fresh: float = 64.0         # best stroke rate when fresh (spm)
    rate_opt_shift: float = 12.0         # best rate lower by this at fatigue 1
    rate_width: float = 8.0              # efficiency hill width (spm, 1 s.d.)
    duty0: float = 0.55                  # drive share of the cycle at 60 spm, fresh
    profile_peak_shift: float = 0.15     # force peak earlier by this share of the drive at fatigue 1
    side_asymmetry: float = 0.06         # (left - right) / mean impulse
    switch_every: Tuple[int, int] = (10, 20)   # strokes per side, uniform range
    switch_stroke_factor: float = 0.85   # impulse of the first stroke after a switch
    consistency_sd: float = 0.03         # stroke-to-stroke impulse s.d. (fraction), fresh
    consistency_fatigue_sd: float = 0.05 # extra s.d. at fatigue 1
    rate_jitter: float = 0.012           # stroke-to-stroke period s.d. (fraction)
    rate_wander_spm: float = 0.6         # slow wander of the held rate (s.d.)
    ama_load_kg: float = 5.0             # weight carried by the ama when fresh (OC1)
    ama_load_fatigue_kg: float = 10.0    # extra weight leant onto the ama at fatigue 1
    ama_load_sd_kg: float = 1.0          # stroke-to-stroke scatter of the ama load
    wobble_power_loss: float = 0.008     # share of power lost to bracing per (deg of roll wobble)^2


@dataclass
class BoatParams:
    """How the boat moves under the paddler (the hull shape and resistance are in
    :class:`paddling_expt.hydro.Craft`)."""

    added_mass_frac: float = 0.06
    yaw_pulse_dps: float = 4.0           # peak yaw rate of one stroke (deg/s)
    roll_pulse_deg: float = 1.2          # roll excursion of one stroke
    body_surge_n: float = 10.0           # internal body-motion force in recovery
    max_turn_dps: float = 15.0
    steer_gain: float = 1.2              # 1/s, heading error to turn rate
    wobble_dps: float = 0.6              # random yaw-rate wobble (s.d., deg/s) when fresh
    wobble_fatigue: float = 1.0          # wobble grows by this factor at fatigue 1
    wobble_tau_s: float = 0.35           # wobble correlation time
    roll_wobble_deg: float = 0.3         # random roll wobble (s.d., deg)


@dataclass
class Segment:
    """``duration_s`` of paddling at ``rate_spm`` and ``effort`` x critical power.

    ``rate_spm == 0`` is a rest (no strokes; the paddle is held still).
    """

    duration_s: float
    rate_spm: float
    effort: float = 0.9
    label: str = ""


@dataclass
class Plan:
    segments: List[Segment] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(sum(s.duration_s for s in self.segments))

    def at(self, t: float) -> Optional[Segment]:
        acc = 0.0
        for s in self.segments:
            acc += s.duration_s
            if t < acc:
                return s
        return None

    def boundaries(self) -> List[Tuple[float, float, Segment]]:
        out, acc = [], 0.0
        for s in self.segments:
            out.append((acc, acc + s.duration_s, s))
            acc += s.duration_s
        return out


def best_rate(p: PaddlerParams, fatigue: float) -> float:
    return p.rate_opt_fresh - p.rate_opt_shift * fatigue


def efficiency(p: PaddlerParams, rate_spm: float, fatigue: float) -> float:
    """Share of shaft power that moves the boat at this rate and fatigue."""
    z = (rate_spm - best_rate(p, fatigue)) / p.rate_width
    return p.eta_max * (1.0 - p.eta_fatigue_loss * fatigue) * math.exp(-0.5 * z * z)


def drive_duty(p: PaddlerParams, rate_spm: float, fatigue: float) -> float:
    return min(0.7, max(0.3, p.duty0 - 0.0025 * (rate_spm - 60.0) - 0.06 * fatigue))


def profile_exponent(p: PaddlerParams, fatigue: float) -> float:
    """Shape exponent g of the drive force ``sin(pi * x**g)`` (x = share of the
    drive): its peak sits at ``0.5 ** (1 / g)``, mid-drive when fresh."""
    peak = 0.5 - p.profile_peak_shift * fatigue
    return math.log(0.5) / math.log(max(0.05, min(0.95, peak)))


def profile_area(g: float, n: int = 200) -> float:
    """Integral of ``sin(pi * x**g)`` over the drive (2/pi for a half sine)."""
    s = 0.0
    for i in range(n):
        x = (i + 0.5) / n
        s += math.sin(math.pi * x ** g)
    return s / n
