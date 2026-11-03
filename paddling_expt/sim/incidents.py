"""Open races: what can happen mid-race that nobody wrote a rule for (planted).

A coach that only paces is doing what a pacing algorithm does.  These races add the
things a race actually throws at a paddler, drawn per race on their own random stream
(so they move no other draw):

* **weed on the rudder** — extra drag of 6–12 % from a random moment until it is
  cleared with a back-stroke (which costs the boat's momentum); half the paddlers say
  the boat feels heavy, a third clear it on their own after 2–7 minutes;
* **a wind shift nobody forecast** — a push of 0.12–0.30 m/s along the course that
  builds over a minute; half the time the escort boat radios it;
* **the paddle pod drifting** — its reading slides to 72–82 % or 120–132 % of the true
  effort over two minutes (the paddler's effort does not change);
* **the race committee changing the course** — one leg more or one leg less, by radio,
  between 35 % and 55 % of the race; the paddler does not hear it unless told;
* **how the paddler talks** — honest, a pessimist (says they are cooked while they are
  not, then eases off unless reassured), or an optimist (says they feel great while
  the reserve is low, then pushes unless held).

The kinds are split in two before any coach sees them.  ``DEVELOPMENT_KINDS`` (weed,
wind shift, talk) may be used to try coaches out; ``HELD_OUT_KINDS`` (pod drift,
course change) appear only in the held-out check, so a coach cannot have been tuned
on them.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .twin import Incident

__all__ = ["DEVELOPMENT_KINDS", "HELD_OUT_KINDS", "TALK_KINDS", "draw_incidents", "draw_talk"]

DEVELOPMENT_KINDS = ("weed", "wind_shift")
HELD_OUT_KINDS = ("pod_drift", "course_change")
TALK_KINDS = ("honest", "pessimist", "optimist")


def draw_talk(seed: int) -> str:
    rng = np.random.default_rng([seed, 59])
    return str(rng.choice(TALK_KINDS, p=[0.3, 0.35, 0.35]))


def draw_incidents(seed: int, distance_m: float, leg_m: float, pool: str = "development") -> List[Incident]:
    """The incidents of one open race.  ``pool``: ``development`` (weed and wind shift
    only) or ``held_out`` (all four kinds; at least one held-out kind in every race)."""
    rng = np.random.default_rng([seed, 53])
    u = lambda a, b: float(rng.uniform(a, b))                       # noqa: E731
    T = distance_m / 3.4                                            # rough race time (s)
    out: List[Incident] = []
    if rng.random() < 0.7:
        out.append(Incident("weed", u(0.15, 0.75) * T, u(0.06, 0.12),
                            remark_after_s=u(30.0, 90.0) if rng.random() < 0.5 else None,
                            self_fix_s=u(120.0, 420.0) if rng.random() < 0.35 else None))
    if rng.random() < 0.6:
        size = u(0.12, 0.30) * (1.0 if rng.random() < 0.5 else -1.0)
        out.append(Incident("wind_shift", u(0.2, 0.7) * T, size,
                            remark_after_s=u(20.0, 60.0) if rng.random() < 0.5 else None))
    if pool == "held_out":
        held: List[Incident] = []
        if rng.random() < 0.6:
            g = u(0.72, 0.82) if rng.random() < 0.5 else u(1.20, 1.32)
            held.append(Incident("pod_drift", u(0.15, 0.6) * T, g))
        if rng.random() < 0.6:
            held.append(_course_change(rng, distance_m, leg_m, T))
        if not held:
            if rng.random() < 0.5:
                held.append(Incident("pod_drift", u(0.15, 0.6) * T,
                                     u(0.72, 0.82) if rng.random() < 0.5 else u(1.20, 1.32)))
            else:
                held.append(_course_change(rng, distance_m, leg_m, T))
        out += held
    return sorted(out, key=lambda i: i.t_race_s)


def _course_change(rng: np.random.Generator, distance_m: float, leg_m: float, T: float) -> Incident:
    frac = float(rng.uniform(0.35, 0.55))
    return Incident("course_change", frac * T, leg_m if rng.random() < 0.5 else -leg_m)


def describe(incidents: List[Incident]) -> List[Tuple[str, float, float]]:
    return [(i.kind, round(i.t_race_s), round(i.size, 3)) for i in incidents]
