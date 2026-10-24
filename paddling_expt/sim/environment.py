"""The water the simulated canoe moves through: tidal current, swell, course.

Everything here is a *planted truth*: the pipeline never sees these objects, only
the sensor records they produce, and the scorers compare its estimates with them.

* :class:`CurrentField` — a tidal current that floods and ebbs on the principal
  lunar (M2, 12.42 h) cycle, weaker close to a straight shoreline, reversed inside
  circular eddies (the "hide behind the point" water), plus an optional constant
  drift.
* :class:`Swell` — a sum of deep-water wave trains.  Gives surface height, the
  slope along any heading (what pushes a canoe down the face of a bump) and the
  surface orbital velocity (which a GNSS receiver sees as speed changes).
* :class:`Course` — waypoints the simulated paddler steers to, in order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from ..geo import G, compass_to_math

__all__ = ["CurrentField", "Eddy", "Swell", "WaveTrain", "Course"]


@dataclass
class Eddy:
    """A circular patch where the tidal stream is reduced or reversed.

    ``strength`` 1 cancels the stream at the centre, 2 reverses it fully.
    """

    x: float
    y: float
    radius_m: float
    strength: float = 2.0


@dataclass
class CurrentField:
    """Tidal stream + shoreline shelter + eddies + constant drift (m/s, ENU)."""

    peak_mps: float = 0.0
    set_deg: float = 90.0            # compass direction the flood flows TOWARDS
    period_s: float = 44_712.0       # M2 tide, 12.42 h
    t_max_flood_s: float = 0.0       # UTC seconds of maximum flood
    shore_point: Tuple[float, float] = (0.0, -1e9)   # a point on the shoreline
    shore_normal_deg: float = 0.0    # compass direction pointing OFFSHORE
    shelter_m: float = 0.0           # 0 = no shoreline shelter
    eddies: List[Eddy] = field(default_factory=list)
    drift: Tuple[float, float] = (0.0, 0.0)          # constant (east, north) m/s

    def tidal_speed(self, t) -> np.ndarray:
        """Signed stream speed along ``set_deg`` (flood positive)."""
        return self.peak_mps * np.cos(2 * np.pi * (np.asarray(t, float) - self.t_max_flood_s)
                                      / self.period_s)

    def velocity(self, x, y, t) -> Tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        s = self.tidal_speed(t) * np.ones_like(x)
        if self.shelter_m > 0:
            a = compass_to_math(self.shore_normal_deg)
            d = (x - self.shore_point[0]) * np.cos(a) + (y - self.shore_point[1]) * np.sin(a)
            s = s * (1.0 - np.exp(-np.clip(d, 0.0, None) / self.shelter_m))
        for e in self.eddies:
            r2 = ((x - e.x) ** 2 + (y - e.y) ** 2) / (e.radius_m ** 2)
            s = s * (1.0 - e.strength * np.exp(-r2))
        a = compass_to_math(self.set_deg)
        return s * np.cos(a) + self.drift[0], s * np.sin(a) + self.drift[1]


@dataclass
class WaveTrain:
    amplitude_m: float
    period_s: float
    towards_deg: float               # compass direction the waves travel towards
    phase: float = 0.0


@dataclass
class Swell:
    """A sum of linear deep-water wave trains (no breaking)."""

    trains: List[WaveTrain] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._omega = np.array([2 * np.pi / w.period_s for w in self.trains])
        self._k = self._omega ** 2 / G
        ang = np.array([compass_to_math(w.towards_deg) for w in self.trains])
        self._dx, self._dy = np.cos(ang), np.sin(ang)
        self._a = np.array([w.amplitude_m for w in self.trains])
        self._ph = np.array([w.phase for w in self.trains])

    @property
    def active(self) -> bool:
        return bool(self.trains) and float(np.max(self._a)) > 0

    def phase_speeds(self) -> np.ndarray:
        return self._omega / self._k if self.trains else np.zeros(0)

    def state(self, x: float, y: float, t: float):
        """Height, gradient (d/dx, d/dy) and surface orbital velocity at one point."""
        if not self.trains:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        th = self._k * (self._dx * x + self._dy * y) - self._omega * t + self._ph
        c, s = np.cos(th), np.sin(th)
        eta = float(np.sum(self._a * c))
        g = -self._a * self._k * s
        ux = self._a * self._omega * c
        return (eta, float(np.sum(g * self._dx)), float(np.sum(g * self._dy)),
                float(np.sum(ux * self._dx)), float(np.sum(ux * self._dy)))


@dataclass
class Course:
    """Waypoints (ENU metres) visited in order; ``arrive_m`` is the rounding radius."""

    waypoints: Sequence[Tuple[float, float]]
    arrive_m: float = 12.0
