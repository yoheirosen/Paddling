"""A personal heart-rate model: how one person's heart rate answers effort, fitted from
their own sessions, instead of the textbook zones (a share of an age-predicted maximum).

Heart rate follows effort with a lag of tens of seconds, settles at a level that depends
on the effort, creeps up slowly over a session, and stops at a ceiling.  The textbook
assumes the settled level rises in a straight line from rest to the maximum, so a share of
the maximum names an effort zone.  For many people it does not: some sit high already at
moderate effort and rise little above it (a "compressed" response).  For them a share of
the maximum says almost nothing about effort near race pace.

:func:`fit_personal_model` fits, from rows of time, effort (paddle power, or any measure of
effort) and heart rate:

* the lag (a first-order response, 15-90 s);
* the settled level at each effort, a saturating curve
  ``hr = rest + rise * (1 - exp(-effort / scale)) / (1 - exp(-ref / scale))``, which covers
  the textbook straight line (a large ``scale``) and the compressed response (a small one);
* the slow creep (beats per 10 minutes) and the highest sustained rate seen.

From the fit: the heart rate expected at an effort (:meth:`PersonalHeartRate.hr_at`), the
effort a heart rate means (:meth:`effort_at`), and how many watts it takes to move the
settled level by a few beats at each effort (:meth:`resolution_w`) — where that is large,
heart rate no longer tells efforts apart and power, pace or feel must.  :func:`zones`
gives personal zones from the person's own sustainable power and flags the ones heart rate
cannot separate; :func:`textbook_zones` gives the share-of-maximum zones for comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["PersonalHeartRate", "fit_personal_model", "zones", "textbook_zones", "zone_of_share",
           "TEXTBOOK_ZONES", "EFFORT_ZONES"]

# Share-of-maximum zones as commonly taught (five zones).
TEXTBOOK_ZONES = [("recovery", 0.50, 0.60), ("endurance", 0.60, 0.70), ("tempo", 0.70, 0.80),
                  ("threshold", 0.80, 0.90), ("above threshold", 0.90, 1.00)]
# Effort zones as a share of the person's critical power (the zones the personal model names).
EFFORT_ZONES = [("recovery", 0.0, 0.55), ("endurance", 0.55, 0.75), ("tempo", 0.75, 0.90),
                ("threshold", 0.90, 1.05), ("above threshold", 1.05, 9.0)]


def zone_of_share(share: float) -> int:
    """The effort zone (an index into :data:`EFFORT_ZONES`) for a share of critical power."""
    return next(i for i, (_, lo, hi) in enumerate(EFFORT_ZONES) if lo <= max(0.0, share) < hi)


@dataclass
class PersonalHeartRate:
    rest: float                 # the settled level the curve extrapolates to at no effort (bpm)
    rise: float                 # the rise from rest to the reference effort (bpm)
    scale_w: float              # the curve's scale: small = compressed, large = nearly straight
    ref_w: float                # the reference effort the rise is measured to (W)
    lag_s: float                # first-order response time (s)
    creep_per_10min: float      # slow upward creep (bpm per 10 minutes of the session)
    ceiling: float              # highest 30-s mean heart rate seen (bpm)
    rms_bpm: float              # fit residual (bpm)
    n: int
    effort_range_w: List[float] = field(default_factory=list)   # efforts the data covered (5th-95th %)
    notes: List[str] = field(default_factory=list)

    def hr_at(self, effort_w: float) -> float:
        """The settled heart rate at this effort, early in a session (no creep)."""
        g = (1.0 - math.exp(-max(0.0, effort_w) / self.scale_w)) / (1.0 - math.exp(-self.ref_w / self.scale_w))
        return min(self.ceiling, self.rest + self.rise * g)

    def slope_bpm_per_w(self, effort_w: float) -> float:
        g1 = math.exp(-max(0.0, effort_w) / self.scale_w) / self.scale_w
        return self.rise * g1 / (1.0 - math.exp(-self.ref_w / self.scale_w))

    def resolution_w(self, effort_w: float, beats: float = 3.0) -> float:
        """Watts it takes to move the settled heart rate by ``beats`` at this effort."""
        s = self.slope_bpm_per_w(effort_w)
        return float("inf") if s <= 1e-9 else beats / s

    def effort_at(self, hr_bpm: float, minutes_in: float = 0.0) -> Optional[float]:
        """The effort a settled heart rate means ``minutes_in`` into a session (the creep
        taken off first); None above the curve's reach."""
        hr_bpm = hr_bpm - self.creep_per_10min * minutes_in / 10.0
        top = self.rest + self.rise / (1.0 - math.exp(-self.ref_w / self.scale_w))
        y = (hr_bpm - self.rest) / max(1e-9, top - self.rest)
        if y <= 0:
            return 0.0
        if y >= 1:
            return None
        return -self.scale_w * math.log(1.0 - y)

    def read(self, hr_bpm: float, cp_w: float, minutes_in: float = 0.0, margin: float = 0.05) -> Dict[str, Any]:
        """What a settled heart rate means ``minutes_in`` into a session: the effort, its share
        of critical power and its zone, or that heart rate cannot tell efforts apart there
        (above :meth:`separates_up_to` plus ``margin``; the effort is then only a lower bound)."""
        e = self.effort_at(hr_bpm, minutes_in=minutes_in)
        sep = self.separates_up_to(cp_w)
        if e is None:
            return {"effort_w": None, "share_of_cp": None, "zone": None,
                    "note": f"above anything this heart rate curve reaches: an effort above {sep:.2f} of "
                            "critical power, which heart rate cannot tell apart; use power, pace or feel"}
        share = e / cp_w
        if share > sep + margin:
            return {"effort_w": None, "at_least_w": round(sep * cp_w), "share_of_cp": None, "zone": None,
                    "note": f"an effort above about {sep:.2f} of critical power; heart rate cannot tell efforts "
                            "apart there (a few beats span a large range of power); use power, pace or feel"}
        return {"effort_w": round(e), "share_of_cp": round(share, 2),
                "zone": EFFORT_ZONES[zone_of_share(share)][0]}

    def summary(self, cp_w: Optional[float] = None) -> Dict[str, Any]:
        out = {"rest_bpm": round(self.rest), "lag_s": round(self.lag_s), "creep_per_10min": round(self.creep_per_10min, 1),
               "ceiling_bpm": round(self.ceiling), "fit_rms_bpm": round(self.rms_bpm, 1),
               "effort_range_w": [round(x) for x in self.effort_range_w], "notes": self.notes}
        if cp_w:
            pts = [0.5, 0.65, 0.8, 0.9, 1.0]
            out["hr_at_share_of_cp"] = {f"{int(100 * p)}%": round(self.hr_at(p * cp_w)) for p in pts}
            out["watts_per_3_beats_at_share_of_cp"] = {f"{int(100 * p)}%": round(self.resolution_w(p * cp_w))
                                                      for p in pts}
            out["heart_rate_separates_efforts_up_to_share_of_cp"] = self.separates_up_to(cp_w)
        return out

    def separates_up_to(self, cp_w: float, beats: float = 3.0, share: float = 0.05) -> float:
        """The highest share of critical power below which ``beats`` of heart rate still
        mean less than ``share`` of critical power (heart rate tells efforts apart there)."""
        top = 0.4
        for p in np.arange(0.4, 1.201, 0.01):
            if self.resolution_w(float(p) * cp_w, beats) <= share * cp_w:
                top = float(p)
            else:
                break
        return round(top, 2)


def _lagged(t: np.ndarray, x: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        dt = 0.0 if i == 0 else max(0.0, t[i] - t[i - 1])
        acc += (x[i] - acc) * (1.0 - math.exp(-dt / tau))
        out[i] = acc
    return out


def fit_personal_model(rows: Sequence[Dict[str, Any]], *, effort_key: str = "shaft_power_w",
                       hr_key: str = "heart_rate_bpm", ref_w: Optional[float] = None,
                       resample_s: float = 5.0,
                       creep_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)) -> Optional[PersonalHeartRate]:
    """Fit a :class:`PersonalHeartRate` from rows carrying ``t`` (s), an effort and a heart rate.
    ``ref_w``: the effort the rise is measured to (default: the 90th percentile effort)."""
    pts = [(float(r["t"]), float(r[effort_key]), float(r[hr_key])) for r in rows
           if r.get(effort_key) is not None and r.get(hr_key) is not None and r.get("t") is not None]
    if len(pts) < 50:
        return None
    a = np.array(sorted(pts))
    t0 = a[0, 0]
    grid = np.arange(a[0, 0], a[-1, 0], resample_s)
    if len(grid) < 30:
        return None
    P = np.interp(grid, a[:, 0], a[:, 1])
    H = np.interp(grid, a[:, 0], a[:, 2])
    # 30-s means of power, then the first-order lag on those
    k = max(1, int(round(30.0 / resample_s)))
    P = np.convolve(P, np.ones(k) / k, mode="same")
    ref = float(ref_w) if ref_w else float(np.percentile(P, 90))
    tt = (grid - t0) / 600.0
    best = None
    # the creep is held to a realistic 0-4 bpm per 10 min: a session whose effort rises with
    # time (a plain step test) cannot otherwise tell a creep from a rise with effort
    for tau in (15.0, 20.0, 30.0, 40.0, 50.0, 65.0, 80.0):
        Pl = _lagged(grid, P, tau)
        for scale in np.geomspace(0.08 * ref, 8.0 * ref, 40):
            g = (1.0 - np.exp(-np.maximum(Pl, 0.0) / scale)) / (1.0 - math.exp(-ref / scale))
            X = np.vstack([np.ones_like(g), g]).T
            for creep in creep_grid:
                y = H - creep * tt
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                res = y - X @ coef
                sse = float(res @ res)
                if coef[1] > 0 and (best is None or sse < best[0]):
                    best = (sse, tau, float(scale), (float(coef[0]), float(coef[1]), float(creep)))
    if best is None:
        return None
    sse, tau, scale, coef = best
    H30 = np.convolve(H, np.ones(k) / k, mode="valid")
    m = PersonalHeartRate(rest=float(coef[0]), rise=float(coef[1]), scale_w=scale, ref_w=ref, lag_s=tau,
                          creep_per_10min=float(coef[2]), ceiling=float(H30.max()) if H30.size else float(H.max()),
                          rms_bpm=math.sqrt(sse / len(H)), n=int(len(H)),
                          effort_range_w=[float(np.percentile(P, 5)), float(np.percentile(P, 95))])
    if scale < 0.6 * ref:
        m.notes.append("compressed response: heart rate is already high at moderate effort and rises little "
                       "near the top of the range; it tells hard efforts apart poorly")
    return m


def zones(model: PersonalHeartRate, cp_w: float) -> List[Dict[str, Any]]:
    """Personal zones from the person's own critical power, each with the heart rate the
    model expects and whether heart rate can tell it from its neighbours (a zone narrower
    than 3 beats cannot be held by heart rate)."""
    bands = EFFORT_ZONES[:-1] + [(EFFORT_ZONES[-1][0], EFFORT_ZONES[-1][1], 1.30)]
    out = []
    for nm, lo, hi in bands:
        h_lo, h_hi = model.hr_at(lo * cp_w), model.hr_at(hi * cp_w)
        # the bottom zone has no lower heart rate: the curve is not fitted down to no effort
        out.append({"zone": nm, "power_w": [round(lo * cp_w), round(hi * cp_w)],
                    "hr_bpm": [None if lo <= 0.0 else round(h_lo), round(h_hi)],
                    "heart_rate_can_hold_it": bool(h_hi - h_lo >= 3.0)})
    return out


def textbook_zones(hr_max: float) -> List[Dict[str, Any]]:
    return [{"zone": nm, "hr_bpm": [round(lo * hr_max), round(hi * hr_max)]} for nm, lo, hi in TEXTBOOK_ZONES]
