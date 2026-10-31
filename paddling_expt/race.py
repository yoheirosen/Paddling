"""Race state, live: how the race is going and what the rest of it allows.

:class:`RaceState` is updated once per stroke (head-unit stroke rows or SpeedCoach
strokes, in arrival order) and never looks ahead.  For a paddler whose critical power
and reserve are known (:class:`Athlete`, from a test or from training best efforts;
:func:`calibrate`) it keeps:

* **Speed through the water**: GPS speed minus the push of current and wind,
  fitted online from how speed depends on heading, forgetting slowly (the push
  changes over minutes).  On a single heading the push cannot be told from speed,
  so it is shrunk towards zero and flagged unknown until the course turns.
* **Power reaching the boat**: from that speed through the hull model.
* **The reserve above critical power** (W′ balance): work above critical power
  drains it; below critical power it refills in proportion to how far below and to
  how much is spent (Skiba's differential model, 2015).
* **Fade**: speed through the water at the stroke rate being paddled, against the
  same rate earlier in the race (a per-rate reference built over the first
  ``reference_s`` of racing).  It needs no calibration and sees what the reserve
  model cannot: a tiring paddler whose blade work moves the boat less.  It is held
  for ``turn_hold_s`` after a turn (the boat slows and the push estimate lags).
* **What the rest of the race allows**, for a known race distance: the steady
  power that just empties the reserve at the finish (``CP + W′ balance / time
  left``, through the hull model), its pace, the gap to the pace being paddled,
  and, if the current power is above that, how far before the finish the reserve
  runs out.
* **Rate position**: with the paddler's best speed at each stroke rate
  (:func:`.sessions.rate_envelope`), the power being paddled as a percentage of
  the power that best speed needed at this rate.

All of it is power reaching the boat, so the calibration must be in the same units:
best efforts measured the same way (GPS speed, push removed, the same hull model).
Pace numbers are through the water: what the paddler controls.

Limits, measured on the simulator (``PADDLING_PLAN.md``): the reserve is right
while the paddler is fresh (an over-fast start shows at once), but power reaching
the boat is effort times blade efficiency, and efficiency falls with fatigue (the
best stroke rate drifts down, more weight goes onto the ama), so later in a race
the model reads a tiring paddler as easing off; and a short test can misjudge
critical power, which shows as a reserve that reads empty while the speed holds.
The fade measure covers both (the race coach weighs them together); the paddle
pod's force would measure effort itself.  Rows that carry ``ama_load_kg``
(the head unit's roll) are powered at that load instead of the assumed one.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .boatspeed import power_at_speed, speed_at_power
from .hydro import Craft

__all__ = ["Athlete", "calibrate", "RaceState", "sustainable_power", "w_balance_step"]


@dataclass
class Athlete:
    """The paddler's calibration, in power reaching the boat."""

    cp_w: float                          # critical power
    w_prime_j: float                     # reserve above it
    envelope: Optional[Dict[str, Any]] = None   # best speed at each rate (sessions.rate_envelope)
    source: str = ""


def calibrate(best: Dict[float, Dict[str, Any]], *, envelope: Optional[Dict[str, Any]] = None,
              min_s: float = 120.0, max_s: float = 1200.0, source: str = "") -> Optional[Athlete]:
    """An :class:`Athlete` from best efforts (:func:`.sessions.best_efforts`, pooled
    or from a test such as an all-out 3-minute and 12-minute piece)."""
    from .sessions import critical_power
    cp = critical_power(best, min_s=min_s, max_s=max_s)
    if cp is None or cp["cp_w"] <= 0 or cp["w_prime_j"] <= 0:
        return None
    return Athlete(cp_w=cp["cp_w"], w_prime_j=cp["w_prime_j"], envelope=envelope,
                   source=source or f"best efforts {', '.join(f'{d:g} s' for d in sorted(best))}")


def w_balance_step(w_bal: float, power_w: float, dt: float, cp_w: float, w0: float) -> float:
    """One step of the reserve: drains by the work above critical power; below it,
    refills at ``(CP - P) (W0 - W) / W0`` (Skiba et al. 2015)."""
    if power_w >= cp_w:
        w_bal -= (power_w - cp_w) * dt
    else:
        w_bal += (cp_w - power_w) * (w0 - w_bal) / w0 * dt
    return min(w0, w_bal)


_SPEED_TABLES: Dict[tuple, tuple] = {}


def _speed_for(craft: Craft, total_kg: float, power_w: float, ama_kg: float) -> float:
    """Still-water speed at a power reaching the boat, from a cached fine table of the
    hull model (within 0.1 % of :func:`.boatspeed.speed_at_power`)."""
    key = (id(craft), round(total_kg, 3), round(ama_kg, 2))
    tab = _SPEED_TABLES.get(key)
    if tab is None:
        v = np.linspace(0.2, 7.5, 3000)
        tab = _SPEED_TABLES[key] = (np.asarray(power_at_speed(craft, total_kg, v, ama_kg), float), v)
    return float(np.interp(power_w, tab[0], tab[1]))


def sustainable_power(craft: Craft, total_kg: float, cp_w: float, w_bal: float,
                      remaining_m: float, *, ama_kg: float = 0.0, v_guess: float = 3.0,
                      cap: float = 3.0) -> Dict[str, float]:
    """The steady power that just empties the reserve over ``remaining_m`` in still
    water: ``P = CP + W / T`` with ``T = remaining / v(P)`` (fixed point)."""
    if remaining_m <= 0:
        return {"power_w": float("nan"), "speed_mps": float("nan"), "seconds": 0.0}
    T = remaining_m / max(0.5, v_guess)
    P = cp_w
    for _ in range(40):
        P = min(cp_w * cap, cp_w + max(0.0, w_bal) / T)
        v = _speed_for(craft, total_kg, P, ama_kg)
        T_new = remaining_m / max(0.3, v)
        if abs(T_new - T) < 1e-3 * T:
            T = T_new
            break
        T = T_new
    v = _speed_for(craft, total_kg, P, ama_kg)
    return {"power_w": float(P), "speed_mps": float(v), "seconds": float(remaining_m / v)}


class _PushFit:
    """Exponentially weighted least squares of ``v = a + b (rate - r0) + e sin h +
    n cos h``: ``(e, n)`` is the push of current and wind (m/s towards east / north).
    The push columns get a small ridge, so on one heading they stay near zero."""

    def __init__(self, tau_s: float = 600.0, ridge: float = 4.0, r0: float = 60.0) -> None:
        self.tau, self.ridge, self.r0 = float(tau_s), float(ridge), float(r0)
        self.A = np.zeros((4, 4))
        self.b = np.zeros(4)
        self.S = np.zeros((2, 2))          # weighted heading scatter (for coverage)
        self.sw = 0.0
        self.t_last: Optional[float] = None

    def update(self, t: float, v: float, rate: float, heading_deg: float) -> None:
        if self.t_last is not None:
            k = math.exp(-max(0.0, t - self.t_last) / self.tau)
            self.A *= k
            self.b *= k
            self.S *= k
            self.sw *= k
        self.t_last = t
        h = math.radians(heading_deg)
        x = np.array([1.0, rate - self.r0, math.sin(h), math.cos(h)])
        self.A += np.outer(x, x)
        self.b += x * v
        u = x[2:]
        self.S += np.outer(u, u)
        self.sw += 1.0

    def push(self) -> Dict[str, Any]:
        if self.sw < 30:
            return {"push_e": 0.0, "push_n": 0.0, "known": False, "coverage": 0.0}
        A = self.A + np.diag([1e-6, 1e-6, self.ridge, self.ridge])
        try:
            beta = np.linalg.solve(A, self.b)
        except np.linalg.LinAlgError:
            return {"push_e": 0.0, "push_n": 0.0, "known": False, "coverage": 0.0}
        # heading coverage: the smaller principal spread of the headings relative to the
        # larger, after removing the mean direction (0 on one heading, ~1 out and back)
        mean_u = self.A[0, 2:] / self.A[0, 0]
        C = self.S / self.sw - np.outer(mean_u, mean_u)
        ev = np.linalg.eigvalsh(C)
        cov = float(max(0.0, ev[-1]) / 0.5) if ev[-1] > 0 else 0.0
        return {"push_e": float(beta[2]), "push_n": float(beta[3]),
                "known": bool(cov > 0.3), "coverage": min(1.0, cov)}


class RaceState:
    """Per-stroke race state for one paddler and boat (see the module docstring).

    ``distance_m``: the race distance (projections need it).  ``push``: a known
    push of current and wind ``(east, north)`` in m/s to use instead of the online
    fit (from a tide model or the paddle pod's water-speed estimate)."""

    def __init__(self, craft: Craft, total_kg: float, athlete: Athlete, *,
                 distance_m: Optional[float] = None, ama_kg: float = 5.0,
                 push: Optional[tuple] = None, push_tau_s: float = 600.0,
                 smooth_s: float = 30.0, start_t: Optional[float] = None,
                 reference_s: float = 300.0, turn_hold_s: float = 45.0,
                 athlete_shaft: Optional[Athlete] = None) -> None:
        self.craft, self.total_kg, self.athlete = craft, float(total_kg), athlete
        # the paddle pod's own reserve: critical power and W' in shaft power (no
        # blade-efficiency drift); used when rows carry ``shaft_power_w``
        self.athlete_shaft = athlete_shaft
        self.cp_shaft_w = float(athlete_shaft.cp_w) if athlete_shaft else None
        self.w_bal_shaft = float(athlete_shaft.w_prime_j) if athlete_shaft else None
        self.recent_shaft: Deque[tuple] = deque()
        self._local: Deque[tuple] = deque()          # (t, v_gps, shaft power) over 20 s
        self.eta_est: Optional[float] = None         # boat power / shaft power, slow average
        self._pos: Dict[str, float] = {}
        self.distance_m = distance_m
        self.ama_kg = ama_kg if craft.outrigger else 0.0
        self.fixed_push = push
        self.fit = _PushFit(push_tau_s)
        self.smooth_s = float(smooth_s)
        self.w_bal = float(athlete.w_prime_j)
        self.start_t = start_t
        self.reference_s = float(reference_s)
        self._ref: Dict[int, List[float]] = {}       # rate band -> [sum v, n] early in the race
        self._d0: Optional[float] = None              # row distance at the start
        self.cp_w = float(athlete.cp_w)
        self.turn_hold_s = float(turn_hold_s)
        self._headings: Deque[tuple] = deque()       # (t, heading) over the last 20 s
        self._turn_until = -1e9
        self.t0: Optional[float] = None
        self.t_last: Optional[float] = None
        self.travelled = 0.0
        self.recent: Deque[tuple] = deque()          # (t, rate, v_gps, v_water, power, push_along)
        self.history: List[Dict[str, Any]] = []      # one compact row per ~5 s, for tools / plots
        self._next_hist = -1e9
        self.last: Dict[str, Any] = {}

    # ── per stroke ───────────────────────────────────────────────────────
    def update(self, row: Dict[str, Any]) -> Dict[str, Any]:
        t = float(row["t"])
        v = row.get("v_mean")
        if v is None or not math.isfinite(v):
            return self.last
        racing = self.start_t is None or t >= self.start_t
        if self.t0 is None and racing:
            self.t0 = t
        dt = 0.0 if self.t_last is None else min(5.0, max(0.0, t - self.t_last))
        self.t_last = t
        rate = row.get("rate_spm")
        hd = row.get("heading_deg")
        if hd is not None:
            self._headings.append((t, float(hd)))
            while self._headings and self._headings[0][0] < t - 20.0:
                self._headings.popleft()
            d = abs((self._headings[0][1] - float(hd) + 180.0) % 360.0 - 180.0)
            if d > 60.0:
                self._turn_until = t + self.turn_hold_s
        if rate and hd is not None and 30 <= rate <= 90 and v > 0.8:
            self.fit.update(t, float(v), float(rate), float(hd))
        if self.fixed_push is not None:
            pe, pn = self.fixed_push
            pk = {"push_e": pe, "push_n": pn, "known": True, "coverage": 1.0}
        else:
            pk = self.fit.push()
            pe, pn = pk["push_e"], pk["push_n"]
        along = 0.0
        if hd is not None:
            h = math.radians(hd)
            along = pe * math.sin(h) + pn * math.cos(h)
        v_water = max(0.0, float(v) - along)
        ama = row.get("ama_load_kg") if row.get("ama_load_kg") is not None else self.ama_kg
        P = float(power_at_speed(self.craft, self.total_kg, v_water, ama)) if v_water > 0.3 else 0.0
        a = self.athlete
        self.w_bal = max(0.0, w_balance_step(self.w_bal, P, dt, self.cp_w, a.w_prime_j))
        Ps = row.get("shaft_power_w")
        if Ps is not None and self.athlete_shaft is not None:
            self.w_bal_shaft = max(0.0, w_balance_step(self.w_bal_shaft, float(Ps), dt, self.cp_shaft_w,
                                                       self.athlete_shaft.w_prime_j))
            self.recent_shaft.append((t, float(Ps)))
            while self.recent_shaft and self.recent_shaft[0][0] < t - self.smooth_s:
                self.recent_shaft.popleft()
            self._local.append((t, float(v), float(Ps)))
            while self._local and self._local[0][0] < t - 20.0:
                self._local.popleft()
            if racing and self.t0 is not None and pk.get("known") and P > 0 and Ps > 20.0:
                r_ = min(1.0, max(0.2, P / float(Ps)))
                k_ = min(1.0, dt / min(600.0, 30.0 + (t - self.t0)))     # quick at first, then slow
                self.eta_est = r_ if self.eta_est is None else self.eta_est + k_ * (r_ - self.eta_est)
        for key in ("line_m", "along_m", "heading_deg"):
            if row.get(key) is not None:
                self._pos[key] = float(row[key])
        if not racing:
            return self.last
        if row.get("distance_m") is not None:
            if self._d0 is None:
                self._d0 = float(row["distance_m"])
            self.travelled = float(row["distance_m"]) - self._d0
        else:
            self.travelled += float(v) * dt
        if rate and 30 <= rate <= 90 and t - self.t0 > 60.0 and t >= self._turn_until:
            band = int(round(rate / 2.0))
            if t - self.t0 <= 60.0 + self.reference_s:
                acc = self._ref.setdefault(band, [0.0, 0])
                acc[0] += v_water
                acc[1] += 1
        self.recent.append((t, rate or float("nan"), float(v), v_water, P, along))
        while self.recent and self.recent[0][0] < t - self.smooth_s:
            self.recent.popleft()
        snap = self._snapshot(t, pk)
        if t >= self._next_hist:
            self.history.append({k: snap.get(k) for k in (
                "elapsed_s", "distance_m", "rate_spm", "speed_gps", "speed_water", "power_w",
                "w_bal_frac", "push_along_mps", "sustainable_power_w", "pace_gap_s500",
                "empties_in_m", "eta_s", "fade_pct", "shaft_power_w", "w_bal_pod_frac",
                "line_m", "along_m", "push_local_mps")})
            self._next_hist = t + 5.0
        self.last = snap
        return snap

    def _snapshot(self, t: float, pk: Dict[str, Any]) -> Dict[str, Any]:
        a = self.athlete
        R = np.array(self.recent, float)
        rate = float(np.nanmedian(R[:, 1])) if np.isfinite(R[:, 1]).any() else float("nan")
        v_gps, v_w, P, along = (float(R[:, k].mean()) for k in (2, 3, 4, 5))
        snap: Dict[str, Any] = {
            "t": t, "elapsed_s": t - self.t0, "distance_m": self.travelled,
            "rate_spm": rate, "speed_gps": v_gps, "speed_water": v_w, "power_w": P,
            "pace_water_s500": 500.0 / v_w if v_w > 0.3 else None,
            "push_known": pk["known"], "push_coverage": pk["coverage"],
            "push_mps": math.hypot(pk["push_e"], pk["push_n"]),
            "push_towards_deg": math.degrees(math.atan2(pk["push_e"], pk["push_n"])) % 360.0,
            "push_along_mps": along, "cp_w": self.cp_w, "w_bal_j": self.w_bal,
            "w_bal_frac": self.w_bal / a.w_prime_j}
        if (math.isfinite(rate) and self._ref and t - self.t0 > 60.0 + self.reference_s
                and t >= self._turn_until):
            # nearest reference rate band with enough strokes
            bands = [b for b, acc in self._ref.items() if acc[1] >= 15]
            if bands:
                b = min(bands, key=lambda x: abs(2.0 * x - rate))
                if abs(2.0 * b - rate) <= 3.0:
                    ref = self._ref[b][0] / self._ref[b][1]
                    snap["fade_pct"] = 100.0 * (v_w / ref - 1.0)
        snap.update(self._pos)
        if self.eta_est is not None and len(self._local) >= 5:
            vg = sum(x[1] for x in self._local) / len(self._local)
            ps = sum(x[2] for x in self._local) / len(self._local)
            snap["eta_est"] = self.eta_est
            snap["push_local_mps"] = vg - _speed_for(self.craft, self.total_kg, self.eta_est * ps, self.ama_kg)
        if self.athlete_shaft is not None and self.recent_shaft:
            Ps = sum(x[1] for x in self.recent_shaft) / len(self.recent_shaft)
            snap["shaft_power_w"] = Ps
            snap["cp_shaft_w"] = self.cp_shaft_w
            snap["w_bal_pod_frac"] = self.w_bal_shaft / self.athlete_shaft.w_prime_j
            if self.distance_m:
                if Ps > self.cp_shaft_w and v_w > 0.3:
                    snap["empties_in_m_pod"] = v_w * self.w_bal_shaft / (Ps - self.cp_shaft_w)
                rem_pod = max(0.0, self.distance_m - self.travelled)
                if rem_pod > 0 and v_w > 0.3:
                    T = rem_pod / v_w
                    snap["sustainable_shaft_power_w"] = self.cp_shaft_w + self.w_bal_shaft / T
        if a.envelope:
            env = a.envelope
            r = min(max(rate, env["rate_min"] - 2), env["rate_max"] + 2) if math.isfinite(rate) else None
            if r is not None:
                vb = float(np.polyval(env["coef"], r))
                pb = float(power_at_speed(self.craft, self.total_kg, vb, self.ama_kg))
                snap["best_speed_at_rate"] = vb
                snap["pct_best_at_rate"] = 100.0 * P / pb if pb > 0 else None
        if self.distance_m:
            rem = max(0.0, self.distance_m - self.travelled)
            snap["remaining_m"] = rem
            if rem > 0 and v_w > 0.3:
                sp = sustainable_power(self.craft, self.total_kg, self.cp_w, self.w_bal, rem,
                                       ama_kg=self.ama_kg, v_guess=v_w)
                snap["sustainable_power_w"] = sp["power_w"]
                snap["sustainable_pace_s500"] = 500.0 / sp["speed_mps"]
                snap["pace_gap_s500"] = 500.0 / v_w - 500.0 / sp["speed_mps"]
                if P > self.cp_w:
                    snap["empties_in_m"] = v_w * max(0.0, self.w_bal) / (P - self.cp_w)
                else:
                    snap["empties_in_m"] = None
                snap["eta_s"] = snap["elapsed_s"] + rem / max(0.3, v_gps)
        return snap

    def recalibrate(self, cp_w: Optional[float] = None, cp_shaft_w: Optional[float] = None) -> Dict[str, Any]:
        """Change the critical power the reserve models use from now on (an agent's
        revision of the calibration, e.g. when the race shows the test read it low)."""
        out = {"cp_w_before": self.cp_w, "cp_shaft_w_before": self.cp_shaft_w}
        if cp_w is not None:
            self.cp_w = float(cp_w)
        if cp_shaft_w is not None and self.athlete_shaft is not None:
            self.cp_shaft_w = float(cp_shaft_w)
        out.update({"cp_w": self.cp_w, "cp_shaft_w": self.cp_shaft_w})
        return out

    def compact(self) -> Dict[str, Any]:
        """The current state, rounded for a model's context."""
        def r(x, n=1):
            return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(x, n)
        s = self.last
        out = {"elapsed_min": r(s.get("elapsed_s", 0) / 60.0, 1),
               "distance_m": r(s.get("distance_m"), 0), "remaining_m": r(s.get("remaining_m"), 0),
               "rate_spm": r(s.get("rate_spm"), 0), "speed_water_mps": r(s.get("speed_water"), 2),
               "pace_water_s500": r(s.get("pace_water_s500"), 1),
               "pace_display_s500": r(500.0 / s["speed_gps"], 1) if (s.get("speed_gps") or 0) > 0.3 else None,
               "power_w": r(s.get("power_w"), 0),
               "critical_power_w": r(s.get("cp_w"), 0), "reserve_left_pct": r(100 * s.get("w_bal_frac", 0), 0),
               "sustainable_pace_s500": r(s.get("sustainable_pace_s500"), 1),
               "pace_gap_s500": r(s.get("pace_gap_s500"), 1),
               "reserve_runs_out_in_m": r(s.get("empties_in_m"), 0),
               "push_along_mps": r(s.get("push_along_mps"), 2) if s.get("push_known") else None,
               "push_known": s.get("push_known"),
               "pct_best_at_rate": r(s.get("pct_best_at_rate"), 0),
               "speed_vs_early_race_same_rate_pct": r(s.get("fade_pct"), 1)}
        return {k: v for k, v in out.items() if v is not None}
