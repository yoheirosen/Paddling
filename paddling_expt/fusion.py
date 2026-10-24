"""One engine for both devices: feed records, get a per-stroke table and a live panel.

:class:`OC1Engine` takes records in arrival order from any transport (see
:mod:`.records`), hands head-unit records to :class:`~.headunit.HeadUnitProcessor`
and pod records to :class:`~.pod.PodProcessor`, and joins their strokes.

A hull stroke is the surge peak; its pod stroke is the catch that comes shortly
before it.  The delay from catch to surge peak is learned from the session (it is
itself a technique number: how quickly blade force turns into boat
acceleration).  Either device may be missing — the table then carries only the
other device's columns — and every number says which device it came from.

Per stroke (``stroke_table``): time (head clock and UTC), stroke rate, mean speed,
the check (speed lost between strokes, absolute and as a share of the mean, and
the strongest deceleration), distance per stroke, side, and — with the pod —
impulse, peak force, where in the drive the peak falls, the share of impulse in
the first half of the drive, drive time, the force profile, mean blade force over
the stroke cycle (impulse x rate) and a power proxy (mean blade force x boat
speed).  The power proxy uses the blade-normal force the gauge sees and speed
over the ground, so it is a relative number, not watts.

The live panel (``panel``) summarises the last few strokes once a second, in the
shape the platform's agent tiers read.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .headunit import HeadConfig, HeadUnitProcessor
from .pod import PodProcessor, PodProcessorConfig, StrainCalibration
from .records import validate

__all__ = ["OC1Engine"]


class OC1Engine:
    def __init__(self, head_cfg: HeadConfig = None, pod_cfg: PodProcessorConfig = None,
                 mount: Optional[np.ndarray] = None,
                 calibration: Optional[StrainCalibration] = None,
                 validate_records: bool = True) -> None:
        self.head = HeadUnitProcessor(head_cfg, mount=mount)
        self.pod = PodProcessor(pod_cfg, calibration=calibration)
        self.validate_records = validate_records
        self.invalid: List[str] = []
        self.capacity = None              # set by analysis.capacity_curve(...)
        self.water_model = None           # set by analysis.water_speed_model(...)

    # ── ingest ───────────────────────────────────────────────────────────
    def feed(self, rec: Dict[str, Any]) -> None:
        if self.validate_records:
            err = validate(rec)
            if err:
                self.invalid.append(err)
                return
        if rec.get("dev") == "pod":
            self.pod.push(rec)
        else:
            self.head.push(rec)

    def feed_all(self, records) -> "OC1Engine":
        for r in records:
            self.feed(r)
        self.finish()
        return self

    def finish(self) -> None:
        self.head.finish()
        self.pod.finish()

    # ── joining the two devices ──────────────────────────────────────────
    def _pod_catches(self) -> np.ndarray:
        return np.array([s.get("catch", np.nan) for s in self.pod.strokes], float)

    def catch_to_surge_s(self) -> Optional[float]:
        """Median delay from the pod's catch to the hull's surge peak."""
        c = self._pod_catches()
        if not len(c) or not self.head.strokes:
            return None
        d = []
        for s in self.head.strokes:
            k = np.searchsorted(c, s["t"]) - 1
            if 0 <= k < len(c) and 0.05 < s["t"] - c[k] < 0.9:
                d.append(s["t"] - c[k])
        return float(np.median(d)) if len(d) >= 5 else None

    def stroke_table(self, last: Optional[int] = None) -> List[Dict[str, Any]]:
        """Joined per-stroke rows (``last``: only the most recent strokes)."""
        hs = [s for s in (self.head.strokes[-last:] if last else self.head.strokes)
              if s.get("final")]
        c = self._pod_catches()
        lag = self.catch_to_surge_s()
        used = set()
        rows: List[Dict[str, Any]] = []
        for s in hs:
            row = {"t": s["t"], "utc": self.head.clock.utc(s["t"]),
                   "rate_spm": s.get("rate_spm"), "cycle_s": s.get("cycle_s"),
                   "v_mean": s.get("v_mean"), "check_mps": s.get("check_mps"),
                   "check_ratio": s.get("check_ratio"), "peak_decel": s.get("peak_decel"),
                   "dps": s.get("dps"), "heading_deg": s.get("heading_deg"),
                   "side": s.get("side"), "side_source": "hull",
                   "profile_speed": s.get("profile")}
            if len(c) and lag is not None:
                target = s["t"] - lag
                k = int(np.nanargmin(np.abs(c - target))) if np.isfinite(c).any() else -1
                if k >= 0 and abs(c[k] - target) < 0.25 * (s.get("cycle_s") or 1.0) and k not in used:
                    used.add(k)
                    p = self.pod.strokes[k]
                    row.update({key: p.get(key) for key in
                                ("impulse", "peak", "peak_pos", "first_half_share", "drive_s",
                                 "profile", "units", "gap_filled")})
                    row["catch"] = p.get("catch")
                    row["catch_to_surge_s"] = s["t"] - p["catch"]
                    if p.get("side") is not None:
                        row["side"], row["side_source"] = p["side"], "pod"
                    if row["rate_spm"] and row["impulse"] is not None:
                        row["mean_force"] = row["impulse"] * row["rate_spm"] / 60.0
                        if row["v_mean"] is not None:
                            row["power_proxy"] = row["mean_force"] * row["v_mean"]
            rows.append(row)
        self._side_runs(rows, self._side_count_before(hs))
        if self.capacity is not None:
            from .analysis import percent_power
            percent_power(rows, self.capacity)
        if self.water_model is not None:
            from .analysis import water_speed
            water_speed(rows, self.water_model)
        return rows

    def _side_count_before(self, hs) -> tuple:
        """(side, strokes on it) just before the first of ``hs``, from the hull."""
        if not hs:
            return None, 0
        k = hs[0]["index"]
        side, n = None, 0
        for s in reversed(self.head.strokes[max(0, k - 40):k]):
            if side is None:
                side = s.get("side")
            if s.get("side") != side:
                break
            n += 1
        return side, n

    @staticmethod
    def _side_runs(rows: List[Dict[str, Any]], start=(None, 0)) -> None:
        """Strokes since the last side switch (1 = first stroke on the new side)."""
        prev, n = start
        for r in rows:
            sd = r.get("side")
            n = n + 1 if sd is not None and sd == prev else 1
            r["stroke_on_side"] = n
            prev = sd

    # ── live panel ───────────────────────────────────────────────────────
    def panel(self, t: Optional[float] = None, last_n: int = 8) -> Dict[str, Any]:
        """Once-a-second summary of the most recent strokes (head-unit clock)."""
        H = self.head.hist
        t = self.head.t_proc if t is None else t
        rows = [r for r in self.stroke_table(last=last_n + 4) if r["t"] <= t][-last_n:]
        out: Dict[str, Any] = {"t": t, "utc": self.head.clock.utc(t), **self.head.quality(t)}
        sl = H.window(t - 1.0, t)
        v1 = float(np.nanmean(H["v"][sl])) if sl.stop > sl.start else float("nan")
        out["speed"] = v1
        out["pace_500m_s"] = 500.0 / v1 if v1 > 0.3 else None
        med = lambda k: (float(np.median([r[k] for r in rows if r.get(k) is not None]))
                         if any(r.get(k) is not None for r in rows) else None)
        for k in ("rate_spm", "dps", "check_ratio", "check_mps", "impulse", "peak_pos",
                  "drive_s", "power_proxy", "pct_power", "water_speed", "current_along"):
            out[k] = med(k)
        if rows:
            out["side"] = rows[-1].get("side")
            out["stroke_on_side"] = rows[-1].get("stroke_on_side")
        out["on_run"] = self.head._run_on is not None
        out["pod_connected"] = bool(self.pod.strokes) and (
            self.pod.strokes[-1].get("catch", -math.inf) > t - 5.0)
        return out
