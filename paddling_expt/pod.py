"""Real-time processing of the paddle pod: the pull-force curve of every stroke.

Causal, fed packet by packet:

1. **Clock**: the pod has no UTC.  Every packet is stamped by the head unit when
   it arrives; arrival = pod time + clock offset + drift + radio latency, and the
   latency is never negative.  So the *lowest* arrival-minus-pod-time values (one
   per 5 s bin) trace offset + drift, and a straight-line fit through them maps
   pod time to head-unit time to within a few milliseconds.
2. **Zero**: the strain bridge has an unknown zero that drifts with temperature.
   With the blade out of the water the shaft carries almost no bending, so the
   10th percentile of the last 3 s of strain is the running zero.
3. **Strokes**: a drive starts when the zeroed strain rises above 15 % of the
   recent stroke peak and ends when it falls below 10 %.  The catch and exit
   times are then refined by extending the rising and falling edges (through
   their 15 % and 40 % crossings) down to zero.
4. **Per stroke**: impulse (area under the force curve over the drive), peak,
   where in the drive the peak falls (0 = catch, 1 = exit), the share of the
   impulse in the first half of the drive, drive time, and the **force profile**
   — the curve resampled at 41 points from catch to exit — for comparing strokes.
   In raw counts until a static-load calibration converts counts to newtons at
   the blade.
5. **Side**: the paddle's sweep axis is the pod's dominant rotation axis (signed
   so the drive is positive); gravity along it during the drive is positive with
   the blade on the left, negative on the right — the shaft leans out over the
   paddling side.

A gap of up to ``max_fill_s`` (two packets) is bridged by straight-line
interpolation and the stroke is flagged; a longer gap inside a drive makes the
stroke incomplete, and it is dropped rather than reported with a hole in it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .headunit import History

__all__ = ["PodProcessorConfig", "PodClock", "PaddleAxes", "StrainCalibration", "PodProcessor"]


@dataclass
class PodProcessorConfig:
    zero_window_s: float = 3.0
    zero_pct: float = 10.0
    on_frac: float = 0.15
    off_frac: float = 0.10
    min_drive_s: float = 0.15
    profile_points: int = 41
    clock_bin_s: float = 5.0
    axes_after_strokes: int = 15
    max_fill_s: float = 0.11


class PodClock:
    """Pod clock -> head-unit clock from packet arrival times (lower envelope)."""

    def __init__(self, bin_s: float = 5.0) -> None:
        self.bin_s = bin_s
        self.bins: Dict[int, Tuple[float, float]] = {}   # bin -> (t_pod, min delay)
        self.coef: Optional[Tuple[float, float]] = None
        self._n = 0

    @property
    def ready(self) -> bool:
        return self.coef is not None

    def add(self, t_pod_last: float, rx: float) -> None:
        d = rx - t_pod_last
        k = int(math.floor(t_pod_last / self.bin_s))
        cur = self.bins.get(k)
        if cur is None or d < cur[1]:
            self.bins[k] = (t_pod_last, d)
        self._n += 1
        if self.coef is None or self._n % 50 == 0:
            self._fit()

    def _fit(self) -> None:
        pts = np.array(sorted(self.bins.values()))
        if len(pts) < 3:
            self.coef = (0.0, float(pts[:, 1].min()))
            return
        x0 = pts[0, 0]
        e, c = np.polyfit(pts[:, 0] - x0, pts[:, 1], 1)
        # one more pass without bins whose minimum still sat on a slow packet
        res = pts[:, 1] - (c + e * (pts[:, 0] - x0))
        keep = res < np.percentile(res, 80) + 1e-4
        if keep.sum() >= 3:
            e, c = np.polyfit(pts[keep, 0] - x0, pts[keep, 1], 1)
        self.coef = (float(e), float(c - e * x0))

    def to_head(self, t_pod):
        e, c = self.coef
        t_pod = np.asarray(t_pod, float)
        return t_pod + c + e * t_pod


class PaddleAxes:
    """The paddle's sweep axis in pod coordinates (signed: drive = positive)."""

    def __init__(self) -> None:
        self.sweep: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self.sweep is not None

    def estimate(self, gyr: np.ndarray, in_drive: np.ndarray) -> None:
        w, v = np.linalg.eigh(np.cov(gyr.T))
        ax = v[:, -1]
        if in_drive.any() and float(np.mean(gyr[in_drive] @ ax)) < 0:
            ax = -ax
        self.sweep = ax


@dataclass
class StrainCalibration:
    """Counts per newton at the blade, from known static loads."""

    counts_per_n: float

    @classmethod
    def from_loads(cls, loads_n: Sequence[float], counts: Sequence[float]) -> "StrainCalibration":
        slope, _ = np.polyfit(np.asarray(loads_n, float), np.asarray(counts, float), 1)
        return cls(float(slope))


class PodProcessor:
    def __init__(self, cfg: PodProcessorConfig = None, calibration: StrainCalibration = None) -> None:
        self.cfg = cfg or PodProcessorConfig()
        self.cal = calibration
        self.clock = PodClock(self.cfg.clock_bin_s)
        self.axes = PaddleAxes()
        self.hist = History(["tp", "strain", "z", "ax", "ay", "az", "gx", "gy", "gz"])
        self.strokes: List[Dict[str, Any]] = []
        self.packets = 0
        self.lost_packets = 0
        self.dropped_strokes = 0
        self.filled_samples = 0
        self._filled_in_drive = False
        self._last_seq: Optional[int] = None
        self._last_tp: Optional[float] = None
        self._fs = 50.0
        self._zero_buf: Deque[float] = deque()
        self._zero = None
        self._peaks: Deque[float] = deque(maxlen=8)
        self._state = "idle"
        self._drive_start_i = 0
        self._gap_in_drive = False
        self._recent_max: Deque[Tuple[float, float]] = deque()

    @property
    def calibrated(self) -> bool:
        return self.cal is not None

    def push(self, rec: Dict[str, Any]) -> None:
        if rec.get("dev") != "pod":
            return
        if rec.get("stream") == "meta":
            self._fs = float(rec.get("fs", self._fs))
            return
        if rec.get("stream") != "pkt":
            return
        fs = float(rec["fs"])
        self._fs = fs
        strain = np.asarray(rec["strain"], float)
        n = len(strain)
        if n == 0:
            return
        tp = float(rec["t0"]) + np.arange(n) / fs
        self.packets += 1
        seq = int(rec["seq"])
        if self._last_seq is not None:
            self.lost_packets += max(0, (seq - self._last_seq - 1) % 65536)
        self._last_seq = seq
        self.clock.add(float(tp[-1]), float(rec["rx"]))
        acc = np.asarray(rec.get("acc") if rec.get("acc") is not None else np.zeros((n, 3)), float)
        gyr = np.asarray(rec.get("gyr") if rec.get("gyr") is not None else np.zeros((n, 3)), float)
        gap = False
        if self._last_tp is not None and tp[0] - self._last_tp > 1.5 / fs:
            span = tp[0] - self._last_tp
            if span <= self.cfg.max_fill_s + 1e-9 and self.hist.n:
                # bridge a lost packet or two with a straight line
                k = int(round(span * fs)) - 1
                tf = self._last_tp + np.arange(1, k + 1) / fs
                s0 = self.hist["strain"][-1]
                sf = s0 + (strain[0] - s0) * (tf - self._last_tp) / span
                tp = np.concatenate([tf, tp])
                strain = np.concatenate([sf, strain])
                acc = np.concatenate([np.repeat(acc[:1], k, 0), acc])
                gyr = np.concatenate([np.repeat(gyr[:1], k, 0), gyr])
                if self._state == "drive":
                    self._filled_in_drive = True
                self.filled_samples += k
            else:
                gap = True
        self._last_tp = float(tp[-1])
        z = self._zeroed(tp, strain)
        i0 = self.hist.n
        self.hist.extend(tp=tp, strain=strain, z=z, ax=acc[:, 0], ay=acc[:, 1], az=acc[:, 2],
                         gx=gyr[:, 0], gy=gyr[:, 1], gz=gyr[:, 2])
        if gap and self._state == "drive":
            self._gap_in_drive = True
        self._segment(i0)

    def _zeroed(self, tp, strain) -> np.ndarray:
        """Subtract the running zero, updated once per packet."""
        c = self.cfg
        for t, s in zip(tp, strain):
            self._zero_buf.append((t, s))
        while self._zero_buf and tp[-1] - self._zero_buf[0][0] > c.zero_window_s:
            self._zero_buf.popleft()
        self._zero = float(np.percentile(np.fromiter((v for _, v in self._zero_buf), float),
                                         c.zero_pct))
        return strain - self._zero

    def _threshold(self, t: float) -> Tuple[float, float]:
        c = self.cfg
        if len(self._peaks) >= 3:
            ref = float(np.median(self._peaks))
            return c.on_frac * ref, c.off_frac * ref
        # before any strokes: a quarter of the largest bend in the last 3 s
        H = self.hist
        sl = slice(max(0, H.n - int(3 * self._fs)), H.n)
        m = float(np.max(H["z"][sl])) if H.n else 0.0
        return max(0.25 * m, 1e-9), max(0.15 * m, 1e-9)

    def _segment(self, i0: int) -> None:
        H = self.hist
        z, tp = H["z"], H["tp"]
        on, off = self._threshold(tp[-1])           # once per packet
        for i in range(max(1, i0), H.n):
            if self._state == "idle":
                if z[i] > on and z[i - 1] <= on:
                    self._state, self._drive_start_i = "drive", i - 1
                    self._gap_in_drive = self._filled_in_drive = False
            elif z[i] < off:
                self._state = "idle"
                self._finish_stroke(self._drive_start_i, i)

    def _edge_time(self, tp, z, idx, level, rising: bool) -> Optional[float]:
        """Interpolated time where ``z`` crosses ``level`` near ``idx``."""
        rng_ = range(idx, len(z) - 1) if rising else range(idx, 0, -1)
        for j in rng_:
            a, b = (j, j + 1) if rising else (j - 1, j)
            if (z[a] - level) * (z[b] - level) <= 0 and z[a] != z[b]:
                return tp[a] + (level - z[a]) / (z[b] - z[a]) * (tp[b] - tp[a])
        return None

    def _finish_stroke(self, a: int, b: int) -> None:
        c = self.cfg
        H = self.hist
        lo = max(0, a - 3)
        hi = min(H.n, b + 3)
        tp, z = H["tp"][lo:hi], H["z"][lo:hi]
        # an unfilled gap anywhere in the stroke's samples (e.g. a radio outage
        # that began in the recovery and ended mid-drive) means it is incomplete
        if self._gap_in_drive or len(z) < 5 or np.max(np.diff(tp)) > 1.5 / self._fs:
            self.dropped_strokes += 1
            return
        k = int(np.argmax(z))
        pk = float(z[k])
        if 0 < k < len(z) - 1:
            den = z[k - 1] - 2 * z[k] + z[k + 1]
            off = 0.5 * (z[k - 1] - z[k + 1]) / den if den != 0 else 0.0
            off = max(-0.5, min(0.5, off))
        else:
            off = 0.0
        dt = tp[1] - tp[0]
        t_pk = tp[k] + off * dt
        # rising edge: last crossings of 15 % and 40 % before the peak, extended to 0
        r15 = self._edge_time(tp[:k + 1], z[:k + 1], k, 0.15 * pk, rising=False)
        r40 = self._edge_time(tp[:k + 1], z[:k + 1], k, 0.40 * pk, rising=False)
        f15 = self._edge_time(tp[k:], z[k:], 0, 0.15 * pk, rising=True)
        f40 = self._edge_time(tp[k:], z[k:], 0, 0.40 * pk, rising=True)
        if None in (r15, r40, f15, f40):
            self.dropped_strokes += 1
            return
        catch = r15 - (r40 - r15) * 0.15 / 0.25
        exit_ = f15 + (f15 - f40) * 0.15 / 0.25
        drive = exit_ - catch
        if drive < c.min_drive_s:
            return
        self._peaks.append(pk)
        grid = np.linspace(catch, exit_, c.profile_points)
        prof = np.clip(np.interp(grid, tp, z), 0.0, None)
        prof[0] = prof[-1] = 0.0
        imp = float(np.trapezoid(prof, grid))
        half = grid <= catch + drive / 2
        scale = 1.0 / self.cal.counts_per_n if self.cal else 1.0
        st = {"catch_pod": float(catch), "exit_pod": float(exit_), "drive_s": float(drive),
              "peak": pk * scale, "impulse": imp * scale,
              "peak_pos": float((t_pk - catch) / drive),
              "first_half_share": float(np.trapezoid(prof[half], grid[half]) / imp) if imp > 0 else float("nan"),
              "profile": (prof * scale).round(4).tolist(),
              "gap_filled": bool(self._filled_in_drive),
              "units": "N" if self.cal else "counts"}
        # side: gravity along the signed sweep axis during the drive
        sl = slice(lo + int(np.searchsorted(tp, catch)), lo + int(np.searchsorted(tp, exit_)))
        acc = np.stack([H["ax"][sl], H["ay"][sl], H["az"][sl]], -1)
        gyr = np.stack([H["gx"][sl], H["gy"][sl], H["gz"][sl]], -1)
        st["_acc"], st["_gyr"] = acc.mean(0) if len(acc) else None, gyr
        self.strokes.append(st)
        if not self.axes.ready and len(self.strokes) >= c.axes_after_strokes:
            self._estimate_axes()
        if self.axes.ready:
            for s in self.strokes:
                if "side" not in s and s.get("_acc") is not None:
                    s["side"] = 1 if float(s["_acc"] @ self.axes.sweep) > 0 else -1
        if self.clock.ready:
            for s in self.strokes[-3:]:
                s["catch"] = float(self.clock.to_head(s["catch_pod"]))
                s["exit"] = float(self.clock.to_head(s["exit_pod"]))

    def _estimate_axes(self) -> None:
        H = self.hist
        n0 = max(0, H.n - int(60 * self._fs))
        gyr = np.stack([H["gx"][n0:], H["gy"][n0:], H["gz"][n0:]], -1)
        z = H["z"][n0:]
        ref = float(np.median(self._peaks)) if self._peaks else float(np.max(z))
        self.axes.estimate(gyr, z > 0.3 * ref)

    def finish(self) -> None:
        """Map every stroke to head-unit time with the final clock fit."""
        if not self.clock.ready:
            return
        self.clock._fit()
        for s in self.strokes:
            s["catch"] = float(self.clock.to_head(s["catch_pod"]))
            s["exit"] = float(self.clock.to_head(s["exit_pod"]))
            s.pop("_gyr", None)

    def asymmetry(self, last: int = 0) -> Dict[str, Any]:
        """Left minus right mean impulse over the mean (positive = left stronger)."""
        S = self.strokes[-last:] if last else self.strokes
        L = [s["impulse"] for s in S if s.get("side") == 1]
        R = [s["impulse"] for s in S if s.get("side") == -1]
        if len(L) < 5 or len(R) < 5:
            return {}
        mL, mR = float(np.mean(L)), float(np.mean(R))
        return {"asymmetry": (mL - mR) / ((mL + mR) / 2), "n_left": len(L), "n_right": len(R),
                "cv_left": float(np.std(L) / mL), "cv_right": float(np.std(R) / mR)}
