"""Real-time processing of the head unit (boat box): from raw records to strokes.

Everything here is **causal** — it uses only data that has already arrived — and
runs as records are fed, so the same code serves a live relay and a log replayed
after the session.  The steps:

1. **Clock**: the box's own clock is mapped to UTC by a running straight-line fit
   over GNSS fixes (each carries both).
2. **Mount calibration**: the box sits at an unknown rotation on a flexing beam.
   From the first ~40 s of paddling, the mean accelerometer reading gives "up";
   the dominant horizontal direction of the stroke surge gives the fore-aft axis;
   its sign comes from matching the integrated surge with the GNSS speed's own
   within-stroke wobble.  Samples from before the calibration are kept and
   processed once it is known, so the first strokes are not lost.  A stored
   calibration can be supplied instead.
3. **Attitude**: pitch and roll from a complementary filter (gyro for fast
   changes, gravity for the slow level), used to remove gravity from the surge
   and to see nose-down surfing.  Wave-slap impacts and saturated samples are
   left out of the gravity reference and the surge.
4. **Strokes**: the fore-aft acceleration, band-passed to 0.35–3 Hz, has one peak
   per stroke; a peak counts when it clears a fraction of the recent peak height
   and is the largest within a refractory window scaled to the recent stroke
   period.  Stroke rate is 60 / the interval between peaks.
5. **Side**: a one-sided stroke yaws the bow away from the paddle side (a left
   stroke turns the bow right).  The yaw-rate pulse during the drive, relative to
   the cycle's mean, gives each stroke's raw side; a switch is accepted after two
   consecutive strokes agree on the new side, and both are relabelled.  This
   physical sign is a planted assumption until checked on a real hull.
6. **Speed**: a two-state Kalman filter (fore-aft ground speed, accelerometer
   bias) integrates the IMU at 100 Hz and is corrected by GNSS speed at 5–10 Hz,
   giving the within-stroke speed profile that GNSS alone is too slow and noisy
   for.  It runs a fixed 0.25 s behind the newest IMU sample so a GNSS fix that
   arrives a little late is applied at its own epoch.
7. **Per stroke** (once the next stroke has been seen): mean / minimum / maximum
   speed over the cycle; the **check** — how much speed the boat loses between
   strokes (maximum minus minimum, in m/s and as a share of the mean; lower is a
   smoother run) and the strongest deceleration; distance per stroke; and the
   within-stroke speed profile.
8. **Downwind runs**: speed well above the recent paddling baseline together with
   forward acceleration and a nose-down pitch (a speed gain alone is often just
   a harder piece starting).

GNSS speed is speed over **ground**: currents, tide and swell orbital motion are
in it.  Separating speed through the water needs the current estimate (a later
milestone); nothing here assumes a water-speed sensor.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .geo import G

__all__ = ["HeadConfig", "ClockMap", "MountCalibration", "HeadUnitProcessor", "History"]


@dataclass
class HeadConfig:
    calib_moving_s: float = 40.0
    calib_min_speed: float = 1.2
    proc_lag_s: float = 0.25
    hp_hz: float = 0.35
    lp_hz: float = 3.0
    peak_frac: float = 0.35
    min_peak: float = 0.12
    refractory_frac: float = 0.6
    attitude_tau_s: float = 10.0
    kf_acc_sd: float = 0.8
    kf_bias_sd: float = 0.01
    gnss_speed_sd: float = 0.05
    run_gain: float = 0.45
    run_end_gain: float = 0.2
    run_min_s: float = 1.5
    run_pitch_deg: float = 1.0
    run_acc: float = 0.05
    base_window_s: float = 45.0
    base_pct: float = 25.0
    side_confirm: int = 2


class History:
    """Append-only numeric columns (amortised growth) for the processed samples."""

    def __init__(self, names, cap: int = 4096) -> None:
        self.names = list(names)
        self._a = {k: np.empty(cap) for k in self.names}
        self.n = 0

    def extend(self, **cols) -> None:
        m = len(next(iter(cols.values())))
        if self.n + m > len(self._a[self.names[0]]):
            cap = max(2 * len(self._a[self.names[0]]), self.n + m)
            for k in self.names:
                b = np.empty(cap)
                b[: self.n] = self._a[k][: self.n]
                self._a[k] = b
        for k in self.names:
            self._a[k][self.n:self.n + m] = cols[k]
        self.n += m

    def __getitem__(self, k: str) -> np.ndarray:
        return self._a[k][: self.n]

    def window(self, t0: float, t1: float) -> slice:
        t = self["t"]
        return slice(int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1, "right")))


class ClockMap:
    """Head-unit clock -> UTC, a running least-squares line over GNSS epochs."""

    def __init__(self, max_pairs: int = 3000) -> None:
        self.pairs: Deque[Tuple[float, float]] = deque(maxlen=max_pairs)
        self.coef: Optional[Tuple[float, float]] = None
        self._since = 0

    def add(self, t_head: float, utc: float) -> None:
        self.pairs.append((t_head, utc))
        self._since += 1
        if self.coef is None or self._since >= 25:
            self._fit()

    def _fit(self) -> None:
        self._since = 0
        a = np.asarray(self.pairs)
        if len(a) == 1:
            self.coef = (1.0, a[0, 1] - a[0, 0])
            return
        t0 = a[0, 0]
        s, c = np.polyfit(a[:, 0] - t0, a[:, 1], 1)
        self.coef = (s, c - s * t0)

    def utc(self, t_head):
        if self.coef is None:
            return None
        s, c = self.coef
        return s * np.asarray(t_head, float) + c


class MountCalibration:
    """Box axes -> hull axes.  ``matrix @ v_box = v_hull`` (rows: forward, left, up)."""

    def __init__(self, matrix: Optional[np.ndarray] = None) -> None:
        self.matrix = None if matrix is None else np.asarray(matrix, float)
        self.info: Dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.matrix is not None

    @staticmethod
    def estimate(t: np.ndarray, acc: np.ndarray, gnss_t: np.ndarray, gnss_sog: np.ndarray,
                 fs: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        from scipy.signal import sosfiltfilt
        up = acc.mean(0)
        up /= np.linalg.norm(up)
        h = acc - np.outer(acc @ up, up)
        sos = butter(2, [0.4, 3.0], btype="band", fs=fs, output="sos")
        hb = sosfiltfilt(sos, h, axis=0)
        # two in-plane basis vectors, then the principal axis of the surge
        e1 = np.cross(up, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 0.3:
            e1 = np.cross(up, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        P = np.stack([hb @ e1, hb @ e2], -1)
        evals, evecs = np.linalg.eigh(np.cov(P.T))
        d = evecs[:, -1]
        fwd = d[0] * e1 + d[1] * e2
        # Sign: the integrated surge must rise and fall with the GNSS speed's own
        # within-stroke wobble.  (The principal axis is tilted by 1-4 degrees by
        # the sideways roll and yaw of one-sided strokes when one side dominates
        # the window; that leaks under 0.02 m/s^2 into the surge, so it is left.)
        hp = lambda x: x - np.convolve(x, np.ones(9) / 9, mode="same")
        vi = hp(np.interp(gnss_t, t, np.cumsum(hb @ fwd) / fs))
        gs = hp(np.asarray(gnss_sog, float))
        m = slice(5, -5) if len(gs) > 12 else slice(None)
        r = float(np.corrcoef(vi[m], gs[m])[0, 1]) if len(gs[m]) > 3 else 0.0
        if r < 0:
            fwd = -fwd
        left = np.cross(up, fwd)
        M = np.stack([fwd, left, up])
        return M, {"surge_axis_share": float(evals[-1] / max(1e-12, evals.sum())),
                   "gnss_corr": abs(r)}


class HeadUnitProcessor:
    """Feed head-unit records (and nothing else); collect strokes, runs, panel rows."""

    def __init__(self, cfg: HeadConfig = None, mount: Optional[np.ndarray] = None) -> None:
        self.cfg = cfg or HeadConfig()
        self.mount = MountCalibration(mount)
        self.clock = ClockMap()
        self.fs = 100.0
        self.acc_range = 8 * G
        self._raw: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # pending IMU
        self._calib: List[Tuple[np.ndarray, np.ndarray]] = []
        self._gnss: Deque[Dict[str, Any]] = deque()                     # pending fixes
        self.fixes = History(["t", "v", "ve", "vn"], cap=1024)         # every GNSS velocity
        self._moving_s = 0.0
        self.t_newest = -math.inf
        self.t_proc = -math.inf
        self.hist = History(["t", "a_fwd", "band", "yaw_rate", "roll", "pitch", "v",
                             "sat"])
        self.strokes: List[Dict[str, Any]] = []
        self.runs: List[Tuple[float, float]] = []
        self.events: List[Dict[str, Any]] = []
        self.last_fix: Optional[Dict[str, Any]] = None
        self.last_fix_t = -math.inf
        self._gnss_hz = 5.0
        self._init_filters()

    # ── setup ────────────────────────────────────────────────────────────
    def _init_filters(self) -> None:
        c = self.cfg
        self._sos = np.vstack([butter(2, c.hp_hz, btype="high", fs=self.fs, output="sos"),
                               butter(4, c.lp_hz, btype="low", fs=self.fs, output="sos")])
        self._zi = None
        self._att = None                      # (pitch, roll)
        self._a_last = 0.0
        self._gbias = np.zeros(3)
        self._gbias_n = 0
        self._kf = None                       # [v, b], P
        self._y_tail: List[Tuple[float, float]] = []
        self._pending_peak: Optional[Tuple[float, float]] = None
        self._peak_heights: Deque[float] = deque(maxlen=9)
        self._intervals: Deque[float] = deque(maxlen=7)
        self._last_peak_t = -math.inf
        self._side = 0
        self._side_pending: List[int] = []
        self._run_on: Optional[float] = None
        self._runs_eval_t = -math.inf
        self._run_off_since: Optional[float] = None

    @property
    def period(self) -> float:
        return float(np.median(self._intervals)) if self._intervals else 1.0

    # ── ingest ───────────────────────────────────────────────────────────
    def push(self, rec: Dict[str, Any]) -> None:
        s = rec.get("stream")
        if s == "imu":
            t = np.asarray(rec["t"], float)
            acc = np.asarray(rec["acc"], float)
            gyr = np.asarray(rec["gyr"], float)
            self._raw.append((t, acc, gyr))
            self.t_newest = max(self.t_newest, float(t[-1]))
            if not self.mount.ready:
                self._calib.append((t, acc))
                self._try_calibrate()
                # bounded memory while waiting for enough paddling to calibrate
                while self._calib and self.t_newest - float(self._calib[0][0][-1]) > 180.0:
                    self._calib.pop(0)
                while self._raw and self.t_newest - float(self._raw[0][0][-1]) > 180.0:
                    self._raw.pop(0)
            if self.mount.ready:
                self._process_until(self.t_newest - self.cfg.proc_lag_s)
        elif s == "gnss":
            if rec.get("utc") is not None:
                self.clock.add(float(rec["t"]), float(rec["utc"]))
            if rec.get("fix") and rec.get("sog") is not None:
                self._gnss.append(rec)
                sog = float(rec["sog"])
                ve, vn = rec.get("vel_e"), rec.get("vel_n")
                if ve is None or vn is None:
                    c = math.radians(float(rec.get("cog") or 0.0))
                    ve, vn = sog * math.sin(c), sog * math.cos(c)
                self.fixes.extend(t=[float(rec["t"])], v=[sog], ve=[float(ve)], vn=[float(vn)])
                if float(rec["sog"]) > self.cfg.calib_min_speed:
                    self._moving_s += 1.0 / max(1.0, self._gnss_hz)
                self.last_fix = rec
        elif s == "meta":
            self.fs = float(rec.get("imu_hz", self.fs))
            self._gnss_hz = float(rec.get("gnss_hz", 5.0))
            self.acc_range = float(rec.get("acc_range", self.acc_range))
            self._init_filters()
        elif s in ("house", "button", "link"):
            self.events.append({"t": rec.get("t"), "type": s,
                                **{k: v for k, v in rec.items()
                                   if k not in ("dev", "stream", "t", "arrival")}})

    def _try_calibrate(self) -> None:
        if self._moving_s < self.cfg.calib_moving_s or self.fixes.n < 20:
            return
        t = np.concatenate([c[0] for c in self._calib])
        acc = np.concatenate([c[1] for c in self._calib])
        gt, gv = self.fixes["t"], self.fixes["v"]
        moving = np.interp(t, gt, gv) > self.cfg.calib_min_speed
        if moving.sum() < self.cfg.calib_moving_s * self.fs * 0.8:
            return
        m = (gt >= t[moving][0]) & (gt <= t[moving][-1])
        M, info = MountCalibration.estimate(t[moving], acc[moving], gt[m], gv[m], self.fs)
        self.mount.matrix, self.mount.info = M, info
        self._calib.clear()
        self.events.append({"t": float(t[-1]), "type": "mount_calibrated", **info})

    # ── the per-sample path ──────────────────────────────────────────────
    def _process_until(self, t_end: float) -> None:
        if not self._raw:
            return
        t = np.concatenate([r[0] for r in self._raw])
        acc = np.concatenate([r[1] for r in self._raw])
        gyr = np.concatenate([r[2] for r in self._raw])
        k = int(np.searchsorted(t, t_end, "right"))
        if k == 0:
            return
        rest = (t[k:], acc[k:], gyr[k:])
        self._raw = [rest] if len(rest[0]) else []
        self._process_block(t[:k], acc[:k], gyr[:k])

    def _process_block(self, t, acc, gyr) -> None:
        c, dt = self.cfg, 1.0 / self.fs
        B = self.mount.matrix
        f = acc @ B.T
        w = gyr @ B.T
        sat = (np.abs(acc) >= 0.98 * self.acc_range).any(1).astype(float)
        # slow gyro bias on roll/pitch axes (their long-run mean rate is zero)
        for row in w:
            self._gbias_n += 1
            a = 1.0 / min(self._gbias_n, int(60 * self.fs))
            self._gbias[:2] += a * (row[:2] - self._gbias[:2])
        w = w - np.array([self._gbias[0], self._gbias[1], 0.0])
        # attitude: complementary filter
        alpha = c.attitude_tau_s / (c.attitude_tau_s + dt)
        # gravity's pitch reference, with the boat's own acceleration (from GNSS
        # speed changes) taken out — otherwise a surf run reads as a nose-up pitch
        pitch_acc = np.arctan2(-(f[:, 0] - self._gnss_accel(t)), np.clip(f[:, 2], 1e-3, None))
        roll_acc = np.arctan2(f[:, 1], np.clip(f[:, 2], 1e-3, None))
        if self._att is None:
            self._att = (float(pitch_acc[0]), float(roll_acc[0]))
        # Impacts (wave slap, a hand on the box) and range saturation: the
        # accelerometer is not reading the boat's motion, so those samples (and a
        # few either side) give no gravity reference and no surge.
        imp = sat > 0
        imp |= np.abs(np.linalg.norm(f, axis=1) - G) > 0.5 * G
        if imp.any():
            k = np.flatnonzero(imp)
            for d in (-3, -2, -1, 1, 2, 3):
                imp[np.clip(k + d, 0, len(t) - 1)] = True
        pitch = np.empty(len(t))
        roll = np.empty(len(t))
        a_fwd = np.empty(len(t))
        pc, rc = self._att
        a_last = self._a_last
        for i in range(len(t)):
            if imp[i]:
                pc += w[i, 1] * dt
                rc += w[i, 0] * dt
                a_fwd[i] = a_last
            else:
                pc = alpha * (pc + w[i, 1] * dt) + (1 - alpha) * pitch_acc[i]
                rc = alpha * (rc + w[i, 0] * dt) + (1 - alpha) * roll_acc[i]
                a_fwd[i] = a_last = f[i, 0] + G * math.sin(pc)
            pitch[i], roll[i] = pc, rc
        self._att, self._a_last = (pc, rc), a_last
        if self._zi is None:
            self._zi = sosfilt_zi(self._sos) * 0.0
        band, self._zi = sosfilt(self._sos, a_fwd, zi=self._zi)
        v = self._kalman(t, a_fwd)
        self.hist.extend(t=t, a_fwd=a_fwd, band=band, yaw_rate=w[:, 2], roll=roll,
                         pitch=pitch, v=v, sat=sat)
        self.t_proc = float(t[-1])
        self._peaks(t, band)
        self._runs(t)

    def _gnss_accel(self, t) -> np.ndarray:
        """Fore-aft acceleration from GNSS speed (difference over ~1 s, so it is
        about half a second late — fine for the slow attitude reference)."""
        gt = self.fixes["t"]
        if len(gt) < 6:
            return np.zeros(len(t))
        gv = self.fixes["v"]
        j = np.clip(np.searchsorted(gt, t, "right") - 1, 5, len(gt) - 1)
        span = gt[j] - gt[j - 5]
        a = (gv[j] - gv[j - 5]) / np.where(span > 0, span, np.inf)
        a[(span > 2.5) | (t - gt[j] > 1.0)] = 0.0     # across a drop-out: no estimate
        return a

    def _kalman(self, t, a) -> np.ndarray:
        c, dt = self.cfg, 1.0 / self.fs
        out = np.empty(len(t))
        if self._kf is None:
            if not self._gnss:
                out[:] = np.nan
                return out
            g0 = self._gnss[0]
            self._kf = [float(g0["sog"]), 0.0, np.diag([0.25, 0.01])]
        v, b, P = self._kf
        q_v, q_b = (c.kf_acc_sd * dt) ** 2, (c.kf_bias_sd * dt) ** 2
        r = c.gnss_speed_sd ** 2
        for i in range(len(t)):
            v += (a[i] - b) * dt
            p00 = P[0, 0] - 2 * dt * P[0, 1] + dt * dt * P[1, 1] + q_v
            p01 = P[0, 1] - dt * P[1, 1]
            p11 = P[1, 1] + q_b
            P = np.array([[p00, p01], [p01, p11]])
            while self._gnss and float(self._gnss[0]["t"]) <= t[i]:
                g = self._gnss.popleft()
                z = float(g["sog"])
                s = P[0, 0] + r
                k0, k1 = P[0, 0] / s, P[0, 1] / s
                innov = z - v
                v += k0 * innov
                b += k1 * innov
                P = np.array([[(1 - k0) * P[0, 0], (1 - k0) * P[0, 1]],
                              [P[0, 1] - k1 * P[0, 0], P[1, 1] - k1 * P[0, 1]]])
                self.last_fix_t = float(g["t"])
            out[i] = v
        self._kf = [v, b, P]
        return out

    # ── strokes ──────────────────────────────────────────────────────────
    def _peaks(self, t, y) -> None:
        c = self.cfg
        tail = self._y_tail + list(zip(t.tolist(), y.tolist()))
        for j in range(1, len(tail) - 1):
            tj, yj = tail[j]
            if not (tail[j - 1][1] < yj >= tail[j + 1][1]):
                continue
            ref = float(np.median(self._peak_heights)) if len(self._peak_heights) >= 3 else None
            thr = max(c.min_peak, c.peak_frac * ref) if ref else c.min_peak * 1.5
            if yj < thr:
                continue
            refr = min(1.2, max(0.35, c.refractory_frac * self.period))
            pp = self._pending_peak
            if pp is None:
                self._pending_peak = (tj, yj)
            elif tj - pp[0] < refr:
                if yj > pp[1]:
                    self._pending_peak = (tj, yj)
            else:
                self._confirm(pp)
                self._pending_peak = (tj, yj)
        self._y_tail = tail[-2:]
        pp = self._pending_peak
        if pp is not None and self.t_proc - pp[0] > min(1.2, max(0.35, c.refractory_frac * self.period)):
            self._confirm(pp)
            self._pending_peak = None
        # finalise a last stroke that was followed by a stop
        if self.strokes and not self.strokes[-1].get("final") and \
                self.t_proc - self.strokes[-1]["t"] > 2.5 * self.period:
            self._finalise(len(self.strokes) - 1, self.strokes[-1]["t"] + self.period)

    def _confirm(self, peak: Tuple[float, float]) -> None:
        tp, h = peak
        if self._last_peak_t > -math.inf:
            dtp = tp - self._last_peak_t
            if dtp < 2.5 * self.period or len(self._intervals) < 3:
                if 0.35 < dtp < 3.0:
                    self._intervals.append(dtp)
        prev_gap = tp - self._last_peak_t
        self._last_peak_t = tp
        self._peak_heights.append(h)
        st = {"t": tp, "height": h, "index": len(self.strokes),
              "rate_spm": (60.0 / prev_gap) if 0.35 < prev_gap < 2.5 * self.period else None}
        if self.strokes and not self.strokes[-1].get("final"):
            self._finalise(len(self.strokes) - 1, tp)
        self.strokes.append(st)

    def _finalise(self, k: int, t_end: float) -> None:
        s = self.strokes[k]
        tp = s["t"]
        T = max(0.4, t_end - tp)
        H = self.hist
        sl = H.window(tp, t_end)
        v = H["v"][sl]
        tt = H["t"][sl]
        if len(v) > 3 and np.isfinite(v).all():
            vm = float(np.mean(v))
            a = np.convolve(H["a_fwd"][sl], np.ones(5) / 5, mode="same")
            s.update(v_mean=vm, v_min=float(v.min()), v_max=float(v.max()),
                     check_mps=float(v.max() - v.min()),
                     check_ratio=float((v.max() - v.min()) / max(0.1, vm)),
                     peak_decel=float(-a.min()),
                     dps=float(np.trapezoid(v, tt)) + vm * max(0.0, T - (tt[-1] - tt[0])),
                     profile=np.interp(np.linspace(tp, t_end, 21), tt, v).round(4).tolist())
        s["cycle_s"] = T
        # direction of travel over the ground (GNSS course), averaged over the cycle
        fw = self.fixes.window(tp - 0.5, t_end + 0.5)
        if fw.stop > fw.start:
            ve, vn = float(np.mean(self.fixes["ve"][fw])), float(np.mean(self.fixes["vn"][fw]))
            if math.hypot(ve, vn) > 0.5:
                s["heading_deg"] = math.degrees(math.atan2(ve, vn)) % 360.0
        # side from the yaw pulse during the drive, relative to the cycle mean
        dw = H.window(tp - 0.35 * T, tp + 0.10 * T)
        cw = H.window(tp - 0.5 * T, tp + 0.5 * T)
        yr = H["yaw_rate"]
        if dw.stop - dw.start > 3 and cw.stop - cw.start > 5:
            feat = float(np.mean(yr[dw]) - np.mean(yr[cw]))
            ro = H["roll"]
            s["yaw_feat"] = feat
            s["roll_feat"] = float(np.mean(ro[dw]) - np.mean(ro[cw]))
            s["side_raw"] = -1 if feat > 0 else 1
            self._side_update(k)
        s["final"] = True

    def _side_update(self, k: int) -> None:
        r = self.strokes[k]["side_raw"]
        if self._side == 0:
            self._side = r
        if r == self._side:
            for j in self._side_pending:
                self.strokes[j]["side"] = self._side
            self._side_pending = []
            self.strokes[k]["side"] = self._side
            return
        self._side_pending.append(k)
        self.strokes[k]["side"] = self._side
        if len(self._side_pending) >= self.cfg.side_confirm:
            self._side = r
            for j in self._side_pending:
                self.strokes[j]["side"] = r
            self._side_pending = []

    # ── downwind runs ────────────────────────────────────────────────────
    def _runs(self, t) -> None:
        """Downwind runs: speed well above what the strokes alone hold, with the
        boat accelerating or nose-down.  "What the strokes hold" is a low
        percentile of the recent speed: on a downwind leg the boat surfs much of
        the time, so the speed on the backs of the waves is the honest baseline."""
        c = self.cfg
        H = self.hist
        if H.n < self.fs * 5:
            return
        tn = float(t[-1])
        if tn - self._runs_eval_t < 0.5:          # evaluated twice a second
            return
        self._runs_eval_t = tn
        if tn - self.last_fix_t > 2.0:            # no run calls on a stale speed
            self._close_run(tn)
            return
        base_sl = H.window(tn - c.base_window_s, tn)
        vb = H["v"][base_sl][::10]
        vb = vb[np.isfinite(vb)]
        if len(vb) < 10 * self.fs / 10:           # under ~10 s of usable speed
            return
        v_base = float(np.percentile(vb, c.base_pct))
        sl = H.window(tn - 1.0, tn)
        v1 = float(np.nanmean(H["v"][sl]))
        a1 = float(np.mean(H["a_fwd"][sl]))
        pitch_dev = math.degrees(float(np.mean(H["pitch"][sl])
                                       - np.median(H["pitch"][base_sl][::10])))
        if self._run_on is None:
            # a surf run is the boat nose-down on a wave face AND going clearly
            # faster than the strokes hold; a speed gain alone is often just a
            # harder piece starting
            if v1 > v_base + c.run_gain and pitch_dev > c.run_pitch_deg and a1 > c.run_acc:
                self._run_on = tn - 1.0
        elif v1 < v_base + c.run_end_gain:
            if self._run_off_since is None:
                self._run_off_since = tn
            elif tn - self._run_off_since > 0.5:
                self._close_run(self._run_off_since)
        else:
            self._run_off_since = None

    def _close_run(self, t_end: float) -> None:
        if self._run_on is not None and t_end - self._run_on >= self.cfg.run_min_s:
            self.runs.append((self._run_on, t_end))
        self._run_on = self._run_off_since = None

    # ── outputs ──────────────────────────────────────────────────────────
    def finish(self) -> None:
        if self.mount.ready:
            self._process_until(math.inf)
        if self._pending_peak is not None:
            self._confirm(self._pending_peak)
            self._pending_peak = None
        if self.strokes and not self.strokes[-1].get("final"):
            self._finalise(len(self.strokes) - 1, self.strokes[-1]["t"] + self.period)
        if self._run_on is not None and self.t_proc - self._run_on >= self.cfg.run_min_s:
            self.runs.append((self._run_on, self.t_proc))
            self._run_on = None

    def quality(self, t: float) -> Dict[str, Any]:
        H = self.hist
        sl = H.window(t - 1.0, t)
        return {"calibrated": self.mount.ready,
                "gnss_age_s": round(t - self.last_fix_t, 2) if self.last_fix_t > -math.inf else None,
                "imu_saturated": bool(H["sat"][sl].any()) if H.n else False}
