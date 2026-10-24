"""Sensor models for the two devices, producing the records the pipeline ingests.

**Head unit** (boat-mounted box on the iako, the crossbeam to the float):

* its own clock (offset + drift, ppm) — GNSS records carry both the head-unit
  time of the fix and its UTC, so the pipeline can map one to the other;
* GNSS at 5 or 10 Hz: position with a slowly wandering error, Doppler velocity
  with white noise, speed and course over ground, and drop-outs (bad sky view)
  reported as fixes without a position;
* a 3-axis accelerometer and gyroscope at 100 Hz in the **box's own axes**, at an
  unknown rotation to the hull and offset from the paddler (lever arm), with
  biases, noise, beam-flex vibration excited by each stroke, wave-slap impacts
  and range saturation;
* housekeeping (a flat battery voltage, temperature) and button events.

**Paddle pod** (on the shaft):

* its own clock with a larger drift; it sleeps when the paddle is still and
  wakes on motion with a short delay, so data starts late;
* a single strain-gauge bridge read as raw counts: unknown zero and gain,
  temperature drift, bending in one plane only;
* a 3-axis accelerometer and gyroscope in the pod's axes, at an unknown roll
  about the shaft;
* a Bluetooth LE link: one packet every 50 ms carrying the samples since the
  last packet, receive-time jitter, single-packet loss (worse while the blade is
  in the water) and longer outages with reconnects.

Every random draw comes from one seeded generator, and everything the pipeline
must estimate (clock offsets and drifts, mount rotations, strain gain) is kept in
``SensorTruth`` for the scorers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..geo import G, enu_to_latlon, euler_zyx, math_to_compass, random_rotation, rx, ry, rz
from .hull import Truth

__all__ = ["HeadUnitConfig", "PodConfig", "SensorTruth", "make_records",
           "strain_calibration_counts"]


@dataclass
class HeadUnitConfig:
    gnss_hz: float = 5.0
    imu_hz: float = 100.0
    imu_batch: int = 10                   # samples per IMU record
    acc_range_g: float = 8.0
    gyr_range_dps: float = 500.0
    acc_noise: float = 0.02               # m/s^2 per sample
    gyr_noise: float = 0.004              # rad/s per sample
    acc_bias_sd: float = 0.05
    gyr_bias_sd_dps: float = 0.3
    vibration: float = 0.25               # beam-flex ringing (m/s^2 scale)
    slap_per_min: float = 0.0             # wave-slap impacts (0 = flat water)
    lever_arm_m: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    mount: Optional[np.ndarray] = None    # box->hull rotation; None = random
    clock_drift_ppm: float = 20.0
    clock_offset_s: float = 4321.0
    gnss_pos_wander_m: float = 1.0
    gnss_pos_white_m: float = 0.3
    gnss_vel_noise: float = 0.04
    gnss_dropouts_per_h: float = 2.0
    origin_latlon: Tuple[float, float] = (36.9514, -122.0262)   # off Santa Cruz


@dataclass
class PodConfig:
    present: bool = True
    fs: float = 50.0
    packet_s: float = 0.05
    clock_drift_ppm: float = -35.0
    clock_offset_s: float = -987.654
    wake_latency_s: Tuple[float, float] = (0.3, 1.0)
    sleep_after_s: float = 45.0
    connect_s: Tuple[float, float] = (0.2, 0.6)
    loss_base: float = 0.01
    loss_in_water: float = 0.08
    outages_per_h: float = 4.0
    supervision_s: float = 1.0
    strain_gain: float = 1200.0           # counts per N*m of bending
    strain_offset: float = 2.1e6
    strain_temp_coef: float = 400.0       # counts per degree C
    strain_noise: float = 50.0
    strain_misalign_deg: float = 5.0
    gauge_lever_m: float = 0.35           # blade centre of pressure to the gauge
    pod_from_grip_m: float = 0.45
    acc_noise: float = 0.05
    gyr_noise: float = 0.01
    mount_roll_deg: Optional[float] = None    # about the shaft; None = random


@dataclass
class SensorTruth:
    head_clock: Tuple[float, float]       # (drift, offset): t_head = (1+d) t + o
    pod_clock: Tuple[float, float]
    mount: np.ndarray                     # box -> hull
    pod_mount: np.ndarray                 # pod -> paddle
    acc_bias: np.ndarray
    gyr_bias: np.ndarray
    gnss_dropouts: List[Tuple[float, float]]
    ble_outages: List[Tuple[float, float]]
    pod_awake: List[Tuple[float, float]]
    slaps: List[float]
    strain: Dict[str, float]
    head_cfg: HeadUnitConfig
    pod_cfg: PodConfig
    extra: Dict[str, Any] = field(default_factory=dict)

    def head_time(self, t):
        d, o = self.head_clock
        return (1.0 + d) * np.asarray(t, float) + o

    def pod_time(self, t):
        d, o = self.pod_clock
        return (1.0 + d) * np.asarray(t, float) + o


def _poisson_intervals(rng, duration, per_h, lo, hi):
    out, t = [], 0.0
    if per_h <= 0:
        return out
    while True:
        t += rng.exponential(3600.0 / per_h)
        if t >= duration:
            return out
        out.append((t, t + rng.uniform(lo, hi)))


def _in_any(t, intervals):
    m = np.zeros(np.shape(t), bool)
    for a, b in intervals:
        m |= (t >= a) & (t <= b)
    return m


def _vee(S):
    return np.stack([S[..., 2, 1] - S[..., 1, 2], S[..., 0, 2] - S[..., 2, 0],
                     S[..., 1, 0] - S[..., 0, 1]], -1) * 0.5


def _hull_kinematics(tr: Truth):
    A, dt = tr.arrays, tr.dt
    yaw = np.unwrap(A["heading"])
    R = euler_zyx(yaw, A["pitch"], A["roll"])
    w = _vee(np.swapaxes(R, -1, -2) @ np.gradient(R, dt, axis=0))   # body rates
    a_ref = np.stack([np.gradient(A["vgx"], dt), np.gradient(A["vgy"], dt),
                      np.gradient(np.gradient(A["heave"], dt), dt)], -1)
    return R, w, a_ref


def _resonator(rng, n, dt, f0, q, drive):
    from scipy.signal import iirpeak, lfilter
    bb, aa = iirpeak(f0, q, fs=1.0 / dt)
    return lfilter(bb, aa, drive + 0.3 * rng.normal(size=n) * np.std(drive + 1e-12))


def make_records(tr: Truth, *, head: HeadUnitConfig = None, pod: PodConfig = None,
                 seed: int = 0) -> Tuple[List[Dict[str, Any]], SensorTruth]:
    """All records both devices would deliver, sorted by head-unit arrival time."""
    hc = head or HeadUnitConfig()
    pc = pod or PodConfig()
    rng = np.random.default_rng(seed + 1009)
    A, dt = tr.arrays, tr.dt
    t = A["t"]
    n = len(t)
    dur = float(t[-1]) if n else 0.0
    hd = (hc.clock_drift_ppm * 1e-6, hc.clock_offset_s)
    pd_ = (pc.clock_drift_ppm * 1e-6, pc.clock_offset_s)
    head_time = lambda tt: (1.0 + hd[0]) * np.asarray(tt, float) + hd[1]

    R, w, a_ref = _hull_kinematics(tr)
    mount = hc.mount if hc.mount is not None else random_rotation(rng)
    records: List[Tuple[float, Dict[str, Any]]] = []

    # ── head IMU ─────────────────────────────────────────────────────────
    lever = np.asarray(hc.lever_arm_m, float)
    wdot = np.gradient(w, dt, axis=0)
    lev_h = np.cross(wdot, lever) + np.cross(w, np.cross(w, lever))
    a_box = a_ref + np.einsum("nij,nj->ni", R, lev_h)
    f_h = np.einsum("nji,nj->ni", R, a_box + np.array([0.0, 0.0, G]))
    # beam flex, excited by changes in stroke force
    dF = np.abs(np.gradient(A["f_prop"], dt)) / 400.0
    flex = _resonator(rng, n, dt, 12.0, 6.0, dF) * hc.vibration
    f_h[:, 2] += flex
    f_h[:, 1] += 0.4 * flex
    slaps: List[float] = []
    if hc.slap_per_min > 0:
        k = rng.poisson(hc.slap_per_min * dur / 60.0)
        slaps = sorted(rng.uniform(0, dur, k).tolist())
        ring = np.arange(int(0.12 / dt)) * dt
        for ts in slaps:
            i0 = int(ts / dt)
            amp = rng.lognormal(math.log(15.0), 0.6)
            sh = amp * np.exp(-ring / 0.03) * np.cos(2 * np.pi * 20.0 * ring)
            j = min(n, i0 + len(sh))
            f_h[i0:j, 2] += sh[: j - i0]
            f_h[i0:j, 0] += 0.3 * sh[: j - i0]
    f_box = f_h @ mount            # = mount^T f_h per row
    w_box = w @ mount
    acc_bias = rng.normal(0, hc.acc_bias_sd, 3)
    gyr_bias = np.radians(rng.normal(0, hc.gyr_bias_sd_dps, 3))
    step = max(1, int(round(1.0 / (hc.imu_hz * dt))))
    m = (n // step) * step
    f_s = f_box[:m].reshape(-1, step, 3).mean(1) + acc_bias \
        + rng.normal(0, hc.acc_noise, (m // step, 3))
    w_s = w_box[:m].reshape(-1, step, 3).mean(1) + gyr_bias \
        + rng.normal(0, hc.gyr_noise, (m // step, 3))
    t_s = t[:m].reshape(-1, step)[:, -1]
    ar, gr = hc.acc_range_g * G, math.radians(hc.gyr_range_dps)
    f_s = np.clip(f_s, -ar, ar)
    w_s = np.clip(w_s, -gr, gr)
    th = head_time(t_s)
    for i0 in range(0, len(t_s), hc.imu_batch):
        sl = slice(i0, i0 + hc.imu_batch)
        records.append((float(th[sl][-1]) + 0.002, {
            "dev": "head", "stream": "imu", "t": th[sl].round(6),
            "acc": f_s[sl].astype(np.float32), "gyr": w_s[sl].astype(np.float32)}))
    records.append((float(head_time(0.0)), {
        "dev": "head", "stream": "meta", "t": float(head_time(0.0)),
        "imu_hz": hc.imu_hz, "gnss_hz": hc.gnss_hz, "acc_range": ar, "gyr_range": gr,
        "fw": "sim-0"}))

    # ── GNSS ─────────────────────────────────────────────────────────────
    drop = _poisson_intervals(rng, dur, hc.gnss_dropouts_per_h, 2.0, 8.0)
    utc0 = tr.utc0
    period = 1.0 / hc.gnss_hz
    first = math.ceil(utc0 / period) * period
    utc = np.arange(first, utc0 + dur, period)
    te = utc - utc0
    k_alpha = math.exp(-period / 60.0)
    wander = np.zeros((len(te), 2))
    e = rng.normal(0, hc.gnss_pos_wander_m, 2)
    for i in range(len(te)):
        e = k_alpha * e + math.sqrt(1 - k_alpha ** 2) * rng.normal(0, hc.gnss_pos_wander_m, 2)
        wander[i] = e
    # receiver velocity is lightly smoothed (tau ~ 0.1 s)
    kk = max(1, int(round(0.1 / dt)))
    sm = np.ones(kk) / kk
    vgx = np.convolve(A["vgx"], sm, mode="same")
    vgy = np.convolve(A["vgy"], sm, mode="same")
    px = np.interp(te, t, A["x"]) + wander[:, 0] + rng.normal(0, hc.gnss_pos_white_m, len(te))
    py = np.interp(te, t, A["y"]) + wander[:, 1] + rng.normal(0, hc.gnss_pos_white_m, len(te))
    vx = np.interp(te, t, vgx) + rng.normal(0, hc.gnss_vel_noise, len(te))
    vy = np.interp(te, t, vgy) + rng.normal(0, hc.gnss_vel_noise, len(te))
    lat, lon = enu_to_latlon(px, py, *hc.origin_latlon)
    sog = np.hypot(vx, vy)
    cog = math_to_compass(np.arctan2(vy, vx))
    lost = _in_any(te, drop)
    th_e = head_time(te) + rng.normal(0, 0.0005, len(te))
    for i in range(len(te)):
        rec = {"dev": "head", "stream": "gnss", "t": float(th_e[i]), "utc": float(utc[i])}
        if lost[i]:
            rec.update(fix=0, lat=None, lon=None, sog=None, cog=None, nsat=3, hdop=None)
        else:
            rec.update(fix=3, lat=float(lat[i]), lon=float(lon[i]), sog=float(sog[i]),
                       cog=float(cog[i]), vel_e=float(vx[i]), vel_n=float(vy[i]),
                       nsat=18 if hc.gnss_hz <= 5 else 11, hdop=0.8)
        records.append((float(th_e[i]) + 0.03, rec))

    # ── housekeeping + a start button press ─────────────────────────────
    for tk in np.arange(0.0, dur, 5.0):
        v = 3.32 - 0.03 * tk / 7200.0 + rng.normal(0, 0.003)
        records.append((float(head_time(tk)), {
            "dev": "head", "stream": "house", "t": float(head_time(tk)), "vbat": round(v, 4),
            "charging": False, "temp_c": 31.0 + 2.0 * tk / 3600.0, "state": "awake"}))
    records.append((float(head_time(1.0)), {"dev": "head", "stream": "button",
                                            "t": float(head_time(1.0)), "button": 1,
                                            "action": "press"}))

    # ── paddle pod ───────────────────────────────────────────────────────
    pod_mount = rz(math.radians(pc.mount_roll_deg if pc.mount_roll_deg is not None
                                else rng.uniform(0, 360)))
    awake: List[Tuple[float, float]] = []
    outages: List[Tuple[float, float]] = []
    strain_info = {"gain": pc.strain_gain, "offset": pc.strain_offset,
                   "temp_coef": pc.strain_temp_coef, "lever_m": pc.gauge_lever_m,
                   "misalign_deg": pc.strain_misalign_deg}
    if pc.present:
        records += _pod_records(tr, pc, rng, R, w, a_ref, pod_mount, head_time, pd_,
                                awake, outages, strain_info)

    records.sort(key=lambda r: r[0])
    out = []
    for arrival, rec in records:
        rec["arrival"] = round(arrival, 6)
        out.append(rec)
    st = SensorTruth(head_clock=hd, pod_clock=pd_, mount=mount, pod_mount=pod_mount,
                     acc_bias=acc_bias, gyr_bias=gyr_bias, gnss_dropouts=drop,
                     ble_outages=outages, pod_awake=awake, slaps=slaps, strain=strain_info,
                     head_cfg=hc, pod_cfg=pc)
    return out, st


def _pod_records(tr, pc, rng, R, w, a_ref, pod_mount, head_time, pd_, awake, outages,
                 strain_info):
    A, dt = tr.arrays, tr.dt
    t = A["t"]
    n = len(t)
    dur = float(t[-1])
    # paddle orientation in the hull: tilt about the hull's fore-aft axis after the
    # sweep about its left axis (drive = positive rotation about +y)
    Rph = rx(A["tilt"]) @ ry(A["sweep"])
    Rpw = R @ Rph
    w_p = _vee(np.swapaxes(Rpw, -1, -2) @ np.gradient(Rpw, dt, axis=0))
    grip = np.array([0.0, 0.0, 0.8])
    r_h = grip + Rph @ np.array([0.0, 0.0, -pc.pod_from_grip_m])
    r_w = np.einsum("nij,nj->ni", R, r_h)
    a_w = a_ref + np.gradient(np.gradient(r_w, dt, axis=0), dt, axis=0)
    f_p = np.einsum("nji,nj->ni", Rpw, a_w + np.array([0.0, 0.0, G]))
    f_pod = f_p @ pod_mount
    w_pod = w_p @ pod_mount

    # motion -> awake intervals
    motion = (np.linalg.norm(w_p - np.einsum("nji,nj->ni", Rph, w), axis=1) > 0.15)
    awake_flag = np.zeros(n, bool)
    asleep, last_motion, i = True, -1e9, 0
    wake_at = None
    for i in range(n):
        if motion[i]:
            last_motion = t[i]
            if asleep and wake_at is None:
                wake_at = t[i] + rng.uniform(*pc.wake_latency_s)
        if asleep and wake_at is not None and t[i] >= wake_at:
            asleep, wake_at = False, None
            awake.append((t[i], t[i]))
        if not asleep:
            if t[i] - last_motion > pc.sleep_after_s:
                asleep = True
            else:
                awake[-1] = (awake[-1][0], t[i])
                awake_flag[i] = True

    # strain counts
    temp = 30.0 + 3.0 * t / max(1.0, dur) + np.cumsum(rng.normal(0, 0.02, n)) * math.sqrt(dt)
    sweep_acc = np.gradient(np.gradient(A["sweep"], dt), dt)
    moment = (A["blade_force"] * pc.gauge_lever_m * math.cos(math.radians(pc.strain_misalign_deg))
              + 0.03 * sweep_acc)
    counts = (pc.strain_offset + pc.strain_temp_coef * (temp - 30.0) + pc.strain_gain * moment)

    # the link: occasional outages with a reconnect
    outages += _poisson_intervals(rng, dur, pc.outages_per_h, 1.0, 4.0)
    outages.sort()
    t_pod_of = lambda tt: (1.0 + pd_[0]) * tt + pd_[1]
    t_true_of = lambda tp: (tp - pd_[1]) / (1.0 + pd_[0])
    recs = []
    seq = 0
    for (a, b) in awake:
        # connection comes up a little after waking; samples start at wake
        tp0 = math.ceil(t_pod_of(a) * pc.fs) / pc.fs
        tp_end = t_pod_of(b)
        sp = np.arange(tp0, tp_end, 1.0 / pc.fs)
        if len(sp) == 0:
            continue
        tt = t_true_of(sp)
        sc = np.interp(tt, t, counts) + rng.normal(0, pc.strain_noise, len(tt))
        fa = np.stack([np.interp(tt, t, f_pod[:, k]) for k in range(3)], -1) \
            + rng.normal(0, pc.acc_noise, (len(tt), 3))
        wa = np.stack([np.interp(tt, t, w_pod[:, k]) for k in range(3)], -1) \
            + rng.normal(0, pc.gyr_noise, (len(tt), 3))
        drive = np.interp(tt, t, A["drive"]) > 0.5
        link_up = a + rng.uniform(*pc.connect_s)
        recs.append((float(head_time(link_up)), {"dev": "head", "stream": "link",
                                                 "t": float(head_time(link_up)), "state": "up"}))
        recs.append((float(head_time(link_up)) + 0.001, {
            "dev": "pod", "stream": "meta", "rx": float(head_time(link_up)), "fs": pc.fs,
            "fw": "pod-sim-0"}))
        spp = pc.packet_s * pc.fs                     # samples per packet (may be fractional)
        k, acc_s = 0, 0.0
        down_until = -1.0
        while k < len(sp):
            acc_s += spp
            m = int(acc_s)
            acc_s -= m
            if m <= 0:
                continue
            sl = slice(k, min(len(sp), k + m))
            k = sl.stop
            t_send = tt[sl.stop - 1] + 0.001
            seq = (seq + 1) % 65536
            if t_send < link_up:
                continue                               # buffered before the link: dropped
            in_out = next(((oa, ob) for oa, ob in outages if oa <= t_send <= ob), None)
            if in_out is not None:
                if in_out[1] - in_out[0] > pc.supervision_s and down_until < in_out[1]:
                    t_dn = in_out[0] + pc.supervision_s
                    recs.append((float(head_time(t_dn)), {"dev": "head", "stream": "link",
                                                          "t": float(head_time(t_dn)),
                                                          "state": "down"}))
                    down_until = in_out[1] + rng.uniform(*pc.connect_s)
                    recs.append((float(head_time(down_until)), {
                        "dev": "head", "stream": "link", "t": float(head_time(down_until)),
                        "state": "up"}))
                continue
            if t_send < down_until:
                continue
            p_loss = pc.loss_base + (pc.loss_in_water if drive[sl].any() else 0.0)
            if rng.random() < p_loss:
                continue
            lat = 0.002 + rng.uniform(0, 0.0075)
            while rng.random() < 0.1 and lat < 0.03:
                lat += 0.0075
            rx_t = float(head_time(t_send + lat))
            recs.append((rx_t, {"dev": "pod", "stream": "pkt", "rx": round(rx_t, 6), "seq": seq,
                                "t0": round(float(sp[sl.start]), 6), "fs": pc.fs,
                                "strain": np.round(sc[sl]).astype(np.int64),
                                "acc": fa[sl].astype(np.float32),
                                "gyr": wa[sl].astype(np.float32)}))
    return recs


def strain_calibration_counts(st: SensorTruth, loads_n, *, lever_m: float = 0.35,
                              seed: int = 0) -> np.ndarray:
    """Readings with known weights hung at ``lever_m`` from the gauge (the shaft
    clamped horizontally, power face down) — the per-paddle calibration step."""
    rng = np.random.default_rng(seed)
    s = st.strain
    loads = np.asarray(loads_n, float)
    return (s["offset"] + s["gain"] * loads * lever_m * math.cos(math.radians(s["misalign_deg"]))
            + rng.normal(0, st.pod_cfg.strain_noise, loads.shape))
