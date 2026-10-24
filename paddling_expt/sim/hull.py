"""Integrate paddler + hull + water into a ground-truth state history.

One explicit time step (default 200 Hz) moves a single surge degree of freedom
(speed through the water along the hull) under the stroke force, quadratic drag
and the swell's gravity push down a wave face; heading follows the paddler's
steering towards the next waypoint plus a yaw pulse from every one-sided stroke;
roll, pitch and heave follow the strokes and the wave surface.  Ground velocity is
speed through the water along the heading plus the current plus the swell's
surface orbital velocity.

The output :class:`Truth` is what the scorers compare the pipeline with.  The
sensor models in :mod:`.sensors` read it to make the records the pipeline sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..geo import G, wrap_pi
from .environment import Course, CurrentField, Swell
from .paddler import (BoatParams, PaddlerParams, Plan, best_rate, drive_duty, efficiency,
                      profile_area, profile_exponent)

__all__ = ["Truth", "simulate_truth"]

_CATCH_SWEEP = math.radians(-25.0)   # paddle sweep angle at the catch (blade forward)
_EXIT_SWEEP = math.radians(35.0)     # at the exit (blade behind the paddler)
_REST_SWEEP = math.radians(70.0)     # paddle held across the boat while resting
_DRIVE_TILT = math.radians(10.0)     # shaft tilt with the blade outboard
_RECOVERY_TILT = math.radians(15.0)  # extra tilt as the blade swings forward


@dataclass
class Truth:
    dt: float
    utc0: float
    arrays: Dict[str, np.ndarray]
    strokes: Dict[str, np.ndarray]
    runs: List[Tuple[float, float]]
    segments: List[Tuple[float, float, str, float]]
    params: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.arrays[k]

    @property
    def t(self) -> np.ndarray:
        return self.arrays["t"]

    def at(self, name: str, t) -> np.ndarray:
        """Linear interpolation of a truth array at times ``t``."""
        return np.interp(t, self.arrays["t"], self.arrays[name])


def _detect_runs(t, u, u_paddle, f_wave, dt, min_s=1.5, gain=0.5):
    """Truth surf runs: speed through the water well above what the strokes alone
    would hold, while the wave pushes forward, for at least ``min_s``."""
    k = max(1, int(round(0.5 / dt)))
    fw = np.convolve(f_wave, np.ones(k) / k, mode="same")
    on = (u - u_paddle > gain) & (fw > 0)
    runs, i, n = [], 0, len(t)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if t[j - 1] - t[i] >= min_s:
                runs.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return runs


def simulate_truth(plan: Plan, *, paddler: PaddlerParams = None, boat: BoatParams = None,
                   current: CurrentField = None, swell: Swell = None,
                   course: Course = None, start_xy=(0.0, 0.0), start_heading_deg=0.0,
                   utc0: float = 1_792_857_600.0,
                   dt: float = 0.005, seed: int = 0) -> Truth:
    p = paddler or PaddlerParams()
    b = boat or BoatParams()
    cur = current or CurrentField()
    sw = swell or Swell()
    rng = np.random.default_rng(seed)
    n = int(round(plan.duration_s / dt))
    m_tot = p.body_kg + b.hull_kg
    m_eff = m_tot * (1.0 + b.added_mass_frac)

    names = ["t", "x", "y", "u", "vgx", "vgy", "cur_x", "cur_y", "orb_x", "orb_y",
             "heading", "pitch", "roll", "heave", "f_prop", "f_body", "f_wave", "f_drag",
             "blade_force", "fatigue", "w_bal", "eta", "rate_opt", "p_shaft",
             "u_paddle", "stroke", "phase", "drive", "sweep", "tilt",
             "resting", "turning"]
    A = {k: np.zeros(n) for k in names}

    # waves as plain lists (a scalar path is much faster than numpy on 2-4 trains)
    tr = [(w.amplitude_m, 2 * math.pi / w.period_s,
           (2 * math.pi / w.period_s) ** 2 / G,
           math.cos(math.radians(90 - w.towards_deg)),
           math.sin(math.radians(90 - w.towards_deg)), w.phase) for w in sw.trains]

    x, y = float(start_xy[0]), float(start_xy[1])
    psi = math.radians(90.0 - start_heading_deg)
    u = 0.0
    wp = list(course.waypoints) if course else []
    wp_i = 0
    w_bal, work = p.w_prime_j, 0.0
    wander = 0.0
    # stroke state
    in_stroke, t0s, T, Td, Fpk, side, gam = False, 0.0, 1.0, 0.5, 0.0, 1, 1.0
    p_shaft_stroke, v_paddle = 0.0, 0.0
    on_side, switch_at = 0, int(rng.integers(p.switch_every[0], p.switch_every[1] + 1))
    last_of_side, first_after_switch = False, False
    st = {k: [] for k in ["catch_t", "exit_t", "end_t", "side", "impulse", "peak",
                          "rate_spm", "duty", "fatigue", "eta", "rate_opt", "p_shaft",
                          "v_paddle", "switch", "turning", "peak_pos"]}
    k_stroke = -1
    cx = cy = 0.0
    sweep_prev = _REST_SWEEP
    tilt_prev = 0.0

    for i in range(n):
        t = i * dt
        seg = plan.at(t)
        fatigue = min(1.0, max(0.0, 0.7 * (1.0 - w_bal / p.w_prime_j)
                               + work / p.work_capacity_j))
        cp_eff = p.critical_power_w * (1.0 - p.slow_cp_loss * min(1.0, work / p.work_capacity_j))

        # steering
        turning = False
        if wp and wp_i < len(wp):
            dx, dy = wp[wp_i][0] - x, wp[wp_i][1] - y
            if math.hypot(dx, dy) < course.arrive_m and wp_i < len(wp) - 1:
                wp_i += 1
                dx, dy = wp[wp_i][0] - x, wp[wp_i][1] - y
            err = float(wrap_pi(math.atan2(dy, dx) - psi))
        else:
            err = 0.0
        turning = abs(err) > math.radians(35.0)

        # start a stroke?
        if in_stroke and t - t0s >= T:
            in_stroke = False
        if (not in_stroke and seg is not None and seg.rate_spm > 0):
            wander += (-wander * (T / 20.0)
                       + p.rate_wander_spm * math.sqrt(2.0 * T / 20.0) * rng.normal())
            rate_cmd = seg.rate_spm + wander
            T = 60.0 / rate_cmd * (1.0 + p.rate_jitter * rng.normal())
            rate = 60.0 / T
            if last_of_side:
                side, on_side, first_after_switch = -side, 0, True
                switch_at = int(rng.integers(p.switch_every[0], p.switch_every[1] + 1))
            else:
                first_after_switch = False
            on_side += 1
            last_of_side = on_side >= switch_at
            P = seg.effort * cp_eff
            if w_bal <= 0.02 * p.w_prime_j:
                P = min(P, cp_eff)
            eta = efficiency(p, rate, fatigue) * (0.6 if turning else 1.0)
            v_paddle = (eta * P / b.drag_k) ** (1.0 / 3.0)
            fbar = b.drag_k * v_paddle * v_paddle
            J = fbar * T
            J *= 1.0 + (0.5 if side > 0 else -0.5) * p.side_asymmetry
            if first_after_switch:
                J *= p.switch_stroke_factor
            J *= max(0.5, 1.0 + (p.consistency_sd + p.consistency_fatigue_sd * fatigue)
                     * rng.normal())
            Td = drive_duty(p, rate, fatigue) * T
            gam = profile_exponent(p, fatigue)
            Fpk = J / (Td * profile_area(gam))
            p_shaft_stroke = P
            t0s, in_stroke = t, True
            k_stroke += 1
            for key, val in (("catch_t", t), ("exit_t", t + Td), ("end_t", t + T),
                             ("side", side), ("impulse", J), ("peak", Fpk),
                             ("rate_spm", rate), ("duty", Td / T), ("fatigue", fatigue),
                             ("eta", eta), ("rate_opt", best_rate(p, fatigue)),
                             ("p_shaft", P), ("v_paddle", v_paddle),
                             ("switch", first_after_switch), ("turning", turning),
                             ("peak_pos", 0.5 ** (1.0 / gam))):
                st[key].append(val)

        # stroke force and paddle kinematics
        F = F_body = 0.0
        drive = False
        phase = 0.0
        if in_stroke:
            tau = t - t0s
            phase = tau / T
            if tau < Td:
                drive = True
                F = Fpk * math.sin(math.pi * (tau / Td) ** gam)
                s = 0.5 * (1.0 - math.cos(math.pi * tau / Td))
                sweep = _CATCH_SWEEP + (_EXIT_SWEEP - _CATCH_SWEEP) * s
                tilt = side * _DRIVE_TILT
            else:
                rho = (tau - Td) / max(1e-6, T - Td)
                F_body = -b.body_surge_n * math.sin(2.0 * math.pi * rho)
                s = 0.5 * (1.0 - math.cos(math.pi * rho))
                sweep = _EXIT_SWEEP + (_CATCH_SWEEP - _EXIT_SWEEP) * s
                if last_of_side:
                    tilt = side * _DRIVE_TILT * math.cos(math.pi * rho)
                else:
                    tilt = side * (_DRIVE_TILT + _RECOVERY_TILT * math.sin(math.pi * rho))
            p_now = p_shaft_stroke
        else:
            # resting: the paddle settles across the boat
            sweep = sweep_prev + (_REST_SWEEP - sweep_prev) * min(1.0, dt / 0.8)
            tilt = tilt_prev * max(0.0, 1.0 - dt / 0.8)
            p_now = 0.0
        sweep_prev, tilt_prev = sweep, tilt

        # fatigue bookkeeping
        if p_now > cp_eff:
            w_bal -= (p_now - cp_eff) * dt
        else:
            tau_r = 546.0 * math.exp(-0.01 * (cp_eff - p_now)) + 316.0
            w_bal += (p.w_prime_j - w_bal) / tau_r * dt
        w_bal = max(0.0, w_bal)
        work += p_now * dt

        # water (the current varies slowly: evaluate it at 20 Hz)
        if i % 10 == 0:
            if cur.peak_mps or tuple(cur.drift) != (0.0, 0.0):
                cxa, cya = cur.velocity(x, y, t + utc0)
                cx, cy = float(cxa), float(cya)
            else:
                cx = cy = 0.0
        eta_w = gx = gy = ox = oy = 0.0
        for (a, om, kk, ddx, ddy, ph) in tr:
            th = kk * (ddx * x + ddy * y) - om * t + ph
            c, s_ = math.cos(th), math.sin(th)
            eta_w += a * c
            gx += -a * kk * s_ * ddx
            gy += -a * kk * s_ * ddy
            ox += a * om * c * ddx
            oy += a * om * c * ddy
        ch, sh = math.cos(psi), math.sin(psi)
        slope_along = gx * ch + gy * sh
        slope_across = -gx * sh + gy * ch
        f_wave = -0.9 * m_tot * G * slope_along

        f_drag = b.drag_k * u * abs(u)
        u += (F + F_body + f_wave - f_drag) / m_eff * dt

        # heading
        yaw_pulse = (-side * math.radians(b.yaw_pulse_dps) * (F / Fpk)) if (drive and Fpk > 0) else 0.0
        turn = max(-math.radians(b.max_turn_dps),
                   min(math.radians(b.max_turn_dps), b.steer_gain * err))
        if u < 0.3:
            turn *= max(0.0, u / 0.3)
        psi += (turn + yaw_pulse) * dt

        vgx = u * ch + cx + ox
        vgy = u * sh + cy + oy
        x += vgx * dt
        y += vgy * dt

        roll = (side * math.radians(b.roll_pulse_deg) * (F / Fpk) if (drive and Fpk > 0) else 0.0)
        roll += -0.35 * math.atan(slope_across)
        pitch = 0.9 * math.atan(-slope_along) + (math.radians(0.3) * F / Fpk if Fpk > 0 else 0.0)

        row = A
        row["t"][i] = t
        row["x"][i], row["y"][i], row["u"][i] = x, y, u
        row["vgx"][i], row["vgy"][i] = vgx, vgy
        row["cur_x"][i], row["cur_y"][i] = cx, cy
        row["orb_x"][i], row["orb_y"][i] = ox, oy
        row["heading"][i], row["pitch"][i], row["roll"][i] = psi, pitch, roll
        row["heave"][i] = eta_w
        row["f_prop"][i], row["f_body"][i] = F, F_body
        row["f_wave"][i], row["f_drag"][i] = f_wave, f_drag
        row["blade_force"][i] = F / max(0.5, math.cos(sweep)) if drive else 0.0
        row["fatigue"][i], row["w_bal"][i] = fatigue, w_bal
        row["eta"][i] = st["eta"][-1] if st["eta"] else 0.0
        row["rate_opt"][i] = best_rate(p, fatigue)
        row["p_shaft"][i] = p_now
        row["u_paddle"][i] = v_paddle if in_stroke else 0.0
        row["stroke"][i] = k_stroke if in_stroke else -1
        row["phase"][i] = phase
        row["drive"][i] = drive
        row["sweep"][i], row["tilt"][i] = sweep, tilt
        row["resting"][i] = (seg is not None and seg.rate_spm == 0) or seg is None
        row["turning"][i] = turning

    strokes = {k: np.asarray(v, dtype=bool if k in ("switch", "turning") else float)
               for k, v in st.items()}
    if len(strokes["side"]):
        strokes["side"] = strokes["side"].astype(int)
    runs = (_detect_runs(A["t"], A["u"], A["u_paddle"], A["f_wave"], dt)
            if sw.active else [])
    segs = [(a, c, s.label, s.rate_spm) for a, c, s in plan.boundaries()]
    return Truth(dt=dt, utc0=utc0, arrays=A, strokes=strokes, runs=runs,
                 segments=segs,
                 params={"paddler": p, "boat": b, "current": cur, "swell": sw,
                         "course": course, "m_tot": m_tot})
