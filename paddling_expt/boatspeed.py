"""What each loss costs in boat speed, from the hull model and the measured motion.

For every stroke (rows of :meth:`.fusion.OC1Engine.stroke_table`, or SpeedCoach
rows where the quantity exists) and a :class:`~.hydro.Craft` with the total mass:

* **Power to the boat** — resistance of this hull at this speed (and, for an OC1,
  at this ama load), plus air drag and side-slip drag, times speed.  Absolute
  watts, given the hull shape and mass; it is the power that reaches the boat,
  not what the paddler produces (blade losses come on top).
* **Ama load (OC1)** — from the stroke's mean roll against the level reference,
  through the outrigger's statics (span, rigged height, ama shape).  Its cost:
  the speed the same power would give with only ``ref_ama_kg`` on the ama.
* **Side-slip (yaw drag)** — the slender-body induced drag of the measured drift
  angle; its cost is the speed the same power would give with none.  The model
  says this is small unless the drift reaches several degrees; most of what a
  wobbly boat costs is the paddler's power (bracing, shortened strokes), so
* **Wobble cost, measured** (:func:`wobble_cost_from_data`) — how distance per
  stroke at a matched rate falls as roll wobble grows, within the session.

Speed is taken through the water when it is known (``water_speed`` from the pod,
or ``v_corrected`` from the current-and-wind fit); otherwise GNSS speed, and the
result says so.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hydro import (Craft, air_drag, ama_load_from_roll, resistance_table, roll_table,
                    speed_for_power, yaw_drag)

__all__ = ["estimate_ama_load", "stroke_physics", "physics_summary", "wobble_cost_from_data"]


def _speed_key(rows) -> str:
    for k in ("water_speed", "v_corrected"):
        if any(r.get(k) is not None for r in rows):
            return k
    return "v_mean"


def estimate_ama_load(rows: List[Dict[str, Any]], craft: Craft, total_kg: float, *,
                      zero_load_roll_deg: Optional[float] = None, window: int = 30) -> int:
    """Adds ``roll_static_deg`` and ``ama_load_kg`` to every OC1 row with a mean
    roll (see :func:`stroke_physics` for ``zero_load_roll_deg``).  Returns the count."""
    if craft.outrigger is None:
        return 0
    rt = roll_table(craft, round(total_kg, 3))
    shift = 0.0
    if zero_load_roll_deg is not None:
        shift = math.degrees(float(np.interp(0.0, rt["loads"], rt["roll"]))) - zero_load_roll_deg
    # The static roll is what holds over both paddling sides: each stroke leans the
    # boat a little towards its own side, so average the left-side and right-side
    # mean roll over a rolling window of ``window`` strokes.
    R = [r for r in rows if r.get("roll_mean_deg") is not None]
    for i, r in enumerate(R):
        win = R[max(0, i - window + 1):i + 1]
        by = {sd: [w["roll_mean_deg"] for w in win if w.get("side") == sd] for sd in (1, -1)}
        if by[1] and by[-1]:
            static = 0.5 * (np.mean(by[1]) + np.mean(by[-1]))
        else:
            static = float(np.mean([w["roll_mean_deg"] for w in win]))
        r["roll_static_deg"] = float(static)
        r["ama_load_kg"] = float(ama_load_from_roll(rt, math.radians(static + shift),
                                                    craft.outrigger.side))
    return len(R)


def stroke_physics(rows: List[Dict[str, Any]], craft: Craft, total_kg: float, *,
                   speed_key: Optional[str] = None, ref_ama_kg: float = 3.0,
                   head_wind: float = 0.0, zero_load_roll_deg: Optional[float] = None,
                   default_ama_kg: Optional[float] = None) -> Dict[str, Any]:
    """Adds per stroke: ``ama_load_kg``, ``drag_n``, ``boat_power_w``,
    ``ama_cost_mps`` / ``ama_cost_s500``, ``yaw_cost_mps`` / ``yaw_cost_s500``.

    ``zero_load_roll_deg``: the roll measured on shore with the paddler seated and
    the ama just touching the water (no load on it).  When given, roll is read
    relative to it, which absorbs both the level-reference error and any
    difference between the real and the modelled rig height.
    ``default_ama_kg``: the ama load to assume for rows without a roll (e.g.
    SpeedCoach data, which has no motion sensor)."""
    key = speed_key or _speed_key(rows)
    table = resistance_table(craft, round(total_kg, 3))
    sp, loads, R = table["speeds"], table["ama_loads"], table["R"]
    rt = roll_table(craft, round(total_kg, 3)) if craft.outrigger else None
    side = craft.outrigger.side if craft.outrigger else 1
    draft = craft.hull_draft(total_kg, ref_ama_kg if craft.outrigger else 0.0)
    yaw_k = 0.5 * math.pi * 1025.0 * draft ** 2

    def curve_at(L: float) -> np.ndarray:
        if len(loads) == 1:
            return R[:, 0]
        L = min(max(L, loads[0]), loads[-1])
        j = max(0, min(len(loads) - 2, int(np.searchsorted(loads, L)) - 1))
        w = (L - loads[j]) / (loads[j + 1] - loads[j])
        return (1 - w) * R[:, j] + w * R[:, j + 1]

    ref_curve = curve_at(ref_ama_kg if craft.outrigger else 0.0)
    estimate_ama_load(rows, craft, total_kg, zero_load_roll_deg=zero_load_roll_deg)
    n = 0
    for r in rows:
        v = r.get(key)
        if v is None or not np.isfinite(v) or v < 0.8:
            continue
        L = r.get("ama_load_kg")
        if L is None:
            L = default_ama_kg if (craft.outrigger is not None and default_ama_kg is not None) else 0.0
        beta = math.radians(r.get("beta_rms_deg") or 0.0)
        curve = curve_at(L)
        Rw = float(np.interp(v, sp, curve))
        Ra = air_drag(v, head_wind, craft.cda_m2)
        Ry = float(yaw_drag(v, beta, draft))
        P = (Rw + Ra + Ry) * v
        r.update(drag_n=Rw + Ra + Ry, boat_power_w=P, yaw_drag_n=Ry, speed_basis=key)
        yc = yaw_k * beta * beta
        if craft.outrigger is not None and L > ref_ama_kg:
            v1 = speed_for_power(sp, ref_curve, P, cda_m2=craft.cda_m2, head_wind=head_wind,
                                 extra_coef=yc)
            r["ama_cost_mps"] = v1 - v
            r["ama_cost_s500"] = 500.0 / v - 500.0 / v1
        elif craft.outrigger is not None:
            r["ama_cost_mps"] = r["ama_cost_s500"] = 0.0
        v2 = speed_for_power(sp, curve, P, cda_m2=craft.cda_m2, head_wind=head_wind)
        r["yaw_cost_mps"] = v2 - v
        r["yaw_cost_s500"] = 500.0 / v - 500.0 / v2
        n += 1
    return {"n": n, "speed_basis": key, "ref_ama_kg": ref_ama_kg, "draft_m": draft}


def physics_summary(rows, t0: Optional[float] = None, t1: Optional[float] = None) -> Dict[str, Any]:
    R = [r for r in rows if r.get("boat_power_w") is not None
         and (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1)]
    if not R:
        return {"n": 0}
    med = lambda k: (float(np.median([r[k] for r in R if r.get(k) is not None]))
                     if any(r.get(k) is not None for r in R) else None)
    return {"n": len(R), "boat_power_w": med("boat_power_w"), "drag_n": med("drag_n"),
            "ama_load_kg": med("ama_load_kg"), "ama_cost_s500": med("ama_cost_s500"),
            "ama_cost_mps": med("ama_cost_mps"), "yaw_cost_s500": med("yaw_cost_s500"),
            "beta_rms_deg": med("beta_rms_deg"), "roll_wobble_deg": med("roll_wobble_deg"),
            "speed_basis": R[0].get("speed_basis")}


def wobble_cost_from_data(rows, *, block: int = 15, min_blocks: int = 12) -> Dict[str, Any]:
    """Within a session: distance per stroke against roll wobble, in blocks of
    ``block`` strokes (the boat's speed answers a change of wobble over several
    strokes), at a matched rate and with a steady trend over time taken out
    (fatigue raises wobble and lowers speed at the same time):
    ``dps = a + b (rate - median) + c * wobble^2 + d * minutes``.  ``c`` (metres
    per degree squared), its standard error, and the implied share of distance
    per stroke lost at the median wobble are the measured cost."""
    R = [r for r in rows if r.get("dps") and r.get("rate_spm") and r.get("roll_wobble_deg") is not None]
    nb = len(R) // block
    if nb < min_blocks:
        return {"n": len(R), "blocks": nb}
    take = lambda k: np.array([r[k] for r in R[: nb * block]]).reshape(nb, block).mean(1)
    d, rt, t = take("dps"), take("rate_spm"), take("t")
    w2 = (np.array([r["roll_wobble_deg"] for r in R[: nb * block]]) ** 2).reshape(nb, block).mean(1)
    X = np.stack([np.ones_like(d), rt - np.median(rt), w2, (t - t.mean()) / 60.0], -1)
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    res = d - X @ beta
    s2 = float(res @ res) / max(1, len(d) - 3)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    wm2 = float(np.median(w2))
    share = -beta[2] * wm2 / beta[0] if beta[0] > 0 else float("nan")
    return {"n": len(R), "blocks": nb, "dps_per_deg2": float(beta[2]), "se": float(se[2]),
            "median_wobble_deg": math.sqrt(wm2), "dps_share_lost_at_median": float(share),
            "conclusive": bool(abs(beta[2]) > 2 * se[2])}
