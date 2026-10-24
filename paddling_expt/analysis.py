"""Performance analyses over the per-stroke table (training and racing).

All functions take the rows of :meth:`.fusion.OC1Engine.stroke_table` (or any list
of dicts with the same keys), so they run the same way on a finished session or on
the trailing window of a live one.  Each returns plain numbers with the stroke
counts behind them; nothing is reported without saying how many strokes support it.

* :func:`rate_speed_curve` — speed against stroke rate: median speed per rate band
  and a quadratic fit, with the rate that gave the best speed.  Run on pieces at a
  fixed effort, that best rate is the paddler's most efficient rate *at that
  level of fatigue*; run early and late in a session, the shift shows fatigue.
* :func:`capacity_curve` / :func:`percent_power` — speed at full effort for each
  rate (from full-effort test pieces), then every other stroke's **share of that
  power** at its own rate: drag rises with the square of speed, so the power
  that moves the boat goes with speed cubed, and
  ``% power = 100 * (speed / full-effort speed at this rate) ** 3``.  This is the
  power that reaches the boat; a paddler working as hard but less efficiently
  (tired, or off their best rate) shows a lower percentage.  The pod's force
  separates the two (:func:`efficiency_by_rate`).
* :func:`profile_change` — whether the pull-force curve has changed against a
  reference stretch, per side: where in the drive the peak falls, how front-loaded
  the pull is, impulse, drive time and the whole-curve shape difference, each with
  a z-score against stroke-to-stroke scatter.
* :func:`fatigue_trends` — distance per stroke at a given rate, impulse, the
  check and consistency over the session, with slopes per 10 minutes.
* :func:`symmetry` — left vs right impulse, peak position, speed and distance per
  stroke, and the cost of the first stroke after a side switch.
* :func:`splits` — per-segment or per-distance summaries (time, distance, pace per
  500 m, rate, distance per stroke, check, impulse, % power).
* :func:`water_speed_model` / :func:`water_speed` — speed through the water from
  the pod's force: in steady paddling mean propulsive force balances drag
  (``F = k v^2``), so ``v_water = sqrt(F / k)`` once ``k`` is fitted on a
  stretch where the current averages out (equal time on out-and-back legs).  The
  current along the course is then GNSS speed minus water speed — how much the
  water is helping or costing the paddler, stroke by stroke.
* :func:`heading_effect` — without the pod: the push of current and wind
  together, from how GNSS speed depends on the direction of travel once stroke
  rate (and, optionally, a steady trend over time) is accounted for.  It needs
  a course with several directions; on a single out-and-back the direction is
  tied to the time in the session, and it says so.
* :func:`runs_summary` — downwind runs: how many, time on them, speed on them.

Speeds come from GNSS, i.e. over the ground.  Unless :func:`water_speed` is used,
a current biases every speed-based number here; out-and-back pieces cancel it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["rate_speed_curve", "capacity_curve", "percent_power", "efficiency_by_rate",
           "profile_change", "fatigue_trends", "symmetry", "splits", "water_speed_model",
           "water_speed", "heading_effect", "legs", "dps_vs_rate", "runs_summary", "select"]


def select(rows, t0: Optional[float] = None, t1: Optional[float] = None, *,
           require: Sequence[str] = (), utc: bool = False) -> List[Dict[str, Any]]:
    """Rows between ``t0`` and ``t1`` (head clock, or UTC with ``utc=True``) that
    have every key in ``require``."""
    key = "utc" if utc else "t"
    out = []
    for r in rows:
        tv = r.get(key)
        if tv is None or (t0 is not None and tv < t0) or (t1 is not None and tv > t1):
            continue
        if all(r.get(k) is not None and np.isfinite(r[k]) if isinstance(r.get(k), float)
               else r.get(k) is not None for k in require):
            out.append(r)
    return out


def _col(rows, k) -> np.ndarray:
    return np.array([r[k] for r in rows], float)


def _distance(g) -> float:
    """Distance covered by a run of rows: the GPS odometer when the rows carry
    one (a SpeedCoach export), else the sum of distance per stroke."""
    a, b = g[0].get("distance_m"), g[-1].get("distance_m")
    if a is not None and b is not None:
        return float(b - a + (g[0].get("dps") or 0.0))
    return float(sum(r["dps"] for r in g if r.get("dps") is not None))


def _fit_quadratic(x, y) -> Optional[Dict[str, float]]:
    if len(x) < 6 or np.ptp(x) < 4:
        return None
    a, b, c = np.polyfit(x, y, 2)
    out = {"a": float(a), "b": float(b), "c": float(c)}
    if a < 0:
        xb = -b / (2 * a)
        out["best_rate"] = float(xb)
        out["best_speed"] = float(c - b * b / (4 * a))
        out["best_inside_range"] = bool(x.min() <= xb <= x.max())
    return out


def rate_speed_curve(rows, *, speed_key: str = "v_mean", bin_spm: float = 2.0,
                     min_n: int = 5) -> Dict[str, Any]:
    R = select(rows, require=("rate_spm", speed_key))
    if not R:
        return {"n": 0, "bins": [], "fit": None}
    r, v = _col(R, "rate_spm"), _col(R, speed_key)
    edges = np.arange(math.floor(r.min() / bin_spm) * bin_spm, r.max() + bin_spm, bin_spm)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() >= min_n:
            q = np.percentile(v[m], [25, 50, 75])
            bins.append({"rate_lo": float(lo), "rate_hi": float(hi), "rate": float(np.median(r[m])),
                         "n": int(m.sum()), "speed_median": float(q[1]),
                         "speed_q25": float(q[0]), "speed_q75": float(q[2])})
    fit = _fit_quadratic(r, v)
    if fit is not None:
        pred = np.polyval([fit["a"], fit["b"], fit["c"]], r)
        fit["rmse"] = float(np.sqrt(np.mean((v - pred) ** 2)))
    return {"n": len(R), "speed_key": speed_key, "bins": bins, "fit": fit}


def capacity_curve(rows, *, speed_key: str = "v_mean", pct: float = 50.0) -> Optional[Dict[str, Any]]:
    """Full-effort speed vs rate, from rows of full-effort test pieces only
    (leave out the first strokes of each piece, while the boat accelerates).

    The median speed per 2-spm band, fitted with a quadratic, is the capacity
    speed at each rate."""
    cur = rate_speed_curve(rows, speed_key=speed_key, min_n=3)
    if not cur["bins"]:
        return None
    R = select(rows, require=("rate_spm", speed_key))
    r, v = _col(R, "rate_spm"), _col(R, speed_key)
    xs, ys = [], []
    for b in cur["bins"]:
        m = (r >= b["rate_lo"]) & (r < b["rate_hi"])
        xs.append(float(np.median(r[m])))
        ys.append(float(np.percentile(v[m], pct)))
    xs, ys = np.array(xs), np.array(ys)
    coef = np.polyfit(xs, ys, 2) if len(xs) >= 3 else np.polyfit(xs, ys, 1)
    return {"coef": coef.tolist(), "rate_min": float(xs.min()), "rate_max": float(xs.max()),
            "points": [{"rate": float(a), "speed": float(b)} for a, b in zip(xs, ys)],
            "speed_key": speed_key}


def capacity_speed(capacity: Dict[str, Any], rate) -> np.ndarray:
    r = np.clip(np.asarray(rate, float), capacity["rate_min"] - 2, capacity["rate_max"] + 2)
    return np.polyval(capacity["coef"], r)


def percent_power(rows, capacity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adds ``pct_power`` (and ``capacity_speed``) to each row in place."""
    key = capacity.get("speed_key", "v_mean")
    for r in rows:
        if r.get("rate_spm") and r.get(key) is not None:
            vm = float(capacity_speed(capacity, r["rate_spm"]))
            r["capacity_speed"] = vm
            r["pct_power"] = 100.0 * (r[key] / vm) ** 3 if vm > 0 else None
            r["rate_outside_test"] = not (capacity["rate_min"] - 1 <= r["rate_spm"]
                                          <= capacity["rate_max"] + 1)
    return rows


def efficiency_by_rate(rows, *, speed_key: str = "v_mean", bin_spm: float = 2.0,
                       min_n: int = 5) -> Dict[str, Any]:
    """Speed cubed per unit pod power proxy, by rate band, relative to the best band.

    Boat power goes with speed cubed; the pod's proxy (mean blade force x speed)
    goes with what the paddler puts in.  Their ratio, normalised to its maximum,
    is a relative efficiency per rate — high where effort turns into speed best.
    """
    R = select(rows, require=("rate_spm", speed_key, "power_proxy"))
    if not R:
        return {"n": 0, "bins": []}
    r, v, p = _col(R, "rate_spm"), _col(R, speed_key), _col(R, "power_proxy")
    e = v ** 3 / p
    edges = np.arange(math.floor(r.min() / bin_spm) * bin_spm, r.max() + bin_spm, bin_spm)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() >= min_n:
            bins.append({"rate": float(np.median(r[m])), "n": int(m.sum()),
                         "eff": float(np.median(e[m]))})
    if bins:
        top = max(b["eff"] for b in bins)
        for b in bins:
            b["eff_rel"] = b["eff"] / top
    return {"n": len(R), "bins": bins,
            "best_rate": max(bins, key=lambda b: b["eff"])["rate"] if bins else None}


# ── pull-force profile ──────────────────────────────────────────────────
def _shape(rows) -> np.ndarray:
    P = np.array([r["profile"] for r in rows], float)
    area = P.mean(1, keepdims=True)
    return P / np.where(area > 0, area, 1.0)


def _feature_change(ref: np.ndarray, cur: np.ndarray) -> Dict[str, float]:
    d = float(np.mean(cur) - np.mean(ref))
    se = math.sqrt(np.var(ref, ddof=1) / len(ref) + np.var(cur, ddof=1) / len(cur))
    return {"ref": float(np.mean(ref)), "now": float(np.mean(cur)), "change": d,
            "z": d / se if se > 0 else float("nan")}


def profile_change(rows, ref: Tuple[float, float], now: Tuple[float, float], *,
                   per_side: bool = True, min_n: int = 10, z_flag: float = 3.0
                   ) -> Dict[str, Any]:
    """Compare the pull-force curve in ``now`` (head-clock interval) with ``ref``.

    Features: peak position in the drive (0 = catch, 1 = exit), share of impulse
    in the first half, impulse, peak force, drive time, and the shape difference
    (root-mean-square difference of the area-normalised curves, relative to the
    reference curve's own stroke-to-stroke scatter)."""
    need = ("profile", "peak_pos", "impulse", "drive_s", "first_half_share", "peak")
    A = select(rows, *ref, require=need)
    B = select(rows, *now, require=need)
    groups = {"both": (A, B)}
    if per_side:
        for sd, name in ((1, "left"), (-1, "right")):
            groups[name] = ([r for r in A if r.get("side") == sd],
                            [r for r in B if r.get("side") == sd])
    out: Dict[str, Any] = {}
    for name, (a, b) in groups.items():
        if len(a) < min_n or len(b) < min_n:
            out[name] = {"n_ref": len(a), "n_now": len(b), "enough": False}
            continue
        g = {"n_ref": len(a), "n_now": len(b), "enough": True}
        for k in ("peak_pos", "first_half_share", "impulse", "peak", "drive_s"):
            g[k] = _feature_change(_col(a, k), _col(b, k))
        sa, sb = _shape(a), _shape(b)
        mean_a = sa.mean(0)
        own = np.sqrt(np.mean((sa - mean_a) ** 2, axis=1))         # scatter within ref
        diff = float(np.sqrt(np.mean((sb.mean(0) - mean_a) ** 2)))
        g["shape_rms"] = diff
        g["shape_ratio"] = diff / (float(np.median(own)) / math.sqrt(len(b)) + 1e-12)
        g["mean_shape_ref"] = mean_a.round(4).tolist()
        g["mean_shape_now"] = sb.mean(0).round(4).tolist()
        g["flags"] = [k for k in ("peak_pos", "first_half_share", "impulse", "peak", "drive_s")
                      if abs(g[k]["z"]) >= z_flag]
        out[name] = g
    return out


# ── trends, symmetry, splits ────────────────────────────────────────────
def _slope_per_10min(t, y) -> Dict[str, float]:
    if len(t) < 10 or np.ptp(t) < 60:
        return {"slope_per_10min": float("nan"), "se": float("nan")}
    X = np.stack([np.ones_like(t), (t - t.mean()) / 600.0], -1)
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / max(1, len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"slope_per_10min": float(beta[1]), "se": float(math.sqrt(cov[1, 1]))}


def fatigue_trends(rows, t0: Optional[float] = None, t1: Optional[float] = None, *,
                   window_s: float = 120.0) -> Dict[str, Any]:
    """Trends over a steady piece: distance per stroke corrected to the piece's
    median rate (so a rate change is not read as fatigue), impulse, the check and
    consistency (coefficient of variation of impulse in each window)."""
    R = select(rows, t0, t1, require=("rate_spm", "dps"))
    if len(R) < 20:
        return {"n": len(R)}
    t, r, d = _col(R, "t"), _col(R, "rate_spm"), _col(R, "dps")
    # distance per stroke vs rate within the piece: remove the rate effect
    b = np.polyfit(r, d, 1)
    d_adj = d - b[0] * (r - np.median(r))
    out: Dict[str, Any] = {"n": len(R), "dps_at_median_rate": _slope_per_10min(t, d_adj),
                           "rate": _slope_per_10min(t, r)}
    for k in ("check_ratio", "impulse", "peak_pos", "v_mean"):
        m = [x.get(k) is not None for x in R]
        if sum(m) >= 20:
            tt = t[np.array(m)]
            out[k] = _slope_per_10min(tt, np.array([x[k] for x in R if x.get(k) is not None], float))
    # consistency per window
    if all(x.get("impulse") is not None for x in R):
        imp = _col(R, "impulse")
        edges = np.arange(t.min(), t.max() + window_s, window_s)
        cv = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t < hi)
            if m.sum() >= 10:
                cv.append((float(np.mean(t[m])), float(np.std(imp[m]) / np.mean(imp[m]))))
        out["impulse_cv_windows"] = cv
    return out


def symmetry(rows) -> Dict[str, Any]:
    L = [r for r in rows if r.get("side") == 1]
    R_ = [r for r in rows if r.get("side") == -1]
    out: Dict[str, Any] = {"n_left": len(L), "n_right": len(R_)}
    for k in ("impulse", "peak", "peak_pos", "v_mean", "dps", "check_ratio"):
        a = [r[k] for r in L if r.get(k) is not None]
        b = [r[k] for r in R_ if r.get(k) is not None]
        if len(a) >= 5 and len(b) >= 5:
            ma, mb = float(np.mean(a)), float(np.mean(b))
            out[k] = {"left": ma, "right": mb, "diff": ma - mb,
                      "rel": (ma - mb) / ((ma + mb) / 2) if (ma + mb) else float("nan")}
    sw = [r for r in rows if r.get("stroke_on_side") == 1 and r.get("dps") is not None]
    rest = [r for r in rows if (r.get("stroke_on_side") or 0) > 2 and r.get("dps") is not None]
    if len(sw) >= 3 and len(rest) >= 10:
        out["switch_stroke_dps_loss_m"] = float(np.mean([r["dps"] for r in rest])
                                                - np.mean([r["dps"] for r in sw]))
        runs, n = [], 0
        for r in rows:
            if r.get("stroke_on_side") == 1 and n:
                runs.append(n)
            n = r.get("stroke_on_side") or n
        out["strokes_per_side_median"] = float(np.median(runs)) if runs else None
    return out


def splits(rows, *, every_m: Optional[float] = None,
           segments: Optional[Sequence[Tuple[float, float, str]]] = None) -> List[Dict[str, Any]]:
    """Summaries per distance split (``every_m``) or per labelled time segment."""
    R = select(rows, require=("dps",))
    if not R:
        return []
    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    if segments:
        for a, b, lab in segments:
            groups.append((lab, [r for r in R if a <= r["t"] < b]))
    else:
        step = every_m or 500.0
        cum, cur, k = 0.0, [], 0
        for r in R:
            cur.append(r)
            cum += r["dps"]
            if cum >= (k + 1) * step:
                groups.append((f"{int(k * step)}-{int((k + 1) * step)} m", cur))
                cur, k = [], k + 1
        if cur:
            groups.append((f"{int(k * step)}- m", cur))
    out = []
    for lab, g in groups:
        if len(g) < 3:
            continue
        dist = _distance(g)
        dur = float(g[-1]["t"] - g[0]["t"] + (g[-1].get("cycle_s") or 0))
        row = {"label": lab, "strokes": len(g), "distance_m": dist, "time_s": dur,
               "pace_500m_s": 500.0 * dur / dist if dist > 0 else None}
        for k in ("rate_spm", "v_mean", "dps", "check_ratio", "impulse", "peak_pos",
                  "pct_power", "power_proxy", "water_speed", "current_along"):
            vals = [r[k] for r in g if r.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


# ── speed through the water and the current ─────────────────────────────
def water_speed_model(rows, *, min_n: int = 30) -> Optional[Dict[str, Any]]:
    """Fit ``mean_force = k * v_water^2`` on rows where the current averages out
    (out-and-back legs of equal time, or slack water).  Returns ``{"k": ...}``."""
    R = select(rows, require=("mean_force", "v_mean"))
    if len(R) < min_n:
        return None
    f, v = _col(R, "mean_force"), _col(R, "v_mean")
    # GNSS speed = sqrt(F / k) + current; over out-and-back legs the current
    # averages out, so regress speed on sqrt(F) through the origin
    s = np.sqrt(np.clip(f, 0, None))
    slope = float(np.sum(v * s) / np.sum(s * s))          # 1 / sqrt(k)
    resid = v - slope * s
    return {"k": 1.0 / slope ** 2, "n": len(R), "mean_residual": float(np.mean(resid)),
            "rmse": float(np.std(resid))}


def water_speed(rows, model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adds ``water_speed`` and ``current_along`` (GNSS speed minus water speed;
    positive = the water is carrying the boat) to each row with a pod force."""
    k = model["k"]
    for r in rows:
        if r.get("mean_force") is not None and r.get("v_mean") is not None and k > 0:
            r["water_speed"] = math.sqrt(max(0.0, r["mean_force"]) / k)
            r["current_along"] = r["v_mean"] - r["water_speed"]
    return rows


def runs_summary(runs: Sequence[Tuple[float, float]], rows, t0: float, t1: float
                 ) -> Dict[str, Any]:
    on = sum(min(b, t1) - max(a, t0) for a, b in runs if b > t0 and a < t1)
    inside = lambda t: any(a <= t <= b for a, b in runs)
    R = select(rows, t0, t1, require=("v_mean",))
    v_on = [r["v_mean"] for r in R if inside(r["t"])]
    v_off = [r["v_mean"] for r in R if not inside(r["t"])]
    return {"n_runs": sum(1 for a, b in runs if b > t0 and a < t1),
            "time_on_runs_s": float(on), "share_of_time": float(on / max(1e-9, t1 - t0)),
            "speed_on_runs": float(np.mean(v_on)) if v_on else None,
            "speed_between": float(np.mean(v_off)) if v_off else None}


def heading_effect(rows, *, speed_key: str = "v_mean", time_trend: bool = True,
                   min_n: int = 60) -> Dict[str, Any]:
    """Fit ``speed = a + b (rate - median) + e sin(heading) + n cos(heading)
    [+ trend * minutes / 10]``: ``(e, n)`` is the east/north push of current and
    wind (m/s) a stroke at the median rate would feel.  Adds ``water_effect``
    (the push along each stroke's direction of travel) and ``v_corrected``
    (speed with the push removed) to the rows.

    Standard errors are widened for the strong stroke-to-stroke correlation of
    the residuals (effective sample size from their lag-1 autocorrelation)."""
    R = select(rows, require=(speed_key, "heading_deg", "rate_spm", "t"))
    if len(R) < min_n:
        return {"n": len(R), "enough": False}
    v, r, t = _col(R, speed_key), _col(R, "rate_spm"), _col(R, "t")
    h = np.radians(_col(R, "heading_deg"))
    # Headings along one line only (out and back on the same bearing) fix the
    # push along that line, not across it: then fit the along-line part alone.
    U = np.stack([np.sin(h), np.cos(h)], -1)
    ev, evec = np.linalg.eigh(U.T @ U / len(U))
    one_axis = ev[0] < 0.05 * ev[1]
    axis = evec[:, -1]                                 # (east, north) of the main line
    if one_axis:
        cols = [np.ones_like(v), r - np.median(r), U @ axis]
        names = ["speed_at_median_rate", "speed_per_spm", "push_along_axis"]
    else:
        cols = [np.ones_like(v), r - np.median(r), np.sin(h), np.cos(h)]
        names = ["speed_at_median_rate", "speed_per_spm", "push_east", "push_north"]
    if time_trend:
        cols.append((t - t.mean()) / 600.0)
        names.append("trend_per_10min")
    X = np.stack(cols, -1)
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    res = v - X @ beta
    rho = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 3 else 0.0
    n_eff = max(5.0, len(v) * (1 - rho) / (1 + rho)) if rho < 0.999 else 5.0
    s2 = float(res @ res) / max(1, len(v) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X) * (len(v) / n_eff)
    se = np.sqrt(np.diag(cov))
    out: Dict[str, Any] = {"n": len(R), "n_effective": n_eff, "enough": True,
                           "residual_sd": float(np.sqrt(s2))}
    for k, b_, e_ in zip(names, beta, se):
        out[k] = {"value": float(b_), "se": float(e_)}
    if one_axis:
        pe, pn = beta[2] * axis[0], beta[2] * axis[1]
        out["heading_coverage"] = "one line: only the push along it is measured"
        out["axis_deg"] = float(math.degrees(math.atan2(axis[0], axis[1])) % 360.0)
    else:
        pe, pn = beta[2], beta[3]
        out["heading_coverage"] = "two or more directions"
    out["push_mps"] = float(math.hypot(pe, pn))
    out["push_towards_deg"] = float(math.degrees(math.atan2(pe, pn)) % 360.0)
    if time_trend:
        along = pe * np.sin(h) + pn * np.cos(h)
        c = float(np.corrcoef(along, t)[0, 1]) if np.std(along) > 0 else 0.0
        out["direction_time_corr"] = c
        out["confounded_with_time"] = abs(c) > 0.7
    for row, hh in zip(R, h):
        row["water_effect"] = float(pe * math.sin(hh) + pn * math.cos(hh))
        row["v_corrected"] = row[speed_key] - row["water_effect"]
    return out


def legs(rows, *, turn_deg: float = 60.0, hold_strokes: int = 20,
         min_strokes: int = 30) -> List[Dict[str, Any]]:
    """Split a session into legs of steady direction: a new leg starts when the
    direction of travel stays more than ``turn_deg`` off the current leg's for
    ``hold_strokes`` strokes in a row."""
    R = [r for r in rows if r.get("heading_deg") is not None]
    if not R:
        return []
    out, start, ref, off = [], 0, None, 0
    ang = lambda a: math.radians(a)
    acc = np.zeros(2)
    for i, r in enumerate(R):
        u = np.array([math.sin(ang(r["heading_deg"])), math.cos(ang(r["heading_deg"]))])
        if ref is None:
            ref, acc = u, u.copy()
            continue
        if float(u @ ref) < math.cos(math.radians(turn_deg)):
            off += 1
        else:
            off = 0
            acc += u
            ref = acc / np.linalg.norm(acc)
        if off >= hold_strokes:
            cut = i - hold_strokes + 1
            out.append((start, cut))
            start, off = cut, 0
            ref = acc = None
            acc = np.zeros(2)
            for rr in R[cut:i + 1]:
                acc += [math.sin(ang(rr["heading_deg"])), math.cos(ang(rr["heading_deg"]))]
            ref = acc / np.linalg.norm(acc)
    out.append((start, len(R)))
    res = []
    for a, b in out:
        g = R[a:b]
        if len(g) < min_strokes:
            continue
        hs = np.radians([r["heading_deg"] for r in g])
        hd = math.degrees(math.atan2(np.mean(np.sin(hs)), np.mean(np.cos(hs)))) % 360.0
        dist = _distance(g)
        dur = float(g[-1]["t"] - g[0]["t"])
        row = {"t0": g[0]["t"], "t1": g[-1]["t"], "strokes": len(g), "heading_deg": hd,
               "distance_m": dist, "time_s": dur,
               "pace_500m_s": 500.0 * dur / dist if dist > 0 else None}
        for k in ("rate_spm", "v_mean", "dps", "v_corrected", "water_effect"):
            vals = [r[k] for r in g if r.get(k) is not None]
            row[k] = float(np.median(vals)) if vals else None
        res.append(row)
    return res


def dps_vs_rate(rows, *, speed_key: str = "v_mean") -> Dict[str, Any]:
    """How distance per stroke and speed change with stroke rate (straight-line
    fits over the rows given): whether extra rate is buying speed or only
    shortening the stroke.  Speed per stroke-per-minute near zero with distance
    per stroke falling means the extra rate is not paying."""
    R = select(rows, require=("rate_spm", "dps", speed_key))
    if len(R) < 30:
        return {"n": len(R)}
    r, d, v = _col(R, "rate_spm"), _col(R, "dps"), _col(R, speed_key)
    bd = np.polyfit(r, d, 1)
    bv = np.polyfit(r, v, 1)
    return {"n": len(R), "dps_per_spm": float(bd[0]), "speed_per_spm": float(bv[0]),
            "dps_at_median_rate": float(np.polyval(bd, np.median(r))),
            "median_rate": float(np.median(r))}
