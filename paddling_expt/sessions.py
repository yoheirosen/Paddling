"""Several sessions together: what one session cannot show.

Works on per-stroke rows with time, stroke rate, GPS speed and direction of
travel: SpeedCoach exports (:mod:`.speedcoach`) or the head unit's stroke table.
For each session the push of current and wind is removed
(:func:`.analysis.heading_effect`) and the hull model gives the power reaching
the boat (:func:`.boatspeed.stroke_physics`).  Across sessions:

* **Best efforts**: the highest average power reaching the boat held for 1,
  2, 5, 10, 20 and 40 minutes of continuous paddling.
* **Critical power**: the standard two-parameter fit to those best efforts,
  ``work = CP x time + W'``.  CP is the power that can be held for a long race;
  W' is the reserve of work above it, spent in surges and the finish.  Together
  with the hull model they predict a time for a race distance in still water.
* **Best speed at each stroke rate**: the upper percentile of one-minute windows
  across sessions, its peak rate, and each window's share of that power.
* **Does extra rate buy speed?** The speed and distance-per-stroke change per
  extra stroke per minute *within* sessions (each session compared with itself,
  then pooled, with an interval from resampling whole sessions), next to the same
  slope *between* sessions.  Within a session rate still moves with effort, so
  this says what a paddler got from going up in rate, not what rate alone buys.
* **Per-session summaries** over the calendar.

None of the sessions need to be planned tests, which also limits what they show.
Best efforts in training are lower bounds on what the paddler can do, so CP and the
predicted times are conservative. "% of best" is relative to the best seen, not to
a true maximum.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .analysis import _fit_quadratic, dps_vs_rate, heading_effect
from .boatspeed import power_at_speed, speed_at_power, stroke_physics
from .hydro import Craft
from .speedcoach import add_headings

__all__ = ["prepare_session", "windows", "best_efforts", "critical_power", "race_prediction",
           "rate_envelope", "percent_of_best", "rate_effect", "session_summary", "place_groups",
           "DURATIONS_S"]

DURATIONS_S = (60, 120, 300, 600, 1200, 2400)


def prepare_session(data: Dict[str, Any], craft: Craft, total_kg: float, *,
                    ama_kg: float = 5.0, warmup_s: float = 60.0,
                    rate_range=(40.0, 75.0)) -> Optional[Dict[str, Any]]:
    """One session's paddling rows with the push removed and power added.

    ``data`` is what :func:`.speedcoach.read_speedcoach` returns.  The push of
    current and wind is fitted on steady paddling (``rows``: rate inside
    ``rate_range``, after ``warmup_s``) and then removed from every stroke
    (``all_rows``), so that a glitch stroke (a missed stroke reads as half the rate,
    the next as double) or a steering stroke does not cut a continuous effort.
    ``ama_kg``: the ama load assumed for an OC1 (SpeedCoach records no roll).
    Returns ``None`` for a session too short to analyse."""
    S = add_headings(list(data["strokes"]))
    lo, hi = rate_range
    R = [r for r in S if r["t"] > warmup_s and r.get("rate_spm") and lo <= r["rate_spm"] <= hi
         and r.get("v_mean") and r.get("heading_deg") is not None]
    if len(R) < 200:
        return None
    fade = heading_effect(R, time_trend=True)        # bracket: push with a steady fade
    push = heading_effect(R, time_trend=False)       # leaves the rows corrected
    a = math.radians(push["push_towards_deg"])
    pe, pn = push["push_mps"] * math.sin(a), push["push_mps"] * math.cos(a)
    A, h = [], None
    for r in S:
        if r.get("heading_deg") is not None:
            h = math.radians(r["heading_deg"])
        if r["t"] <= warmup_s or r.get("v_mean") is None or h is None:
            continue
        r["water_effect"] = pe * math.sin(h) + pn * math.cos(h)
        r["v_corrected"] = r["v_mean"] - r["water_effect"]
        A.append(r)
    stroke_physics(A, craft, total_kg, speed_key="v_corrected", default_ama_kg=ama_kg)
    pos = np.array([(r["lat"], r["lon"]) for r in R if r.get("lat") is not None
                    and r.get("lon") is not None], float)
    return {"start_local": data.get("start_local"), "info": data.get("info", {}),
            "rows": R, "all_rows": A, "push": push, "push_fade": fade,
            "centre": tuple(np.median(pos, 0)) if len(pos) else None,
            "last": data["strokes"][-1]}


def _blocks(t: np.ndarray, max_gap_s: float) -> List[slice]:
    """Runs of continuous paddling: split wherever strokes are ``max_gap_s`` apart."""
    cut = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    edges = np.r_[0, cut, len(t)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


def _weights(t: np.ndarray) -> np.ndarray:
    """Time each stroke stands for: half the gap to each neighbour."""
    if len(t) < 2:
        return np.ones_like(t)
    d = np.diff(t)
    w = np.r_[d[0], 0.5 * (d[:-1] + d[1:]), d[-1]]
    return np.clip(w, 0.3, 3.0)


def _arrays(rows, speed_key):
    R = [r for r in rows if r.get(speed_key) is not None and r.get("boat_power_w") is not None]
    t = np.array([r["t"] for r in R], float)
    v = np.array([r[speed_key] for r in R], float)
    p = np.array([r["boat_power_w"] for r in R], float)
    rr = np.array([r["rate_spm"] if r.get("rate_spm") else np.nan for r in R], float)
    return t, v, p, rr


def windows(rows, *, win_s: float = 60.0, step_s: float = 15.0, max_gap_s: float = 5.0,
            speed_key: str = "v_corrected", rate_range=(40.0, 75.0)) -> List[Dict[str, Any]]:
    """Time-weighted averages over sliding windows of continuous paddling (no gap
    between strokes over ``max_gap_s``): ``speed``, ``power`` (mean of per-stroke
    boat power), ``rate`` (median stroke rate, which a glitch stroke does not move),
    ``dps`` (distance per stroke).  Windows whose rate is outside ``rate_range``
    are left out."""
    t, v, p, rr = _arrays(rows, speed_key)
    if len(t) < 3:
        return []
    out = []
    for b in _blocks(t, max_gap_s):
        tb, vb, pb, rb = t[b], v[b], p[b], rr[b]
        wb = _weights(tb)
        cw, cv, cp = (np.r_[0, np.cumsum(x)] for x in (wb, wb * vb, wb * pb))
        starts = np.arange(tb[0], tb[-1] - win_s + 1e-9, step_s)
        for s0 in starts:
            i, j = np.searchsorted(tb, s0), np.searchsorted(tb, s0 + win_s)
            if j - i < 5:
                continue
            W = cw[j] - cw[i]
            rate = float(np.nanmedian(rb[i:j])) if np.isfinite(rb[i:j]).any() else float("nan")
            if not rate_range[0] <= rate <= rate_range[1]:
                continue
            speed = (cv[j] - cv[i]) / W
            out.append({"t0": float(tb[i]), "t1": float(tb[j - 1]), "rate": float(rate),
                        "speed": float(speed), "power": float((cp[j] - cp[i]) / W),
                        "dps": float(speed * 60.0 / rate)})
    return out


def best_efforts(rows, durations_s: Sequence[float] = DURATIONS_S, *, max_gap_s: float = 5.0,
                 speed_key: str = "v_corrected") -> Dict[float, Dict[str, Any]]:
    """For each duration, the window of continuous paddling with the highest mean
    boat power (and its speed and rate).  Durations longer than any continuous run
    are left out."""
    t, v, p, rr = _arrays(rows, speed_key)
    out: Dict[float, Dict[str, Any]] = {}
    if len(t) < 3:
        return out
    for b in _blocks(t, max_gap_s):
        tb, vb, pb, rb = t[b], v[b], p[b], rr[b]
        wb = _weights(tb)
        cw, cv, cp = (np.r_[0, np.cumsum(x)] for x in (wb, wb * vb, wb * pb))
        end = tb + wb / 2.0                                # each stroke's time span ends here
        begin = tb - wb / 2.0
        for dur in durations_s:
            if end[-1] - begin[0] < dur:
                continue
            # window starting at stroke i covers strokes i..j-1, j the first that ends past begin[i]+dur
            j = np.searchsorted(end, begin + dur, side="left") + 1
            ok = j <= len(tb)
            if not ok.any():
                continue
            i = np.flatnonzero(ok)
            j = j[ok]
            W = cw[j] - cw[i]
            P = (cp[j] - cp[i]) / W
            k = int(np.argmax(P))
            best = out.get(dur)
            if best is None or P[k] > best["power"]:
                ii, jj = int(i[k]), int(j[k])
                out[dur] = {"power": float(P[k]), "speed": float((cv[jj] - cv[ii]) / W[k]),
                            "rate": float(np.nanmedian(rb[ii:jj])), "t0": float(tb[ii]),
                            "seconds": float(W[k])}
    return dict(sorted(out.items()))


def critical_power(best: Dict[float, Dict[str, Any]], *, min_s: float = 120.0,
                   max_s: float = 1200.0) -> Optional[Dict[str, Any]]:
    """Two-parameter critical-power fit ``work = CP x time + W'`` to best efforts
    lasting ``min_s`` to ``max_s`` (the usual range: shorter efforts are limited by
    other things, longer ones by fatigue the model does not include)."""
    pts = [(d, b["power"]) for d, b in best.items() if min_s <= d <= max_s]
    if len(pts) < 3:
        return None
    t = np.array([d for d, _ in pts], float)
    P = np.array([p for _, p in pts], float)
    work = P * t
    X = np.stack([t, np.ones_like(t)], -1)
    (cp, wp), *_ = np.linalg.lstsq(X, work, rcond=None)
    pred = X @ np.array([cp, wp])
    ss = float(((work - work.mean()) ** 2).sum())
    r2 = 1.0 - float(((work - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    return {"cp_w": float(cp), "w_prime_j": float(wp), "r2": r2,
            "points": [{"seconds": float(a), "power": float(b)} for a, b in pts],
            "range_s": (min_s, max_s)}


def race_prediction(cp: Dict[str, Any], craft: Craft, total_kg: float,
                    distances_m: Sequence[float], *, ama_kg: float = 5.0) -> List[Dict[str, Any]]:
    """Time for each distance in still water if paddled at the power the fit allows
    for that time (``CP + W'/time``), through the hull model.  Past about 30–40
    minutes the two-parameter model is optimistic: fatigue it does not include sets in."""
    CP, Wp = cp["cp_w"], max(cp["w_prime_j"], 0.0)
    out = []
    for D in distances_m:
        lo, hi = 30.0, 6 * 3600.0
        for _ in range(60):
            T = math.sqrt(lo * hi)
            v = speed_at_power(craft, total_kg, CP + Wp / T, ama_kg)
            if v * T < D:
                lo = T
            else:
                hi = T
        T = math.sqrt(lo * hi)
        P = CP + Wp / T
        out.append({"distance_m": float(D), "seconds": float(T), "power_w": float(P),
                    "speed_mps": float(D / T), "pace_500m_s": float(500.0 * T / D),
                    "beyond_model": T > 2400.0})
    return out


def rate_envelope(wins: List[Dict[str, Any]], *, pct: float = 95.0, bin_spm: float = 2.0,
                  min_n: int = 8) -> Optional[Dict[str, Any]]:
    """Best speed at each stroke rate: the ``pct``-th percentile of window speed per
    rate band (bands with at least ``min_n`` windows), fitted with a quadratic.  The
    peak of that curve is the rate that has given the most speed; it is reported only
    when it lies inside the rates covered."""
    if not wins:
        return None
    r = np.array([w["rate"] for w in wins], float)
    v = np.array([w["speed"] for w in wins], float)
    edges = np.arange(math.floor(r.min() / bin_spm) * bin_spm, r.max() + bin_spm, bin_spm)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() >= min_n:
            pts.append({"rate": float(np.median(r[m])), "speed": float(np.percentile(v[m], pct)),
                        "speed_median": float(np.median(v[m])), "n": int(m.sum())})
    if len(pts) < 3:
        return None
    xs = np.array([p["rate"] for p in pts])
    ys = np.array([p["speed"] for p in pts])
    q = _fit_quadratic(xs, ys)
    coef = np.polyfit(xs, ys, 2)
    peak = q.get("best_rate") if q else None
    inside = peak is not None and xs.min() <= peak <= xs.max()
    lin = np.polyfit(xs, ys, 1)
    return {"pct": pct, "points": pts, "coef": coef.tolist(), "rate_min": float(xs.min()),
            "rate_max": float(xs.max()), "peak_rate": float(peak) if inside else None,
            "peak_speed": float(np.polyval(coef, peak)) if inside else None,
            "slope_mps_per_spm": float(lin[0])}


def percent_of_best(wins: List[Dict[str, Any]], env: Dict[str, Any], craft: Craft,
                    total_kg: float, *, ama_kg: float = 5.0) -> List[Dict[str, Any]]:
    """Adds ``pct_best``: each window's boat power as a percentage of the power
    needed for the best speed seen at its rate (hull model, not speed cubed)."""
    for w in wins:
        r = min(max(w["rate"], env["rate_min"] - 2), env["rate_max"] + 2)
        vb = float(np.polyval(env["coef"], r))
        pb = float(power_at_speed(craft, total_kg, vb, ama_kg))
        w["best_speed"] = vb
        w["pct_best"] = 100.0 * float(power_at_speed(craft, total_kg, w["speed"], ama_kg)) / pb \
            if pb > 0 else None
        w["rate_outside"] = not (env["rate_min"] - 1 <= w["rate"] <= env["rate_max"] + 1)
    return wins


def rate_effect(session_windows: List[List[Dict[str, Any]]], *,
                split_rate: Optional[float] = None, n_boot: int = 2000,
                seed: int = 0) -> Dict[str, Any]:
    """Speed and distance per stroke against stroke rate, within and between sessions.

    ``session_windows``: one list of non-overlapping :func:`windows` per session.
    Within: each session's windows centred on that session's means, pooled into one
    least-squares slope (so differences between sessions in conditions, boat or
    effort level do not enter); 95 % interval by resampling whole sessions.
    Between: the slope of session-mean speed on session-mean rate.  With
    ``split_rate``, the within slope is also given below and above that rate, to
    show whether the gain from extra rate tails off."""
    sess = []
    for w in session_windows:
        if len(w) >= 8:
            r = np.array([x["rate"] for x in w], float)
            sess.append((r, np.array([x["speed"] for x in w], float),
                         np.array([x["dps"] for x in w], float)))
    if len(sess) < 2:
        return {"n_sessions": len(sess)}

    def pooled(group, mask_fn=None):
        num_v = num_d = den = 0.0
        for r, v, d in group:
            m = mask_fn(r) if mask_fn else np.ones_like(r, bool)
            if m.sum() < 4:
                continue
            rc = r[m] - r[m].mean()
            num_v += float(rc @ (v[m] - v[m].mean()))
            num_d += float(rc @ (d[m] - d[m].mean()))
            den += float(rc @ rc)
        return (num_v / den, num_d / den) if den > 0 else (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)

    def with_ci(mask_fn=None):
        b = pooled(sess, mask_fn)
        boot = np.array([pooled([sess[i] for i in rng.integers(0, len(sess), len(sess))], mask_fn)
                         for _ in range(n_boot)])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        return {"speed_per_spm": b[0], "speed_ci": (float(lo[0]), float(hi[0])),
                "dps_per_spm": b[1], "dps_ci": (float(lo[1]), float(hi[1]))}

    out: Dict[str, Any] = {"n_sessions": len(sess), "n_windows": int(sum(len(x[0]) for x in sess)),
                           "within": with_ci(),
                           "per_session": [{"speed_per_spm": float(np.polyfit(r, v, 1)[0]),
                                            "dps_per_spm": float(np.polyfit(r, d, 1)[0]),
                                            "rate_sd": float(r.std()), "n": len(r)}
                                           for r, v, d in sess]}
    mr = np.array([r.mean() for r, _, _ in sess])
    mv = np.array([v.mean() for _, v, _ in sess])
    out["between_speed_per_spm"] = float(np.polyfit(mr, mv, 1)[0]) if np.ptp(mr) > 0 else None
    if split_rate is not None:
        out["split_rate"] = float(split_rate)
        out["within_below"] = with_ci(lambda r: r < split_rate)
        out["within_above"] = with_ci(lambda r: r >= split_rate)
    return out


def place_groups(sessions: List[Dict[str, Any]], *, radius_km: float = 5.0) -> List[int]:
    """Group sessions paddled on the same water (centres within ``radius_km``);
    group 0 is the place with the most sessions."""
    labels = [-1] * len(sessions)
    centres = []
    for k, s in enumerate(sessions):
        c = s.get("centre")
        if c is None:
            continue
        for g, cg in enumerate(centres):
            dy = (c[0] - cg[0]) * 111.2
            dx = (c[1] - cg[1]) * 111.2 * math.cos(math.radians(c[0]))
            if math.hypot(dx, dy) <= radius_km:
                labels[k] = g
                break
        else:
            centres.append(c)
            labels[k] = len(centres) - 1
    counts = np.bincount([g for g in labels if g >= 0]) if any(g >= 0 for g in labels) else []
    order = list(np.argsort(-np.asarray(counts), kind="stable"))
    remap = {int(g): i for i, g in enumerate(order)}
    return [remap.get(g, -1) for g in labels]


def session_summary(s: Dict[str, Any], wins: Optional[List[Dict[str, Any]]] = None
                    ) -> Dict[str, Any]:
    """One line per session: distance, time, pace, rate, distance per stroke, power,
    push of current and wind, fade, best efforts and median % of best."""
    R, last = s["rows"], s["last"]
    A = s.get("all_rows", R)
    rate = np.array([r["rate_spm"] for r in R], float)
    pw = np.array([r["boat_power_w"] for r in R if r.get("boat_power_w") is not None], float)
    dv = dps_vs_rate(R, speed_key="v_corrected")
    fade = s["push_fade"].get("trend_per_10min", {})
    be = best_efforts(A)
    out = {"date": s["start_local"].strftime("%Y-%m-%d") if s.get("start_local") else None,
           "start_hour": s["start_local"].hour if s.get("start_local") else None,
           "minutes": float(last["t"]) / 60.0, "km": float(last["distance_m"] or 0.0) / 1000.0,
           "pace_500m_s": 500.0 * float(last["t"]) / float(last["distance_m"])
           if last.get("distance_m") else None,
           "rate_median": float(np.median(rate)),
           "rate_p10": float(np.percentile(rate, 10)), "rate_p90": float(np.percentile(rate, 90)),
           "dps_median": float(np.median([r["dps"] for r in R if r.get("dps")])),
           "power_median_w": float(np.median(pw)) if len(pw) else None,
           "push_mps": s["push"].get("push_mps"), "push_towards_deg": s["push"].get("push_towards_deg"),
           "fade_mps_per_10min": fade.get("value"),
           "fade_confounded": bool(s["push_fade"].get("confounded_with_time")),
           "dps_per_spm": dv.get("dps_per_spm"),
           "best_w": {int(d): b["power"] for d, b in be.items()}}
    if wins:
        pb = [w["pct_best"] for w in wins if w.get("pct_best") is not None]
        if pb:
            out["pct_best_median"] = float(np.median(pb))
            out["pct_best_p90"] = float(np.percentile(pb, 90))
    return out
