"""Score the pipeline's outputs against the simulator's planted truth.

Only simulator runs have truth; these functions are how every derived quantity is
checked before it is trusted on real data.  Each returns a flat dict of plain
numbers so tests can assert on them and the demo can print them.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .geo import rotation_angle_deg
from .sim.hull import Truth
from .sim.sensors import SensorTruth

__all__ = ["true_time", "match_strokes", "score_head", "score_pod", "interval_overlap_score"]


def true_time(st: SensorTruth, t_head) -> np.ndarray:
    d, o = st.head_clock
    return (np.asarray(t_head, float) - o) / (1.0 + d)


def match_strokes(det_t: Sequence[float], catch: np.ndarray, exit_: np.ndarray,
                  after_s: float = 0.35) -> Tuple[np.ndarray, int]:
    """For each true stroke, the index of the first detection within its drive
    (catch .. exit + ``after_s``), or -1.  Returns (matches, n_false)."""
    det = np.asarray(det_t, float)
    used = np.zeros(len(det), bool)
    out = np.full(len(catch), -1)
    for i, (c, e) in enumerate(zip(catch, exit_)):
        lo, hi = np.searchsorted(det, c), np.searchsorted(det, e + after_s, "right")
        for j in range(lo, hi):
            if not used[j]:
                out[i], used[j] = j, True
                break
    return out, int((~used).sum())


def interval_overlap_score(det: List[Tuple[float, float]], truth: List[Tuple[float, float]]
                           ) -> Dict[str, float]:
    """Event precision / recall: a detection is right if it overlaps a true event."""
    ov = lambda a, b: min(a[1], b[1]) - max(a[0], b[0]) > 0
    tp_t = sum(any(ov(t, d) for d in det) for t in truth)
    tp_d = sum(any(ov(d, t) for t in truth) for d in det)
    tot = lambda iv: sum(b - a for a, b in iv)
    inter = sum(max(0.0, min(a[1], b[1]) - max(a[0], b[0])) for a in det for b in truth)
    return {"n_true": len(truth), "n_detected": len(det),
            "recall": tp_t / len(truth) if truth else float("nan"),
            "precision": tp_d / len(det) if det else float("nan"),
            "time_recall": inter / tot(truth) if truth else float("nan"),
            "time_precision": inter / tot(det) if det else float("nan")}


def _true_fwd_ground_speed(tr: Truth) -> np.ndarray:
    A = tr.arrays
    return A["vgx"] * np.cos(A["heading"]) + A["vgy"] * np.sin(A["heading"])


def score_head(hp, tr: Truth, st: SensorTruth) -> Dict[str, Any]:
    S = tr.strokes
    det_t = true_time(st, [s["t"] for s in hp.strokes])
    # speed is scored while paddling: GNSS speed is unsigned, so at a standstill
    # (a boat rocked back and forth by the swell) it cannot follow the sign
    moving = tr.arrays["u"] > 1.0
    keep = np.ones(len(S["catch_t"]), bool)
    catch, exit_ = S["catch_t"], S["exit_t"]
    m, n_false = match_strokes(det_t, catch, exit_)
    hit = m >= 0
    out: Dict[str, Any] = {"n_true_strokes": int(len(catch)),
                           "n_detected": int(len(det_t)),
                           "stroke_recall": float(hit.mean()) if len(hit) else float("nan"),
                           "stroke_precision": (1.0 - n_false / len(det_t)) if len(det_t) else float("nan")}
    lag = det_t[m[hit]] - catch[hit]
    out["detect_lag_ms_median"] = float(np.median(lag) * 1e3) if len(lag) else float("nan")
    out["detect_lag_ms_sd"] = float(np.std(lag) * 1e3) if len(lag) else float("nan")
    # stroke rate: detected interval vs the true interval of the same strokes
    errs = []
    idx = np.where(hit)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1 and m[b] == m[a] + 1:
            r_det = 60.0 / (det_t[m[b]] - det_t[m[a]])
            r_true = 60.0 / (catch[b] - catch[a])
            errs.append(r_det - r_true)
    errs = np.abs(np.asarray(errs))
    out["rate_err_spm_median"] = float(np.median(errs)) if len(errs) else float("nan")
    out["rate_err_spm_p95"] = float(np.percentile(errs, 95)) if len(errs) else float("nan")
    # side
    side_true = S["side"][keep]
    got = [(hp.strokes[m[i]].get("side"), side_true[i]) for i in idx if "side" in hp.strokes[m[i]]]
    out["side_accuracy"] = float(np.mean([a == b for a, b in got])) if got else float("nan")
    # speed: the fused fore-aft ground speed against the truth
    H = hp.hist
    vt = np.interp(true_time(st, H["t"]), tr.t, _true_fwd_ground_speed(tr))
    ok = np.isfinite(H["v"]) & (np.interp(true_time(st, H["t"]), tr.t, moving.astype(float)) > 0.5)
    # with a recent GNSS fix vs dead-reckoning through a GNSS drop-out
    th = H["t"]
    fix_t = np.sort(hp.fixes["t"])
    j = np.searchsorted(fix_t, th, "right") - 1
    age = np.where(j >= 0, th - fix_t[np.clip(j, 0, None)], np.inf) if len(fix_t) else np.full(len(th), np.inf)
    for name, sel in (("speed_rmse", ok & (age < 1.0)), ("speed_rmse_coasting", ok & (age >= 1.0))):
        err = H["v"][sel] - vt[sel]
        out[name] = float(np.sqrt(np.mean(err ** 2))) if sel.any() else float("nan")
    # the check and distance per stroke over the same detected cycles
    fe, de = [], []
    vtrue = _true_fwd_ground_speed(tr)
    for s in hp.strokes:
        if "check_ratio" not in s:
            continue
        a, b = true_time(st, [s["t"], s["t"] + s["cycle_s"]])
        sl = slice(np.searchsorted(tr.t, a), np.searchsorted(tr.t, b))
        v = vtrue[sl]
        if len(v) < 10:
            continue
        fe.append(s["check_ratio"] - (v.max() - v.min()) / v.mean())
        de.append(s["dps"] - np.trapezoid(v, tr.t[sl]))
    out["check_ratio_err_median"] = float(np.median(np.abs(fe))) if fe else float("nan")
    out["check_ratio_bias"] = float(np.mean(fe)) if fe else float("nan")
    out["dps_err_m_median"] = float(np.median(np.abs(de))) if de else float("nan")
    out["mount_err_deg"] = (rotation_angle_deg(hp.mount.matrix, st.mount)
                            if hp.mount.ready else float("nan"))
    runs = [tuple(true_time(st, r)) for r in hp.runs]
    rs = interval_overlap_score(runs, tr.runs)
    out.update({f"runs_{k}": v for k, v in rs.items()})
    return out


def score_pod(pp, tr: Truth, st: SensorTruth) -> Dict[str, Any]:
    """Pod: clock alignment, stroke timing, force curve, side, left/right balance."""
    S = tr.strokes
    out: Dict[str, Any] = {}
    dp, op = st.pod_clock
    if pp.clock.ready:
        t_true = np.linspace(tr.t[0], tr.t[-1], 400)
        awake = np.zeros(len(t_true), bool)
        for a, b in st.pod_awake:
            awake |= (t_true >= a + 5) & (t_true <= b)
        if awake.any():
            err = pp.clock.to_head(st.pod_time(t_true[awake])) - st.head_time(t_true[awake])
            out["clock_err_ms_median"] = float(np.median(np.abs(err)) * 1e3)
            out["clock_err_ms_max"] = float(np.max(np.abs(err)) * 1e3)
    done = [s for s in pp.strokes if "catch" in s]
    det_c = true_time(st, [s["catch"] for s in done])
    keep = np.zeros(len(S["catch_t"]), bool)
    for a, b in st.pod_awake:
        keep |= (S["catch_t"] > a + 1.0) & (S["exit_t"] < b)
    for a, b in st.ble_outages:
        keep &= ~((S["catch_t"] > a - 1.0) & (S["catch_t"] < b + 1.0))
    idx = np.flatnonzero(keep)
    m = np.full(len(idx), -1)
    used = set()
    for r, i in enumerate(idx):
        if not len(det_c):
            break
        j = int(np.argmin(np.abs(det_c - S["catch_t"][i])))
        if abs(det_c[j] - S["catch_t"][i]) < 0.25 and j not in used:
            m[r] = j
            used.add(j)
    hit = m >= 0
    out["n_true_strokes"] = int(len(idx))
    out["stroke_recall"] = float(hit.mean()) if len(hit) else float("nan")
    out["stroke_precision"] = (len(used) / len(det_c)) if len(det_c) else float("nan")
    out["lost_packets"] = pp.lost_packets
    out["dropped_strokes"] = pp.dropped_strokes
    if hit.any():
        ti, dj = idx[hit], m[hit]
        dc = det_c[dj] - S["catch_t"][ti]
        de = true_time(st, [done[j]["exit"] for j in dj]) - S["exit_t"][ti]
        out["catch_err_ms_median"] = float(np.median(np.abs(dc)) * 1e3)
        out["catch_bias_ms"] = float(np.mean(dc) * 1e3)
        out["exit_err_ms_median"] = float(np.median(np.abs(de)) * 1e3)
        imp = np.array([done[j]["impulse"] for j in dj])
        out["impulse_corr"] = float(np.corrcoef(imp, S["impulse"][ti])[0, 1]) if len(imp) > 3 else float("nan")
        if pp.calibrated:
            out["impulse_ratio_median"] = float(np.median(imp / S["impulse"][ti]))
        pk = np.array([done[j]["peak_pos"] for j in dj])
        out["peak_pos_err_median"] = float(np.median(np.abs(pk - S["peak_pos"][ti])))
        out["peak_pos_corr"] = float(np.corrcoef(pk, S["peak_pos"][ti])[0, 1]) if np.std(S["peak_pos"][ti]) > 0 else float("nan")
        sides = [(done[j].get("side"), S["side"][i]) for i, j in zip(ti, dj)]
        out["side_accuracy"] = float(np.mean([a == b for a, b in sides]))
    out["asymmetry_true"] = float(tr.params["paddler"].side_asymmetry)
    a = pp.asymmetry()
    out["asymmetry_est"] = a.get("asymmetry", float("nan")) if a else float("nan")
    if pp.axes.ready:
        c = abs(float(pp.axes.sweep @ st.pod_mount[1]))
        out["sweep_axis_err_deg"] = float(np.degrees(np.arccos(min(1.0, c))))
    return out


def score_training(session, rows, *, settle: int = 8) -> Dict[str, Any]:
    """Score the session analyses of :func:`.sim.scenarios.training_session`.

    Fits the water-speed model on the steady piece, then compares with the truth:
    the current along the course, the best stroke rate fresh and tired (on speed
    through the water, and on GNSS speed to show the current's bias), the share of
    full-effort boat power during the steady piece, and the change in where the
    force peak falls in the drive between the start and end of the steady piece.
    Adds ``water_speed``, ``current_along`` and ``pct_power`` to ``rows``.
    """
    from . import analysis as an
    tr, st = session.truth, session.sensors
    segs = session.head_segments()
    tsegs = session.segments
    steady = [s for s in segs if s[2] == "steady"][0]
    out: Dict[str, Any] = {}
    W = an.water_speed_model(an.select(rows, steady[0], steady[1]))
    an.water_speed(rows, W)
    out["water_k"] = W["k"]
    # current along the hull's heading
    R = an.select(rows, steady[0], steady[1], require=("current_along",))
    tt = true_time(st, [r["t"] for r in R])
    c_true = np.interp(tt, tr.t, tr["cur_x"] * np.cos(tr["heading"]) + tr["cur_y"] * np.sin(tr["heading"]))
    c_est = np.array([r["current_along"] for r in R])
    out["current_rmse_per_stroke"] = float(np.sqrt(np.mean((c_est - c_true) ** 2)))
    legs = []
    for sgn in (1, -1):
        m = np.sign(c_true) == sgn
        if m.sum() > 20:
            legs.append(abs(float(c_est[m].mean() - c_true[m].mean())))
    out["current_leg_mean_err"] = max(legs) if legs else float("nan")

    def pieces(prefix):
        return [r for a, b, lab in segs if lab.startswith(prefix)
                for r in an.select(rows, a, b)[settle:]]
    for tag in ("fresh", "tired"):
        P = pieces(f"test_{tag}")
        t_p = true_time(st, [r["t"] for r in P])
        out[f"best_rate_true_{tag}"] = float(np.median(np.interp(t_p, tr.t, tr["rate_opt"])))
        for key, name in (("water_speed", "water"), ("v_mean", "ground")):
            fit = an.rate_speed_curve(P, speed_key=key)["fit"] or {}
            out[f"best_rate_{name}_{tag}"] = fit.get("best_rate", float("nan"))
    # % of full-effort boat power over the steady piece
    cap = an.capacity_curve(pieces("test_fresh"), speed_key="water_speed")
    an.percent_power(rows, cap)
    R = an.select(rows, steady[0], steady[1], require=("pct_power",))
    t_r = true_time(st, [r["t"] for r in R])
    # truth: (true water speed / true full-effort water speed at that rate)^3
    rt, ut = [], []
    for a, b, lab in tsegs:
        if lab.startswith("test_fresh"):
            m = (tr.strokes["catch_t"] > a + 8) & (tr.strokes["catch_t"] < b)
            rt.append(float(np.median(tr.strokes["rate_spm"][m])))
            mm = (tr.t > a + 8) & (tr.t < b)
            ut.append(float(np.median(tr["u"][mm])))
    cc = np.polyfit(rt, ut, 2)
    u_now = np.array([np.mean(tr["u"][(tr.t >= x) & (tr.t < x + (r.get("cycle_s") or 1.0))])
                      for x, r in zip(t_r, R)])
    truth_pct = 100.0 * (u_now / np.polyval(cc, [r["rate_spm"] for r in R])) ** 3
    est = np.array([r["pct_power"] for r in R])
    out["pct_power_est_median"] = float(np.median(est))
    out["pct_power_true_median"] = float(np.median(truth_pct))
    out["pct_power_err_median"] = float(np.median(np.abs(est - truth_pct)))
    # the planted profile change over the steady piece
    a, b = steady[0], steady[1]
    pc = an.profile_change(rows, (a + 30, a + 210), (b - 180, b))
    g = pc["both"]
    S = tr.strokes
    ta, tb = true_time(st, [a + 30, a + 210]), true_time(st, [b - 180, b])
    m0 = (S["catch_t"] > ta[0]) & (S["catch_t"] < ta[1])
    m1 = (S["catch_t"] > tb[0]) & (S["catch_t"] < tb[1])
    out["peak_pos_shift_true"] = float(S["peak_pos"][m1].mean() - S["peak_pos"][m0].mean())
    out["peak_pos_shift_est"] = g["peak_pos"]["change"] if g.get("enough") else float("nan")
    out["peak_pos_shift_z"] = g["peak_pos"]["z"] if g.get("enough") else float("nan")
    out["profile_flags"] = g.get("flags")
    return out
