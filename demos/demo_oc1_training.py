#!/usr/bin/env python3
"""Simulated OC1 training session through both devices, scored against the truth.

    python "demos/demo_oc1_training.py" [--seed N] [--figure NAME]

Simulates a rate-ladder session (full-effort pieces at interleaved stroke rates,
a long steady piece, the ladder again tired) on out-and-back legs in a tidal
current; produces the records the head unit and paddle pod would send; feeds them
through the real-time pipeline; and scores every measure against the planted
answers: strokes, rate, speed, the check, distance per stroke, side, the pod's
stroke timing and force, left/right balance, the current along the course, the
best stroke rate fresh and tired, % of full-effort power, and the change in the
pull-force profile with fatigue.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from paddling_expt import analysis as an                                 # noqa: E402
from paddling_expt.boatspeed import physics_summary, stroke_physics      # noqa: E402
from paddling_expt.fusion import OC1Engine                               # noqa: E402
from paddling_expt.headunit import HeadConfig                            # noqa: E402
from paddling_expt.pod import StrainCalibration                          # noqa: E402
from paddling_expt.scoring import score_head, score_pod, score_training, true_time  # noqa: E402
from paddling_expt.sim.scenarios import training_session                 # noqa: E402
from paddling_expt.sim.sensors import strain_calibration_counts          # noqa: E402


def plot_oc1_training(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = ax[0, 0]
    for tag, col in (("fresh", "C0"), ("tired", "C3")):
        d = data["ladder"][tag]
        a.scatter(d["rate"], d["water"], s=6, color=col, alpha=0.35)
        rr = np.linspace(min(d["rate"]), max(d["rate"]), 50)
        if d["fit"]:
            a.plot(rr, np.polyval(d["fit"], rr), color=col,
                   label=f"{tag}: best {d['best_est']:.1f} (true {d['best_true']:.1f})")
        a.axvline(d["best_true"], color=col, ls=":", lw=1)
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("speed through the water (m/s)")
    a.set_title("Full-effort speed vs rate, fresh and tired")
    a.legend(fontsize=8)
    a = ax[0, 1]
    x = np.linspace(0, 1, len(data["shape_ref"]))
    a.plot(x, data["shape_ref"], color="C0", label="start of steady piece")
    a.plot(x, data["shape_now"], color="C3", label="end of steady piece")
    a.set_xlabel("share of the drive (catch = 0, exit = 1)")
    a.set_ylabel("force / mean force")
    a.set_title(f"Pull-force profile: peak moved {data['peak_shift_est']:+.3f} "
                f"(true {data['peak_shift_true']:+.3f})")
    a.legend(fontsize=8)
    a = ax[1, 0]
    a.plot(data["cur_t"], data["cur_true"], color="0.3", lw=1.2, label="true")
    a.plot(data["cur_t"], data["cur_est"], color="C2", lw=1.2, label="from pod force + GNSS")
    a.set_xlabel("time in steady piece (min)")
    a.set_ylabel("current along the course (m/s)")
    a.set_title("Water helping (+) or costing (-) the paddler, 20-stroke means")
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(data["pct_t"], data["pct_true"], color="0.3", lw=1.2, label="true")
    a.plot(data["pct_t"], data["pct_est"], color="C4", lw=1.2, label="estimated")
    a.set_xlabel("time in steady piece (min)")
    a.set_ylabel("% of full-effort boat power at this rate")
    a.set_title("Effort during the steady piece, 20-stroke means")
    a.legend(fontsize=8)
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


def _means(x, n=20):
    x = np.asarray(x, float)
    k = len(x) // n
    return x[: k * n].reshape(k, n).mean(1).tolist()


def run(seed: int = 1):
    S = training_session(seed=seed)
    loads = [0, 50, 100, 150, 200]
    cal = StrainCalibration.from_loads(loads, strain_calibration_counts(S.sensors, loads))
    t0 = time.time()
    E = OC1Engine(head_cfg=HeadConfig(craft=S.craft, total_kg=S.total_kg),
                  calibration=cal).feed_all(S.records)
    rows = E.stroke_table()
    proc = time.time() - t0
    head = score_head(E.head, S.truth, S.sensors)
    pod = score_pod(E.pod, S.truth, S.sensors)
    sess = score_training(S, rows)
    stroke_physics(rows, S.craft, S.total_kg)
    return S, E, rows, head, pod, sess, proc


def figure_data(S, rows, sess, seed):
    tr, st = S.truth, S.sensors
    segs = S.head_segments()
    lad = {}
    for tag in ("fresh", "tired"):
        P = [r for a, b, lab in segs if lab.startswith(f"test_{tag}")
             for r in an.select(rows, a, b, require=("water_speed",))[8:]]
        c = an.rate_speed_curve(P, speed_key="water_speed")
        f = c["fit"]
        lad[tag] = {"rate": [r["rate_spm"] for r in P], "water": [r["water_speed"] for r in P],
                    "fit": [f["a"], f["b"], f["c"]] if f else None,
                    "best_est": sess[f"best_rate_water_{tag}"],
                    "best_true": sess[f"best_rate_true_{tag}"]}
    steady = [s for s in segs if s[2] == "steady"][0]
    a, b = steady[0], steady[1]
    pc = an.profile_change(rows, (a + 30, a + 210), (b - 180, b))["both"]
    R = an.select(rows, a, b, require=("current_along", "pct_power"))
    tt = true_time(st, [r["t"] for r in R])
    cur_true = np.interp(tt, tr.t, tr["cur_x"] * np.cos(tr["heading"]) + tr["cur_y"] * np.sin(tr["heading"]))
    tm = (np.array([r["t"] for r in R]) - a) / 60.0
    # truth of % power, as in scoring.score_training
    rt, ut = [], []
    for x0, x1, lab in S.segments:
        if lab.startswith("test_fresh"):
            m = (tr.strokes["catch_t"] > x0 + 8) & (tr.strokes["catch_t"] < x1)
            rt.append(float(np.median(tr.strokes["rate_spm"][m])))
            ut.append(float(np.median(tr["u"][(tr.t > x0 + 8) & (tr.t < x1)])))
    cc = np.polyfit(rt, ut, 2)
    # mean true water speed over each stroke's cycle (as the scorer does)
    u_now = np.array([np.mean(tr["u"][(tr.t >= x) & (tr.t < x + (r.get("cycle_s") or 1.0))])
                      for x, r in zip(tt, R)])
    pct_true = 100 * (u_now / np.polyval(cc, [r["rate_spm"] for r in R])) ** 3
    return {"title": f"Simulated OC1 training session (seed {seed}): both devices, scored against the planted truth",
            "ladder": lad, "shape_ref": pc["mean_shape_ref"], "shape_now": pc["mean_shape_now"],
            "peak_shift_est": sess["peak_pos_shift_est"], "peak_shift_true": sess["peak_pos_shift_true"],
            "cur_t": _means(tm), "cur_true": _means(cur_true),
            "cur_est": _means([r["current_along"] for r in R]),
            "pct_t": _means(tm), "pct_true": _means(pct_true),
            "pct_est": _means([r["pct_power"] for r in R])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--figure", default=None, help="artifact base name to write")
    a = ap.parse_args(argv)
    S, E, rows, head, pod, sess, proc = run(a.seed)
    dur = S.plan.duration_s
    f = lambda d, keys: "  ".join(f"{k}={d[k]:.3g}" if isinstance(d[k], float) else f"{k}={d[k]}"
                                  for k in keys if k in d)
    print(f"Simulated {dur / 60:.0f} min session; processed in {proc:.0f} s "
          f"({dur / proc:.0f}x faster than real time); {len(rows)} strokes")
    print("\nHead unit vs truth:")
    print(" ", f(head, ["stroke_recall", "stroke_precision", "rate_err_spm_median", "side_accuracy",
                        "speed_rmse", "check_ratio_err_median", "dps_err_m_median", "mount_err_deg"]))
    print("Paddle pod vs truth:")
    print(" ", f(pod, ["clock_err_ms_median", "stroke_recall", "catch_err_ms_median",
                       "exit_err_ms_median", "impulse_corr", "peak_pos_err_median", "side_accuracy",
                       "asymmetry_est", "asymmetry_true"]))
    print("Session analyses vs truth:")
    print(" ", f(sess, ["current_leg_mean_err", "best_rate_true_fresh", "best_rate_water_fresh",
                        "best_rate_ground_fresh", "best_rate_true_tired", "best_rate_water_tired"]))
    print(" ", f(sess, ["pct_power_true_median", "pct_power_est_median", "pct_power_err_median",
                        "peak_pos_shift_true", "peak_pos_shift_est", "peak_pos_shift_z"]))
    steady = [x for x in S.head_segments() if x[2] == "steady"][0]
    mid = 0.5 * (steady[0] + steady[1])
    ph0, ph1 = physics_summary(rows, steady[0] + 60, mid), physics_summary(rows, mid, steady[1])
    print("Hull model over the steady piece (first half -> second half):")
    print(f"  power to the boat {ph0['boat_power_w']:.0f} -> {ph1['boat_power_w']:.0f} W; "
          f"ama load {ph0['ama_load_kg']:.1f} -> {ph1['ama_load_kg']:.1f} kg, costing "
          f"{ph0['ama_cost_s500']:.2f} -> {ph1['ama_cost_s500']:.2f} s per 500 m; "
          f"side-slip {ph0['beta_rms_deg']:.2f} deg ({ph1['yaw_cost_s500']:.3f} s per 500 m)")
    if a.figure:
        from agent_lib.figure_artifacts import emit       # agentic_experiments (a dependency)
        paths = emit(a.figure, data=figure_data(S, rows, sess, a.seed), plot_func=plot_oc1_training,
                     artifacts_dir=os.path.join(os.path.dirname(HERE), "artifacts"))
        print("\nwrote", ", ".join(os.path.basename(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
