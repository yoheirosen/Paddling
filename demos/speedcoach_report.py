#!/usr/bin/env python3
"""Performance report for a real NK SpeedCoach session export (CSV).

    python "paddling/demos/speedcoach_report.py" SESSION.csv [--figure NAME]

Reads the per-stroke data, works out the direction of travel from the
positions, splits the session into legs, estimates the push of current and wind
from how speed depends on direction (once stroke rate is accounted for), and
reports: legs, splits per kilometre, how speed and distance per stroke change
with stroke rate (on speed with the push removed), and pacing over the session.
With ``--figure`` it writes the four-file figure artifact to ``paddling/artifacts``
(binned and time-averaged numbers only — no positions).

The raw export holds the paddler's GPS track: keep it out of the repository.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from paddling_expt import analysis as an                       # noqa: E402
from paddling_expt.speedcoach import add_headings, read_speedcoach   # noqa: E402


def plot_speedcoach_session(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = ax[0, 0]
    t = np.array(data["time_min"])
    a.plot(t, data["speed_gps"], color="0.6", lw=1.2, label="GPS speed")
    a.plot(t, data["speed_corrected"], color="C0", lw=1.6,
           label="current and wind removed")
    for lg in data["legs"]:
        a.axvspan(lg["t0_min"], lg["t1_min"], color="C1" if lg["return"] else "C2",
                  alpha=0.08, lw=0)
    a.set_xlabel("time (min)")
    a.set_ylabel("speed (m/s, 30 s means)")
    a.set_title("Speed over the session (shading: outbound / return legs)")
    a.legend(fontsize=8)
    a = ax[0, 1]
    b = data["rate_bins"]
    r = [x["rate"] for x in b]
    a.errorbar(r, [x["speed_median"] for x in b],
               yerr=[[x["speed_median"] - x["speed_q25"] for x in b],
                     [x["speed_q75"] - x["speed_median"] for x in b]],
               fmt="o", color="C0", capsize=2)
    for x in b:
        a.annotate(str(x["n"]), (x["rate"], x["speed_q75"]), fontsize=6, ha="center",
                   va="bottom", color="0.4")
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("speed, current and wind removed (m/s)")
    a.set_title("Speed vs stroke rate (median, quartiles; n strokes)")
    a = ax[1, 0]
    db = data["dps_bins"]
    a.plot([x["rate"] for x in db], [x["dps"] for x in db], "o-", color="C3")
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("distance per stroke (m, median)")
    a.set_title(f"Distance per stroke vs rate: {data['dps_per_spm']*100:+.1f} cm per stroke/min")
    a = ax[1, 1]
    sp = data["splits"]
    a.bar(range(len(sp)), [s["pace_500m_s"] for s in sp], color="C4")
    a.set_xticks(range(len(sp)))
    a.set_xticklabels([s["label"].replace(" m", "") for s in sp], rotation=45, fontsize=7)
    lo = min(s["pace_500m_s"] for s in sp)
    a.set_ylim(lo * 0.9, max(s["pace_500m_s"] for s in sp) * 1.03)
    a.set_ylabel("pace (s per 500 m)")
    a.set_title("Pace per kilometre (lower is faster)")
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


def build(path: str):
    d = read_speedcoach(path)
    S = add_headings(d["strokes"])
    R = [r for r in S if r["t"] > 60 and r.get("rate_spm") and 40 <= r["rate_spm"] <= 75
         and r.get("v_mean")]
    he0 = an.heading_effect(R, time_trend=True)
    he = an.heading_effect(R, time_trend=False)          # leaves the rows corrected
    L = an.legs(R)
    rs = an.rate_speed_curve(R, speed_key="v_corrected")
    dv = an.dps_vs_rate(R, speed_key="v_corrected")
    sp = an.splits(R, every_m=1000)
    return d, R, he, he0, L, rs, dv, sp


def report(path: str) -> dict:
    d, R, he, he0, L, rs, dv, sp = build(path)
    info = d["info"]
    last = d["strokes"][-1]
    dur, dist = last["t"], last["distance_m"]
    print(f"Session {info.get('Name', '')}  ({info.get('Model', '')}, start {d['start_local']} local)")
    print(f"  {dur / 60:.1f} min, {dist / 1000:.2f} km, {len(d['strokes'])} strokes; "
          f"average pace {500 * dur / dist:.0f} s per 500 m")
    print(f"  recorded: heart rate {'yes' if d['has']['hr'] else 'no'}, water-speed impeller "
          f"{'yes' if d['has']['water_speed_imp'] else 'no'}, power {'yes' if d['has']['power_w'] else 'no'}")
    print("\nLegs (steady direction):")
    for lg in L:
        print(f"  {lg['t0'] / 60:5.1f}-{lg['t1'] / 60:5.1f} min  heading {lg['heading_deg']:4.0f} deg  "
              f"{lg['distance_m']:5.0f} m  pace {lg['pace_500m_s']:5.1f} s/500 m  "
              f"rate {lg['rate_spm']:4.1f}  dist/stroke {lg['dps']:.2f} m  "
              f"push of water+wind {lg['water_effect']:+.2f} m/s")
    print(f"\nCurrent and wind together: {he['push_mps']:.2f} m/s towards {he['push_towards_deg']:.0f} deg "
          f"(assuming no fade over the session).")
    print(f"  Allowing a steady fade: {he0['push_mps']:.2f} m/s towards {he0['push_towards_deg']:.0f} deg "
          f"and speed changing {he0['trend_per_10min']['value']:+.3f} m/s per 10 min.")
    if he0.get("confounded_with_time"):
        print("  On this course the direction is tied to the time in the session "
              f"(correlation {he0['direction_time_corr']:.2f}), so a fade and the push cannot be "
              "fully told apart; the two lines bracket the answer.")
    print(f"\nStroke rate vs speed (current and wind removed): speed changes "
          f"{dv['speed_per_spm']:+.4f} m/s and distance per stroke {100 * dv['dps_per_spm']:+.1f} cm "
          f"per extra stroke/min (median rate {dv['median_rate']:.1f}).")
    for b in rs["bins"]:
        print(f"  {b['rate_lo']:4.0f}-{b['rate_hi']:<4.0f} spm  n={b['n']:4d}  speed {b['speed_median']:.2f} m/s")
    print("\nSplits per km:")
    for s in sp:
        print(f"  {s['label']:>12}  pace {s['pace_500m_s']:5.1f} s/500 m  rate {s['rate_spm']:4.1f}  "
              f"dist/stroke {s['dps']:.2f} m")
    return {"d": d, "R": R, "he": he, "he0": he0, "legs": L, "rate_speed": rs, "dps": dv, "splits": sp}


def figure_data(res: dict) -> dict:
    R, L = res["R"], res["legs"]
    t = np.array([r["t"] for r in R])
    v = np.array([r["v_mean"] for r in R])
    vc = np.array([r["v_corrected"] for r in R])
    edges = np.arange(t.min(), t.max() + 30, 30)
    tm, gm, cm = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() >= 5:
            tm.append(float((a + b) / 2 / 60))
            gm.append(float(v[m].mean()))
            cm.append(float(vc[m].mean()))
    r = np.array([x["rate_spm"] for x in R])
    dps = np.array([x["dps"] for x in R])
    db = []
    for lo in range(40, 76, 2):
        m = (r >= lo) & (r < lo + 2)
        if m.sum() >= 10:
            db.append({"rate": float(np.median(r[m])), "dps": float(np.median(dps[m])),
                       "n": int(m.sum())})
    ret = [lg["water_effect"] > 0 for lg in L]
    info = res["d"]["info"]
    return {"title": f"SpeedCoach session {res['d']['start_local']:%Y-%m-%d} "
                     f"({info.get('Model', '')}): {len(res['d']['strokes'])} strokes",
            "time_min": tm, "speed_gps": gm, "speed_corrected": cm,
            "legs": [{"t0_min": lg["t0"] / 60, "t1_min": lg["t1"] / 60, "return": bool(rt)}
                     for lg, rt in zip(L, ret)],
            "rate_bins": res["rate_speed"]["bins"], "dps_bins": db,
            "dps_per_spm": res["dps"]["dps_per_spm"],
            "splits": [{"label": s["label"], "pace_500m_s": s["pace_500m_s"]} for s in res["splits"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--figure", default=None, help="artifact base name to write")
    a = ap.parse_args(argv)
    res = report(a.path)
    if a.figure:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
        from agent_lib.figure_artifacts import emit
        paths = emit(a.figure, data=figure_data(res), plot_func=plot_speedcoach_session,
                     artifacts_dir=os.path.join(os.path.dirname(HERE), "artifacts"))
        print("\nwrote", ", ".join(os.path.basename(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
