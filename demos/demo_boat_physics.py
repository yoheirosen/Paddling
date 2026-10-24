#!/usr/bin/env python3
"""Hull-shape physics: what the ama load and the wobble cost in boat speed.

    python "paddling/demos/demo_boat_physics.py" [--figure NAME]

1. Resistance of the typical OC1 (at several ama loads) and surf ski against
   speed, split into skin friction, wave-making and air.
2. A simulated OC1 race where the paddler leans more onto the ama as they tire:
   the ama load read from the head unit's roll (against the on-shore level
   reference) against the planted load, and what it costs in seconds per 500 m.
3. A simulated surf ski race with bow wobble: side-slip read from the gyro
   against the truth, and its hydrodynamic cost (small) next to the data-driven
   estimate of what wobble costs the paddler.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from paddling_expt.boats import OC1_TYPICAL, SURFSKI_TYPICAL            # noqa: E402
from paddling_expt.boatspeed import physics_summary, stroke_physics, wobble_cost_from_data  # noqa: E402
from paddling_expt.fusion import OC1Engine                               # noqa: E402
from paddling_expt.headunit import HeadConfig                            # noqa: E402
from paddling_expt.hydro import roll_table                               # noqa: E402
from paddling_expt.scoring import true_time                              # noqa: E402
from paddling_expt.sim.paddler import PaddlerParams                      # noqa: E402
from paddling_expt.sim.scenarios import steady_race                      # noqa: E402


def plot_oc1_boat_physics(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = ax[0, 0]
    U = np.array(data["speeds"])
    a.plot(U, data["ski_total"], color="k", lw=1.8, label="surf ski, total")
    for name, col in (("ski_friction", "C0"), ("ski_waves", "C1"), ("ski_air", "C2")):
        a.plot(U, data[name], color=col, lw=1, ls="--", label="surf ski, " + name.split("_")[1])
    for L, col in zip(data["oc1_loads"], ("C3", "C4", "C5")):
        a.plot(U, data["oc1_total"][str(L)], color=col, lw=1.4, label=f"OC1, {L} kg on ama")
    a.set_xlabel("speed through the water (m/s)")
    a.set_ylabel("resistance (N)")
    a.set_title("Resistance from hull shape (typical boats)")
    a.legend(fontsize=7)
    a = ax[0, 1]
    t = np.array(data["oc1_t_min"])
    a.plot(t, data["oc1_load_true"], color="0.3", label="planted ama load")
    a.plot(t, data["oc1_load_est"], color="C3", label="from roll (30 strokes)")
    a.set_xlabel("time (min)")
    a.set_ylabel("weight on the ama (kg)")
    a2 = a.twinx()
    a2.plot(t, data["oc1_cost_s500"], color="C4", lw=1, alpha=0.7)
    a2.set_ylabel("cost vs a light ama (s per 500 m)", color="C4")
    a.set_title("OC1: leaning on the ama as the paddler tires")
    a.legend(fontsize=7, loc="upper left")
    a = ax[1, 0]
    a.plot(data["roll_curve_load"], data["roll_curve_deg"], color="C3")
    a.set_xlabel("weight on the ama (kg)")
    a.set_ylabel("static roll (deg, + lifts the ama)")
    a.set_title(f"Roll vs ama load: about {data['deg_per_kg']:.2f} deg per kg")
    a.axhline(0, color="0.8", lw=0.8)
    a = ax[1, 1]
    a.scatter(data["ski_beta_true"], data["ski_beta_est"], s=5, alpha=0.4, color="C0")
    lim = max(max(data["ski_beta_true"]), max(data["ski_beta_est"]))
    a.plot([0, lim], [0, lim], color="0.6", lw=0.8)
    a.set_xlabel("true side-slip, rms per stroke (deg)")
    a.set_ylabel("estimated from the gyro (deg)")
    a.set_title(f"Surf ski side-slip: drag cost {data['ski_yaw_cost_s500']:.2f} s per 500 m")
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


def _run(kind, seed, dur, **kw):
    s = steady_race(seed=seed, duration_s=dur, kind=kind, **kw)
    e = OC1Engine(head_cfg=HeadConfig(craft=s.craft, total_kg=s.total_kg)).feed_all(s.records)
    rows = e.stroke_table()
    stroke_physics(rows, s.craft, s.total_kg)
    return s, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figure", default=None)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args(argv)

    U = np.round(np.linspace(1.5, 5.0, 15), 3)
    ski = [SURFSKI_TYPICAL.resistance(u, 89.0) for u in U]
    loads = (3, 10, 20)
    oc1 = {str(L): [OC1_TYPICAL.resistance(u, 93.0, ama_load_kg=L)["total"] for u in U]
           for L in loads}
    print("Resistance at 3.3 m/s (11.9 km/h):")
    r = SURFSKI_TYPICAL.resistance(3.3, 89.0)
    print(f"  surf ski: {r['total']:.1f} N = friction {r['hull_friction']:.1f} + waves "
          f"{r['waves']:.1f} + air {r['air']:.1f}; {r['total'] * 3.3:.0f} W to the boat")
    for L in loads:
        r = OC1_TYPICAL.resistance(3.3, 93.0, ama_load_kg=L)
        print(f"  OC1, {L:2d} kg on the ama: {r['total']:.1f} N ({r['total'] * 3.3:.0f} W)")

    s, rows = _run("oc1", a.seed, 600, effort=1.0, paddler=PaddlerParams(work_capacity_j=3.5e5))
    R = [x for x in rows if x.get("ama_load_kg") is not None][40:]
    tt = true_time(s.sensors, [x["t"] for x in R])
    St = s.truth.strokes
    idx = np.array([int(np.argmin(np.abs(St["catch_t"] - (x - 0.34)))) for x in tt])
    tru = np.convolve(St["ama_load"][idx], np.ones(30) / 30, mode="full")[:len(idx)]
    est = np.array([x["ama_load_kg"] for x in R])
    cost = np.array([x["ama_cost_s500"] for x in R])
    k = 30
    print("\nOC1 race, paddler leaning onto the ama as they tire:")
    print(f"  ama load first 2 min {np.median(est[k:k + 120]):.1f} kg (true {tru[k:k + 120].mean():.1f}), "
          f"last 2 min {np.median(est[-120:]):.1f} kg (true {tru[-120:].mean():.1f})")
    print(f"  30-stroke estimate within {np.median(np.abs(est[k:] - tru[k:])):.2f} kg of the truth (median)")
    print(f"  cost against 3 kg on the ama: {np.median(cost[k:k + 120]):.2f} -> "
          f"{np.median(cost[-120:]):.2f} s per 500 m")
    print("  ", physics_summary(R))

    s2, rows2 = _run("surfski", a.seed, 600)
    R2 = [x for x in rows2 if x.get("boat_power_w") is not None][40:]
    tt2 = true_time(s2.sensors, [x["t"] for x in R2])
    St2 = s2.truth.strokes
    idx2 = np.array([int(np.argmin(np.abs(St2["catch_t"] - (x - 0.34)))) for x in tt2])
    b_est = np.array([x["beta_rms_deg"] for x in R2])
    b_true = np.degrees(St2["beta_rms"][idx2])
    summ = physics_summary(R2)
    w = wobble_cost_from_data(R2)
    print("\nSurf ski race with bow wobble:")
    print(f"  side-slip rms {np.median(b_est):.2f} deg (true {np.median(b_true):.2f}); "
          f"its drag costs {summ['yaw_cost_s500']:.3f} s per 500 m (hull model)")
    print(f"  roll wobble {summ['roll_wobble_deg']:.2f} deg; measured cost from the session: "
          f"{w.get('dps_share_lost_at_median', float('nan')) * 100:+.2f} % distance per stroke "
          f"({'conclusive' if w.get('conclusive') else 'not conclusive in one session'})")

    if a.figure:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
        from agent_lib.figure_artifacts import emit
        rt = roll_table(OC1_TYPICAL, 93.0)
        sl = np.polyfit(rt["loads"][2:21], np.degrees(rt["roll"][2:21]), 1)[0]
        t_min = (np.array([x["t"] for x in R]) - R[0]["t"]) / 60.0
        sub = slice(k, None, 3)
        data = {"title": "Hull-shape physics: ama load (OC1) and wobble (surf ski), simulated against planted truth",
                "speeds": U.tolist(),
                "ski_total": [x["total"] for x in ski], "ski_friction": [x["hull_friction"] for x in ski],
                "ski_waves": [x["waves"] for x in ski], "ski_air": [x["air"] for x in ski],
                "oc1_loads": list(loads), "oc1_total": oc1,
                "oc1_t_min": t_min[sub].round(3).tolist(), "oc1_load_true": tru[sub].round(3).tolist(),
                "oc1_load_est": np.array([np.median(est[max(0, i - k + 1):i + 1])
                                          for i in range(len(est))])[sub].round(3).tolist(),
                "oc1_cost_s500": np.convolve(cost, np.ones(k) / k, mode="full")[:len(cost)][sub].round(3).tolist(),
                "roll_curve_load": rt["loads"].tolist(),
                "roll_curve_deg": np.degrees(rt["roll"]).round(4).tolist(), "deg_per_kg": abs(float(sl)),
                "ski_beta_true": b_true[::2].round(4).tolist(), "ski_beta_est": b_est[::2].round(4).tolist(),
                "ski_yaw_cost_s500": summ["yaw_cost_s500"]}
        paths = emit(a.figure, data=data, plot_func=plot_oc1_boat_physics,
                     artifacts_dir=os.path.join(os.path.dirname(HERE), "artifacts"))
        print("\nwrote", ", ".join(os.path.basename(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
