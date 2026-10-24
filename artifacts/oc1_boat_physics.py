#!/usr/bin/env python3
"""Standalone, editable recreation of the `oc1_boat_physics` figure.

Loads `oc1_boat_physics.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python oc1_boat_physics.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "oc1_boat_physics.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_oc1_boat_physics(_data)
    _fig.savefig(os.path.join(_here, "oc1_boat_physics.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "oc1_boat_physics.pdf"), bbox_inches='tight')
    print("wrote oc1_boat_physics.png and oc1_boat_physics.pdf")
