#!/usr/bin/env python3
"""Standalone, editable recreation of the `oc1_sim_training` figure.

Loads `oc1_sim_training.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python oc1_sim_training.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "oc1_sim_training.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_oc1_training(_data)
    _fig.savefig(os.path.join(_here, "oc1_sim_training.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "oc1_sim_training.pdf"), bbox_inches='tight')
    print("wrote oc1_sim_training.png and oc1_sim_training.pdf")
