#!/usr/bin/env python3
"""Standalone, editable recreation of the `oc1_speedcoach_session` figure.

Loads `oc1_speedcoach_session.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python oc1_speedcoach_session.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "oc1_speedcoach_session.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_speedcoach_session(_data)
    _fig.savefig(os.path.join(_here, "oc1_speedcoach_session.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "oc1_speedcoach_session.pdf"), bbox_inches='tight')
    print("wrote oc1_speedcoach_session.png and oc1_speedcoach_session.pdf")
