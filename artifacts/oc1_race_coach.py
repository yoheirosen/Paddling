#!/usr/bin/env python3
"""Standalone, editable recreation of the `oc1_race_coach` figure.

Loads `oc1_race_coach.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python oc1_race_coach.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_oc1_race_coach(data):
    import numpy as np
    import matplotlib.pyplot as plt
    races = data["races"]
    fig, ax = plt.subplots(3, len(races), figsize=(6.0 * len(races), 9), sharex="col", squeeze=False)
    for j, r in enumerate(races):
        t = np.array(r["t_min"])
        a = ax[0, j]
        a.plot(t, r["reserve_planted"], color="0.3", lw=1.6, label="planted (paddler's effort)")
        a.plot(t, r["reserve_est"], color="C0", lw=1.4, label="estimated (boat speed + hull model)")
        if r.get("exhausted_min") is not None:
            a.axvline(r["exhausted_min"], color="C3", ls="--", lw=1, label="planted reserve runs out")
        a.set_ylim(-0.05, 1.05)
        a.set_ylabel("reserve above critical power")
        a.set_title(r["title"])
        a.legend(fontsize=7, loc="lower left")
        a = ax[1, j]
        a.plot(t, r["fade_pct"], color="C2", lw=1.4)
        a.axhline(-4.0, color="C3", lw=0.8, ls=":")
        a.axhline(0.0, color="0.7", lw=0.8)
        a.set_ylabel("speed vs same rate early in race (%)")
        a = ax[2, j]
        kinds = ["over_pace", "fading", "reserve_conflict", "finish_window"]
        for e in r["screen"]:
            y = kinds.index(e["kind"]) if e["kind"] in kinds else -1
            a.plot(e["t_min"], y, "o", color="C1", ms=6)
        for c in r["cues"]:
            a.axvline(c["t_min"], color="C0", lw=0.8, alpha=0.6)
            a.annotate(c["text"], (c["t_min"], len(kinds) - 0.4), rotation=90, fontsize=6.5,
                       va="top", ha="right", color="C0")
        if r.get("exhausted_min") is not None:
            a.axvline(r["exhausted_min"], color="C3", ls="--", lw=1)
        a.set_yticks(range(len(kinds)))
        a.set_yticklabels(kinds, fontsize=7)
        a.set_ylim(-0.6, len(kinds) + 0.4)
        a.set_xlabel("race time (min)")
        a.set_title("screen events (dots) and spoken cues (lines)", fontsize=9)
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "oc1_race_coach.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_oc1_race_coach(_data)
    _fig.savefig(os.path.join(_here, "oc1_race_coach.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "oc1_race_coach.pdf"), bbox_inches='tight')
    print("wrote oc1_race_coach.png and oc1_race_coach.pdf")
