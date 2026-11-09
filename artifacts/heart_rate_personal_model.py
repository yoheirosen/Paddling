#!/usr/bin/env python3
"""Standalone, editable recreation of the `heart_rate_personal_model` figure.

Loads `heart_rate_personal_model.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python heart_rate_personal_model.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_heart_rate(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, shape in zip(axes[:2], ("typical", "compressed")):
        e = data["examples"][shape]
        xs = [float(k.rstrip("%")) / 100.0 for k in e["hr_at"]]
        ax.plot(xs, list(e["hr_at"].values()), "o-", color="#3b6ea8", label="personal model (fitted)")
        px, py = zip(*e["points"])
        ax.plot(px, py, "s", color="k", ms=5, label="measured, held-out steady efforts\n(the slow creep taken off)")
        for nm, lo, hi in data["textbook_zones"]:
            ax.axhspan(lo * e["hr_max"], hi * e["hr_max"], color="#c46a2b", alpha=0.05 + 0.05 * (lo > 0.75))
            ax.text(0.405, (lo + hi) / 2 * e["hr_max"], f"textbook {nm}", fontsize=6.5, va="center", color="#c46a2b")
        if e.get("separates_up_to", 2.0) < 1.02:
            ax.axvspan(e["separates_up_to"], 1.02, color="#999999", alpha=0.15)
            ax.text(e["separates_up_to"] + 0.01, ax.get_ylim()[0] + 4, "heart rate cannot\ntell efforts apart", fontsize=7, color="#555555")
        ax.set_xlim(0.4, 1.02)
        ax.set_xlabel("effort (share of critical power)")
        ax.set_ylabel("heart rate (bpm)")
        ax.set_title(f"a paddler with a {shape} heart-rate response", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax = axes[2]
    shapes = ["typical", "compressed"]
    w = 0.26
    x = np.arange(len(shapes))
    S = data["summary"]
    tb = [100 * S[s]["textbook_zone_right"] for s in shapes]
    th = [100 * S[s]["threshold_zone_right"] for s in shapes]
    pm = [100 * S[s]["personal_zone_right"] for s in shapes]
    ab = [100 * S[s]["personal_says_cannot_tell"] for s in shapes]
    ax.bar(x - w, tb, w, color="#c46a2b", label="textbook zones (share of maximum)")
    ax.bar(x, th, w, color="#e0a458", label="zones from heart rate at threshold")
    ax.bar(x + w, pm, w, color="#3b6ea8", label="personal model: zone named right")
    ax.bar(x + w, ab, w, bottom=pm, color="#bbbbbb", label="personal model: says heart rate cannot tell")
    for i in range(len(shapes)):
        ax.text(x[i] - w, tb[i] + 1, f"{tb[i]:.0f}%", ha="center", fontsize=8)
        ax.text(x[i], th[i] + 1, f"{th[i]:.0f}%", ha="center", fontsize=8)
        ax.text(x[i] + w, pm[i] / 2, f"{pm[i]:.0f}%", ha="center", fontsize=8, color="white")
        if ab[i] > 3:
            ax.text(x[i] + w, pm[i] + ab[i] / 2, f"{ab[i]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} response\n({S[s]['paddlers']} paddlers)" for s in shapes], fontsize=9)
    ax.set_ylabel("held-out steady efforts (%)")
    ax.set_ylim(0, 112)
    ax.set_title("which effort zone a heart rate means", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(data.get("title", ""), fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "heart_rate_personal_model.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_heart_rate(_data)
    _fig.savefig(os.path.join(_here, "heart_rate_personal_model.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "heart_rate_personal_model.pdf"), bbox_inches='tight')
    print("wrote heart_rate_personal_model.png and heart_rate_personal_model.pdf")
