#!/usr/bin/env python3
"""Standalone, editable recreation of the `coaching_open_race` figure.

Loads `coaching_open_race.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python coaching_open_race.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_open_race(data):
    """Open races: race time against the baselines, and what happened to each incident."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    cols = {"incident_rules": "#c46a2b", "no_coach": "#777777"}
    for i, k in enumerate(("incident_rules", "no_coach")):
        d = np.asarray(data["diffs"][k], float)
        rng = np.random.default_rng(i)
        ax.scatter(i + rng.uniform(-0.15, 0.15, d.size), d, s=16, color=cols[k], alpha=0.6, lw=0)
        lo, hi = data["ci95"][k]
        ax.errorbar([i + 0.32], [d.mean()], yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color="k", capsize=4)
        ax.text(i + 0.38, d.mean(), f" {d.mean():+.1f} s", va="center", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rules written for\nthe practice incidents", "no coach"], fontsize=8)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("open coach minus comparison (s); below 0 = coach faster")
    ax.set_title(f"race time, {data['n']} held-out races", fontsize=10)
    ax = axes[1]
    arms = ["no_coach", "autopilot_pod", "incident_rules", "open_coach"]
    names = ["no coach", "fixed rules", "incident rules", "open coach"]
    w = [data["weed_seconds"].get(a) for a in arms]
    ax.barh(range(4), [x or 0 for x in w], color=["#777777", "#999999", "#c46a2b", "#3b6ea8"])
    for i, (x, c) in enumerate(zip(w, [data["weed_cleared"].get(a) for a in arms])):
        ax.text((x or 0) + 10, i, f"{(x or 0):.0f} s; cleared in {c}", va="center", fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("mean seconds raced with weed on the rudder")
    ax.set_title(f"weed (practice kind; {data['n_weed']} races)", fontsize=10)
    ax.set_xlim(0, max(x or 0 for x in w) * 1.6 + 1)
    ax = axes[2]
    rows = [("course changed by radio:\npaddler told the new finish", "told", data["n_course"]),
            ("course changed:\nsprinted for the wrong finish", "overran", data["n_course"]),
            ("pod drifting: cues the faulty\nreading called for", "pod_wrong", None),
            ("pod drifting: cues against\nthe faulty reading", "pod_right", None)]
    y = np.arange(len(rows))
    for j, (a, c) in enumerate((("incident_rules", "#c46a2b"), ("open_coach", "#3b6ea8"))):
        vals = [data["held_out"][a][r[1]] for r in rows]
        ax.barh(y + (0.2 if j == 0 else -0.2), vals, height=0.38, color=c, label=["incident rules", "open coach"][j])
        for yy, v in zip(y + (0.2 if j == 0 else -0.2), vals):
            ax.text(v + 0.2, yy, str(v), va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] + (f" (of {r[2]})" if r[2] else "") for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"held-out kinds (course change {data['n_course']} races, pod drift {data['n_pod']})", fontsize=9)
    fig.suptitle("Open races (simulated paddler, planted incidents): open coach role-played by Claude subagents",
                 fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "coaching_open_race.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_open_race(_data)
    _fig.savefig(os.path.join(_here, "coaching_open_race.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "coaching_open_race.pdf"), bbox_inches='tight')
    print("wrote coaching_open_race.png and coaching_open_race.pdf")
