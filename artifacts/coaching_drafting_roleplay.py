#!/usr/bin/env python3
"""Standalone, editable recreation of the `coaching_drafting_roleplay` figure.

Loads `coaching_drafting_roleplay.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python coaching_drafting_roleplay.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_roleplay_drafting(data):
    """Per-race differences in race time, role-played coach and API coach (drafting races)."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    groups = [("role-played coach\n(30 fresh races)", "#3b6ea8"), ("API coach\n(12 races)", "#c46a2b")]
    for ax, key, title in ((axes[0], "vs_pack_rule", "coach minus pack rule"),
                           (axes[1], "vs_no_coach", "coach minus no coach")):
        for i, (g, (label, col)) in enumerate(zip(("roleplay", "api"), groups)):
            d = np.asarray(data[g][key], float)
            rng = np.random.default_rng(i)
            x = i + rng.uniform(-0.15, 0.15, d.size)
            ax.scatter(x, d, s=16, color=col, alpha=0.6, lw=0)
            lo, hi = data[g][key + "_ci95"]
            ax.errorbar([i + 0.32], [d.mean()], yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color="k",
                        capsize=4, ms=5)
            ax.text(i + 0.38, d.mean(), f" {d.mean():+.1f} s", va="center", fontsize=8)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_xlim(-0.5, 1.9)
        ax.set_ylabel("race time difference (s); below 0 = coach faster")
        ax.set_title(title, fontsize=10)
    ax = axes[2]
    sp = data["split"]
    names = ["let the pack go\nin the first 5 min", "kept it longer\nor never cued"]
    vals = [sp["early"], sp["late"]]
    for i, v in enumerate(vals):
        d = np.asarray(v, float)
        rng = np.random.default_rng(10 + i)
        ax.scatter(i + rng.uniform(-0.15, 0.15, d.size), d, s=16, color="#3b6ea8", alpha=0.6, lw=0)
        ax.plot([i - 0.25, i + 0.25], [d.mean(), d.mean()], color="k", lw=2)
        ax.text(i + 0.27, d.mean(), f" {d.mean():+.1f} s (n={d.size})", va="center", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.5, 1.9)
    ax.set_title("role play, coach minus no coach\n(exploratory split; the API coach let go early in "
                 f"{data['api_early_let_go']} of 12)", fontsize=9)
    fig.suptitle("Drafting races: the role-played coach did not repeat the API coach's gain (simulated paddler, planted pack)",
                 fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "coaching_drafting_roleplay.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_roleplay_drafting(_data)
    _fig.savefig(os.path.join(_here, "coaching_drafting_roleplay.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "coaching_drafting_roleplay.pdf"), bbox_inches='tight')
    print("wrote coaching_drafting_roleplay.png and coaching_drafting_roleplay.pdf")
