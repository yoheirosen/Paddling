#!/usr/bin/env python3
"""Standalone, editable recreation of the `coaching_trial_dev_stand_in` figure.

Loads `coaching_trial_dev_stand_in.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python coaching_trial_dev_stand_in.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_coaching_trial(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.25], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    names = data["comparison_names"]
    for i, n in enumerate(names):
        d = np.array(data["paired_d"][n], float)
        jitter = (np.arange(len(d)) % 7 - 3) * 0.04
        ax.scatter(d, i + jitter, s=12, color="0.55", zorder=2)
        c = data["comparisons"][n]
        if c.get("ci95"):
            ax.plot(c["ci95"], [i, i], color="C0", lw=3, zorder=3)
        ax.plot([c["mean_d"]], [i], "o", color="C0", ms=7, zorder=4)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([data["labels"].get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel(f"{data['agent_label']} minus comparator, race time (s)\n(negative = coach faster; dot = seed, bar = 95% CI of the mean)")
    ax.set_title(data["title"], fontsize=10)
    ex = data["example"]
    t_min = None
    axes = [fig.add_subplot(gs[k, 1]) for k in range(3)]
    for arm, tr in ex["traces"].items():
        t_min = np.array(tr["race_s"], float) / 60.0
        axes[0].plot(t_min, tr["reserve_true"], lw=1.3, label=f"{data['labels'].get(arm, arm)} ({tr['race_time_min']:.2f} min)")
        axes[1].plot(t_min, tr["effort"], lw=1.1)
        axes[2].plot(t_min, np.array(tr["rate"]) - np.array(tr["rate_best"]), lw=1.1)
        for c in tr["cues"]:
            axes[0].axvline(c / 60.0, color=axes[0].lines[-1].get_color(), lw=0.5, alpha=0.5)
    axes[0].set_ylabel("true reserve left")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].set_title(f"example race (seed {ex['seed']}): thin lines = cues spoken", fontsize=9)
    axes[1].set_ylabel("effort (x critical power)")
    axes[2].set_ylabel("rate minus best rate (spm)")
    axes[2].axhline(0, color="0.5", lw=0.8)
    axes[2].set_xlabel("race time (min)")
    fig.suptitle(data["suptitle"], fontsize=11)
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "coaching_trial_dev_stand_in.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_coaching_trial(_data)
    _fig.savefig(os.path.join(_here, "coaching_trial_dev_stand_in.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "coaching_trial_dev_stand_in.pdf"), bbox_inches='tight')
    print("wrote coaching_trial_dev_stand_in.png and coaching_trial_dev_stand_in.pdf")
