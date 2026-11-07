#!/usr/bin/env python3
"""Standalone, editable recreation of the `coaching_heldout_experimenter` figure.

Loads `coaching_heldout_experimenter.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python coaching_heldout_experimenter.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_experimenter(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    ax = axes[0]
    labels = {"scheduled_tests": "scheduled tests\n(fixed rules)", "incident_rules": "rules written for\nthe practice incidents"}
    for i, (k, d) in enumerate(data["diffs"].items()):
        x = i + 0.12 * np.random.default_rng(i).uniform(-1, 1, len(d))
        ax.plot(x, d, "o", color="#c46a2b" if k == "incident_rules" else "#6a8f3b", alpha=0.6, ms=4)
        m = float(np.mean(d))
        lo, hi = data["ci95"][k]
        ax.errorbar(i + 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o", color="k", capsize=4)
        ax.text(i + 0.38, m, f"{m:+.1f} s", va="center", fontsize=9)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(data["diffs"])))
    ax.set_xticklabels([labels[k] for k in data["diffs"]], fontsize=9)
    ax.set_xlim(-0.5, len(data["diffs"]) - 0.2)
    ax.set_ylabel("coach minus comparison (s); below 0 = coach faster")
    ax.set_title("race time, held-out races", fontsize=10)
    ax = axes[1]
    names = list(data["cp_error_mean"])
    cols = ["#777777", "#c46a2b", "#6a8f3b", "#3b6ea8"]
    for i, nm in enumerate(names):
        d = [x for x in data["cp_error_mean"][nm] if x is not None]
        x = i + 0.12 * np.random.default_rng(10 + i).uniform(-1, 1, len(d))
        ax.plot(x, d, "o", color=cols[i], alpha=0.5, ms=4)
        ax.plot([i - 0.25, i + 0.25], [np.mean(d)] * 2, color="k", lw=2)
        ax.text(i + 0.28, np.mean(d), f"{np.mean(d):.1f}", va="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ax.set_ylabel("mean critical-power error over the race (% of the truth)")
    ax.set_title("how well each arm knew the paddler's critical power", fontsize=10)
    ax = axes[2]
    for i, (nm, v) in enumerate(data["cp_error"].items()):
        ax.plot([0, 1, 2], v, "o-", color=cols[i + 1], label=nm)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["start", "halfway", "three quarters"], fontsize=9)
    ax.set_ylabel("critical power error (% of the truth)")
    ax.set_title("the computer's critical power through the race", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    ax = axes[3]
    names = list(data.get("pod_gain_error", {}))
    for i, nm in enumerate(names):
        d = [100.0 * x for x in data["pod_gain_error"][nm]]
        if not d:
            continue
        x = i + 0.12 * np.random.default_rng(20 + i).uniform(-1, 1, len(d))
        ax.plot(x, d, "o", color=cols[i], alpha=0.5, ms=4)
        ax.plot([i - 0.25, i + 0.25], [np.mean(d)] * 2, color="k", lw=2)
        ax.text(i + 0.28, np.mean(d), f"{np.mean(d):.0f} %", va="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ax.set_ylabel("error of the paddle sensor's correction (% of effort)")
    ax.set_title("races where the paddle sensor drifted", fontsize=10)
    fig.suptitle(data.get("title", ""), fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "coaching_heldout_experimenter.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_experimenter(_data)
    _fig.savefig(os.path.join(_here, "coaching_heldout_experimenter.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "coaching_heldout_experimenter.pdf"), bbox_inches='tight')
    print("wrote coaching_heldout_experimenter.png and coaching_heldout_experimenter.pdf")
