#!/usr/bin/env python3
"""Standalone, editable recreation of the `coaching_heldout_analyst` figure.

Loads `coaching_heldout_analyst.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python coaching_heldout_analyst.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_analyst(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    labels = {"incident_rules": "the pacer alone\n(rules for the practice incidents)",
              "scheduled_tests": "scheduled tests\n(fixed rules)"}
    pcs = np.array(data["pieces"])
    for i, (k, d) in enumerate(data["diffs"].items()):
        d = np.asarray(d)
        x = i + 0.12 * np.random.default_rng(i).uniform(-1, 1, len(d))
        ax.plot(x[~pcs], d[~pcs], "o", color="#3b6ea8", alpha=0.6, ms=4, label="no test piece" if i == 0 else None)
        ax.plot(x[pcs], d[pcs], "^", color="#c46a2b", alpha=0.7, ms=5, label="analyst ran test pieces" if i == 0 else None)
        m = float(np.mean(d))
        lo, hi = data["ci95"][k]
        ax.errorbar(i + 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o", color="k", capsize=4)
        ax.text(i + 0.38, m, f"{m:+.1f} s", va="center", fontsize=9)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(data["diffs"])))
    ax.set_xticklabels([labels[k] for k in data["diffs"]], fontsize=9)
    ax.set_xlim(-0.5, len(data["diffs"]) - 0.2)
    ax.set_ylabel("analyst + pacer minus comparison (s); below 0 = faster")
    ax.set_title("race time, held-out races", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    names = list(data["pod_gain_error"])
    cols = ["#777777", "#c46a2b", "#6a8f3b", "#3b6ea8"]
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
    with open(os.path.join(_here, "coaching_heldout_analyst.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_analyst(_data)
    _fig.savefig(os.path.join(_here, "coaching_heldout_analyst.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "coaching_heldout_analyst.pdf"), bbox_inches='tight')
    print("wrote coaching_heldout_analyst.png and coaching_heldout_analyst.pdf")
