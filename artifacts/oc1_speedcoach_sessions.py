#!/usr/bin/env python3
"""Standalone, editable recreation of the `oc1_speedcoach_sessions` figure.

Loads `oc1_speedcoach_sessions.data.json` (the source data) and rebuilds the figure as PNG + PDF.
Edit the plotting function below and re-run:  python oc1_speedcoach_sessions.py
Self-contained — needs only numpy + matplotlib.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_speedcoach_sessions(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    cols = plt.get_cmap("tab10")
    S = data["sessions"]
    a = ax[0, 0]
    w = data["windows"]
    for k in range(len(S)):
        m = [i for i, s in enumerate(w["session"]) if s == k]
        a.scatter([w["rate"][i] for i in m], [w["speed"][i] for i in m], s=4, alpha=0.25,
                  color=cols(k % 10), lw=0)
    env = data["envelope"]
    a.plot([p["rate"] for p in env["points"]], [p["speed"] for p in env["points"]], "ko", ms=4,
           label=f"best (top {100 - env['pct']:.0f} % of one-minute windows)")
    rr = np.linspace(env["rate_min"], env["rate_max"], 50)
    a.plot(rr, np.polyval(env["coef"], rr), "k-", lw=1.4)
    a.plot([p["rate"] for p in env["points"]], [p["speed_median"] for p in env["points"]], "--",
           color="0.4", lw=1, label="typical (median)")
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("speed, current and wind removed (m/s)")
    a.set_title("Best speed at each stroke rate, all sessions")
    a.legend(fontsize=7, loc="upper left")
    a = ax[0, 1]
    for k, s in enumerate(S):
        if s["best"]:
            d = sorted(s["best"], key=float)
            a.plot([float(x) / 60 for x in d], [s["best"][x] for x in d], "-", color=cols(k % 10),
                   lw=0.8, alpha=0.5)
    pb = data["pooled_best"]
    a.plot([p["seconds"] / 60 for p in pb], [p["power"] for p in pb], "ko", ms=5,
           label="best of all sessions")
    cp = data["cp"]
    tt = np.linspace(2.0, 45, 100)
    a.plot(tt, cp["cp_w"] + cp["w_prime_j"] / (tt * 60), "k-", lw=1.2,
           label=f"fit: CP {cp['cp_w']:.0f} W + reserve {cp['w_prime_j'] / 1000:.1f} kJ / time")
    a.axhline(cp["cp_w"], color="0.6", lw=0.8, ls=":")
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 20, 40])
    a.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    a.set_xlabel("duration of continuous effort (min)")
    a.set_ylabel("best average power to the boat (W)")
    a.set_title("Best efforts and critical power")
    a.legend(fontsize=7)
    a = ax[1, 0]
    for k, s in enumerate(S):
        ln = s.get("line")
        if ln:
            x = np.array([ln["rate_lo"], ln["rate_hi"]])
            a.plot(x, ln["intercept"] + ln["slope"] * x, color=cols(k % 10), lw=1.6,
                   label=s["date"][5:])
    re = data["rate_effect"]["within"]
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("speed, current and wind removed (m/s)")
    a.set_title(f"Within each session: {re['speed_per_spm'] * 100:+.1f} cm/s per extra stroke/min\n"
                f"(95 % {re['speed_ci'][0] * 100:+.1f} to {re['speed_ci'][1] * 100:+.1f}); "
                f"distance per stroke {re['dps_per_spm'] * 100:+.1f} cm", fontsize=10)
    a.legend(fontsize=6, ncol=3, loc="upper left")
    a = ax[1, 1]
    x = np.arange(len(S))
    pct = [s.get("pct_best_median") or 0 for s in S]
    bars = a.bar(x, pct, color=[cols(k % 10) for k in range(len(S))], alpha=0.7)
    for b, s in zip(bars, S):
        if s["place"] != 0:
            b.set_hatch("//")
    a.set_ylim(0, 110)
    a.set_ylabel("median % of best power at the rate paddled")
    a.set_xticks(x)
    a.set_xticklabels([s["date"][5:] for s in S], rotation=45, fontsize=7)
    a2 = a.twinx()
    a2.plot(x, [s["best"].get("300", np.nan) for s in S], "kD", ms=5, label="best 5-min power")
    a2.set_ylabel("best 5-minute power (W)")
    a2.legend(fontsize=7, loc="lower right")
    a.set_title("Session by session (hatched: other water)")
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "oc1_speedcoach_sessions.data.json")) as _f:
        _data = json.load(_f)
    _fig = plot_speedcoach_sessions(_data)
    _fig.savefig(os.path.join(_here, "oc1_speedcoach_sessions.png"), dpi=140, bbox_inches='tight')
    _fig.savefig(os.path.join(_here, "oc1_speedcoach_sessions.pdf"), bbox_inches='tight')
    print("wrote oc1_speedcoach_sessions.png and oc1_speedcoach_sessions.pdf")
