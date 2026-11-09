#!/usr/bin/env python3
"""The personal heart-rate model against the textbook zones, on simulated paddlers whose
heart rate does or does not follow the textbook (``sim.twin.heart_rate_profile``: about 2 in
5 have a compressed response).  No model calls.

Each paddler does one step-test session on the simulator (warm-up, then five 5-minute
steps at 55-100 % of critical power, out of order so a slow creep cannot pass for a rise
with effort).  The personal model is fitted from that session alone
(``paddling_expt.heart_rate.fit_personal_model``) and then asked, at steady efforts it was
not fitted on, which zone a heart rate means.  The textbook reads the zone from the share of
the maximum heart rate (taken as the highest rate the person reached, its best case).

    python experiments/heart_rate_model.py run --paddlers 60
    python experiments/heart_rate_model.py summary --figure
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from paddling_expt.heart_rate import TEXTBOOK_ZONES, fit_personal_model, zone_of_share   # noqa: E402
from paddling_expt.sim.twin import STEP_TEST_SHARES, draw_traits, twin_step_test          # noqa: E402

OUT = os.path.join(ROOT, "results", "heart_rate")
STEPS = STEP_TEST_SHARES                     # out of order: a creep then cannot pass for a rise with effort
TEST_SHARES = (0.50, 0.62, 0.72, 0.84, 0.95)        # held-out steady efforts (shares of critical power)
# Zones as a share of the heart rate at threshold (Friel's cycling bands, merged to the same five
# zones); the threshold heart rate is the settled rate in the step at critical power.
THRESHOLD_ZONES = [("recovery", 0.0, 0.81), ("endurance", 0.81, 0.90), ("tempo", 0.90, 0.94),
                   ("threshold", 0.94, 1.02), ("above threshold", 1.02, 9.0)]


def one(seed: int) -> Dict[str, Any]:
    tr = draw_traits(seed)
    rows, tw = twin_step_test(tr, seed=seed)
    cp = tr.paddler.critical_power_w
    model = fit_personal_model(rows)
    hp = tw._hr_par
    # held-out steady efforts: a separate session, the settled heart rate in each last 2 minutes
    trows, _ = twin_step_test(tr, seed=seed + 100_000, shares=TEST_SHARES, hr_par=hp)
    t = np.array([r["t"] for r in trows])
    hr = np.array([r["heart_rate_bpm"] for r in trows])
    hr_max = float(hp["max"])           # the textbook's best case: the person's true maximum
    # the threshold heart rate: the settled rate over the last 2 minutes of the step at critical power
    ft = np.array([r["t"] for r in rows])
    fh = np.array([r["heart_rate_bpm"] for r in rows])
    i_cp = STEPS.index(1.00)
    end_cp = 300.0 + 300.0 * (i_cp + 1)
    hr_thr = float(fh[(ft >= end_cp - 120.0) & (ft < end_cp)].mean())
    out = {"seed": seed, "shape": hp.get("shape", "typical"), "cp_w": round(cp, 1), "hr_max_seen": round(hr_max), "hr_threshold": round(hr_thr, 1),
           "model": model.summary(cp_w=cp) if model else None, "points": []}
    for i, s in enumerate(TEST_SHARES):
        end = 300.0 + 300.0 * (i + 1)
        m = (t >= end - 120.0) & (t < end)
        h = float(hr[m].mean())
        truth = zone_of_share(s)
        tb_share = h / hr_max
        tb = next((j for j, (_, lo, hi) in enumerate(TEXTBOOK_ZONES) if lo <= tb_share < hi),
                  0 if tb_share < TEXTBOOK_ZONES[0][1] else len(TEXTBOOK_ZONES) - 1)
        th = next(j for j, (_, lo, hi) in enumerate(THRESHOLD_ZONES) if lo <= h / hr_thr < hi)
        e = model.effort_at(h, minutes_in=(end - 60.0) / 60.0) if model else None
        sep = model.separates_up_to(cp) if model else 0.0
        # the personal model says "cannot tell from heart rate" where it no longer separates efforts
        pm = None if (e is None or e / cp > sep + 0.05) else zone_of_share(e / cp)
        mins = (end - 60.0) / 60.0
        h_nc = h - (model.creep_per_10min * mins / 10.0 if model else 0.0)
        out["points"].append({"share_of_cp": s, "hr": round(h, 1), "hr_creep_removed": round(h_nc, 1), "true_zone": truth, "textbook_zone": tb, "threshold_zone": th,
                              "personal_zone": pm,
                              "personal_effort_error_pct": None if e is None else round(100.0 * (e - s * cp) / cp, 1)})
    return out


def run(n: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = [one(s) for s in range(1, n + 1)]
    with open(os.path.join(OUT, "step_tests.json"), "w") as f:
        json.dump(rows, f)
    print(f"{len(rows)} paddlers")


def summary(figure: bool) -> Dict[str, Any]:
    rows = json.load(open(os.path.join(OUT, "step_tests.json")))
    out: Dict[str, Any] = {}
    for shape in ("typical", "compressed"):
        R = [r for r in rows if r["shape"] == shape and r["model"]]
        pts = [p for r in R for p in r["points"]]
        tb = float(np.mean([p["textbook_zone"] == p["true_zone"] for p in pts]))
        pm = float(np.mean([p["personal_zone"] == p["true_zone"] for p in pts]))
        pm_abst = float(np.mean([p["personal_zone"] is None for p in pts]))
        pm_wrong = float(np.mean([p["personal_zone"] is not None and p["personal_zone"] != p["true_zone"] for p in pts]))
        tb_wrong = 1.0 - tb
        tb1 = float(np.mean([abs(p["textbook_zone"] - p["true_zone"]) <= 1 for p in pts]))
        th = float(np.mean([p["threshold_zone"] == p["true_zone"] for p in pts]))
        errs = [abs(p["personal_effort_error_pct"]) for p in pts if p["personal_effort_error_pct"] is not None]
        sep = [r["model"]["heart_rate_separates_efforts_up_to_share_of_cp"] for r in R]
        out[shape] = {"paddlers": len(R), "textbook_zone_right": round(tb, 3), "personal_zone_right": round(pm, 3),
                      "personal_says_cannot_tell": round(pm_abst, 3), "personal_zone_wrong": round(pm_wrong, 3),
                      "textbook_zone_wrong": round(tb_wrong, 3), "threshold_zone_right": round(th, 3),
                      "textbook_within_one_zone": round(tb1, 3),
                      "personal_effort_error_pct_median": round(float(np.median(errs)), 1) if errs else None,
                      "hr_separates_up_to_share_of_cp_median": round(float(np.median(sep)), 2)}
        print(f"{shape:10s} paddlers {len(R)}: zone named right — textbook {100 * tb:.0f} % (wrong {100 * tb_wrong:.0f} %), "
              f"threshold-anchored {100 * th:.0f} %, "
              f"personal model {100 * pm:.0f} % (wrong {100 * pm_wrong:.0f} %, says it cannot tell {100 * pm_abst:.0f} %); "
              f"personal effort error (median) {out[shape]['personal_effort_error_pct_median']} % "
              f"of critical power; heart rate separates efforts up to {out[shape]['hr_separates_up_to_share_of_cp_median']} "
              f"of critical power (median)")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    if figure:
        print("figure:", _figure(rows, out))
    return out


def _figure(rows, out):
    from agent_lib.figure_artifacts import emit
    ex = {}
    for shape in ("typical", "compressed"):
        r = next(r for r in rows if r["shape"] == shape and r["model"])
        ex[shape] = {"hr_at": r["model"]["hr_at_share_of_cp"], "hr_max": r["hr_max_seen"],
                     "separates_up_to": r["model"]["heart_rate_separates_efforts_up_to_share_of_cp"],
                     "points": [(p["share_of_cp"], p["hr_creep_removed"]) for p in r["points"]]}
    data = {"title": "A personal heart-rate model against textbook and threshold-anchored zones "
                     "(simulated paddlers; planted physiology)",
            "examples": ex, "summary": out,
            "textbook_zones": [[nm, lo, hi] for nm, lo, hi in TEXTBOOK_ZONES]}
    return emit("heart_rate_personal_model", data=data, plot_func=plot_heart_rate,
                artifacts_dir=os.path.join(ROOT, "artifacts"))


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--paddlers", type=int, default=60)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.paddlers)
    else:
        summary(a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
