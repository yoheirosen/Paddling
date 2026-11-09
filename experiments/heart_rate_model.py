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

from paddling_expt.heart_rate import TEXTBOOK_ZONES, fit_personal_model          # noqa: E402
from paddling_expt.sim.twin import PaddlerTwin, RaceSetup, draw_traits         # noqa: E402

OUT = os.path.join(ROOT, "results", "heart_rate")
STEPS = (0.70, 0.55, 0.90, 0.80, 1.00)      # out of order: a creep then cannot pass for a rise with effort
ZONE_BANDS = [("recovery", 0.0, 0.55), ("endurance", 0.55, 0.75), ("tempo", 0.75, 0.90),
              ("threshold", 0.90, 1.05), ("above threshold", 1.05, 9.0)]
TEST_SHARES = (0.50, 0.62, 0.72, 0.84, 0.95)        # held-out steady efforts (shares of critical power)


def _zone(share: float) -> int:
    return next(i for i, (_, lo, hi) in enumerate(ZONE_BANDS) if lo <= share < hi)


def _session(traits, seed: int, shares) -> PaddlerTwin:
    plan = [(300.0, 56.0, 0.5)] + [(300.0, traits.habit_rate, s) for s in shares]
    tw = PaddlerTwin(traits, RaceSetup(current_mps=0.0, plan=plan), seed=seed)
    tw.enable_heart_rate(profiles=True)
    return tw


def one(seed: int) -> Dict[str, Any]:
    tr = draw_traits(seed)
    tw = _session(tr, seed, STEPS)
    rows = tw.run_alone(step_s=10.0)
    cp = tr.paddler.critical_power_w
    model = fit_personal_model(rows)
    hp = tw._hr_par
    # held-out steady efforts: a separate session, the settled heart rate in each last 2 minutes
    test = _session(tr, seed + 100_000, TEST_SHARES)
    trows = test.run_alone(step_s=10.0)
    t = np.array([r["t"] for r in trows])
    hr = np.array([r["heart_rate_bpm"] for r in trows])
    hr_max = float(hp["max"])           # the textbook's best case: the person's true maximum
    out = {"seed": seed, "shape": hp.get("shape", "typical"), "cp_w": round(cp, 1), "hr_max_seen": round(hr_max),
           "model": model.summary(cp_w=cp) if model else None, "points": []}
    for i, s in enumerate(TEST_SHARES):
        end = 300.0 + 300.0 * (i + 1)
        m = (t >= end - 120.0) & (t < end)
        h = float(hr[m].mean())
        truth = _zone(s)
        tb_share = h / hr_max
        tb = next((j for j, (_, lo, hi) in enumerate(TEXTBOOK_ZONES) if lo <= tb_share < hi),
                  0 if tb_share < TEXTBOOK_ZONES[0][1] else len(TEXTBOOK_ZONES) - 1)
        e = model.effort_at(h, minutes_in=(end - 60.0) / 60.0) if model else None
        sep = model.separates_up_to(cp) if model else 0.0
        # the personal model says "cannot tell from heart rate" where it no longer separates efforts
        pm = None if (e is None or e / cp > sep + 0.05) else _zone(e / cp)
        out["points"].append({"share_of_cp": s, "hr": round(h, 1), "true_zone": truth, "textbook_zone": tb,
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
        errs = [abs(p["personal_effort_error_pct"]) for p in pts if p["personal_effort_error_pct"] is not None]
        sep = [r["model"]["heart_rate_separates_efforts_up_to_share_of_cp"] for r in R]
        out[shape] = {"paddlers": len(R), "textbook_zone_right": round(tb, 3), "personal_zone_right": round(pm, 3),
                      "personal_says_cannot_tell": round(pm_abst, 3), "personal_zone_wrong": round(pm_wrong, 3),
                      "textbook_zone_wrong": round(tb_wrong, 3),
                      "textbook_within_one_zone": round(tb1, 3),
                      "personal_effort_error_pct_median": round(float(np.median(errs)), 1) if errs else None,
                      "hr_separates_up_to_share_of_cp_median": round(float(np.median(sep)), 2)}
        print(f"{shape:10s} paddlers {len(R)}: zone named right — textbook {100 * tb:.0f} % (wrong {100 * tb_wrong:.0f} %), "
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
                     "points": [(p["share_of_cp"], p["hr"]) for p in r["points"]]}
    data = {"title": "A personal heart-rate model against the textbook zones (simulated paddlers; planted physiology)",
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
        ax.plot(px, py, "s", color="k", ms=5, label="measured, steady efforts (held out)")
        for nm, lo, hi in data["textbook_zones"]:
            ax.axhspan(lo * e["hr_max"], hi * e["hr_max"], color="#c46a2b", alpha=0.06 + 0.04 * (lo > 0.75))
            ax.text(0.42, (lo + hi) / 2 * e["hr_max"], f"textbook {nm}", fontsize=7, va="center", color="#c46a2b")
        ax.set_xlim(0.4, 1.02)
        ax.set_xlabel("effort (share of critical power)")
        ax.set_ylabel("heart rate (bpm)")
        ax.set_title(f"a paddler with a {shape} heart-rate response", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax = axes[2]
    shapes = ["typical", "compressed"]
    w = 0.35
    x = np.arange(len(shapes))
    tb = [100 * data["summary"][s]["textbook_zone_right"] for s in shapes]
    pm = [100 * data["summary"][s]["personal_zone_right"] for s in shapes]
    ax.bar(x - w / 2, tb, w, color="#c46a2b", label="textbook zones (share of max)")
    ax.bar(x + w / 2, pm, w, color="#3b6ea8", label="personal model")
    for i in range(len(shapes)):
        ax.text(x[i] - w / 2, tb[i] + 1, f"{tb[i]:.0f}%", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, pm[i] + 1, f"{pm[i]:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} response\n({data['summary'][s]['paddlers']} paddlers)" for s in shapes], fontsize=9)
    ax.set_ylabel("effort zone named correctly from heart rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("which zone a heart rate means", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
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
