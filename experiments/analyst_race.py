#!/usr/bin/env python3
"""The analyst beside the rule pacer: the held-out summary and figure
(``coaching_trial.HELDOUT_ANALYST``).

    python experiments/experimenter_race.py baselines --pool held_out --seeds 5090-5119 --tag heldout_analyst
    python experiments/roleplay_race.py ... --incidents held_out --coach-version 11 --tag heldout_analyst
    python experiments/analyst_race.py summary --tag heldout_analyst --figure

The analyst is role-played; the baselines are the rule arms on the same seeds."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from paddling_expt import coaching_trial as ct            # noqa: E402
from experimenter_race import OUT, _load                  # noqa: E402


def summary(tag: str, figure: bool) -> Dict[str, Any]:
    from open_race import _incident_table
    rp, nc, rules, sched, _rows = _load(tag)
    S = sorted(set(rp) & set(nc) & set(rules) & set(sched))
    out: Dict[str, Any] = {"tag": tag, "n": len(S), "seeds": S,
                           "not_finished": sorted(set(ct.HELDOUT_ANALYST["seeds"]) - set(rp))}
    if not S:
        print("no finished races")
        return out
    t = lambda d: [d[s]["race_time_s"] for s in S]                                         # noqa: E731
    comps = {"incident_rules": ct.paired(t(rp), t(rules)), "scheduled_tests": ct.paired(t(rp), t(sched)),
             "no_coach": ct.paired(t(rp), t(nc))}
    h = ct.holm({k: comps[k]["wilcoxon_p"] for k in ("incident_rules", "scheduled_tests")})
    for k in ("incident_rules", "scheduled_tests"):
        comps[k]["holm_p"] = h[k]
    out["race_time"] = comps
    print(f"the analyst beside the pacer, {len(S)} held-out races: race time")
    for k, c in comps.items():
        print(f"  analyst minus {k:16s} {c['mean_d']:+6.1f} s  95% CI [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"
              f"  p {c['wilcoxon_p']}" + (f"  Holm {c['holm_p']}" if "holm_p" in c else "")
              + f"  analyst faster in {c['n_a_faster']}/{c['n']}")
    le = lambda d, k: [d[s].get("learning", {}).get(k) for s in S]                         # noqa: E731
    sec = {}
    for nm, d in (("analyst", rp), ("scheduled_tests", sched), ("incident_rules", rules), ("no_coach", nc)):
        g = [x for x in le(d, "pod_gain_error") if x is not None]
        sec[nm] = {"pod_gain_error_drift_races": round(float(np.mean(g)), 4) if g else None,
                   "n_drift_races": len(g),
                   "reserve_error_pts": round(float(np.mean([x for x in le(d, "reserve_error_pts") if x is not None])), 2),
                   "blown_s": round(float(np.mean([d[s]["blown_s"] for s in S])), 1)}
    out["secondary"] = sec
    print("pod gain error in drift races, reserve error, seconds blown up:")
    for nm, v in sec.items():
        print(f"  {nm:16s} {v}")
    for kind in sorted({i["kind"] for s in S for i in rp[s].get("incidents", [])}):
        K = [s for s in S if any(i["kind"] == kind for i in rp[s].get("incidents", []))]
        if len(K) >= 5:
            c1 = ct.paired([rp[s]["race_time_s"] for s in K], [rules[s]["race_time_s"] for s in K])
            c2 = ct.paired([rp[s]["race_time_s"] for s in K], [sched[s]["race_time_s"] for s in K])
            out[f"races_with_{kind}"] = {"n": len(K), "vs_incident_rules": c1, "vs_scheduled_tests": c2}
            print(f"  races with {kind} ({len(K)}): analyst minus incident rules {c1['mean_d']:+.1f} s "
                  f"[{c1['ci95'][0]:+.1f}, {c1['ci95'][1]:+.1f}]; minus scheduled tests {c2['mean_d']:+.1f} s")
    P = [s for s in S if rp[s].get("pieces")]
    N = [s for s in S if not rp[s].get("pieces")]
    use = {"pieces_per_race": round(float(np.mean([len(rp[s].get("pieces") or []) for s in S])), 2),
           "races_with_pieces": len(P),
           "beliefs_set_per_race": round(float(np.mean([len(rp[s].get("belief_log") or []) for s in S])), 2),
           "model_calls_per_race": round(float(np.mean([rp[s].get("roleplay", {}).get("answers", 0) for s in S])), 1),
           "minus_incident_rules_with_pieces_s": (round(float(np.mean([rp[s]["race_time_s"] - rules[s]["race_time_s"]
                                                                      for s in P])), 1) if P else None),
           "minus_incident_rules_without_pieces_s": (round(float(np.mean([rp[s]["race_time_s"] - rules[s]["race_time_s"]
                                                                         for s in N])), 1) if N else None)}
    out["analyst_use"] = use
    print("  the analyst's use of its tools:", use)
    out["incidents"] = _incident_table({"analyst": rp, "scheduled_tests": sched, "incident_rules": rules,
                                        "no_coach": nc}, S)
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if figure:
        print("figure:", _figure(tag, rp, sched, rules, nc, S, out))
    return out


def _figure(tag, rp, sched, rules, nc, S, out):
    from agent_lib.figure_artifacts import emit
    data = {
        "title": "An analyst beside the rule pacer, held-out races (simulated paddler, planted incidents; analyst role-played)",
        "diffs": {"incident_rules": [rp[s]["race_time_s"] - rules[s]["race_time_s"] for s in S],
                  "scheduled_tests": [rp[s]["race_time_s"] - sched[s]["race_time_s"] for s in S]},
        "ci95": {k: out["race_time"][k]["ci95"] for k in ("incident_rules", "scheduled_tests")},
        "pieces": [bool(rp[s].get("pieces")) for s in S],
        "pod_gain_error": {nm: [x for x in (d[s].get("learning", {}).get("pod_gain_error") for s in S) if x is not None]
                           for nm, d in (("no coach", nc), ("the pacer alone", rules), ("scheduled tests", sched),
                                         ("analyst + pacer", rp))},
    }
    return emit(f"coaching_{tag}", data=data, plot_func=plot_analyst, artifacts_dir=os.path.join(ROOT, "artifacts"))


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["summary"])
    ap.add_argument("--tag", default="heldout_analyst")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args(argv)
    summary(a.tag, a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
