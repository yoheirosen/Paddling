#!/usr/bin/env python3
"""The experimenting coach on open races: tuning the scheduled-test rules, the baselines of
the held-out check, and its summary and figure.

    python experiments/experimenter_race.py tune --seeds 101-124
    python experiments/experimenter_race.py baselines --pool held_out --seeds 5060-5089 --tag heldout_experimenter
    python experiments/experimenter_race.py summary --tag heldout_experimenter --figure

Every arm races with the heart-rate strap (it changes nothing in the race; the paddler's
other draws are untouched) and is scored on race time and on the learning measures
(``coaching_trial.score_learning``).  The coach itself is role-played
(``roleplay_race.py --incidents held_out --coach-version 10``)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from paddling_expt import coaching_trial as ct            # noqa: E402

OUT = os.path.join(ROOT, "results", "coaching_trial")
HR_RISES = (0.4, 0.6, 0.9)
K_CHANGES = (0.07, 0.10)


def _seeds(spec: str) -> List[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _prep(pool: str, seed: int) -> ct.Preparation:
    path = os.path.join(OUT, f"preparations_open_{pool}.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if str(seed) not in cache:
        cache[str(seed)] = ct.draw_preparation(seed, incidents=pool).to_dict()
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    return ct.Preparation.from_dict(cache[str(seed)])


def _slim(r: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(r)
    r.pop("series", None)
    return r


def _run(pool: str, seeds: List[int], tag: str, arms_fn) -> None:
    path = os.path.join(OUT, f"episodes_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["seed"], r["arm"], json.dumps(r["setting"], sort_keys=True)) for r in map(json.loads, open(path))}
    with open(path, "a") as f:
        for s in seeds:
            prep = _prep(pool, s)
            for arm in arms_fn():
                key = (s, arm.name, json.dumps(arm.setting, sort_keys=True))
                if key in done:
                    continue
                r = ct.run_episode(prep, arm, heart_rate=True)
                f.write(json.dumps(_slim(r)) + "\n")
                f.flush()
            print(f"seed {s}: done", flush=True)


def tune_arms():
    return ([ct.NoCoach(), ct.IncidentRules(drop_mps=0.06)]
            + [ct.ScheduledTests(hr_rise_bpm_min=h, k_change=k) for h in HR_RISES for k in K_CHANGES])


def heldout_arms():
    p = ct.HELDOUT_EXPERIMENTER["scheduled_tests_primary"]
    out = [ct.NoCoach(), ct.IncidentRules(drop_mps=0.06), ct.ScheduledTests(**p)]
    out += [ct.ScheduledTests(hr_rise_bpm_min=h, k_change=p["k_change"]) for h in HR_RISES if h != p["hr_rise_bpm_min"]]
    return out


def _rows(tag: str) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in open(os.path.join(OUT, f"episodes_{tag}.jsonl"))]


def tune_summary(tag: str) -> Dict[str, Any]:
    rows = _rows(tag)
    by: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        key = r["arm"] + ("" if r["arm"] != "scheduled_tests" else
                          f" hr{r['setting']['hr_rise_bpm_min']} k{r['setting']['k_change']}")
        by.setdefault(key, {})[r["seed"]] = r
    S = sorted(set.intersection(*[set(d) for d in by.values()]))
    base = by["incident_rules"]
    out = {}
    for k, d in sorted(by.items()):
        diff = [d[s]["race_time_s"] - base[s]["race_time_s"] for s in S]
        le = [d[s].get("learning", {}).get("reserve_error_pts") for s in S]
        out[k] = {"n": len(S), "minus_incident_rules_s": round(float(np.mean(diff)), 1),
                  "reserve_error_pts": round(float(np.mean([x for x in le if x is not None])), 1),
                  "blown_s": round(float(np.mean([d[s]["blown_s"] for s in S])), 1)}
        print(f"  {k:32s} minus incident rules {out[k]['minus_incident_rules_s']:+6.1f} s   "
              f"reserve error {out[k]['reserve_error_pts']:5.1f} pts   blown {out[k]['blown_s']:6.1f} s")
    return out


# ── the held-out summary ────────────────────────────────────────────────────

def _load(tag: str):
    rows = _rows(tag)
    pr = ct.HELDOUT_EXPERIMENTER["scheduled_tests_primary"]
    by = lambda arm, cond=lambda r: True: {r["seed"]: r for r in rows if r["arm"] == arm and cond(r)}   # noqa: E731
    sched = by("scheduled_tests", lambda r: abs(r["setting"]["hr_rise_bpm_min"] - pr["hr_rise_bpm_min"]) < 1e-9
               and abs(r["setting"]["k_change"] - pr["k_change"]) < 1e-9)
    rp = {}
    base = os.path.join(OUT, "roleplay", tag)
    for s in (os.listdir(base) if os.path.isdir(base) else []):
        p = os.path.join(base, s, "result.json")
        if os.path.exists(p):
            r = json.load(open(p))
            rp[r["seed"]] = r
    return rp, by("no_coach"), by("incident_rules"), sched, rows


def summary(tag: str, figure: bool) -> Dict[str, Any]:
    from open_race import _incident_table
    rp, nc, rules, sched, rows = _load(tag)
    S = sorted(set(rp) & set(nc) & set(rules) & set(sched))
    out: Dict[str, Any] = {"tag": tag, "n": len(S), "seeds": S}
    if not S:
        print("no finished races")
        return out
    t = lambda d: [d[s]["race_time_s"] for s in S]                                         # noqa: E731
    le = lambda d, k="reserve_error_pts": [d[s].get("learning", {}).get(k) for s in S]      # noqa: E731
    comps = {"scheduled_tests": ct.paired(t(rp), t(sched)), "incident_rules": ct.paired(t(rp), t(rules)),
             "no_coach": ct.paired(t(rp), t(nc))}
    h = ct.holm({k: comps[k]["wilcoxon_p"] for k in ("scheduled_tests", "incident_rules")})
    for k in ("scheduled_tests", "incident_rules"):
        comps[k]["holm_p"] = h[k]
    out["race_time"] = comps
    print(f"experimenting coach on {len(S)} races: race time")
    for k, c in comps.items():
        print(f"  coach minus {k:16s} {c['mean_diff']:+6.1f} s  95% CI [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"
              f"  p {c['wilcoxon_p']}" + (f"  Holm {c['holm_p']}" if "holm_p" in c else "")
              + f"  coach faster in {sum(1 for a, b in zip(t(rp), t({'scheduled_tests': sched, 'incident_rules': rules, 'no_coach': nc}[k])) if a < b)}/{len(S)}")
    lc = {k: ct.paired(le(rp), le(d)) for k, d in (("scheduled_tests", sched), ("incident_rules", rules), ("no_coach", nc))}
    h2 = ct.holm({k: lc[k]["wilcoxon_p"] for k in ("scheduled_tests", "incident_rules")})
    for k in ("scheduled_tests", "incident_rules"):
        lc[k]["holm_p"] = h2[k]
    out["reserve_error"] = lc
    out["reserve_error_means"] = {nm: round(float(np.mean(le(d))), 2) for nm, d in
                                  (("open_coach", rp), ("scheduled_tests", sched), ("incident_rules", rules), ("no_coach", nc))}
    print("learning: mean error of the pod's reserve reading (percentage points)", out["reserve_error_means"])
    for k, c in lc.items():
        print(f"  coach minus {k:16s} {c['mean_diff']:+6.2f} pts  95% CI [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
              f"  p {c['wilcoxon_p']}" + (f"  Holm {c['holm_p']}" if "holm_p" in c else ""))
    cpm = {}
    for nm, d in (("open_coach", rp), ("scheduled_tests", sched), ("incident_rules", rules)):
        cpm[nm] = {k: round(float(np.mean(le(d, k))), 2) for k in ("cp_error_pct_start", "cp_error_pct_half", "cp_error_pct_three_quarters")}
        g = [x for x in le(d, "pod_gain_error") if x is not None]
        cpm[nm]["pod_gain_error_drift_races"] = round(float(np.mean(g)), 4) if g else None
        cpm[nm]["blown_s"] = round(float(np.mean([d[s]["blown_s"] for s in S])), 1)
    out["learning_secondary"] = cpm
    print("critical power error (% of the truth), pod gain error, seconds blown up:")
    for nm, v in cpm.items():
        print(f"  {nm:16s} {v}")
    for kind in ("pod_drift", "course_change", "weed"):
        K = [s for s in S if any(i["kind"] == kind for i in rp[s].get("incidents", []))]
        if len(K) >= 5:
            c1 = ct.paired([rp[s]["race_time_s"] for s in K], [sched[s]["race_time_s"] for s in K])
            c2 = ct.paired([rp[s]["race_time_s"] for s in K], [rules[s]["race_time_s"] for s in K])
            out[f"races_with_{kind}"] = {"n": len(K), "vs_scheduled_tests": c1, "vs_incident_rules": c2}
            print(f"  races with {kind} ({len(K)}): coach minus scheduled tests {c1['mean_diff']:+.1f} s "
                  f"[{c1['ci95'][0]:+.1f}, {c1['ci95'][1]:+.1f}]; minus incident rules {c2['mean_diff']:+.1f} s")
    pr = ct.HELDOUT_EXPERIMENTER["scheduled_tests_primary"]
    sens = {}
    for hrr in HR_RISES:
        d = {r["seed"]: r for r in rows if r["arm"] == "scheduled_tests" and abs(r["setting"]["hr_rise_bpm_min"] - hrr) < 1e-9
             and abs(r["setting"]["k_change"] - pr["k_change"]) < 1e-9}
        if all(s in d for s in S):
            sens[str(hrr)] = round(float(np.mean([rp[s]["race_time_s"] - d[s]["race_time_s"] for s in S])), 1)
    out["vs_scheduled_tests_by_threshold"] = sens
    print("  coach minus scheduled tests by heart-rate threshold:", sens)
    out["incidents"] = _incident_table({"open_coach": rp, "scheduled_tests": sched, "incident_rules": rules,
                                        "no_coach": nc}, S)
    out["coach_use"] = {"pieces_per_race": round(float(np.mean([len(rp[s].get("pieces") or []) for s in S])), 2),
                        "beliefs_set_per_race": round(float(np.mean([len(rp[s].get("belief_log") or []) for s in S])), 2),
                        "model_calls_per_race": round(float(np.mean([rp[s].get("roleplay", {}).get("answers", 0) for s in S])), 1)}
    print("  coach's use of its tools:", out["coach_use"])
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if figure:
        print("figure:", _figure(tag, rp, sched, rules, nc, S, out))
    return out


def _figure(tag, rp, sched, rules, nc, S, out):
    from agent_lib.figure_artifacts import emit
    data = {
        "title": "Open races with an experimenting coach (simulated paddler, planted incidents; coach role-played)",
        "diffs": {"scheduled_tests": [rp[s]["race_time_s"] - sched[s]["race_time_s"] for s in S],
                  "incident_rules": [rp[s]["race_time_s"] - rules[s]["race_time_s"] for s in S]},
        "ci95": {k: out["race_time"][k]["ci95"] for k in ("scheduled_tests", "incident_rules")},
        "reserve_error": {nm: [d[s].get("learning", {}).get("reserve_error_pts") for s in S]
                          for nm, d in (("no coach", nc), ("incident rules", rules), ("scheduled tests", sched),
                                        ("experimenting coach", rp))},
        "cp_error": {nm: [out["learning_secondary"][k]["cp_error_pct_start"], out["learning_secondary"][k]["cp_error_pct_half"],
                          out["learning_secondary"][k]["cp_error_pct_three_quarters"]]
                     for nm, k in (("incident rules", "incident_rules"), ("scheduled tests", "scheduled_tests"),
                                   ("experimenting coach", "open_coach"))},
    }
    return emit(f"coaching_{tag}", data=data, plot_func=plot_experimenter,
                artifacts_dir=os.path.join(ROOT, "artifacts"))


def plot_experimenter(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
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
    names = list(data["reserve_error"])
    cols = ["#777777", "#c46a2b", "#6a8f3b", "#3b6ea8"]
    for i, nm in enumerate(names):
        d = [x for x in data["reserve_error"][nm] if x is not None]
        x = i + 0.12 * np.random.default_rng(10 + i).uniform(-1, 1, len(d))
        ax.plot(x, d, "o", color=cols[i], alpha=0.5, ms=4)
        ax.plot([i - 0.25, i + 0.25], [np.mean(d)] * 2, color="k", lw=2)
        ax.text(i + 0.28, np.mean(d), f"{np.mean(d):.1f}", va="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ax.set_ylabel("mean |reserve read from the pod - true reserve| (points)")
    ax.set_title("how well each arm knew the paddler's reserve", fontsize=10)
    ax = axes[2]
    for i, (nm, v) in enumerate(data["cp_error"].items()):
        ax.plot([0, 1, 2], v, "o-", color=cols[i + 1], label=nm)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["start", "halfway", "three quarters"], fontsize=9)
    ax.set_ylabel("critical power error (% of the truth)")
    ax.set_title("the computer's critical power through the race", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    fig.suptitle(data.get("title", ""), fontsize=10)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["tune", "tune-summary", "baselines", "summary"])
    ap.add_argument("--seeds", default="101-124")
    ap.add_argument("--pool", default="development")
    ap.add_argument("--tag", default="tune_scheduled")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "tune":
        _run("development", _seeds(a.seeds), a.tag, tune_arms)
        tune_summary(a.tag)
    elif a.cmd == "tune-summary":
        tune_summary(a.tag)
    elif a.cmd == "baselines":
        _run(a.pool, _seeds(a.seeds), a.tag, heldout_arms)
    else:
        summary(a.tag, a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
