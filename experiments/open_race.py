#!/usr/bin/env python3
"""Open races: the baselines, and the summary of a role-played open coach against them.

    python experiments/open_race.py baselines --pool held_out --seeds 5000-5029 --tag heldout_open
    python experiments/open_race.py summary   --tag heldout_open [--figure]

``baselines`` races no coach, the pod autopilot (fixed rules written before incidents
existed) and the incident rules (written knowing the development incidents; three
detection thresholds) on each seed and writes ``results/coaching_trial/
episodes_<tag>.jsonl``.  ``summary`` reads those and the role-played open-coach races
(``results/coaching_trial/roleplay/<tag>/<seed>/result.json``) and prints the
registered comparisons (``coaching_trial.HELDOUT_OPEN``) and the per-incident scores.
"""
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

from paddling_expt import coaching_trial as ct      # noqa: E402

OUT = os.path.join(ROOT, "results", "coaching_trial")
RULE_DROPS = (0.04, 0.06, 0.08)
PRIMARY_DROP = 0.06


def _seeds(spec: str) -> List[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _prep(pool: str, seed: int) -> ct.Preparation:
    path = os.path.join(OUT, f"preparations_open_{pool}.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if str(seed) not in cache:
        cache[str(seed)] = ct.draw_preparation(seed, incidents=pool).to_dict()
        with open(path, "w") as f:
            json.dump(cache, f)
    return ct.Preparation.from_dict(cache[str(seed)])


def baselines(pool: str, seeds: List[int], tag: str) -> None:
    path = os.path.join(OUT, f"episodes_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["seed"], r["arm"], json.dumps(r["setting"], sort_keys=True)) for r in map(json.loads, open(path))}
    with open(path, "a") as f:
        for s in seeds:
            prep = _prep(pool, s)
            for arm in [ct.NoCoach(), ct.PodAutopilot()] + [ct.IncidentRules(drop_mps=d) for d in RULE_DROPS]:
                key = (s, arm.name, json.dumps(arm.setting, sort_keys=True))
                if key in done:
                    continue
                r = ct.run_episode(prep, arm)
                r.pop("series", None)
                f.write(json.dumps(r) + "\n")
                f.flush()
            print(f"seed {s}: baselines done", flush=True)


def _load(tag: str):
    rows = [json.loads(x) for x in open(os.path.join(OUT, f"episodes_{tag}.jsonl"))]
    by = lambda arm, cond=lambda r: True: {r["seed"]: r for r in rows if r["arm"] == arm and cond(r)}   # noqa: E731
    rp = {}
    base = os.path.join(OUT, "roleplay", tag)
    for s in (os.listdir(base) if os.path.isdir(base) else []):
        p = os.path.join(base, s, "result.json")
        if os.path.exists(p):
            r = json.load(open(p))
            rp[r["seed"]] = r
    rules = by("incident_rules", lambda r: abs(r["setting"].get("drop_mps", 0) - PRIMARY_DROP) < 1e-9)
    return rp, by("no_coach"), by("autopilot_pod"), rules, rows


def _incident_table(arms: Dict[str, Dict[int, Dict[str, Any]]], seeds: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, d in arms.items():
        sc = [d[s]["incident_score"] for s in seeds if s in d]
        weed = [x["weed"] for x in sc if "weed" in x]
        pod = [x["pod_drift"] for x in sc if "pod_drift" in x]
        cc = [x["course_change"] for x in sc if "course_change" in x]
        talk = [x["talk"] for x in sc]
        out[name] = {
            "weed": {"n": len(weed), "mean_seconds_with_weed": round(float(np.mean([w["seconds_with_weed"] for w in weed])), 1) if weed else None,
                     "cleared_by_cue": sum(1 for w in weed if w["cleared_by"] == "cue"),
                     "cleared_by_paddler": sum(1 for w in weed if w["cleared_by"] == "self"),
                     "never_cleared": sum(1 for w in weed if w["cleared_by"] == "never"),
                     "median_first_clear_cue_after_s": (float(np.median([w["first_clear_cue_after_s"] for w in weed
                                                                         if w["first_clear_cue_after_s"] is not None]))
                                                        if any(w["first_clear_cue_after_s"] is not None for w in weed) else None),
                     "false_clears": sum(x["false_clears"] for x in sc)},
            "pod_drift": {"n": len(pod), "cues_as_faulty_pod_suggests": sum(p["cues_as_the_faulty_pod_suggests"] for p in pod),
                          "cues_against_it": sum(p["cues_against_it"] for p in pod)},
            "course_change": {"n": len(cc), "paddler_told": sum(1 for c in cc if c["paddler_told"]),
                              "overran_believed_finish": sum(1 for c in cc if c["overran_believed_finish"]),
                              "sprint_start_m_from_true_finish": [c["sprint_started_m_from_true_finish"] for c in cc]},
            "talk": {k: sum(t.get(k, 0) for t in talk) for k in ("self_ease", "self_push", "reassured_in_time",
                                                                 "reassured_late", "held_in_time", "held_late")},
            "questions_asked": sum(x.get("questions_asked", 0) for x in sc)}
    return out


def plot_open_race(data):
    """Open races: race time against the baselines, and what happened to each incident."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]},
                             constrained_layout=True)
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
        ax.text((x or 0) + 15, i, f"{(x or 0):.0f} s\ncleared: {c}", va="center", fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean seconds raced with weed on the rudder")
    ax.set_title(f"weed, a practice kind ({data['n_weed']} races)", fontsize=10)
    ax.set_xlim(0, max(x or 0 for x in w) * 1.9 + 1)
    ax = axes[2]
    rows = [(f"course changed by radio:\npaddler told the new finish\n(of {data['n_course']} races)", "told"),
            (f"course changed: paddler sprinted\nfor the old finish (of {data['n_course']})", "overran"),
            (f"pod drifting: the coach's notes\nname the faulty pod* (of {data['n_pod']})", "pod_diag")]
    y = np.arange(len(rows))
    for j, (a, c) in enumerate((("incident_rules", "#c46a2b"), ("open_coach", "#3b6ea8"))):
        vals = [data["held_out"][a].get(r[1]) for r in rows]
        for yy, v in zip(y + (-0.2 if j == 1 else 0.2), vals):
            if v is None:
                ax.text(0.3, yy, "(rules keep no notes)", va="center", fontsize=7, color=c)
                continue
            ax.barh(yy, v, height=0.38, color=c)
            ax.text(v + 0.3, yy, str(v), va="center", fontsize=7)
    from matplotlib.patches import Patch
    handles = [Patch(color="#c46a2b", label="incident rules"), Patch(color="#3b6ea8", label="open coach")]
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(len(rows) - 0.25, -0.55)
    ax.set_xlim(0, max(data["n_course"], data["n_pod"]) * 1.2)
    ax.legend(handles=handles, fontsize=7, loc="center right")
    ax.set_title("held-out kinds, never used in development", fontsize=10)
    ax.text(0.0, -0.2, f"* exploratory, read from the coach's notes; such a note also appears in "
            f"{data['pod_diag_false']} of {data['n_no_pod']} races without drift", transform=ax.transAxes, fontsize=6.5)
    fig.suptitle(data.get("title", "Open races (simulated paddler, planted incidents): open coach role-played"),
                 fontsize=10)
    return fig


_POD_NOTE = None


def pod_diagnosis_from_notes(tag: str, seed: int) -> bool:
    """Exploratory: did the coach write a note naming the paddle pod as faulty or drifting?"""
    import re
    global _POD_NOTE
    if _POD_NOTE is None:
        _POD_NOTE = re.compile(r"(pod|paddle sensor|shaft).{0,80}(drift|fault|faulty|reading high|reads high|reading low|"
                               r"reads low|unreliable|broken|offset|miscalibrat)", re.I)
    p = os.path.join(OUT, "roleplay", tag, str(seed), "answers.json")
    for a in json.load(open(p)) if os.path.exists(p) else []:
        for tc in (a.get("tool_calls") or []) if isinstance(a, dict) else []:
            if tc.get("name") == "take_note" and _POD_NOTE.search(str(tc.get("arguments", {}).get("text", ""))):
                return True
    return False


def _figure(out: Dict[str, Any], rp, nc, rules, S, tag: str = "") -> None:
    from agent_lib.figure_artifacts import emit
    inc = out["incidents"]
    pod = [s for s in S if any(i["kind"] == "pod_drift" for i in rp[s]["incidents"])]
    nopod = [s for s in S if s not in pod]
    data = {"n": len(S),
            "diffs": {"incident_rules": [rp[s]["race_time_s"] - rules[s]["race_time_s"] for s in S],
                      "no_coach": [rp[s]["race_time_s"] - nc[s]["race_time_s"] for s in S]},
            "ci95": {"incident_rules": out["comparisons"]["incident_rules"]["ci95"],
                     "no_coach": out["comparisons"]["no_coach"]["ci95"]},
            "weed_seconds": {a: inc[a]["weed"]["mean_seconds_with_weed"] for a in inc},
            "weed_cleared": {a: f"{inc[a]['weed']['cleared_by_cue']} by cue, {inc[a]['weed']['cleared_by_paddler']} alone"
                             for a in inc},
            "n_weed": inc["open_coach"]["weed"]["n"],
            "n_course": inc["open_coach"]["course_change"]["n"], "n_pod": inc["open_coach"]["pod_drift"]["n"],
            "held_out": {a: {"told": inc[a]["course_change"]["paddler_told"],
                             "overran": inc[a]["course_change"]["overran_believed_finish"],
                             "pod_diag": (sum(pod_diagnosis_from_notes(tag, s) for s in pod) if a == "open_coach" else None)}
                         for a in inc},
            "n_no_pod": len(nopod),
            "pod_diag_false": sum(pod_diagnosis_from_notes(tag, s) for s in nopod),
            "note": "simulated races with planted incidents; per-race differences and counts only; no positions",
            "title": ("Open races (simulated paddler, planted incidents): open coach role-played by Claude subagents"
                      if tag == "heldout_open" else
                      "Open races, the later open coach (simulated paddler, planted incidents; role-played)")}
    name = "coaching_open_race" if tag == "heldout_open" else f"coaching_{tag}"
    paths = emit(name, data=data, plot_func=plot_open_race, artifacts_dir=os.path.join(ROOT, "artifacts"))
    print("figure:", paths)


def summary(tag: str, figure: bool = False) -> Dict[str, Any]:
    rp, nc, pod, rules, rows = _load(tag)
    S = sorted(s for s in rp if s in nc and s in rules)
    t = lambda d: [d[s]["race_time_s"] for s in S]                       # noqa: E731
    out: Dict[str, Any] = {"n": len(S)}
    print(f"open coach on {len(S)} races")
    if len(S) >= 2:
        comps = {"incident_rules": ct.paired(t(rp), t(rules)), "no_coach": ct.paired(t(rp), t(nc)),
                 "autopilot_pod": ct.paired(t(rp), t(pod))}
        h = ct.holm({k: comps[k]["wilcoxon_p"] for k in ("incident_rules", "no_coach")})
        for k, c in comps.items():
            print(f"  coach minus {k:15s} {c['mean_d']:+7.1f} s  95% CI [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]  "
                  f"p {c['wilcoxon_p']:.4f}  coach faster in {c['n_a_faster']}/{c['n']}")
        print("  Holm (primary):", {k: round(v, 4) for k, v in h.items()})
        out.update({"comparisons": comps, "holm": h})
        for kind in ("pod_drift", "course_change", "weed"):
            K = [s for s in S if any(i["kind"] == kind for i in rp[s]["incidents"])]
            if len(K) >= 2:
                c1 = ct.paired([rp[s]["race_time_s"] for s in K], [rules[s]["race_time_s"] for s in K])
                c2 = ct.paired([rp[s]["race_time_s"] for s in K], [nc[s]["race_time_s"] for s in K])
                print(f"  races with {kind} ({len(K)}): coach minus rules {c1['mean_d']:+.1f} s "
                      f"[{c1['ci95'][0]:+.1f}, {c1['ci95'][1]:+.1f}]; minus no coach {c2['mean_d']:+.1f} s")
                out[f"races_with_{kind}"] = {"n": len(K), "vs_incident_rules": c1, "vs_no_coach": c2}
        sens = {}
        for d in RULE_DROPS:
            rd = {r["seed"]: r for r in rows if r["arm"] == "incident_rules" and abs(r["setting"]["drop_mps"] - d) < 1e-9}
            if all(s in rd for s in S):
                sens[str(d)] = ct.paired(t(rp), [rd[s]["race_time_s"] for s in S])["mean_d"]
        out["vs_incident_rules_by_threshold"] = sens
        print("  coach minus incident rules by threshold:", {k: round(v, 1) for k, v in sens.items()})
        out["incidents"] = _incident_table({"open_coach": rp, "incident_rules": rules, "autopilot_pod": pod,
                                            "no_coach": nc}, S)
        for k, v in out["incidents"].items():
            print(f"  {k}: {json.dumps(v)}")
        if figure:
            _figure(out, rp, nc, rules, S, tag)
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["baselines", "summary"])
    ap.add_argument("--pool", default="held_out", choices=["development", "held_out"])
    ap.add_argument("--seeds", default="5000-5029")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "baselines":
        baselines(a.pool, _seeds(a.seeds), a.tag)
    else:
        summary(a.tag, a.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
