#!/usr/bin/env python3
"""Exploratory: the experimenting coach's findings, applied to rule-paced races.

The registered check (``coaching_trial.HELDOUT_EXPERIMENTER``) found that the coach learnt
the instruments and the course well but lost race time on its own pacing.  This asks, at
no model cost, what its findings are worth when the incident rules do the pacing.  For each
held-out race the coach's findings are read from its recorded race (pod-gain corrections,
the race distance it set, its messages telling the paddler the new distance, its rudder
clears, and optionally its critical-power beliefs), and applied at the same race seconds to
a race paced by the incident rules.  The paddler, water and incidents are the same race.

    python experiments/findings_counterfactual.py extract   # replay the coach's races once
    python experiments/findings_counterfactual.py run       # race the variants
    python experiments/findings_counterfactual.py summary

Not registered: the coach's findings came from its own races, where the paddler was
paced differently, so their timing is the coach's, not an answer to these races.
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
sys.path.insert(0, HERE)

from paddling_expt import coaching_trial as ct            # noqa: E402

OUT = os.path.join(ROOT, "results", "coaching_trial")
TAG = "heldout_experimenter"
FINDINGS = os.path.join(OUT, f"findings_{TAG}.json")
VARIANTS = {
    "pod": dict(pod=True),
    "pod_course": dict(pod=True, course=True),
    "pod_course_weed": dict(pod=True, course=True, weed=True),
    "pod_course_weed_cp": dict(pod=True, course=True, weed=True, cp=True),
}


class CoachFindingsRules(ct.IncidentRules):
    """The incident rules (the registered primary setting), plus the experimenting coach's
    findings from its own race of this seed, applied at the same race seconds."""

    name = "rules_with_coach_findings"
    heart_rate = True

    def __init__(self, findings: Dict[str, Any], *, pod=False, course=False, weed=False, cp=False,
                 drop_mps: float = 0.06) -> None:
        super().__init__(drop_mps)
        self.use = {"pod": pod, "course": course, "weed": weed, "cp": cp}
        self.setting.update({"findings": "+".join(k for k, v in self.use.items() if v)})
        items = []
        if pod:
            items += [dict(f, kind="pod_gain") for f in findings.get("pod_gain", [])]
        if cp:
            items += [dict(f, kind="calibration") for f in findings.get("calibration", [])]
        if course:
            items += [dict(f, kind="distance") for f in findings.get("distance", [])]
            items += [dict(f, kind="tell") for f in findings.get("tell", [])]
        if weed:
            items += [dict(f, kind="clear") for f in findings.get("clear", [])]
        self._todo = sorted(items, key=lambda x: x["race_s"])
        self.applied: List[Dict[str, Any]] = []

    def _cue(self, ep, act) -> bool:
        return bool(ep.gate.deliver({**act, "source": self.name}, source=self.name).get("delivered"))

    def on_tick(self, ep, t, alerts) -> None:
        super().on_tick(ep, t, alerts)
        st = ep.state
        el = st.last.get("elapsed_s")
        if el is None:
            return
        keep = []
        for f in self._todo:
            if f["race_s"] > el:
                keep.append(f)
                continue
            k = f["kind"]
            done = True
            if k == "pod_gain":
                st.set_pod_gain(float(f["value"]), f.get("since_race_s"))
            elif k == "calibration":
                kw = {"critical_power_pod_w": "cp_shaft_w", "reserve_size_pod_kj": "w_prime_shaft_j",
                      "critical_power_boat_w": "cp_w", "reserve_size_boat_kj": "w_prime_j"}[f["name"]]
                v = float(f["value"]) * (1000.0 if f["name"].endswith("_kj") else 1.0)
                st.recalibrate(recompute=True, **{kw: v})
            elif k == "distance":
                st.distance_m = float(f["new_m"])
            elif k == "tell":
                left = max(100.0, float(st.distance_m or 0.0) - float(st.travelled))
                done = self._cue(ep, {"text": f"Course changed: {left:.0f} metres to go.", "ask": "tell",
                                      "distance_left_m": left})
            elif k == "clear":
                done = self._cue(ep, {"text": "Clear the rudder. One back-stroke.", "ask": "clear_rudder"})
                if done:
                    self._last_clear = el
            if not done and el - f["race_s"] < 90.0:       # the cue gate refused it: try again for 90 s
                keep.append(f)
            else:
                self.applied.append({"race_s": round(el), "kind": k, "delivered": done})
        self._todo = keep

    def summary(self) -> Dict[str, Any]:
        return {"applied": self.applied}


def extract() -> None:
    """Replay each coached race from its answers and keep the findings it acted on."""
    import roleplay_race as rr
    orig = ct.AgentArm.summary

    def summary(self):
        out = orig(self)
        if self.coach is not None:
            out["distance_changes"] = self.coach.distance_changes
        return out
    ct.AgentArm.summary = summary
    base = os.path.join(OUT, "roleplay", TAG)
    found: Dict[str, Any] = {}
    for s in sorted(os.listdir(base), key=int):
        a = argparse.Namespace(seed=int(s), tag=TAG, pack=False, coastal=False, incidents="held_out",
                               coach_version=10, label="agent_experimenter_roleplay", periodic_s=180.0,
                               model_label="roleplay-subagent", latency_s=5.0)
        r = rr._replay(a, json.load(open(os.path.join(base, s, "answers.json"))))
        bl = r.get("belief_log") or []
        found[s] = {
            "pod_gain": [{"race_s": b["set_at_race_s"], "value": b["value"], "since_race_s": b.get("since_race_s")}
                         for b in bl if b["name"] == "pod_gain"],
            "calibration": [{"race_s": b["set_at_race_s"], "name": b["name"], "value": b["value"]}
                            for b in bl if b["name"] in ("critical_power_pod_w", "reserve_size_pod_kj",
                                                         "critical_power_boat_w", "reserve_size_boat_kj")],
            "distance": [{"race_s": round(d["elapsed_s"] or 0.0), "new_m": d["new_m"]} for d in r.get("distance_changes", [])],
            "tell": [{"race_s": c["race_s"]} for c in r["cue_log"] if c["ask"] == "tell"],
            "clear": [{"race_s": c["race_s"]} for c in r["cue_log"] if c["ask"] == "clear_rudder"],
            "race_time_s": r["race_time_s"],
        }
        print(f"seed {s}: {sum(len(v) for k, v in found[s].items() if isinstance(v, list))} findings", flush=True)
    with open(FINDINGS, "w") as f:
        json.dump(found, f, indent=1)


def run() -> None:
    import experimenter_race as er
    found = json.load(open(FINDINGS))
    path = os.path.join(OUT, f"episodes_{TAG}_findings.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["seed"], r["setting"]["findings"]) for r in map(json.loads, open(path))}
    with open(path, "a") as f:
        for s, fd in sorted(found.items(), key=lambda x: int(x[0])):
            prep = er._prep("held_out", int(s))
            for vname, kw in VARIANTS.items():
                arm = CoachFindingsRules(fd, **kw)
                if (int(s), arm.setting["findings"]) in done:
                    continue
                r = ct.run_episode(prep, arm, heart_rate=True)
                r.pop("series", None)
                f.write(json.dumps(r) + "\n")
                f.flush()
            print(f"seed {s}: done", flush=True)


def summary() -> Dict[str, Any]:
    import experimenter_race as er
    rp, nc, rules, sched, _ = er._load(TAG)
    rows = [json.loads(x) for x in open(os.path.join(OUT, f"episodes_{TAG}_findings.jsonl"))]
    out: Dict[str, Any] = {}
    S = sorted(rp)
    for vname, kw in VARIANTS.items():
        key = "+".join(k for k, v in {"pod": False, "course": False, "weed": False, "cp": False, **kw}.items() if v)
        d = {r["seed"]: r for r in rows if r["setting"].get("findings") == key}
        if not all(s in d for s in S):
            continue
        t = [d[s]["race_time_s"] for s in S]
        c_rules = ct.paired(t, [rules[s]["race_time_s"] for s in S])
        c_coach = ct.paired(t, [rp[s]["race_time_s"] for s in S])
        c_sched = ct.paired(t, [sched[s]["race_time_s"] for s in S])
        c_none = ct.paired(t, [nc[s]["race_time_s"] for s in S])
        out[vname] = {"minus_incident_rules": c_rules, "minus_coach": c_coach, "minus_scheduled_tests": c_sched,
                      "minus_no_coach": c_none, "blown_s": round(float(np.mean([d[s]["blown_s"] for s in S])), 1),
                      "cp_error_pct_mean": round(float(np.mean([d[s]["learning"]["cp_error_pct_mean"] for s in S])), 2)}
        print(f"rules + coach's findings [{vname}]  minus incident rules {c_rules['mean_d']:+.1f} s "
              f"[{c_rules['ci95'][0]:+.1f}, {c_rules['ci95'][1]:+.1f}] p {c_rules['wilcoxon_p']:.3g} "
              f"(faster in {c_rules['n_a_faster']}/{c_rules['n']});  minus the coach {c_coach['mean_d']:+.1f} s "
              f"[{c_coach['ci95'][0]:+.1f}, {c_coach['ci95'][1]:+.1f}] p {c_coach['wilcoxon_p']:.3g};  "
              f"minus no coach {c_none['mean_d']:+.1f} s;  blown {out[vname]['blown_s']} s;  "
              f"critical-power error {out[vname]['cp_error_pct_mean']} %")
    with open(os.path.join(OUT, f"summary_{TAG}_findings.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["extract", "run", "summary"])
    a = ap.parse_args(argv)
    {"extract": extract, "run": run, "summary": summary}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
