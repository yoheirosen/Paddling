#!/usr/bin/env python3
"""Summarise the role-played drafting check (coaching_trial.HELDOUT_PACK_ROLEPLAY).

    python experiments/roleplay_summary.py

Reads the finished role-played races (results/coaching_trial/roleplay/<tag>/<seed>/
result.json) and the baselines, and prints the registered comparisons, the pooled
secondary analysis with the 12 API-coached races, and the fidelity check.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from paddling_expt.coaching_trial import holm, paired      # noqa: E402

OUT = os.path.join(ROOT, "results", "coaching_trial")


def _rp(tag):
    base = os.path.join(OUT, "roleplay", tag)
    out = {}
    if os.path.isdir(base):
        for s in os.listdir(base):
            p = os.path.join(base, s, "result.json")
            if os.path.exists(p):
                r = json.load(open(p))
                out[r["seed"]] = r
    return out


def _rows(tag):
    p = os.path.join(OUT, f"episodes_{tag}.jsonl")
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def _by(rows, arm):
    return {r["seed"]: r for r in rows if r["arm"] == arm and r.get("race_time_s")}


def main() -> int:
    out = {}
    rp = _rp("heldout_pack_rp")
    base = _rows("heldout_pack_rp")
    nc, pr, pod = _by(base, "no_coach"), _by(base, "pack_rule"), _by(base, "autopilot_pod")
    S = sorted(s for s in rp if s in nc)
    t = lambda d: [d[s]["race_time_s"] for s in S]                    # noqa: E731
    print(f"role-played coach on {len(S)} of 30 fresh pack races")
    if len(S) >= 2:
        comps = {"pack_rule": paired(t(rp), t(pr)), "no_coach": paired(t(rp), t(nc)),
                 "autopilot_pod": paired(t(rp), t(pod))}
        h = holm({k: comps[k]["wilcoxon_p"] for k in ("pack_rule", "no_coach")})
        for k, c in comps.items():
            print(f"  coach minus {k:14s} {c['mean_d']:+6.1f} s  95% CI [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]  "
                  f"p {c['wilcoxon_p']:.4f}  coach faster in {c['n_a_faster']}/{c['n']}")
        print("  Holm (primary):", {k: round(v, 4) for k, v in h.items()})
        print(f"  rules vs no coach: pack rule {np.mean([pr[s]['race_time_s'] - nc[s]['race_time_s'] for s in S]):+.1f} s, "
              f"pod autopilot {np.mean([pod[s]['race_time_s'] - nc[s]['race_time_s'] for s in S]):+.1f} s")
        asks = {}
        for s in S:
            for k, v in (rp[s].get("asks") or {}).items():
                asks[k] = asks.get(k, 0) + v
        calls = [rp[s]["roleplay"]["answers"] for s in S]
        print(f"  cues {np.mean([rp[s]['cues_spoken'] for s in S]):.1f} a race, asks {asks}; model calls a race {np.mean(calls):.1f}")
        out["roleplay"] = {"n": len(S), "comparisons": comps, "holm": h, "asks": asks}
        # pooled with the API-coached races (secondary; mixed coaches)
        api = _rows("heldout_pack")
        anc, apr, aag = _by(api, "no_coach"), _by(api, "pack_rule"), _by(api, "agent_v7")
        A = sorted(aag)
        coach = t(rp) + [aag[s]["race_time_s"] for s in A]
        rule = t(pr) + [apr[s]["race_time_s"] for s in A]
        none = t(nc) + [anc[s]["race_time_s"] for s in A]
        pp, pn = paired(coach, rule), paired(coach, none)
        print(f"  pooled with the {len(A)} API races (mixed): vs pack rule {pp['mean_d']:+.1f} s "
              f"[{pp['ci95'][0]:+.1f}, {pp['ci95'][1]:+.1f}] p {pp['wilcoxon_p']:.4f} ({pp['n_a_faster']}/{pp['n']}); "
              f"vs no coach {pn['mean_d']:+.1f} s [{pn['ci95'][0]:+.1f}, {pn['ci95'][1]:+.1f}] p {pn['wilcoxon_p']:.4f}")
        out["pooled"] = {"vs_pack_rule": pp, "vs_no_coach": pn}
    fid = _rp("pack_rp_fidelity")
    if fid:
        api = {r["seed"]: r for r in _rows("heldout_pack") if r["arm"] == "agent_v7"}
        base_api = _rows("heldout_pack")
        anc = _by(base_api, "no_coach")
        print("fidelity (role play vs the API coach on the same races):")
        rows = []
        for s in sorted(fid):
            a, b = fid[s], api.get(s)
            if b is None:
                continue
            rows.append({"seed": s, "roleplay_vs_no_coach_s": a["race_time_s"] - anc[s]["race_time_s"],
                         "api_vs_no_coach_s": b["race_time_s"] - anc[s]["race_time_s"],
                         "roleplay_asks": a.get("asks"), "api_asks": b.get("asks")})
            print(f"  seed {s}: role play {rows[-1]['roleplay_vs_no_coach_s']:+.1f} s, API {rows[-1]['api_vs_no_coach_s']:+.1f} s "
                  f"vs no coach | asks role play {a.get('asks')} | API {b.get('asks')}")
        out["fidelity"] = rows
    with open(os.path.join(OUT, "summary_heldout_pack_rp.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
