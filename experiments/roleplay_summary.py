#!/usr/bin/env python3
"""Summarise the role-played drafting check (coaching_trial.HELDOUT_PACK_ROLEPLAY).

    python experiments/roleplay_summary.py [--figure]

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


def plot_roleplay_drafting(data):
    """Per-race differences in race time, role-played coach and API coach (drafting races)."""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    groups = [("role-played coach\n(30 fresh races)", "#3b6ea8"), ("API coach\n(12 races)", "#c46a2b")]
    for ax, key, title in ((axes[0], "vs_pack_rule", "coach minus pack rule"),
                           (axes[1], "vs_no_coach", "coach minus no coach")):
        for i, (g, (label, col)) in enumerate(zip(("roleplay", "api"), groups)):
            d = np.asarray(data[g][key], float)
            rng = np.random.default_rng(i)
            x = i + rng.uniform(-0.15, 0.15, d.size)
            ax.scatter(x, d, s=16, color=col, alpha=0.6, lw=0)
            lo, hi = data[g][key + "_ci95"]
            ax.errorbar([i + 0.32], [d.mean()], yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color="k",
                        capsize=4, ms=5)
            ax.text(i + 0.38, d.mean(), f" {d.mean():+.1f} s", va="center", fontsize=8)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_xlim(-0.5, 1.9)
        ax.set_ylabel("race time difference (s); below 0 = coach faster")
        ax.set_title(title, fontsize=10)
    ax = axes[2]
    sp = data["split"]
    names = ["let the pack go\nin the first 5 min", "kept it longer\nor never cued"]
    vals = [sp["early"], sp["late"]]
    for i, v in enumerate(vals):
        d = np.asarray(v, float)
        rng = np.random.default_rng(10 + i)
        ax.scatter(i + rng.uniform(-0.15, 0.15, d.size), d, s=16, color="#3b6ea8", alpha=0.6, lw=0)
        ax.plot([i - 0.25, i + 0.25], [d.mean(), d.mean()], color="k", lw=2)
        ax.text(i + 0.27, d.mean(), f" {d.mean():+.1f} s (n={d.size})", va="center", fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.5, 1.9)
    ax.set_title("role play, coach minus no coach\n(exploratory split; the API coach let go early in "
                 f"{data['api_early_let_go']} of 12)", fontsize=9)
    fig.suptitle("Drafting races: the role-played coach did not repeat the API coach's gain (simulated paddler, planted pack)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def _figure(rp, nc, pr, S, api_rows):
    from agent_lib.figure_artifacts import emit
    anc, apr, aag = _by(api_rows, "no_coach"), _by(api_rows, "pack_rule"), _by(api_rows, "agent_v7")
    A = sorted(aag)

    def early(r):
        return any(c["ask"] == "let_go" and c["race_s"] < 300 for c in r["cue_log"])
    d_rp_pr = [rp[s]["race_time_s"] - pr[s]["race_time_s"] for s in S]
    d_rp_nc = [rp[s]["race_time_s"] - nc[s]["race_time_s"] for s in S]
    d_api_pr = [aag[s]["race_time_s"] - apr[s]["race_time_s"] for s in A]
    d_api_nc = [aag[s]["race_time_s"] - anc[s]["race_time_s"] for s in A]
    ci = lambda a, b: paired(a, b)["ci95"]                                 # noqa: E731
    data = {
        "roleplay": {"vs_pack_rule": d_rp_pr, "vs_no_coach": d_rp_nc,
                     "vs_pack_rule_ci95": ci([rp[s]["race_time_s"] for s in S], [pr[s]["race_time_s"] for s in S]),
                     "vs_no_coach_ci95": ci([rp[s]["race_time_s"] for s in S], [nc[s]["race_time_s"] for s in S])},
        "api": {"vs_pack_rule": d_api_pr, "vs_no_coach": d_api_nc,
                "vs_pack_rule_ci95": ci([aag[s]["race_time_s"] for s in A], [apr[s]["race_time_s"] for s in A]),
                "vs_no_coach_ci95": ci([aag[s]["race_time_s"] for s in A], [anc[s]["race_time_s"] for s in A])},
        "split": {"early": [d for s, d in zip(S, d_rp_nc) if early(rp[s])],
                  "late": [d for s, d in zip(S, d_rp_nc) if not early(rp[s])]},
        "api_early_let_go": sum(1 for s in A if early(aag[s])),
        "note": "simulated paddler twin with a planted pack; per-race differences in seconds; no positions",
    }
    paths = emit("coaching_drafting_roleplay", data=data, plot_func=plot_roleplay_drafting,
                 artifacts_dir=os.path.join(ROOT, "artifacts"))
    print("figure:", paths)
    return data["split"], data["api_early_let_go"]


def main() -> int:
    figure = "--figure" in sys.argv[1:]
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
        if figure:
            split, api_early = _figure(rp, nc, pr, S, api)
            out["exploratory_let_go_split"] = {k: {"n": len(v), "mean_vs_no_coach_s": float(np.mean(v))}
                                               for k, v in split.items()}
            out["exploratory_let_go_split"]["api_races_let_go_in_first_5_min"] = api_early
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
