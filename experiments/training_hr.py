#!/usr/bin/env python3
"""Reading effort zones through a training block: the fixed arms, the summary and the figure.

    python experiments/training_hr.py run --pool development --seeds 1001-1012 --tag dev
    python experiments/training_hr.py summary --tag dev [--tag dev_rp ...] [--figure training_hr_dev]

``run`` plays every block with the fixed arms (textbook zones from the maximum heart rate,
the personal heart-rate model as calibrated, the model refitted after every session, pace
only, the warm-up rule; ``--oracle`` adds a
reference that is told the planted day state, not a competitor) and appends one line per block
and arm to ``results/training_hr/blocks_<tag>.jsonl``.  Role-played agent blocks are added
with ``roleplay_training.py collect``.  ``summary`` pools the tags, scores each arm on steady
effort (share of checkpoints in the right zone, in a wrong zone, "cannot tell"), splits by
day kind and heart-rate shape, compares the agent with every fixed arm per block (paired
Wilcoxon, bootstrap interval, Holm), and writes the figure.  Everything here is simulated:
the paddlers, their heart rates and the day effects are planted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from paddling_expt import hr_training as ht                        # noqa: E402
from paddling_expt.coaching_trial import holm, paired              # noqa: E402
from paddling_expt.sim.training import draw_block                  # noqa: E402

OUT = os.path.join(ROOT, "results", "training_hr")
ARMS = {"textbook": ht.TextbookArm, "hr_model": ht.HrModelArm, "pace": ht.PaceArm, "pace_forecast": ht.PaceForecastArm,
        "warmup_rule": ht.WarmupRuleArm, "refit": ht.RefitArm, "oracle_day": ht.OracleDayArm}
FIXED = ["textbook", "hr_model", "refit", "pace", "pace_forecast", "warmup_rule"]


def _seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _one(job) -> List[Dict[str, Any]]:
    seed, pool, arms = job
    b = draw_block(seed, pool=pool)
    cal = ht.calibrate_block(b)
    return [ht.run_block(b, ARMS[a](), cal=cal) for a in arms]


def cmd_run(a) -> int:
    arms = FIXED + (["oracle_day"] if a.oracle else [])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"blocks_{a.tag}.jsonl")
    jobs = [(s, a.pool, arms) for s in _seeds(a.seeds)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(path, "a") as f:
        for res in ex.map(_one, jobs):
            for r in res:
                f.write(json.dumps(r) + "\n")
            print(f"seed {res[0]['seed']} ({res[0]['shape']}): " +
                  "  ".join(f"{r['arm']} {r['total']['right_frac']:.2f}/{r['total']['wrong_frac']:.2f}" for r in res),
                  flush=True)
    return 0


def _load(tags: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for t in tags:
        p = os.path.join(OUT, f"blocks_{t}.jsonl")
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _frac(sessions: List[Dict[str, Any]], key: str) -> float:
    n = sum(s["n"] for s in sessions)
    return sum(s[key] for s in sessions) / n if n else float("nan")


def summarize(rows: List[Dict[str, Any]], agent: str = None) -> Dict[str, Any]:
    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["seed"]] = r            # the last line for an arm and seed wins
    arms = sorted(by_arm, key=lambda k: (k not in FIXED, FIXED.index(k) if k in FIXED else 0, k))
    out: Dict[str, Any] = {"arms": {}, "by_day": {}, "by_shape": {}, "paired": {}}
    for arm in arms:
        B = list(by_arm[arm].values())
        S = [s for b in B for s in b["sessions"]]
        out["arms"][arm] = {"blocks": len(B), "checkpoints": sum(s["n"] for s in S),
                            "right": round(_frac(S, "right"), 3), "wrong": round(_frac(S, "wrong"), 3),
                            "cannot_tell": round(_frac(S, "cannot_tell"), 3),
                            "consultations": sum(b.get("consultations", 0) for b in B)}
        days: Dict[str, List] = defaultdict(list)
        shapes: Dict[str, List] = defaultdict(list)
        for b in B:
            for s in b["sessions"]:
                days[s["day"]].append(s)
            shapes[b["shape"]] += b["sessions"]
        out["by_day"][arm] = {d: {"sessions": len(v), "right": round(_frac(v, "right"), 3),
                                  "wrong": round(_frac(v, "wrong"), 3)} for d, v in sorted(days.items())}
        out["by_shape"][arm] = {k: {"right": round(_frac(v, "right"), 3), "wrong": round(_frac(v, "wrong"), 3)}
                                for k, v in sorted(shapes.items())}
    if agent and agent in by_arm:
        seeds = sorted(by_arm[agent])
        for measure in ("right", "wrong"):
            res, ps = {}, {}
            for arm in FIXED:
                if arm not in by_arm or not all(s in by_arm[arm] for s in seeds):
                    continue
                x = [_frac(by_arm[agent][s]["sessions"], measure) for s in seeds]
                y = [_frac(by_arm[arm][s]["sessions"], measure) for s in seeds]
                res[arm] = paired(x, y)
                ps[arm] = res[arm].get("wilcoxon_p")
            for arm, p in holm(ps).items():
                res[arm]["holm_p"] = p
            out["paired"][measure] = res
    return out


def plot_training_hr(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    arms = data["arms"]
    days = data["days"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    y = np.arange(len(arms))
    right = [data["overall"][a]["right"] for a in arms]
    wrong = [data["overall"][a]["wrong"] for a in arms]
    cant = [data["overall"][a]["cannot_tell"] for a in arms]
    ax.barh(y, right, color="#2e7d32", label="right zone")
    ax.barh(y, cant, left=right, color="#bdbdbd", label="cannot tell")
    ax.barh(y, wrong, left=np.add(right, cant), color="#c62828", label="wrong zone")
    ax.set_yticks(y, [data["labels"][a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of steady-effort checkpoints")
    ax.set_title("All sessions")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax = axes[1]
    w = 0.8 / len(arms)
    x = np.arange(len(days))
    for i, a in enumerate(arms):
        v = [data["by_day"][a].get(d, {}).get("right", np.nan) for d in days]
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, v, w, label=data["labels"][a])
    ax.set_xticks(x, [d.replace("_", " ") for d in days])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share in the right zone")
    ax.set_title("By the day's planted state")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(data["title"], fontsize=10)
    fig.tight_layout()
    return fig


LABELS = {"textbook": "textbook zones (max heart rate)", "hr_model": "personal model, as calibrated",
          "refit": "personal model, refitted each session",
          "pace_forecast": "GPS pace + forecast current",
          "pace": "GPS pace only", "warmup_rule": "warm-up offset rule", "oracle_day": "told the day (reference)"}


def cmd_summary(a) -> int:
    rows = _load(a.tag)
    s = summarize(rows, agent=a.agent)
    print(json.dumps({k: s[k] for k in ("arms", "by_shape")}, indent=1))
    print(json.dumps(s["by_day"], indent=1))
    if s["paired"]:
        for m, res in s["paired"].items():
            for arm, r in res.items():
                print(f"{a.agent} - {arm} ({m}): mean {r['mean_d']:+.3f} CI {r.get('ci95')} "
                      f"p {r.get('wilcoxon_p')} Holm {r.get('holm_p')}")
    if a.figure:
        from agent_lib.figure_artifacts import emit
        arms = list(s["arms"])
        days = sorted({d for arm in arms for d in s["by_day"][arm]})
        data = {"title": a.title or "Zone read through a simulated training block (planted paddlers and days)",
                "arms": arms, "days": days, "labels": {k: LABELS.get(k, k) for k in arms},
                "overall": s["arms"], "by_day": s["by_day"]}
        emit(a.figure, data=data, plot_func=plot_training_hr, artifacts_dir=os.path.join(ROOT, "artifacts"))
        print("figure", a.figure)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pool", choices=["development", "held_out"], default="development")
    r.add_argument("--seeds", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--oracle", action="store_true")
    r.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("summary")
    s.add_argument("--tag", action="append", required=True)
    s.add_argument("--agent", default=None, help="the arm compared with every fixed arm per block")
    s.add_argument("--figure", default=None)
    s.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    raise SystemExit(main())
