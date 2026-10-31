#!/usr/bin/env python3
"""Run and summarise the closed-loop coaching experiment on the paddler twin.

    python experiments/run_coaching_trial.py run --seeds dev --arms baselines,stand-in --tag dev
    python experiments/run_coaching_trial.py run --seeds eval --arms agent --tag eval \\
        --provider anthropic --model MODEL --api-key-env PADDLING_COACH_API_KEY
    python experiments/run_coaching_trial.py summarize --tag eval [--figure coaching_trial]
    python experiments/run_coaching_trial.py selftest

Design, outcomes and tests: ``paddling_expt.coaching_trial.PREREGISTERED`` and the
closed-loop coaching section of ``PADDLING_PLAN.md``.  Each seed's preparation is
cached in ``results/coaching_trial/preparations.json``; every episode is appended to
``episodes_<tag>.jsonl`` (full, with time series; not tracked) and the compact table
``episodes_<tag>.json`` (tracked), so an interrupted run resumes where it stopped.
Model calls of the agent arm are recorded per seed in ``records/`` for re-execution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from paddling_expt import coaching_trial as ct                      # noqa: E402
from paddling_expt.coach import make_coach_provider, stand_in_coach  # noqa: E402

OUT = os.path.join(ROOT, "results", "coaching_trial")
PREP_CACHE = os.path.join(OUT, "preparations.json")
COMPACT_DROP = ("series", "definitions")
# list prices, USD per million tokens (input, output, cache read, cache write), 2026-10
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
          "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5)}


def estimate_usd(usage: Dict[str, Any], model: str) -> float:
    pi, po, pr, pw = PRICES.get(model, (4.0, 20.0, 0.40, 5.0))
    return (usage.get("input_tokens", 0) * pi + usage.get("output_tokens", 0) * po
            + usage.get("cache_read_tokens", 0) * pr + usage.get("cache_create_tokens", 0) * pw) / 1e6


def seed_list(spec: str) -> List[int]:
    pre = ct.PREREGISTERED["seeds"]
    if spec == "dev":
        return list(pre["development"])
    if spec == "eval":
        return list(pre["evaluation"])
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ── preparations ────────────────────────────────────────────────────────────

def _load_preps() -> Dict[str, Any]:
    if os.path.exists(PREP_CACHE):
        with open(PREP_CACHE) as f:
            return json.load(f)
    return {}


COASTAL = False


def _prep_one_coastal(seed: int) -> Dict[str, Any]:
    return ct.draw_preparation(seed, coastal=True).to_dict()


def _prep_one(seed: int) -> Dict[str, Any]:
    return ct.draw_preparation(seed, coastal=COASTAL).to_dict()


def preparations(seeds: List[int], workers: int) -> Dict[int, ct.Preparation]:
    cache = _load_preps()
    todo = [s for s in seeds if str(s) not in cache]
    if todo:
        print(f"preparing {len(todo)} seeds ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for d in ex.map(_prep_one_coastal if COASTAL else _prep_one, todo):
                cache[str(d["seed"])] = d
        os.makedirs(OUT, exist_ok=True)
        with open(PREP_CACHE, "w") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
    lacking = [s for s in seeds if "athlete_shaft" not in cache[str(s)]]
    if lacking:
        print(f"adding the pod calibration to {len(lacking)} cached seeds ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s, d in zip(lacking, ex.map(_pod_one, [cache[str(s)] for s in lacking])):
                cache[str(s)]["athlete_shaft"] = d
        with open(PREP_CACHE, "w") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
    return {s: ct.Preparation.from_dict(cache[str(s)]) for s in seeds}


def _pod_one(d: Dict[str, Any]) -> Dict[str, Any]:
    prep = ct.Preparation.from_dict(d)
    a = ct.shaft_athlete(prep.traits, prep.seed)
    return {"cp_w": a.cp_w, "w_prime_j": a.w_prime_j, "source": a.source}


# ── arms ────────────────────────────────────────────────────────────────────

def _arm_key(r: Dict[str, Any]) -> str:
    return f"{r['seed']}|{r['arm']}|{json.dumps(r['setting'], sort_keys=True)}"


def _make_arms(spec: str, a: argparse.Namespace) -> List[Any]:
    arms: List[Any] = []
    for part in spec.split(","):
        if part == "baselines":
            arms += ct.baseline_arms()
        elif part == "no_coach":
            arms.append(ct.NoCoach())
        elif part == "line_rule":
            arms += [ct.LineRule(60.0), ct.LineRule(120.0)]
        elif part == "autopilot_pod":
            pre = ct.PREREGISTERED
            arms += [ct.PodAutopilot(m, f) for m in pre["autopilot_sweep"]["overpace_margin_m"]
                     for f in pre["autopilot_sweep"]["fade_pct"]]
        elif part == "stand-in":
            arms.append(("stand-in", None))
        elif part == "agent":
            arms.append(("agent", None))
        else:
            raise SystemExit(f"unknown arm {part!r}")
    return arms


def _episode(prep_d: Dict[str, Any], arm_spec: Any, opts: Dict[str, Any]) -> Dict[str, Any]:
    prep = ct.Preparation.from_dict(prep_d)
    counting = None
    record_path = None
    if isinstance(arm_spec, tuple):
        kind = arm_spec[0]
        if kind == "stand-in":
            arm = ct.AgentArm(stand_in_coach(), label="agent_stand_in")
            latency = 6.0
        else:
            prov = make_coach_provider(opts["provider"], opts.get("model"),
                                       api_key_env=opts["api_key_env"], api_base_url=opts["api_base_url"])
            counting = ct.CountingProvider(prov)
            arm = ct.AgentArm(counting, label=opts.get("agent_label") or "agent",
                              coach_version=opts.get("coach_version", 1))
            arm.setting["model"] = getattr(prov, "model", "?")
            latency = None                       # each call takes its measured time
            os.makedirs(os.path.join(OUT, "records"), exist_ok=True)
            record_path = os.path.join(OUT, "records", f"{opts['tag']}_{arm.name}_{prep.seed}.jsonl")
    else:
        arm, latency = arm_spec, 6.0
    t0 = time.perf_counter()
    r = ct.run_episode(prep, arm, latency_s=latency, record_path=record_path)
    r["wall_s"] = round(time.perf_counter() - t0, 2)
    if counting is not None:
        r["model_usage"] = counting.totals()
    return r


def run(a: argparse.Namespace) -> None:
    global COASTAL, PREP_CACHE
    if getattr(a, "coastal", False):
        COASTAL = True
        PREP_CACHE = os.path.join(OUT, "preparations_coastal.json")
    seeds = seed_list(a.seeds)
    preps = preparations(seeds, a.workers)
    arms = _make_arms(a.arms, a)
    full = os.path.join(OUT, f"episodes_{a.tag}.jsonl")
    done = set()
    if os.path.exists(full):
        with open(full) as f:
            for line in f:
                done.add(_arm_key(json.loads(line)))
    opts = {"provider": a.provider, "model": a.model, "api_key_env": a.api_key_env,
            "api_base_url": a.api_base_url, "tag": a.tag, "agent_label": a.agent_label,
            "coach_version": a.coach_version}
    jobs = []
    for s in seeds:
        for arm in arms:
            name = arm[0] if isinstance(arm, tuple) else arm.name
            setting = {} if isinstance(arm, tuple) else arm.setting
            label = {"stand-in": "agent_stand_in", "agent": a.agent_label}.get(name, name)
            key_prefix = f"{s}|{label}|"
            if isinstance(arm, tuple):
                if any(k.startswith(key_prefix) for k in done):
                    continue
            elif f"{key_prefix}{json.dumps(setting, sort_keys=True)}" in done:
                continue
            jobs.append((s, arm))
    print(f"{len(jobs)} episodes to run ({len(done)} done)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    spent = 0.0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_episode, preps[s].to_dict(), arm, opts): (s, arm) for s, arm in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as exc:              # noqa: BLE001
                s, arm = futs[fut]
                print(f"  seed {s} arm {arm}: FAILED {exc!r}", flush=True)
                continue
            if r.get("model_usage", {}).get("errors"):
                # a race in which the coach's model failed is not a coached race: keep it out
                with open(os.path.join(OUT, f"failed_{a.tag}.jsonl"), "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  seed {r['seed']} {r['arm']}: {r['model_usage']['errors']} failed model calls "
                      f"({r.get('consult_error_example', '')}) - not recorded; stopping", flush=True)
                for f_ in futs:
                    f_.cancel()
                break
            if r.get("model_usage"):
                r["model_usd_estimate"] = round(estimate_usd(r["model_usage"], r["setting"].get("model", "")), 4)
                spent += r["model_usd_estimate"]
                u = r["model_usage"]
                print(f"  seed {r['seed']} {r['arm']}: {r['race_time_s'] / 60:.2f} min, {r['cues_spoken']} cues, "
                      f"{u['calls']} model calls ({u['errors']} errors), median call {u.get('latency_median_s', 0):.1f} s, "
                      f"~${r['model_usd_estimate']:.2f} (total ~${spent:.2f})", flush=True)
            with open(full, "a") as f:
                f.write(json.dumps(r) + "\n")
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes, {time.perf_counter() - t0:.0f} s", flush=True)
            if a.max_usd is not None and spent > a.max_usd:
                print(f"  spend ~${spent:.2f} passed --max-usd {a.max_usd}: cancelling the rest", flush=True)
                for f_ in futs:
                    f_.cancel()
                break
    write_compact(a.tag)


def load(tag: str) -> List[Dict[str, Any]]:
    full = os.path.join(OUT, f"episodes_{tag}.jsonl")
    with open(full) as f:
        return [json.loads(line) for line in f]


def write_compact(tag: str) -> None:
    rows = [{k: v for k, v in r.items() if k not in COMPACT_DROP} for r in load(tag)]
    rows.sort(key=lambda r: (r["seed"], r["arm"], json.dumps(r["setting"], sort_keys=True)))
    with open(os.path.join(OUT, f"episodes_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=0)


# ── summary ─────────────────────────────────────────────────────────────────

def _by(rows, arm, setting=None):
    return {r["seed"]: r for r in rows if r["arm"] == arm
            and (setting is None or r["setting"] == setting) and r.get("race_time_s")}


def _select(rows, family: str, seeds: List[int]) -> Dict[str, Any]:
    settings = []
    for r in rows:
        if r["arm"] == family and r["setting"] not in settings:
            settings.append(r["setting"])
    ranked = []
    for st in settings:
        b = _by(rows, family, st)
        if all(s in b for s in seeds):
            ranked.append((float(np.mean([b[s]["race_time_s"] for s in seeds])), st))
    ranked.sort(key=lambda x: x[0])
    if not ranked:
        return {}
    return {"best": ranked[0][1], "typical": ranked[(len(ranked) - 1) // 2][1],
            "ranking": [{"setting": st, "mean_race_time_s": m} for m, st in ranked]}


def summarize(tag: str, agent_arm: str = "agent", figure: Optional[str] = None) -> Dict[str, Any]:
    rows = load(tag)
    write_compact(tag)
    ag = _by(rows, agent_arm)
    nc = _by(rows, "no_coach")
    seeds = sorted(s for s in ag if s in nc)
    fams = [f for f in ("fixed_plan", "autopilot", "autopilot_pod", "line_rule") if any(r["arm"] == f for r in rows)]
    sel = {fam: _select(rows, fam, seeds) for fam in fams}
    comps = {"no_coach": nc}
    for fam in fams:
        for which in ("best", "typical"):
            if sel[fam]:
                comps[f"{fam}_{which}"] = _by(rows, fam, sel[fam][which])
    for other in ("agent", "agent_haiku", "agent_stand_in", "agent_v2", "agent_v3"):
        if other != agent_arm and _by(rows, other):
            comps[other] = _by(rows, other)
    t = lambda d: [d[s]["race_time_s"] for s in seeds]                           # noqa: E731
    out: Dict[str, Any] = {"tag": tag, "agent_arm": agent_arm, "n_seeds": len(seeds), "seeds": seeds,
                           "selection": sel, "comparisons": {}}
    if agent_arm == "agent_stand_in":
        out["honesty"] = ct.HONESTY_STAND_IN
    elif agent_arm != "agent":
        out["note"] = ("secondary arm: the Holm-adjusted p values are shown for reference; the "
                       "pre-registered primary family belongs to the 'agent' arm")
    for name, d in comps.items():
        if all(s in d for s in seeds):
            out["comparisons"][name] = ct.paired(t(ag), t(d))
    ref = [ag[s]["reference_s"] for s in seeds]
    out["comparisons"]["reference"] = ct.paired(t(ag), ref)
    primary = {k: out["comparisons"].get(k, {}).get("wilcoxon_p")
               for k in ("no_coach", "fixed_plan_best", "autopilot_best")}
    out["holm_primary"] = ct.holm(primary)
    arms_table = {}
    for name, d in [(agent_arm, ag)] + list(comps.items()):
        if not all(s in d for s in seeds):
            continue
        gaps = [(nc[s]["race_time_s"] - d[s]["race_time_s"]) / (nc[s]["race_time_s"] - d[s]["reference_s"])
                for s in seeds if nc[s]["race_time_s"] - d[s]["reference_s"] >= 10.0]
        arms_table[name] = {
            "mean_race_time_s": float(np.mean(t(d))),
            "mean_vs_no_coach_s": float(np.mean([d[s]["race_time_s"] - nc[s]["race_time_s"] for s in seeds])),
            "mean_blown_s": float(np.mean([d[s]["blown_s"] for s in seeds])),
            "mean_reserve_left": float(np.mean([d[s]["reserve_left"] for s in seeds])),
            "mean_cues": float(np.mean([d[s]["cues_spoken"] for s in seeds])),
            "followed_share": (float(np.sum([d[s]["cues_followed"] for s in seeds]))
                               / max(1.0, float(np.sum([d[s]["cues_spoken"] for s in seeds])))),
            "median_gap_closed": float(np.median(gaps)) if gaps else None, "n_gap": len(gaps)}
    out["arms"] = arms_table
    out["mean_reference_vs_no_coach_s"] = float(np.mean([nc[s]["reference_s"] - nc[s]["race_time_s"] for s in seeds]))
    out["breakdown"] = breakdown(rows, seeds, {agent_arm: ag, **comps})
    usage = [ag[s].get("model_usage") for s in seeds if ag[s].get("model_usage")]
    if usage:
        out["model_usage"] = {k: int(sum(u[k] for u in usage)) for k in (
            "calls", "errors", "input_tokens", "output_tokens", "cache_create_tokens", "cache_read_tokens")}
        out["model_usage"]["latency_median_s"] = float(np.median([u["latency_median_s"] for u in usage]))
    path = os.path.join(OUT, f"summary_{tag}_{agent_arm}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    _print(out)
    if figure:
        make_figure(tag, agent_arm, figure, out)
    return out


def breakdown(rows, seeds, arms: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """Gains vs no coach split by the paddler's own pacing (over- vs under-pacers: their
    planted body effort above or below their even pace), and what each coach defined."""
    preps = _load_preps()
    nc = arms.get("no_coach") or _by(rows, "no_coach")
    over = [s for s in seeds if preps[str(s)]["traits"]["body_bias"] > 1.0]
    under = [s for s in seeds if s not in over]
    out: Dict[str, Any] = {"n_over_pacers": len(over), "n_under_pacers": len(under), "arms": {}}
    for name, d in arms.items():
        if name == "no_coach" or not all(s in d for s in seeds):
            continue
        g = lambda ss: float(np.mean([d[s]["race_time_s"] - nc[s]["race_time_s"] for s in ss])) if ss else None  # noqa: E731
        row = {"vs_no_coach_over_pacers_s": g(over), "vs_no_coach_under_pacers_s": g(under)}
        defs = [d[s].get("definitions") for s in seeds if d[s].get("definitions")]
        if defs and name.startswith("agent") and name != "agent_stand_in":
            start_w = {"over_pace", "fading", "reserve_conflict", "finish_window"}
            n_w = [len([w for w in x.get("watches", []) if w["watch_id"] not in start_w]) for x in defs]
            n_m = [len([m for m in x.get("metrics", []) if m["name"] != "fade_min_5min"]) for x in defs]
            row.update({"own_watches_per_race": float(np.mean(n_w)), "own_metrics_per_race": float(np.mean(n_m)),
                        "races_with_own_definitions": int(sum(1 for a, b in zip(n_w, n_m) if a + b > 0))})
            asks: Dict[str, int] = {}
            for s in seeds:
                for k, v in (d[s].get("asks") or {}).items():
                    asks[k] = asks.get(k, 0) + v
            row["asks"] = asks
        out["arms"][name] = row
    return out


def _print(out: Dict[str, Any]) -> None:
    print(f"\n{out['agent_arm']} on {out['n_seeds']} seeds ({out['tag']})")
    if out.get("honesty"):
        print("  " + out["honesty"])
    if out.get("note"):
        print("  " + out["note"])
    print(f"  perfect-knowledge even pacer vs no coach: {out['mean_reference_vs_no_coach_s']:+.1f} s (mean)")
    for name, a in out["arms"].items():
        gc = "" if a["median_gap_closed"] is None else f", median share of the gap closed {a['median_gap_closed']:+.2f}"
        print(f"  {name:22s} mean {a['mean_race_time_s'] / 60:6.2f} min ({a['mean_vs_no_coach_s']:+6.1f} s vs no coach), "
              f"blown {a['mean_blown_s']:5.0f} s, cues {a['mean_cues']:4.1f} ({100 * a['followed_share']:.0f} % followed){gc}")
    for name, c in out["comparisons"].items():
        ci = c.get("ci95")
        print(f"  agent - {name:18s} mean {c['mean_d']:+6.1f} s  95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]  "
              f"p {c.get('wilcoxon_p') if c.get('wilcoxon_p') is None else round(c['wilcoxon_p'], 4)}  "
              f"agent faster on {c['n_a_faster']}/{c['n']}" if ci else f"  agent - {name}: n={c['n']}")
    print("  Holm (primary):", {k: (None if v is None else round(v, 4)) for k, v in out["holm_primary"].items()})
    b = out.get("breakdown") or {}
    if b:
        print(f"  by own pacing ({b['n_over_pacers']} over-pacers, {b['n_under_pacers']} under-pacers):")
        for name, row in b["arms"].items():
            extra = ""
            if "own_watches_per_race" in row:
                extra = (f"; wrote {row['own_watches_per_race']:.1f} watch rules and {row['own_metrics_per_race']:.1f} "
                         f"metrics per race (in {row['races_with_own_definitions']} races); asks {row['asks']}")
            print(f"    {name:22s} over-pacers {row['vs_no_coach_over_pacers_s']:+6.1f} s, "
                  f"under-pacers {row['vs_no_coach_under_pacers_s']:+6.1f} s{extra}")
    if out.get("model_usage"):
        print("  model usage:", out["model_usage"])


# ── figure ──────────────────────────────────────────────────────────────────

def plot_coaching_trial(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.25], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    names = data["comparison_names"]
    for i, n in enumerate(names):
        d = np.array(data["paired_d"][n], float)
        jitter = (np.arange(len(d)) % 7 - 3) * 0.04
        ax.scatter(d, i + jitter, s=12, color="0.55", zorder=2)
        c = data["comparisons"][n]
        if c.get("ci95"):
            ax.plot(c["ci95"], [i, i], color="C0", lw=3, zorder=3)
        ax.plot([c["mean_d"]], [i], "o", color="C0", ms=7, zorder=4)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([data["labels"].get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel(f"{data['agent_label']} minus comparator, race time (s)\n(negative = coach faster; dot = seed, bar = 95% CI of the mean)")
    ax.set_title(data["title"], fontsize=10)
    ex = data["example"]
    t_min = None
    axes = [fig.add_subplot(gs[k, 1]) for k in range(3)]
    for arm, tr in ex["traces"].items():
        t_min = np.array(tr["race_s"], float) / 60.0
        axes[0].plot(t_min, tr["reserve_true"], lw=1.3, label=f"{data['labels'].get(arm, arm)} ({tr['race_time_min']:.2f} min)")
        axes[1].plot(t_min, tr["effort"], lw=1.1)
        axes[2].plot(t_min, np.array(tr["rate"]) - np.array(tr["rate_best"]), lw=1.1)
        for c in tr["cues"]:
            axes[0].axvline(c / 60.0, color=axes[0].lines[-1].get_color(), lw=0.5, alpha=0.5)
    axes[0].set_ylabel("true reserve left")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].set_title(f"example race (seed {ex['seed']}): thin lines = cues spoken", fontsize=9)
    axes[1].set_ylabel("effort (x critical power)")
    axes[2].set_ylabel("rate minus best rate (spm)")
    axes[2].axhline(0, color="0.5", lw=0.8)
    axes[2].set_xlabel("race time (min)")
    fig.suptitle(data["suptitle"], fontsize=11)
    return fig


def make_figure(tag: str, agent_arm: str, name: str, summary: Dict[str, Any]) -> None:
    rows = load(tag)
    ag = _by(rows, agent_arm)
    seeds = summary["seeds"]
    sel = summary["selection"]
    sources = {"no_coach": _by(rows, "no_coach")}
    for fam in ("fixed_plan", "autopilot"):
        if sel.get(fam):
            sources[f"{fam}_best"] = _by(rows, fam, sel[fam]["best"])
    labels = {"no_coach": "no coach", "fixed_plan_best": "fixed plan (best setting)",
              "fixed_plan_typical": "fixed plan (typical)", "autopilot_best": "rule autopilot (best)",
              "autopilot_typical": "rule autopilot (typical)", "reference": "perfect-knowledge even pacer",
              "agent_stand_in": "scripted stand-in coach", "agent": "language-model coach (Opus 5.5)",
              "agent_haiku": "language-model coach (Haiku 5.5)"}
    names = [n for n in ("no_coach", "fixed_plan_best", "fixed_plan_typical", "autopilot_best",
                         "autopilot_typical", "agent", "agent_haiku", "agent_stand_in", "reference")
             if n in summary["comparisons"] and n != agent_arm]
    paired_d = {}
    for n in names:
        if n == "reference":
            paired_d[n] = [ag[s]["race_time_s"] - ag[s]["reference_s"] for s in seeds]
        else:
            d = _by(rows, "no_coach") if n == "no_coach" else None
            if d is None:
                fam, which = n.rsplit("_", 1) if n.endswith(("_best", "_typical")) else (n, None)
                d = _by(rows, fam, sel[fam][which]) if which else _by(rows, n)
            paired_d[n] = [ag[s]["race_time_s"] - d[s]["race_time_s"] for s in seeds]
    # example: the seed with the median no-coach gap to the reference
    gaps = sorted(seeds, key=lambda s: sources["no_coach"][s]["race_time_s"] - ag[s]["reference_s"])
    ex_seed = gaps[len(gaps) // 2]
    traces = {}
    for arm, d in [("no_coach", sources["no_coach"])] + [(k, v) for k, v in sources.items() if k != "no_coach"] + [(agent_arm, ag)]:
        r = d[ex_seed]
        S = r["series"]
        traces[arm] = {"race_s": [x["race_s"] for x in S], "reserve_true": [x["reserve_true"] for x in S],
                       "effort": [x["effort"] for x in S], "rate": [x["rate"] for x in S],
                       "rate_best": [x["rate_best"] for x in S], "race_time_min": r["race_time_s"] / 60.0,
                       "cues": [c["race_s"] for c in r["cue_log"]]}
    agent_label = labels.get(agent_arm, agent_arm)
    data = {"title": f"{agent_label} vs comparators, {len(seeds)} simulated paddlers",
            "suptitle": ("Closed-loop coaching on the paddler twin (planted physiology and cue response: "
                         "what coaching is worth if paddlers respond as modelled)"
                         + ("  [stand-in coach: harness check, not evidence]" if agent_arm == "agent_stand_in" else "")),
            "agent_label": agent_label, "labels": labels, "comparison_names": names,
            "comparisons": {n: summary["comparisons"][n] for n in names}, "paired_d": paired_d,
            "example": {"seed": ex_seed, "traces": traces}}
    from agent_lib.figure_artifacts import emit       # agentic_experiments (a dependency)
    paths = emit(name, data=data, plot_func=plot_coaching_trial, artifacts_dir=os.path.join(ROOT, "artifacts"))
    print("wrote", ", ".join(os.path.basename(p) for p in paths.values()))


# ── self-test ───────────────────────────────────────────────────────────────

def selftest() -> None:
    global OUT, PREP_CACHE
    import tempfile
    OUT = tempfile.mkdtemp(prefix="coaching_selftest_")
    PREP_CACHE = os.path.join(OUT, "preparations.json")
    a = argparse.Namespace(seeds="0-2", arms="no_coach,stand-in", tag="selftest", workers=3,
                           provider="stand-in", model=None, api_key_env="ANTHROPIC_API_KEY",
                           api_base_url="https://api.anthropic.com", agent_label="agent", max_usd=None,
                           coach_version=1)
    run(a)
    rows = load("selftest")
    assert len(rows) == 6, len(rows)
    assert all(r["race_time_s"] for r in rows)
    st = [r for r in rows if r["arm"] == "agent_stand_in"]
    assert st and all(r["consultations"] > 0 for r in st)
    print("selftest ok:", OUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", default="dev")
    r.add_argument("--arms", default="baselines,stand-in")
    r.add_argument("--tag", default="dev")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--provider", default="anthropic")
    r.add_argument("--model", default=None)
    r.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    r.add_argument("--api-base-url", default="https://api.anthropic.com")
    r.add_argument("--agent-label", default="agent", help="arm name for this model (agent = primary)")
    r.add_argument("--coach-version", type=int, default=1, help="1 = registered coach; 2+ = later rounds")
    r.add_argument("--coastal", action="store_true", help="water that varies across the course (round 2)")
    r.add_argument("--max-usd", type=float, default=None,
                   help="stop scheduling new episodes once the agent arm's estimated spend passes this")
    s = sub.add_parser("summarize")
    s.add_argument("--tag", default="dev")
    s.add_argument("--agent-arm", default="agent")
    s.add_argument("--figure", default=None)
    s.add_argument("--coastal", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summarize":
        global PREP_CACHE
        if a.coastal:
            PREP_CACHE = os.path.join(OUT, "preparations_coastal.json")
        summarize(a.tag, a.agent_arm, a.figure)
    else:
        selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
