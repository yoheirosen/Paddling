#!/usr/bin/env python3
"""Report across several NK SpeedCoach sessions (CSV exports, or a folder of them).

    python "paddling/demos/speedcoach_sessions.py" FOLDER_OR_FILES... [--boat oc1]
        [--total-kg 93] [--ama-kg 5] [--figure NAME]

For each session it removes the push of current and wind (from how speed depends
on direction of travel) and turns speed into power reaching the boat with the
hull model.  Across sessions it reports:
- best efforts from 1 to 40 minutes;
- a critical-power fit, with predicted race times in still water;
- the best speed seen at each stroke rate, and each session's share of that;
- whether going up in rate bought speed within sessions.

With ``--figure`` it writes the four-file figure artifact to ``paddling/artifacts``
(averages only: no positions).

The raw exports hold the paddler's GPS tracks: keep them out of the repository.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from paddling_expt import sessions as ss                       # noqa: E402
from paddling_expt.boats import CRAFT                           # noqa: E402
from paddling_expt.speedcoach import read_speedcoach            # noqa: E402


def plot_speedcoach_sessions(data):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    cols = plt.get_cmap("tab10")
    S = data["sessions"]
    a = ax[0, 0]
    w = data["windows"]
    for k in range(len(S)):
        m = [i for i, s in enumerate(w["session"]) if s == k]
        a.scatter([w["rate"][i] for i in m], [w["speed"][i] for i in m], s=4, alpha=0.25,
                  color=cols(k % 10), lw=0)
    env = data["envelope"]
    a.plot([p["rate"] for p in env["points"]], [p["speed"] for p in env["points"]], "ko", ms=4,
           label=f"best (top {100 - env['pct']:.0f} % of one-minute windows)")
    rr = np.linspace(env["rate_min"], env["rate_max"], 50)
    a.plot(rr, np.polyval(env["coef"], rr), "k-", lw=1.4)
    a.plot([p["rate"] for p in env["points"]], [p["speed_median"] for p in env["points"]], "--",
           color="0.4", lw=1, label="typical (median)")
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("speed, current and wind removed (m/s)")
    a.set_title("Best speed at each stroke rate, all sessions")
    a.legend(fontsize=7, loc="upper left")
    a = ax[0, 1]
    for k, s in enumerate(S):
        if s["best"]:
            d = sorted(s["best"], key=float)
            a.plot([float(x) / 60 for x in d], [s["best"][x] for x in d], "-", color=cols(k % 10),
                   lw=0.8, alpha=0.5)
    pb = data["pooled_best"]
    a.plot([p["seconds"] / 60 for p in pb], [p["power"] for p in pb], "ko", ms=5,
           label="best of all sessions")
    cp = data["cp"]
    tt = np.linspace(2.0, 45, 100)
    a.plot(tt, cp["cp_w"] + cp["w_prime_j"] / (tt * 60), "k-", lw=1.2,
           label=f"fit: CP {cp['cp_w']:.0f} W + reserve {cp['w_prime_j'] / 1000:.1f} kJ / time")
    a.axhline(cp["cp_w"], color="0.6", lw=0.8, ls=":")
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 20, 40])
    a.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    a.set_xlabel("duration of continuous effort (min)")
    a.set_ylabel("best average power to the boat (W)")
    a.set_title("Best efforts and critical power")
    a.legend(fontsize=7)
    a = ax[1, 0]
    for k, s in enumerate(S):
        ln = s.get("line")
        if ln:
            x = np.array([ln["rate_lo"], ln["rate_hi"]])
            a.plot(x, ln["intercept"] + ln["slope"] * x, color=cols(k % 10), lw=1.6,
                   label=s["date"][5:])
    re = data["rate_effect"]["within"]
    a.set_xlabel("stroke rate (strokes/min)")
    a.set_ylabel("speed, current and wind removed (m/s)")
    a.set_title(f"Within each session: {re['speed_per_spm'] * 100:+.1f} cm/s per extra stroke/min\n"
                f"(95 % {re['speed_ci'][0] * 100:+.1f} to {re['speed_ci'][1] * 100:+.1f}); "
                f"distance per stroke {re['dps_per_spm'] * 100:+.1f} cm", fontsize=10)
    a.legend(fontsize=6, ncol=3, loc="upper left")
    a = ax[1, 1]
    x = np.arange(len(S))
    pct = [s.get("pct_best_median") or 0 for s in S]
    bars = a.bar(x, pct, color=[cols(k % 10) for k in range(len(S))], alpha=0.7)
    for b, s in zip(bars, S):
        if s["place"] != 0:
            b.set_hatch("//")
    a.set_ylim(0, 110)
    a.set_ylabel("median % of best power at the rate paddled")
    a.set_xticks(x)
    a.set_xticklabels([s["date"][5:] for s in S], rotation=45, fontsize=7)
    a2 = a.twinx()
    a2.plot(x, [s["best"].get("300", np.nan) for s in S], "kD", ms=5, label="best 5-min power")
    a2.set_ylabel("best 5-minute power (W)")
    a2.legend(fontsize=7, loc="lower right")
    a.set_title("Session by session (hatched: other water)")
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


def load(paths, craft, total_kg, ama_kg):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    S, skipped = [], []
    for f in files:
        try:
            s = ss.prepare_session(read_speedcoach(f), craft, total_kg, ama_kg=ama_kg)
        except ValueError as e:
            skipped.append((os.path.basename(f), str(e)))
            continue
        if s is None:
            skipped.append((os.path.basename(f), "too short"))
        else:
            S.append(s)
    S.sort(key=lambda s: s["start_local"] or 0)
    return S, skipped


def analyse(S, craft, total_kg, ama_kg):
    for s, g in zip(S, ss.place_groups(S)):
        s["place"] = g
        s["wins"] = ss.windows(s["all_rows"])
        s["wins_sep"] = ss.windows(s["all_rows"], step_s=60.0)    # non-overlapping, for slopes
    env = ss.rate_envelope([w for s in S for w in s["wins"]])
    best = {}
    for s in S:
        ss.percent_of_best(s["wins"], env, craft, total_kg, ama_kg=ama_kg)
        s["summary"] = ss.session_summary(s, s["wins"])
        for d, b in ss.best_efforts(s["all_rows"]).items():
            if d not in best or b["power"] > best[d]["power"]:
                best[d] = dict(b, date=s["summary"]["date"])
    cp = ss.critical_power(best)
    split = float(np.median([w["rate"] for s in S for w in s["wins_sep"]]))
    re = ss.rate_effect([s["wins_sep"] for s in S], split_rate=round(split))
    return env, best, cp, re


def report(S, skipped, env, best, cp, re, craft, total_kg, ama_kg):
    print(f"{len(S)} sessions ({craft.name}, {total_kg:.0f} kg with paddler"
          f"{f', {ama_kg:.0f} kg assumed on the ama' if craft.outrigger else ''}); "
          f"skipped: {', '.join(f'{a} ({b})' for a, b in skipped) or 'none'}")
    other = [s["summary"]["date"] for s in S if s["place"] != 0]
    if other:
        print(f"  on other water (more than 5 km from the usual place): {', '.join(other)}")
    print("\n  date        km   min  pace/500  rate (10-90%)   power  push of current+wind   % of best")
    for s in S:
        m = s["summary"]
        print(f"  {m['date']}  {m['km']:4.1f}  {m['minutes']:4.0f}  {m['pace_500m_s']:6.0f} s  "
              f"{m['rate_median']:3.0f} ({m['rate_p10']:.0f}-{m['rate_p90']:.0f})   "
              f"{m['power_median_w']:4.0f} W  {m['push_mps']:.2f} m/s towards {m['push_towards_deg']:3.0f}°   "
              f"{m.get('pct_best_median', float('nan')):4.0f} %")
    pushes = [s["summary"]["push_mps"] for s in S]
    v0 = float(np.median([s["summary"]["km"] * 1000 / (s["summary"]["minutes"] * 60) for s in S]))
    print(f"\nCurrent and wind: {min(pushes):.2f}-{max(pushes):.2f} m/s; at {v0:.1f} m/s a push of "
          f"{max(pushes):.2f} m/s is worth {500 / (v0 - max(pushes)) - 500 / v0:.0f} s per 500 m into it.")
    print("\nBest efforts (continuous paddling, power to the boat, current and wind removed):")
    for d, b in best.items():
        print(f"  {d / 60:4.0f} min  {b['power']:5.0f} W  {b['speed']:.2f} m/s  rate {b['rate']:4.0f}  ({b['date']})")
    if cp:
        print(f"\nCritical power {cp['cp_w']:.0f} W, reserve above it {cp['w_prime_j'] / 1000:.1f} kJ "
              f"(fit to 2-20 min best efforts, r² {cp['r2']:.3f}).")
        print("  Predicted times in still water (training best efforts, so conservative):")
        for x in ss.race_prediction(cp, craft, total_kg, [1000, 5000, 10000, 15000, 20000], ama_kg=ama_kg):
            t = x["seconds"]
            flag = "  (past ~40 min the model is optimistic)" if x["beyond_model"] else ""
            print(f"    {x['distance_m'] / 1000:4.0f} km  {int(t // 3600)}:{int(t % 3600 // 60):02d}:"
                  f"{int(t % 60):02d}  {x['pace_500m_s']:5.1f} s/500 m  {x['power_w']:4.0f} W{flag}")
    print(f"\nBest speed at each stroke rate (top {100 - env['pct']:.0f} % of one-minute windows):")
    for p in env["points"]:
        print(f"  {p['rate']:4.1f} spm  best {p['speed']:.2f} m/s  typical {p['speed_median']:.2f}  (n={p['n']})")
    print(f"  {'peak at ' + format(env['peak_rate'], '.1f') + ' spm' if env['peak_rate'] else 'no peak inside the rates paddled'}; "
          f"overall {env['slope_mps_per_spm'] * 100:+.1f} cm/s per stroke/min")
    w = re["within"]
    print(f"\nGoing up in rate within a session ({re['n_sessions']} sessions, {re['n_windows']} one-minute windows):")
    print(f"  speed {w['speed_per_spm'] * 100:+.1f} cm/s per extra stroke/min (95 % {w['speed_ci'][0] * 100:+.1f} "
          f"to {w['speed_ci'][1] * 100:+.1f}); distance per stroke {w['dps_per_spm'] * 100:+.1f} cm "
          f"({w['dps_ci'][0] * 100:+.1f} to {w['dps_ci'][1] * 100:+.1f})")
    for k, lab in (("within_below", "below"), ("within_above", "at or above")):
        x = re[k]
        print(f"  {lab} {re['split_rate']:.0f} spm: speed {x['speed_per_spm'] * 100:+.1f} cm/s per stroke/min "
              f"({x['speed_ci'][0] * 100:+.1f} to {x['speed_ci'][1] * 100:+.1f}), distance per stroke "
              f"{x['dps_per_spm'] * 100:+.1f} cm")
    print(f"  between sessions: {re['between_speed_per_spm'] * 100:+.1f} cm/s per stroke/min of session mean rate")
    print("  Rate moved with effort in these sessions, so this is what going up in rate gave,\n"
          "  not what rate alone buys; a paddle force sensor would separate the two.")


def figure_data(S, env, best, cp, re):
    wr, wv, wk = [], [], []
    for k, s in enumerate(S):
        for x in s["wins"][::2]:
            wr.append(round(x["rate"], 2))
            wv.append(round(x["speed"], 3))
            wk.append(k)
    sess = []
    for k, s in enumerate(S):
        m = s["summary"]
        ws = s["wins_sep"]
        line = None
        if len(ws) >= 8:
            r = np.array([x["rate"] for x in ws])
            v = np.array([x["speed"] for x in ws])
            b1, b0 = np.polyfit(r, v, 1)
            line = {"slope": float(b1), "intercept": float(b0),
                    "rate_lo": float(np.percentile(r, 5)), "rate_hi": float(np.percentile(r, 95))}
        sess.append({"date": m["date"], "place": int(s["place"]), "km": round(m["km"], 2),
                     "pct_best_median": m.get("pct_best_median"),
                     "best": {str(d): round(p, 1) for d, p in m["best_w"].items()},
                     "push_mps": round(m["push_mps"], 3), "line": line})
    return {"title": f"{len(S)} SpeedCoach sessions, {S[0]['summary']['date']} to {S[-1]['summary']['date']}: "
                     "speed with current and wind removed, power from the hull model",
            "windows": {"rate": wr, "speed": wv, "session": wk},
            "envelope": {k: env[k] for k in ("pct", "points", "coef", "rate_min", "rate_max", "peak_rate")},
            "pooled_best": [{"seconds": float(d), "power": b["power"]} for d, b in best.items()],
            "cp": {"cp_w": cp["cp_w"], "w_prime_j": cp["w_prime_j"]},
            "rate_effect": {"within": re["within"]}, "sessions": sess}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="SpeedCoach CSV exports, or folders of them")
    ap.add_argument("--boat", choices=sorted(CRAFT), default="oc1")
    ap.add_argument("--total-kg", type=float, default=93.0, help="paddler + boat (kg)")
    ap.add_argument("--ama-kg", type=float, default=5.0, help="weight assumed on the ama (OC1)")
    ap.add_argument("--figure", default=None, help="artifact base name to write")
    a = ap.parse_args(argv)
    craft = CRAFT[a.boat]
    S, skipped = load(a.paths, craft, a.total_kg, a.ama_kg)
    if len(S) < 2:
        print("need at least two sessions long enough to analyse")
        return 1
    env, best, cp, re = analyse(S, craft, a.total_kg, a.ama_kg)
    report(S, skipped, env, best, cp, re, craft, a.total_kg, a.ama_kg)
    if a.figure:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
        from agent_lib.figure_artifacts import emit
        paths = emit(a.figure, data=figure_data(S, env, best, cp, re), plot_func=plot_speedcoach_sessions,
                     artifacts_dir=os.path.join(os.path.dirname(HERE), "artifacts"))
        print("\nwrote", ", ".join(os.path.basename(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
