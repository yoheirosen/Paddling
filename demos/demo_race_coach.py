#!/usr/bin/env python3
"""The in-race coach, replayed: race state, screen and coach on a virtual clock.

    python demos/demo_race_coach.py [--seed N] [--figure NAME]
        [--provider stand-in|anthropic|openai|ollama] [--model M] [--latency-s 6]
    python demos/demo_race_coach.py --speedcoach SESSION.csv --calibrate-from FOLDER
        [--distance-m 15000]

Simulated (default): the paddler does the critical-power test (all-out 3 and
12 minutes), which calibrates the race state; then two races on out-and-back legs
in a current, one over-paced (its planted reserve runs out before the finish) and
one evenly paced.  Each is replayed through the race state, the screen and the
coach, every model call taking ``--latency-s`` of race time.  The estimated
reserve is compared with the planted one; the screen events and cues are listed.

Real (``--speedcoach``): replays a SpeedCoach session as if live, calibrated from
the best efforts of the other sessions in ``--calibrate-from``.

The default coach is a scripted stand-in (no network): it shows the loop and the
timing, not what a model would say.  ``--provider anthropic`` puts a real model
in the coach's seat; the key is read from ``--api-key-env`` (default
``ANTHROPIC_API_KEY``; Claude Code cloud sessions withhold that name, so store the
key under another one there) and sent only to ``--api-base-url``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from paddling_expt import sessions as ss                                     # noqa: E402
from paddling_expt.boats import CRAFT                                         # noqa: E402
from paddling_expt.coach import (CueSink, RaceCoach, make_coach_provider,  # noqa: E402
                                 make_cue_gate, replay_race)
from paddling_expt.fusion import OC1Engine                                    # noqa: E402
from paddling_expt.headunit import HeadConfig                                 # noqa: E402
from paddling_expt.race import RaceState, calibrate                           # noqa: E402
from paddling_expt.scoring import true_time                                   # noqa: E402
from paddling_expt.sim.paddler import PaddlerParams                           # noqa: E402
from paddling_expt.sim.scenarios import cp_test, paced_race                   # noqa: E402
from paddling_expt.speedcoach import add_headings, read_speedcoach            # noqa: E402


def plot_oc1_race_coach(data):
    import numpy as np
    import matplotlib.pyplot as plt
    races = data["races"]
    fig, ax = plt.subplots(3, len(races), figsize=(6.0 * len(races), 9), sharex="col", squeeze=False)
    for j, r in enumerate(races):
        t = np.array(r["t_min"])
        a = ax[0, j]
        a.plot(t, r["reserve_planted"], color="0.3", lw=1.6, label="planted (paddler's effort)")
        a.plot(t, r["reserve_est"], color="C0", lw=1.4, label="estimated (boat speed + hull model)")
        if r.get("exhausted_min") is not None:
            a.axvline(r["exhausted_min"], color="C3", ls="--", lw=1, label="planted reserve runs out")
        a.set_ylim(-0.05, 1.05)
        a.set_ylabel("reserve above critical power")
        a.set_title(r["title"])
        a.legend(fontsize=7, loc="lower left")
        a = ax[1, j]
        a.plot(t, r["fade_pct"], color="C2", lw=1.4)
        a.axhline(-4.0, color="C3", lw=0.8, ls=":")
        a.axhline(0.0, color="0.7", lw=0.8)
        a.set_ylabel("speed vs same rate early in race (%)")
        a = ax[2, j]
        kinds = ["over_pace", "fading", "reserve_conflict", "finish_window"]
        for e in r["screen"]:
            y = kinds.index(e["kind"]) if e["kind"] in kinds else -1
            a.plot(e["t_min"], y, "o", color="C1", ms=6)
        for c in r["cues"]:
            a.axvline(c["t_min"], color="C0", lw=0.8, alpha=0.6)
            a.annotate(c["text"], (c["t_min"], len(kinds) - 0.4), rotation=90, fontsize=6.5,
                       va="top", ha="right", color="C0")
        if r.get("exhausted_min") is not None:
            a.axvline(r["exhausted_min"], color="C3", ls="--", lw=1)
        a.set_yticks(range(len(kinds)))
        a.set_yticklabels(kinds, fontsize=7)
        a.set_ylim(-0.6, len(kinds) + 0.4)
        a.set_xlabel("race time (min)")
        a.set_title("screen events (dots) and spoken cues (lines)", fontsize=9)
    fig.suptitle(data["title"], fontsize=11)
    fig.tight_layout()
    return fig


def _provider(a):
    return make_coach_provider(a.provider, a.model, api_key_env=a.api_key_env,
                               api_base_url=a.api_base_url)


def _engine_rows(session):
    return OC1Engine(head_cfg=HeadConfig(craft=session.craft, total_kg=session.total_kg)
                     ).feed_all(session.records).stroke_table()


def simulated(a):
    pp = PaddlerParams(rate_width=12.0)
    T = cp_test(seed=a.seed, paddler=pp)
    prep = ss.prepare_rows(_engine_rows(T), T.craft, T.total_kg)
    be = ss.best_efforts(prep["all_rows"], (180, 720))
    ath = calibrate(be, source="all-out 3 and 12 minutes")
    print(f"Critical-power test: best 3 min {be[180]['power']:.0f} W, best 12 min {be[720]['power']:.0f} W "
          f"(power reaching the boat) -> critical power {ath.cp_w:.0f} W, reserve {ath.w_prime_j / 1000:.1f} kJ")
    races = []
    for label, kw in (("over-paced", {}), ("evenly paced",
                                          dict(start_effort=1.15, body_effort=0.98, finish_effort=1.25))):
        S = paced_race(seed=a.seed + 1, paddler=pp, **kw)
        rows = _engine_rows(S)
        A = S.truth.arrays
        t_start = [x for x, _, l in S.segments if l == "start"][0]
        m = (A["t"] >= t_start) & (A["t"] <= S.plan.duration_s)
        dist = float(np.sum(np.hypot(np.diff(A["x"][m]), np.diff(A["y"][m]))))
        st = RaceState(S.craft, S.total_kg, ath, distance_m=dist, start_t=S.head_segments("start")[0][0])
        sink = CueSink(st)
        coach = RaceCoach(_provider(a), st, make_cue_gate(sink, st))
        res = replay_race(rows, st, coach, latency_s=a.latency_s)
        H = [h for h in res["history"] if h.get("elapsed_s") is not None]
        el = np.array([h["elapsed_s"] for h in H])
        # history times are head-unit seconds from the start; map to true time for the planted reserve
        t_head = np.array([h["elapsed_s"] for h in H]) + S.head_segments("start")[0][0]
        tt = true_time(S.sensors, t_head)
        planted = np.interp(tt, A["t"], A["w_bal"]) / pp.w_prime_j
        ex = np.flatnonzero((A["w_bal"] <= 0.02 * pp.w_prime_j) & (A["t"] >= t_start))
        exhausted = float((A["t"][ex[0]] - t_start) / 60.0) if len(ex) else None
        est = np.array([h["w_bal_frac"] for h in H])
        print(f"\n{label} race ({dist / 1000:.2f} km): planted reserve runs out "
              f"{'at %.1f min' % exhausted if exhausted else 'never'}")
        for e in res["screen"]:
            print(f"  {e['elapsed_s'] / 60:5.1f} min  screen  {e['kind']:16s} {e['detail']}")
        for c in sink.log:
            print(f"  {c['elapsed_s'] / 60:5.1f} min  CUE     {c['text']}")
        lat = [c["t_answered"] - c["t_asked"] for c in res["consultations"] if c["t_answered"]]
        print(f"  {len(res['consultations'])} consultations, coach turnaround {np.median(lat):.0f} s (median);"
              f" estimated vs planted reserve, first 5 min: mean error "
              f"{np.mean(np.abs(est[el <= 300] - planted[el <= 300])):.2f}")
        races.append({"title": f"{label} race, {dist / 1000:.1f} km", "t_min": (el / 60).round(3).tolist(),
                      "reserve_est": est.round(3).tolist(), "reserve_planted": planted.round(3).tolist(),
                      "fade_pct": [None if h.get("fade_pct") is None else round(h["fade_pct"], 2) for h in H],
                      "exhausted_min": exhausted,
                      "screen": [{"t_min": e["elapsed_s"] / 60, "kind": e["kind"]} for e in res["screen"]],
                      "cues": [{"t_min": c["elapsed_s"] / 60, "text": c["text"]} for c in sink.log]})
    coach_label = "scripted stand-in coach" if a.provider == "stand-in" else f"{a.provider} coach"
    return {"title": f"In-race coach replayed on simulated races (critical power from a 3/12-min test; "
                     f"{coach_label}; {a.latency_s:g} s per model call)", "races": races}


def real(a):
    craft = CRAFT[a.boat]
    files = sorted(glob.glob(os.path.join(a.calibrate_from, "*.csv")))
    target = os.path.abspath(a.speedcoach)
    S = [x for x in (ss.prepare_session(read_speedcoach(f), craft, a.total_kg)
                     for f in files if os.path.abspath(f) != target) if x]
    best = {}
    for x in S:
        for d, b in ss.best_efforts(x["all_rows"]).items():
            if d not in best or b["power"] > best[d]["power"]:
                best[d] = b
    env = ss.rate_envelope([w for x in S for w in ss.windows(x["all_rows"])])
    ath = calibrate(best, envelope=env, source=f"{len(S)} other sessions")
    print(f"Calibrated from {len(S)} other sessions: critical power {ath.cp_w:.0f} W, "
          f"reserve {ath.w_prime_j / 1000:.1f} kJ")
    rows = add_headings(list(read_speedcoach(a.speedcoach)["strokes"]))
    dist = a.distance_m or rows[-1]["distance_m"]
    st = RaceState(craft, a.total_kg, ath, distance_m=dist)
    sink = CueSink(st)
    coach = RaceCoach(_provider(a), st, make_cue_gate(sink, st))
    res = replay_race(rows, st, coach, latency_s=a.latency_s)
    for e in res["screen"]:
        print(f"  {e['elapsed_s'] / 60:5.1f} min {e['distance_m'] / 1000:5.2f} km  screen  "
              f"{e['kind']:16s} {e['detail']}")
    for c in sink.log:
        print(f"  {c['elapsed_s'] / 60:5.1f} min  CUE     {c['text']}")
    H = [h for h in res["history"] if h.get("eta_s")]
    for q in (0.2, 0.4, 0.6, 0.8):
        h = H[int(q * (len(H) - 1))]
        print(f"  at {h['elapsed_s'] / 60:4.1f} min the projected finish was {h['eta_s'] / 60:.1f} min "
              f"(actual {rows[-1]['t'] / 60:.1f})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--provider", default="stand-in", choices=["stand-in", "anthropic", "openai", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                    help="variable holding the Anthropic key (cloud sessions withhold ANTHROPIC_API_KEY)")
    ap.add_argument("--api-base-url", default="https://api.anthropic.com",
                    help="where the anthropic coach sends its requests")
    ap.add_argument("--latency-s", type=float, default=6.0, help="race time each model call takes")
    ap.add_argument("--speedcoach", default=None, help="replay this SpeedCoach session instead")
    ap.add_argument("--calibrate-from", default=None, help="folder of other SpeedCoach sessions")
    ap.add_argument("--distance-m", type=float, default=None)
    ap.add_argument("--boat", choices=sorted(CRAFT), default="oc1")
    ap.add_argument("--total-kg", type=float, default=93.0)
    a = ap.parse_args(argv)
    if a.speedcoach:
        if not a.calibrate_from:
            ap.error("--speedcoach needs --calibrate-from")
        real(a)
        return 0
    data = simulated(a)
    if a.figure:
        from agent_lib.figure_artifacts import emit       # agentic_experiments (a dependency)
        paths = emit(a.figure, data=data, plot_func=plot_oc1_race_coach,
                     artifacts_dir=os.path.join(os.path.dirname(HERE), "artifacts"))
        print("\nwrote", ", ".join(os.path.basename(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
