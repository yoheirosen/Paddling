"""Several sessions together (best efforts, critical power, best speed at each
rate, within- vs between-session rate effect), each against a planted answer."""
import math

import numpy as np
import pytest

from paddling_expt import sessions as ss
from paddling_expt.boats import SURFSKI_TYPICAL
from paddling_expt.boatspeed import speed_at_power
from paddling_expt.speedcoach import parse_speedcoach

HEAD = ("Interval,Distance (GPS),Distance (IMP),Elapsed Time,Split (GPS),Speed (GPS),Split (IMP),"
        "Speed (IMP),Stroke Rate,Total Strokes,Distance/Stroke (GPS),Distance/Stroke (IMP),Heart Rate,"
        "Power,Catch,Slip,Finish,Wash,Force Avg,Work,Force Max,Max Force Angle,GPS Lat.,GPS Lon.")
UNITS = ("(Interval),(Miles),(Miles),(HH:MM:SS.tenths),(/Mile),(MPH),(/Mile),(MPH),(SPM),(Strokes),"
         "(Feet),(Feet),(BPM),(Watts),(Degrees),(Degrees),(Degrees),(Degrees),(Newtons),(Joules),"
         "(Newtons),(Degrees),(Degrees),(Degrees)")
R_E = 6_371_008.8


def _hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def _out_and_back(legs=6, n_leg=150, water=3.0, push_n=0.2, glitch_at=300):
    """Out-and-back legs due north and south at a steady speed through the water,
    in a current pushing north; one missed stroke reads as 29 then 113 spm."""
    lines = ["Session Information:,,,,Device Information:,,,,Oarlock Information:,,,,Oarlock Settings:",
             "", "Name:,Synthetic,,,Name:,SpdCoach 1,,,Firmware:,---,,,Boat ID:,---",
             "Start Time:,10/17/2026 11:00:00,,,Model:,SpeedCoach GPS Pro,,,,,,,Seat Number:,---",
             "", "Per-Stroke Data:", "", HEAD, UNITS]
    lat, lon, dist, t, k = 36.95, -122.0, 0.0, 60.0, 0
    for leg in range(legs):
        north = leg % 2 == 0
        g = water + (push_n if north else -push_n)
        for _ in range(n_leg):
            rate = 60.0 + 2.0 * math.sin(k * 0.37)
            cyc = 60.0 / rate
            t += cyc
            step = g * cyc
            lat += (step if north else -step) / R_E * 180 / math.pi
            dist += step
            shown = {glitch_at: 29.0, glitch_at + 1: 113.0}.get(k, rate)
            lines.append(f"1,{dist / 1609.344:.5f},0.000,{_hms(t)},00:10:00.0,{g / 0.44704:.4f},"
                         f"00:00:00.0,0.000,{shown:.1f},{k + 1},{step / 0.3048:.3f},0.0,---,---,---,---,"
                         f"---,---,---,---,---,---,{lat:.7f},{lon:.7f}")
            k += 1
    return "\n".join(lines) + "\n"


def test_prepare_session_removes_the_current_from_every_stroke():
    s = ss.prepare_session(parse_speedcoach(_out_and_back()), SURFSKI_TYPICAL, 89.0)
    assert s["push"]["push_mps"] == pytest.approx(0.2, abs=0.03)
    assert min(s["push"]["push_towards_deg"], 360 - s["push"]["push_towards_deg"]) < 10
    vc = np.array([r["v_corrected"] for r in s["all_rows"]])
    assert np.median(np.abs(vc - 3.0)) < 0.02
    glitch = [r for r in s["all_rows"] if r["rate_spm"] in (29.0, 113.0)]
    assert len(glitch) == 2 and not any(r in s["rows"] for r in glitch)
    assert all(r.get("boat_power_w") for r in glitch)          # kept in continuous efforts
    w = ss.windows(s["all_rows"])
    near = [x for x in w if x["t0"] <= glitch[0]["t"] <= x["t1"]]
    assert near and all(abs(x["rate"] - 60.0) < 2.5 for x in near)   # median rate ignores it
    be = ss.best_efforts(s["all_rows"])
    assert 600 in be and be[600]["speed"] == pytest.approx(3.0, abs=0.03)


def _rows(segments):
    """Rows at one stroke per second: segments of (seconds, power) separated by
    ``None`` for a stop (a 20 s gap)."""
    rows, t = [], 0.0
    for seg in segments:
        if seg is None:
            t += 20.0
            continue
        dur, p = seg
        for _ in range(int(dur)):
            t += 1.0
            rows.append({"t": t, "boat_power_w": p, "v_corrected": 3.0, "rate_spm": 60.0})
    return rows


def test_best_efforts_find_the_planted_blocks_and_respect_stops():
    rows = _rows([(400, 100.0), (300, 150.0), (800, 100.0), None, (200, 300.0)])
    be = ss.best_efforts(rows, (60, 300, 1200, 2400))
    assert be[60]["power"] == pytest.approx(300.0)              # the short block after the stop
    assert be[300]["power"] == pytest.approx(150.0, rel=0.01)   # the 300 s block; 300 W is too short
    assert be[1200]["power"] == pytest.approx((300 * 150 + 900 * 100) / 1200, rel=0.01)
    assert 2400 not in be                                       # no continuous run that long
    assert be[300]["rate"] == pytest.approx(60.0)


def test_critical_power_recovers_the_planted_values():
    best = {d: {"power": 200.0 + 15000.0 / d} for d in ss.DURATIONS_S}
    cp = ss.critical_power(best)
    assert cp["cp_w"] == pytest.approx(200.0, rel=1e-6)
    assert cp["w_prime_j"] == pytest.approx(15000.0, rel=1e-6)
    assert cp["r2"] == pytest.approx(1.0)


def test_race_prediction_is_self_consistent():
    cp = {"cp_w": 110.0, "w_prime_j": 8000.0}
    pr = ss.race_prediction(cp, SURFSKI_TYPICAL, 89.0, [1000, 5000, 20000])
    for x in pr:
        v = speed_at_power(SURFSKI_TYPICAL, 89.0, x["power_w"])
        assert v * x["seconds"] == pytest.approx(x["distance_m"], rel=1e-3)
        assert x["power_w"] == pytest.approx(110.0 + 8000.0 / x["seconds"], rel=1e-6)
    assert pr[0]["pace_500m_s"] < pr[1]["pace_500m_s"] < pr[2]["pace_500m_s"]
    assert not pr[0]["beyond_model"] and pr[2]["beyond_model"]


def test_rate_envelope_finds_the_planted_best_rate():
    rng = np.random.default_rng(3)
    r = rng.uniform(45, 75, 4000)
    best = 3.0 - 0.002 * (r - 62.0) ** 2
    v = best - np.abs(rng.normal(0, 0.15, r.size))
    wins = [{"rate": a, "speed": b, "dps": b * 60 / a} for a, b in zip(r, v)]
    env = ss.rate_envelope(wins, pct=99.0, min_n=20)
    assert env["peak_rate"] == pytest.approx(62.0, abs=1.5)
    probe = [{"rate": 62.0, "speed": 3.0}, {"rate": 50.0, "speed": 3.0 - 0.002 * 144}]
    ss.percent_of_best(probe, env, SURFSKI_TYPICAL, 89.0)
    assert all(97.0 < p["pct_best"] < 104.0 for p in probe)


def test_rate_effect_tells_within_from_between_sessions():
    rng = np.random.default_rng(5)
    sess = []
    for k in range(7):
        mr, mv = 50.0 + 3.0 * k, 2.5 + 0.09 * k          # harder sessions at higher rates
        r = mr + rng.uniform(-4, 4, 40)
        v = mv + rng.normal(0, 0.03, 40)                  # but rate buys nothing within a session
        sess.append([{"rate": a, "speed": b, "dps": b * 60 / a} for a, b in zip(r, v)])
    e = ss.rate_effect(sess, split_rate=60.0)
    lo, hi = e["within"]["speed_ci"]
    assert lo < 0.0 < hi and abs(e["within"]["speed_per_spm"]) < 0.005
    assert e["between_speed_per_spm"] == pytest.approx(0.03, abs=0.005)
    assert e["within"]["dps_per_spm"] < 0                 # same speed at more rate: shorter strokes


def test_place_groups():
    sess = [{"centre": (36.95, -122.00)}, {"centre": (36.954, -122.003)},
            {"centre": (36.30, -121.90)}, {"centre": (36.951, -122.001)}]
    assert ss.place_groups(sess) == [0, 0, 1, 0]
