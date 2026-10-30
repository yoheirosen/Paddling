"""The live race state (reserve, sustainable pace, push, fade), each against a
planted answer."""
import math

import pytest

from paddling_expt.boats import SURFSKI_TYPICAL
from paddling_expt.boatspeed import power_at_speed, speed_at_power
from paddling_expt.race import Athlete, RaceState, sustainable_power, w_balance_step

CRAFT, KG = SURFSKI_TYPICAL, 89.0


def _rows(minutes=20.0, v_water=3.0, push_n=0.0, leg_s=240.0, rate=60.0, fade_after_s=None,
          fade=0.0, heading_fixed=None, t0=100.0):
    """One stroke a second; out-and-back legs due north / south (or one heading),
    in a current pushing north; optionally the speed through the water drops by
    ``fade`` after ``fade_after_s``."""
    rows, t, d = [], t0, 0.0
    for k in range(int(minutes * 60)):
        north = (int(k // leg_s) % 2 == 0) if heading_fixed is None else heading_fixed == 0.0
        vw = v_water * (1.0 - (fade if fade_after_s is not None and k >= fade_after_s else 0.0))
        v = vw + (push_n if north else -push_n)
        d += v
        rows.append({"t": t, "v_mean": v, "rate_spm": rate, "heading_deg": 0.0 if north else 180.0,
                     "distance_m": d})
        t += 1.0
    return rows


def test_reserve_drains_above_and_refills_below_critical_power():
    w = w_balance_step(10_000.0, 150.0, 10.0, 100.0, 10_000.0)
    assert w == pytest.approx(10_000.0 - 50.0 * 10.0)
    w2 = w_balance_step(5_000.0, 60.0, 1.0, 100.0, 10_000.0)
    assert w2 == pytest.approx(5_000.0 + 40.0 * 0.5)            # (CP-P)(W0-W)/W0
    assert w_balance_step(10_000.0, 50.0, 100.0, 100.0, 10_000.0) == 10_000.0


def test_sustainable_power_spends_the_reserve_exactly_at_the_finish():
    sp = sustainable_power(CRAFT, KG, 110.0, 6000.0, 3000.0)
    v = speed_at_power(CRAFT, KG, sp["power_w"])
    T = 3000.0 / v
    assert sp["power_w"] == pytest.approx(110.0 + 6000.0 / T, rel=2e-3)
    assert sp["seconds"] == pytest.approx(T, rel=2e-3)


def test_push_is_learned_out_and_back_and_unknown_on_one_heading():
    ath = Athlete(cp_w=150.0, w_prime_j=8000.0)
    st = RaceState(CRAFT, KG, ath, distance_m=10_000.0)
    for r in _rows(push_n=0.2):
        s = st.update(r)
    assert s["push_known"]
    assert s["push_mps"] == pytest.approx(0.2, abs=0.04)
    assert min(s["push_towards_deg"], 360 - s["push_towards_deg"]) < 15
    assert s["speed_water"] == pytest.approx(3.0, abs=0.05)
    st1 = RaceState(CRAFT, KG, ath, distance_m=10_000.0)
    for r in _rows(push_n=0.2, heading_fixed=0.0, minutes=8):
        s1 = st1.update(r)
    assert not s1["push_known"] and s1["push_mps"] < 0.05     # cannot be told from speed


def test_reserve_and_projection_at_a_known_power_above_critical_power():
    v = 3.0
    P = float(power_at_speed(CRAFT, KG, v))
    ath = Athlete(cp_w=P - 20.0, w_prime_j=8000.0)
    st = RaceState(CRAFT, KG, ath, distance_m=8000.0, push=(0.0, 0.0), start_t=160.0)
    snaps = [dict(st.update(r)) for r in _rows(minutes=6)]
    racing = [s for s in snaps if s.get("elapsed_s") is not None]
    assert racing[0]["distance_m"] < 5.0                        # distance counts from the start
    s = racing[-1]
    elapsed_total = s["t"] - 100.0                              # reserve drains from the first stroke
    assert s["w_bal_j"] == pytest.approx(8000.0 - 20.0 * elapsed_total, rel=0.02)
    assert s["empties_in_m"] == pytest.approx(v * s["w_bal_j"] / 20.0, rel=0.02)
    assert s["pace_gap_s500"] < 0                               # this pace cannot be held to the finish


def test_fade_is_measured_against_the_same_rate_early_in_the_race():
    ath = Athlete(cp_w=150.0, w_prime_j=8000.0)
    st = RaceState(CRAFT, KG, ath, distance_m=20_000.0, push=(0.0, 0.0), reference_s=300.0)
    for r in _rows(minutes=15, fade_after_s=600, fade=0.08):
        s = st.update(r)
    assert s["fade_pct"] == pytest.approx(-8.0, abs=0.5)
    st2 = RaceState(CRAFT, KG, ath, distance_m=20_000.0, push=(0.0, 0.0))
    for r in _rows(minutes=15):
        s2 = st2.update(r)
    assert abs(s2["fade_pct"]) < 0.5


def test_compact_state_is_small_and_rounded():
    ath = Athlete(cp_w=150.0, w_prime_j=8000.0)
    st = RaceState(CRAFT, KG, ath, distance_m=8000.0)
    for r in _rows(minutes=3):
        st.update(r)
    c = st.compact()
    assert len(c) <= 17 and "reserve_left_pct" in c and "pace_water_s500" in c
