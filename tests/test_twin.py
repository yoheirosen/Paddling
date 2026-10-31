"""The paddler twin: its planted physiology, own pacing and cue response."""
import pytest

from paddling_expt.sim.twin import (PaddlerTwin, RaceSetup, Traits, draw_setup, draw_traits,
                                    even_effort, oracle_effort)
from paddling_expt.sim.paddler import PaddlerParams

SHORT = RaceSetup(distance_m=3000.0, leg_m=1000.0, current_mps=0.0, warmup_s=120.0)


def _traits(**kw):
    base = dict(paddler=PaddlerParams(rate_width=13.0, rate_opt_shift=5.0), habit_rate=62.0,
                start_effort=1.0, start_s=60.0, kick_m=0.0, wander_sd=0.0, compliance=1.0,
                response_s=5.0, adherence_s=300.0)
    base.update(kw)
    return Traits(**base)


def _race(tr, su=SHORT, body=None, cues=(), noise=False, seed=0):
    tw = PaddlerTwin(tr, su, seed=seed, body_effort=body, noise=noise, even=body)
    t, cues = 0.0, sorted(cues)
    while tw.finish_t is None and t < 4 * 3600:
        t += 1.0
        tw.advance(t)
        while cues and tw.racing and tw.t - tw.start_t >= cues[0][0]:
            _, ask, pace = cues.pop(0)
            tw.cue(ask, pace)
    return tw


def test_the_reserve_drains_above_and_refills_below_critical_power():
    tr = _traits()
    hard = _race(tr, body=1.15)
    easy = _race(tr, body=0.9)
    assert hard.blown_s > 0 and easy.blown_s == 0
    assert easy.w > 0.9 * tr.paddler.w_prime_j


def test_blowing_up_costs_time_so_even_pacing_wins():
    tr = _traits()
    e = even_effort(tr, SHORT)
    times = {k: _race(tr, body=k * e).race_time_s for k in (0.95, 1.0, 1.12)}
    assert times[1.0] < times[1.12] and times[1.0] < times[0.95]
    assert _race(tr, body=1.0 * e).blown_s == 0


def test_the_perfect_knowledge_pacer_is_at_least_as_fast_as_even_pacing():
    tr, su = draw_traits(3), draw_setup(3)
    ref = oracle_effort(tr, su)
    own_even = _race(tr, su, body=even_effort(tr, su)).race_time_s
    assert ref["race_time_s"] <= own_even + 1.0
    assert ref["reserve_left"] < 0.15


def test_a_followed_ease_cue_lowers_effort_after_the_response_time_and_fades_back():
    tr = _traits(effort_step=0.05, adherence_s=200.0)
    tw = _race(tr, body=1.0, cues=[(300.0, "ease", None)])
    T = tw.truth()
    import numpy as np
    rs, e = T["race_s"], T["effort"]
    before = e[(rs > 250) & (rs < 300)].mean()
    soon = e[(rs > 310) & (rs < 330)].mean()
    late = e[(rs > 1200) & (rs < 1300)].mean() if (rs > 1200).any() else e[-50:].mean()
    assert before - soon == pytest.approx(0.05, abs=0.01)
    assert abs(late - before) < abs(soon - before)
    assert tw.cues[0]["followed"]


def test_an_ignored_cue_changes_nothing():
    tr = _traits(compliance=0.0)
    a = _race(tr, body=1.0)
    b = _race(tr, body=1.0, cues=[(200.0, "ease", None), (400.0, "rate_down", None)])
    assert a.race_time_s == pytest.approx(b.race_time_s)
    assert not any(c["followed"] for c in b.cues)


def test_a_pace_target_is_chased_within_reason_and_given_up_on_blowing_up():
    import numpy as np
    tr = _traits(pace_gain=0.8)
    tw = _race(tr, body=0.95, cues=[(200.0, "hold", 150.0)])     # 2:30 is out of reach
    T = tw.truth()
    chased = T["effort"][(T["race_s"] > 210) & (T["race_s"] < 400)]
    assert chased.max() > 1.0 and chased.max() <= 1.12 * 0.95 + 1e-6
    hot = _race(tr, body=1.12, cues=[(100.0, "hold", 150.0)])
    assert hot.blown_s > 0 and hot._pace is None


def test_rate_down_helps_a_tired_paddler_whose_best_rate_has_fallen():
    tr = _traits(paddler=PaddlerParams(rate_width=11.0, rate_opt_shift=9.0), habit_rate=66.0)
    su = RaceSetup(distance_m=8000.0, current_mps=0.0, warmup_s=120.0)
    e = even_effort(tr, su)
    plain = _race(tr, su, body=e).race_time_s
    lower = _race(tr, su, body=e, cues=[(900.0, "rate_down", None), (1500.0, "rate_down", None)]).race_time_s
    assert lower < plain


def test_draws_are_reproducible_and_noise_is_seeded():
    assert draw_traits(5) == draw_traits(5) and draw_setup(5) == draw_setup(5)
    tr, su = draw_traits(5), draw_setup(5)
    e = even_effort(tr, su)
    a = PaddlerTwin(tr, su, seed=5, even=e)
    b = PaddlerTwin(tr, su, seed=5, even=e)
    assert a.advance(600.0) == b.advance(600.0)


def test_coastal_water_is_weaker_inshore_and_eddies_run_against_it():
    from paddling_expt.sim.twin import CoastalWater
    w = CoastalWater(c0=0.4, c1=0.0, eddies=[(500.0, 150.0, 120.0, 0.8)])
    assert w.at(40.0, 0.0, 0.0) < w.at(400.0, 0.0, 0.0)          # weaker close to shore
    assert w.at(40.0, 500.0, 0.0) < 0 < w.at(400.0, 500.0, 0.0)   # the eddy reverses the inshore water
    assert CoastalWater(c0=0.3, c1=-0.6, trend_s=100.0).stream(100.0) == pytest.approx(-0.3)   # it can turn


def test_a_line_cue_moves_the_paddler_across_at_a_cost_and_inshore_pays_against_the_stream():
    from dataclasses import replace
    from paddling_expt.sim.twin import draw_setup, draw_traits
    su = draw_setup(9, coastal=True)
    tr = replace(draw_traits(9), compliance=1.0, wander_sd=0.0)
    e = even_effort(tr, su)

    def go(cue_at_turn):
        tw = PaddlerTwin(tr, su, seed=9, even=e)
        t, leg = 0.0, None
        while tw.finish_t is None and t < 4 * 3600:
            t += 1.0
            tw.advance(t)
            if cue_at_turn and tw.racing and tw.heading != leg:
                leg = tw.heading
                against = su.water.stream(tw.t - tw.start_t) * (1 if leg < 90 else -1) < 0
                if against:
                    tw.cue("line_in", line_m=60.0)
        return tw
    plain, inside = go(False), go(True)
    assert min(inside.log["line_m"]) < 70.0 and min(plain.log["line_m"]) == pytest.approx(tr.habit_line_m)
    assert inside.race_time_s < plain.race_time_s


def test_the_forecast_calculator_follows_the_forecast():
    from paddling_expt.sim.twin import forecast_stream
    fc = {"stream_at_start_mps": 0.4, "stream_change_mps": 0.0, "over_s": 1000,
          "inshore_profile": {"share_at_shore": 0.25, "scale_m": 150.0}, "eddies": []}
    assert forecast_stream(fc, 50.0, 0.0, 0.0) < forecast_stream(fc, 400.0, 0.0, 0.0) <= 0.4
