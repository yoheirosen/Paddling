"""The experimenting coach: test pieces and a heart-rate strap on the twin, measuring stretches
of a race, beliefs that set the boat computer, the scheduled-test rules, the learning scores."""
import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from agent_lib.providers.fake_provider import FakeProvider

from paddling_expt import coaching_trial as ct
from paddling_expt.coach import CueSink, RaceCoach, make_cue_gate, measure_windows
from paddling_expt.race import RaceState
from paddling_expt.sim.twin import PaddlerTwin, check_piece


@pytest.fixture(scope="module")
def prep():
    p = ct.draw_preparation(1, incidents="development")
    p.traits = replace(p.traits, talk="quiet")
    p.setup.incidents = []
    return p


def _twin(prep, hr=False, compliance=1.0):
    tw = PaddlerTwin(replace(prep.traits, compliance=compliance), prep.setup, seed=prep.seed, even=prep.even)
    if hr:
        tw.enable_heart_rate()
    return tw


def _until(tw, race_s, state=None):
    t = tw.t
    while tw.t - tw.start_t < race_s and tw.finish_t is None:
        t += 1.0
        for r in tw.advance(t):
            if state is not None:
                state.update(r)


def test_piece_limits_are_checked():
    assert check_piece([{"seconds": 90, "effort_pct": 8}, {"seconds": 90}]) is None
    assert "segments" in check_piece([])
    assert "at most 3" in check_piece([{"seconds": 30}] * 4)
    assert "seconds" in check_piece([{"seconds": 5}])
    assert "effort_pct" in check_piece([{"seconds": 60, "effort_pct": 40}])
    assert "480" in check_piece([{"seconds": 300}, {"seconds": 300}])


def test_a_followed_piece_changes_effort_by_the_clock_then_hands_back(prep):
    tw = PaddlerTwin(replace(prep.traits, compliance=1.0), prep.setup, seed=prep.seed, even=prep.even,
                     body_effort=0.9)                 # below critical power: the reserve refills after the start
    _until(tw, 900.0)
    assert not tw.blown and tw.w > 0.6 * tw.p.w_prime_j
    tw.cue("piece", segments=[{"seconds": 90, "effort_pct": 10}, {"seconds": 60, "effort_pct": -10}])
    t0 = tw.t + tw.tr.response_s
    _until(tw, tw.t - tw.start_t + 200.0)
    L = tw.truth()
    def mean_effort(a, b):
        m = (L["t"] >= t0 + a) & (L["t"] < t0 + b)
        return float(L["effort"][m].mean())
    before = float(L["effort"][(L["t"] >= t0 - 60) & (L["t"] < t0 - 5)].mean())
    assert mean_effort(10, 85) / before == pytest.approx(1.10, abs=0.04)
    assert mean_effort(100, 145) / before == pytest.approx(0.90, abs=0.04)
    events = [e["event"] for e in tw.incident_log]
    assert "piece_started" in events and "piece_done" in events


def test_a_piece_on_an_empty_reserve_ends_in_a_blow_up_and_is_abandoned(prep):
    tw = PaddlerTwin(replace(prep.traits, compliance=1.0), prep.setup, seed=prep.seed, even=prep.even,
                     body_effort=1.04)
    _until(tw, 700.0)
    while tw.blown:
        _until(tw, tw.t - tw.start_t + 5.0)
    tw.cue("piece", segments=[{"seconds": 120, "effort_pct": 15}])
    _until(tw, tw.t - tw.start_t + 140.0)
    assert [e["event"] for e in tw.incident_log][-1] == "piece_abandoned"


def test_heart_rate_settles_below_critical_power_and_climbs_above_it(prep):
    def trend(effort):
        tw = PaddlerTwin(prep.traits, replace(prep.setup, plan=[(700.0, 62.0, effort)]), seed=3,
                         noise=False)               # no wandering effort: the heart rate's own trend
        tw.enable_heart_rate()
        rows = tw.run_alone(step_s=10.0)
        assert not tw.blown
        t = np.array([r["t"] for r in rows])
        hr = np.array([r["heart_rate_bpm"] for r in rows])
        m = (t > 300) & (t < 650)
        return np.polyfit(t[m] / 60.0, hr[m], 1)[0]
    assert abs(trend(0.9)) < 0.4                 # steady below critical power (a slow creep only)
    assert trend(1.06) > 0.6                     # keeps climbing above it


def test_the_heart_rate_strap_changes_nothing_else_in_the_race(prep):
    a, b = _twin(prep), _twin(prep, hr=True)
    _until(a, 1e9)
    _until(b, 1e9)
    assert a.race_time_s == b.race_time_s


def test_a_pod_gain_correction_recomputes_power_and_the_reserve(prep):
    tw = _twin(prep)
    st = RaceState(tw.craft, tw.total_kg, prep.athlete, distance_m=prep.setup.distance_m,
                   start_t=tw.start_t, athlete_shaft=prep.athlete_shaft)
    _until(tw, 600.0, st)
    p0, w0 = st.last["shaft_power_w"], st.w_bal_shaft
    out = st.set_pod_gain(1.25, since_elapsed_s=0.0)
    assert out["pod_gain"] == 1.25
    _until(tw, 620.0, st)
    assert st.last["shaft_power_w"] == pytest.approx(p0 / 1.25, rel=0.08)
    assert st.last["shaft_power_raw_w"] == pytest.approx(p0, rel=0.08)
    assert st.w_bal_shaft > w0                    # less power read above critical power: more reserve left


def test_measure_splits_speed_into_share_and_current_from_out_and_back_legs(prep):
    tw = _twin(prep, hr=True)
    st = RaceState(tw.craft, tw.total_kg, prep.athlete, distance_m=prep.setup.distance_m,
                   start_t=tw.start_t, athlete_shaft=prep.athlete_shaft)
    leg = prep.setup.leg_m
    while tw.race_m < 3.1 * leg:
        _until(tw, tw.t - tw.start_t + 10.0, st)
    L = tw.truth()
    def span(k):
        m = (L["race_m"] >= k * leg + 150) & (L["race_m"] < (k + 1) * leg - 20) & (L["race_s"] > 0)
        return {"start_race_s": float(L["race_s"][m].min()), "end_race_s": float(L["race_s"][m].max())}
    res = measure_windows(st, leg, [span(1), span(2)])
    assert res["ok"] and len(res["windows"]) == 2
    w = res["windows"][0]
    assert w["direction"] == "back" and "heart_rate_trend_bpm_per_min" in w and len(w["pod_power_w"]) == 2
    push = res["hull_fit"]["push_along_out_direction_mps"][0]
    assert push == pytest.approx(prep.setup.current_mps, abs=0.06)
    share = res["hull_fit"]["share_of_pod_power_reaching_boat"]
    assert share[1] <= share[0] <= share[2]


def _coach(prep, version=10):
    tw = _twin(prep, hr=True)
    st = RaceState(tw.craft, tw.total_kg, prep.athlete, distance_m=prep.setup.distance_m,
                   start_t=tw.start_t, athlete_shaft=prep.athlete_shaft)
    sink = CueSink(st, act=lambda a: tw.cue(a.get("ask", "none"), a.get("pace_s500"), a.get("line_m"),
                                            a.get("distance_left_m"), a.get("segments")))
    c = RaceCoach(FakeProvider(lambda *a: "no cue", model="t"), st, make_cue_gate(sink, st, cooldown_s=60.0, max_cues=20),
                  version=version)
    c.race_info = {"leg_m": prep.setup.leg_m}
    return tw, st, c


def test_beliefs_set_the_computer_and_show_at_every_wake_up(prep):
    tw, st, c = _coach(prep)
    _until(tw, 700.0, st)
    names = set(c.policy.allowed) if hasattr(c.policy, "allowed") else None
    r = c.set_belief("critical_power_pod_w", 1.05 * st.cp_shaft_w, evidence="heart rate settled")
    assert r["ok"] and r["computer"]["recomputed_from_start"]
    assert c.set_belief("pod_gain", 1.2, since_race_s=300)["ok"] and st.pod_gain == 1.2
    assert c.set_belief("talk", "pessimist", evidence="says cooked at 40 % reserve")["ok"]
    assert not c.set_belief("critical_power_pod_w", 5.0 * st.cp_shaft_w)["ok"]
    pk = c._open_packet([], "periodic")
    assert set(pk["your_beliefs"]) == {"critical_power_pod_w", "pod_gain", "talk"}
    assert "heart_rate_bpm" in pk["state"] and "race_s" in pk["state"]
    if names is not None:
        assert {"measure", "test_cost", "set_belief"} <= names and "set_critical_power" not in names


def test_a_piece_goes_through_the_cue_gate_and_is_priced(prep):
    tw, st, c = _coach(prep)
    _until(tw, 700.0, st)
    cost = c.test_cost(8, 120)
    assert cost["ok"] and cost["metres_gained_during"] > 0 and "cp_10pct_lower" in cost
    bad = c.cue("Piece now", ask="piece", segments=[{"seconds": 5}])
    assert not bad["spoken"] and "seconds" in bad["reason"]
    ok = c.cue("Ninety seconds harder, then back.", ask="piece",
               segments=[{"seconds": 90, "effort_pct": 8}, {"seconds": 90, "effort_pct": 0}])
    assert ok["spoken"]
    pk = c._open_packet([], "periodic")
    assert pk["next_cue_allowed_at_race_s"] != "now"
    _until(tw, 760.0, st)
    assert any(e["event"] == "piece_started" for e in tw.incident_log)


def test_scheduled_tests_run_their_schedule_and_every_arm_gets_learning_scores():
    prep = ct.draw_preparation(22, incidents="development")
    r = ct.run_episode(prep, ct.ScheduledTests(), heart_rate=True)
    kinds = {x["test"] for x in r["tests"]}
    assert kinds == {"heart_rate", "pod_vs_gps"}
    for key in ("reserve_error_pts", "cp_error_pct_start", "cp_error_pct_half", "cp_error_pct_three_quarters"):
        assert key in r["learning"]
    r0 = ct.run_episode(prep, ct.NoCoach(), heart_rate=True)
    assert "reserve_error_pts" in r0["learning"]
    assert ct.run_episode(prep, ct.NoCoach())["race_time_s"] == r0["race_time_s"]


def test_the_role_play_driver_points_the_experimenting_coach_to_its_instructions(tmp_path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tag = f"_test_v10_{os.getpid()}"
    try:
        out = subprocess.run([sys.executable, os.path.join(here, "experiments", "roleplay_race.py"), "start",
                              "--incidents", "development", "--coach-version", "10", "--seed", "21", "--tag", tag],
                             capture_output=True, text=True, timeout=600)
        pk = json.loads(out.stdout.strip().splitlines()[-1])
        assert "read_first" in pk and "system_prompt" not in pk
        sysout = subprocess.run([sys.executable, os.path.join(here, "experiments", "roleplay_race.py"), "system",
                                 "--incidents", "development", "--coach-version", "10", "--seed", "21", "--tag", tag],
                                capture_output=True, text=True, timeout=120).stdout
        assert "Experiments" in sysout and "Heart rate" in sysout
    finally:
        import shutil
        shutil.rmtree(os.path.join(here, "results", "coaching_trial", "roleplay", tag), ignore_errors=True)


def test_after_a_piece_the_coach_sees_each_segment_and_can_cancel_one_wake_up(prep):
    tw, st, c = _coach(prep)
    tw.tr = replace(tw.tr, compliance=1.0)
    _until(tw, 700.0, st)
    c.cue("Ninety seconds harder, then back.", ask="piece",
          segments=[{"seconds": 90, "effort_pct": 8}, {"seconds": 90, "effort_pct": 0}])
    _until(tw, 720.0, st)
    assert "piece running" in str(c._seen_after(c.cue_attempts[-1]))
    _until(tw, 920.0, st)
    seen = c._seen_after(c.cue_attempts[-1])
    assert len(seen["segments"]) == 2 and "pod_power_w" in seen["before"]
    assert c.wake_me(in_s=60, reason="check the piece")["ok"] and c.wake_me(in_s=120, reason="finish plan")["ok"]
    out = c.wake_me(cancel="piece")
    assert out["cancelled"] == 1 and [w["reason"] for w in c.timers] == ["finish plan"]
    assert "set_at_race_s" in c.timers[0]


def test_measure_at_a_fixed_share_and_the_reserve_across_a_belief_range(prep):
    tw, st, c = _coach(prep)
    _until(tw, 900.0, st)
    res = c.measure([{"start_race_s": 500, "end_race_s": 800}], share=0.65)
    row = res["at_fixed_share"]["windows"][0]
    assert "gps_minus_that_mps" in row
    cp = st.cp_shaft_w
    c.set_belief("critical_power_pod_w", cp, low=0.95 * cp, high=1.05 * cp, evidence="test")
    rng = c._open_packet([], "periodic")["state"]["reserve_pod_pct_across_your_cp_range"]
    lo, hi = list(rng.values())
    assert lo <= hi


def test_a_new_piece_replaces_a_running_one_from_the_effort_being_held(prep):
    tw = PaddlerTwin(replace(prep.traits, compliance=1.0), prep.setup, seed=prep.seed, even=prep.even,
                     body_effort=0.9)
    _until(tw, 900.0)
    tw.cue("ease", pace_s500=200.0)                    # a pace target: paused while pieces run
    _until(tw, 960.0)
    tw.cue("piece", segments=[{"seconds": 200, "effort_pct": -10}])
    _until(tw, 1060.0)
    L = tw.truth()
    e1 = float(L["effort"][-20:].mean())
    tw.cue("piece", segments=[{"seconds": 120, "effort_pct": 6}])
    _until(tw, 1140.0)
    L = tw.truth()
    e2 = float(L["effort"][-20:].mean())
    assert e2 / e1 == pytest.approx(1.06, abs=0.03)
    assert "piece_replaced" in [e["event"] for e in tw.incident_log]


def test_the_analyst_cannot_cue_effort_and_sees_the_pacer():
    prep = ct.draw_preparation(22, incidents="development")
    seen = {}

    def answer(system, messages, tools):
        import json as _j
        if len(messages) == 1:
            pk = _j.loads(messages[0].text)
            if "pacer_recent_cues" in pk:
                seen["pacer"] = pk["pacer_recent_cues"]
            if "first" not in seen:
                seen["first"] = True
                return {"tool_calls": [{"name": "cue", "arguments": {"text": "Ease off.", "ask": "ease"}},
                                       {"name": "set_belief", "arguments": {"name": "critical_power_pod_w",
                                                                            "value": 220}}]}
            return "no cue"
        seen.setdefault("results", [str(x.content) for x in messages[-1].tool_results])
        return "no cue"
    arm = ct.AnalystArm(FakeProvider(answer, model="t"), periodic_s=180.0)
    r = ct.run_episode(prep, arm, latency_s=5.0)
    assert "pacer's" in seen["results"][0] and "pacer keeps" in seen["results"][1]
    assert r["pacer_cues"] >= 1 and "pacer" in seen
    assert arm.coach.system.startswith("You are the analyst")
    rules = ct.run_episode(prep, ct.IncidentRules(drop_mps=0.06), heart_rate=True)
    assert r["race_time_s"] == rules["race_time_s"]          # an analyst who does nothing = the pacer alone


def test_the_analyst_asks_where_the_finish_is_ramps_a_pod_correction_and_writes_valid_metrics(prep):
    tw, st, c = _coach(prep, version=11)
    c.ask_finish_fn = tw.answer_finish
    c.ask_fn = tw.answer
    _until(tw, 700.0, st)
    a = c.ask_paddler(question="finish")
    assert a["ok"] and a["thinks_finish_in_m"] % 50 == 0
    r = c.set_belief("pod_gain", 1.2, since_race_s=300, ramp_s=200)
    assert r["ok"] and st.gain_at(300) == 1.0 and abs(st.gain_at(400) - 1.1) < 1e-9 and st.gain_at(600) == 1.2
    for i in range(9):
        c.take_note(f"note {i}")
    pk = c._open_packet([], "periodic")
    assert len(pk["your_notes"]) == 6 and pk["older_notes_not_shown"] == 3


def test_the_analyst_is_refused_a_metric_whose_last_line_is_an_assignment():
    from paddling_expt.coach import _last_line_checked
    f = _last_line_checked(lambda **kw: {"ok": True})
    assert not f(name="a", source="w = window('x', 60)\nslope_v = slope(w)")["ok"]
    assert f(name="a", source="w = window('x', 60)\nslope(w) * 60")["ok"]
    assert f(name="a", source="mean(window('x', 60)) == 3")["ok"]


def test_the_role_play_driver_races_the_analyst_with_the_pacer(tmp_path):
    import argparse
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "experiments"))
    import roleplay_race as rr
    a = argparse.Namespace(coach_version=11, label="x", periodic_s=180.0)
    assert isinstance(rr._arm(a, FakeProvider(lambda *x: "no cue", model="t")), ct.AnalystArm)
    a.coach_version = 10
    assert type(rr._arm(a, FakeProvider(lambda *x: "no cue", model="t"))) is ct.AgentArm


def test_the_heart_rate_coach_gets_this_paddlers_own_heart_rate_model(prep):
    from types import SimpleNamespace
    from paddling_expt.sim.twin import heart_rate_profile
    seed = next(s for s in range(1, 200) if heart_rate_profile(s).get("shape") == "compressed")
    p = ct.draw_preparation(seed, incidents="development")
    tw = PaddlerTwin(p.traits, p.setup, seed=p.seed, even=p.even)
    tw.enable_heart_rate(profiles=True)
    st = RaceState(tw.craft, tw.total_kg, p.athlete, distance_m=p.setup.distance_m, start_t=tw.start_t,
                   athlete_shaft=p.athlete_shaft)
    sink = CueSink(st, act=lambda a: None)
    arm = ct.AgentArm(FakeProvider(lambda *a: "no cue", model="t"), coach_version=12)
    from paddling_expt.race_panel import RacePanel
    ep = SimpleNamespace(prep=p, twin=tw, state=st, gate=make_cue_gate(sink, st), panel=RacePanel(st))
    arm.setup(ep)
    c = arm.coach
    m = c.heart_rate_model()
    assert m["ok"] and m["heart_rate_separates_efforts_up_to_share_of_cp"] < 0.75
    assert any(not z["heart_rate_can_hold_it"] for z in m["zones"])
    near_cp = m["hr_at_share_of_cp"]["100%"]
    r = c.heart_rate_model(hr_bpm=near_cp, race_min=0.0)["reading"]
    assert r["zone"] is None and "cannot tell" in r["note"]
