"""Open races: incidents, a talking paddler, and the open coach's tools."""
from dataclasses import replace

import pytest

from agent_lib.providers.fake_provider import FakeProvider

from paddling_expt import coaching_trial as ct
from paddling_expt.sim.incidents import HELD_OUT_KINDS, draw_incidents, draw_talk
from paddling_expt.sim.twin import Incident, PaddlerTwin


@pytest.fixture(scope="module")
def prep():
    p = ct.draw_preparation(1, incidents="development")
    p.traits = replace(p.traits, talk="quiet")
    p.setup.incidents = []
    return p


def _race(prep, incidents, talk="quiet", cues=()):
    """Race the twin alone with these incidents; ``cues``: (race seconds, ask, kwargs)."""
    tr = replace(prep.traits, talk=talk)
    su = replace(prep.setup, incidents=list(incidents))
    tw = PaddlerTwin(tr, su, seed=prep.seed, even=prep.even)
    todo = sorted(cues)
    t = 0.0
    while tw.finish_t is None and t < 3 * 3600:
        t += 1.0
        tw.advance(t)
        while todo and t - tw.start_t >= todo[0][0]:
            _, ask, kw = todo.pop(0)
            tw.tr = replace(tw.tr, compliance=1.0)
            tw.cue(ask, **kw)
    return tw


def test_open_races_are_drawn_the_same_every_time_and_held_out_races_hold_a_held_out_kind():
    a = draw_incidents(5000, 8000.0, 1000.0, "held_out")
    assert a == draw_incidents(5000, 8000.0, 1000.0, "held_out")
    assert draw_talk(7) == draw_talk(7)
    for s in range(5000, 5030):
        kinds = {i.kind for i in draw_incidents(s, 8000.0, 1000.0, "held_out")}
        assert kinds & set(HELD_OUT_KINDS)
    for s in range(0, 30):
        assert not {i.kind for i in draw_incidents(s, 8000.0, 1000.0, "development")} & set(HELD_OUT_KINDS)


def test_weed_slows_the_boat_until_a_back_stroke_clears_it(prep):
    clean = _race(prep, [])
    weed = [Incident("weed", 300.0, 0.10)]
    dragged = _race(prep, weed)
    cleared = _race(prep, weed, cues=[(330.0, "clear_rudder", {})])
    assert dragged.race_time_s > clean.race_time_s + 10.0
    assert clean.race_time_s < cleared.race_time_s < dragged.race_time_s - 10.0
    assert any(e["event"] == "cleared_by_cue" for e in cleared.incident_log)


def test_a_pessimist_reassured_in_time_does_not_ease_off():
    prep = ct.draw_preparation(7, incidents="development")         # a paddler who complains while fine
    tw = _race(prep, [], talk="pessimist")
    said = [e for e in tw.incident_log if e["event"] == "said"]
    assert said and any(e["event"] == "self_ease" for e in tw.incident_log)
    first = said[0]["race_s"]
    tw2 = _race(prep, [], talk="pessimist", cues=[(first + 1.0, "reassure", {})])
    assert any(e["event"] == "reassured_in_time" for e in tw2.incident_log)


def test_a_course_change_reaches_the_paddler_only_when_told(prep):
    D = prep.setup.distance_m
    ext = [Incident("course_change", 400.0, prep.setup.leg_m)]
    alone = _race(prep, ext)
    assert alone.su.distance_m == D + prep.setup.leg_m
    assert any(e["event"] == "overran_believed_finish" for e in alone.incident_log)
    said = [m["text"] for m in alone.outbox]
    assert any("extended" in x for x in said) and any("finish" in x.lower() for x in said)
    told = _race(prep, ext, cues=[(420.0, "tell", {"distance_left_m": None})])   # no distance: nothing learnt
    assert any(e["event"] == "overran_believed_finish" for e in told.incident_log)
    tw = PaddlerTwin(prep.traits, replace(prep.setup, incidents=ext), seed=prep.seed, even=prep.even)
    tw.tr = replace(tw.tr, compliance=1.0)
    t = 0.0
    while tw.t - tw.start_t < 430.0:
        t += 1.0
        tw.advance(t)
    tw.cue("tell", distance_left_m=tw.su.distance_m - tw.race_m)
    while tw.finish_t is None:
        t += 1.0
        tw.advance(t)
    assert not any(e["event"] == "overran_believed_finish" for e in tw.incident_log)
    assert tw.su.distance_m == D + prep.setup.leg_m and prep.setup.distance_m == D   # the shared setup is untouched


def test_a_drifting_pod_misreports_effort_without_changing_it(prep):
    clean = _race(prep, [])
    drift = PaddlerTwin(prep.traits, replace(prep.setup, incidents=[Incident("pod_drift", 200.0, 0.75)]),
                        seed=prep.seed, even=prep.even)
    rows, t = [], 0.0
    while drift.finish_t is None:                      # stepped like _race (the step changes the noise draws)
        t += 1.0
        rows += drift.advance(t)
    assert drift.race_time_s == pytest.approx(clean.race_time_s, abs=1e-9)
    late = [r for r in rows if r["t"] - drift.start_t > 400.0]
    t_idx = [i for i, t in enumerate(drift.log["t"]) if t - drift.start_t > 400.0]
    ratio = sum(r["shaft_power_w"] for r in late) / sum(drift.log["p_shaft"][i] for i in t_idx)
    assert 0.70 < ratio < 0.80


def test_the_open_coach_gets_notes_its_recent_cues_and_the_paddler():
    prep = ct.draw_preparation(7, incidents="development")
    seen = []

    def answer(system, messages, tools):
        seen.append(messages[-1].text if len(messages) == 1 else None)
        n = len(seen)
        if n == 2:
            return {"tool_calls": [{"name": "take_note", "arguments": {"text": "plan: even to 1 km out"}},
                                   {"name": "ask_paddler", "arguments": {}}]}
        if n == 4:
            return {"tool_calls": [{"name": "cue", "arguments": {"text": "Settle and breathe.", "ask": "hold"}}]}
        return "no cue: fine"
    arm = ct.AgentArm(FakeProvider(answer), coach_version=8, periodic_s=180.0, label="open_test")
    r = ct.run_episode(prep, arm, latency_s=5.0)
    assert arm.coach.system.startswith("You are coaching one paddler")
    assert arm.watches() == []
    import json
    packets = [json.loads(x) for x in seen if x]
    assert "race" in packets[0] and "your_notes" in packets[0]
    later = packets[-1]
    assert later["your_notes"][0]["note"] == "plan: even to 1 km out"
    assert later["your_recent_cues"] and later["your_recent_cues"][0]["ask"] == "hold"
    assert any(e["event"] == "asked" for e in r["incident_log"])
    assert "incident_score" in r and r["talk"] == prep.traits.talk


def test_set_race_distance_corrects_the_computer_and_refuses_the_implausible():
    prep = ct.draw_preparation(7, incidents="development")
    arm = ct.AgentArm(FakeProvider(lambda s, m, t: "no cue"), coach_version=8, label="t")
    ct.run_episode(prep, arm, latency_s=5.0)
    c = arm.coach
    done = c.state.last["distance_m"]
    assert c.set_race_distance(done - 50.0)["ok"] is False
    out = c.set_race_distance(done + 1000.0, reason="course shortened by radio")
    assert out["ok"] and c.state.distance_m == done + 1000.0


def _v9_race(seed, script, compliance=None):
    """Race the open coach's later version with a scripted stand-in: ``script`` maps a
    model-call index to an answer; every other call answers "no cue"."""
    from dataclasses import replace as _r
    prep = ct.draw_preparation(seed, incidents="development")
    if compliance is not None:
        prep.traits = _r(prep.traits, compliance=compliance)
    seen = []

    def answer(system, messages, tools):
        seen.append(messages[-1].text if len(messages) == 1 else [tr.content for tr in messages[-1].tool_results])
        return script.get(len(seen) - 1, "no cue: fine")
    arm = ct.AgentArm(FakeProvider(answer), coach_version=9, periodic_s=180.0, label="v9_test")
    r = ct.run_episode(prep, arm, latency_s=5.0)
    return r, arm, seen


def test_the_later_open_coach_keeps_a_plan_wakes_itself_and_sees_what_a_cue_did():
    import json
    script = {0: {"tool_calls": [{"name": "set_plan", "arguments": {"text": "even to 1 km to go"}},
                                 {"name": "wake_me", "arguments": {"in_s": 70, "reason": "check the ease"}},
                                 {"name": "cue", "arguments": {"text": "Ease a touch.", "ask": "ease"}}]}}
    r, arm, seen = _v9_race(7, script)
    packets = [json.loads(x) for x in seen if isinstance(x, str)]
    woke = next(p for p in packets if any(e["kind"] == "wake_me" for e in p["events"]))
    assert woke["your_plan"]["plan"] == "even to 1 km to go"
    after = woke["your_recent_cues"][0]["instruments_after"]
    assert set(after) >= {"pod_power_w", "rate_spm", "speed_gps"} and len(after["pod_power_w"]) == 2
    assert "next_turn_in_m" in packets[0]["state"] and "your_notes" in packets[0]
    assert arm.coach.system.startswith("You are coaching one paddler") and "wake_me" in arm.coach.system


def test_a_back_stroke_shows_on_the_instruments_only_when_the_paddler_makes_it():
    script = {0: {"tool_calls": [{"name": "cue", "arguments": {"text": "Clear the rudder.", "ask": "clear_rudder"}}]}}
    _, done, _ = _v9_race(7, script, compliance=1.0)
    _, ignored, _ = _v9_race(7, script, compliance=0.0)
    assert done.coach.cue_attempts[0]["_seen"]["back_stroke_seen"] is True
    assert ignored.coach.cue_attempts[0]["_seen"]["back_stroke_seen"] is False


def test_the_later_coach_reads_legs_summaries_and_corrects_the_whole_reserve():
    import json
    calls = [{"name": "legs", "arguments": {}},
             {"name": "get_metric_series", "arguments": {"metric": "shaft_power_w", "window_s": 600}},
             {"name": "set_critical_power", "arguments": {"cp_shaft_w": 200, "w_prime_shaft_kj": 20, "reason": "test"}}]
    r, arm, seen = _v9_race(7, {3: {"tool_calls": calls}})
    res = [json.loads(x) for x in seen[4]]
    legs, series, cal = res
    assert legs["ok"] and legs["legs"][0]["direction"] == "out" and "pod_power_w" in legs["legs"][0]
    assert 5 <= len(series["points"]) <= 60 and series["race_s"][1] <= 700        # race seconds, not the data clock
    assert cal["ok"] and cal["recomputed_from_start"] and cal["cp_shaft_w"] == 200.0
    assert arm.coach.state.athlete_shaft.w_prime_j == 20000.0
