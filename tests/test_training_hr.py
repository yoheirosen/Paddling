"""Training blocks with planted day effects, the live zone reader, and the agent arm."""
import json
import math

import pytest

from agent_lib.providers.fake_provider import FakeProvider

from paddling_expt import hr_training as ht
from paddling_expt.hr_agent import AgentGuideArm
from paddling_expt.heart_rate import zone_of_share
from paddling_expt.sim.training import (DEVELOPMENT_DAY_KINDS, HELD_OUT_DAY_KINDS, StrapFault, answer,
                                        draw_block, executed_plan)


@pytest.fixture(scope="module")
def block():
    return draw_block(1001)          # a typical paddler; the first session has a strap locked on the stroke rate


@pytest.fixture(scope="module")
def cal(block):
    return ht.calibrate_block(block)


def test_blocks_are_reproducible_and_keep_held_out_days_apart():
    a, b = draw_block(1003), draw_block(1003)
    assert [(s.kind, s.day.kind, s.plan) for s in a.sessions] == [(s.kind, s.day.kind, s.plan) for s in b.sessions]
    assert executed_plan(a, a.sessions[0]) == executed_plan(b, b.sessions[0])
    dev = {s.day.kind for seed in range(1001, 1021) for s in draw_block(seed).sessions}
    assert dev <= set(DEVELOPMENT_DAY_KINDS) and not dev & set(HELD_OUT_DAY_KINDS)
    held = {s.day.kind for seed in range(5001, 5011) for s in draw_block(seed, pool="held_out").sessions}
    assert held & set(HELD_OUT_DAY_KINDS)


def test_calibration_reads_critical_power_and_heart_rate_from_gps(block, cal):
    true_cp = block.traits.paddler.critical_power_w
    assert abs(cal.athlete.cp_w / true_cp - 1.0) < 0.08                         # the power paddle reads critical power
    assert cal.athlete_boat.cp_w < cal.athlete.cp_w                             # less of it reaches the boat
    assert 0.5 < cal.efficiency(block.traits.habit_rate) < 0.8
    assert cal.model is not None and cal.separates_up_to >= 0.9                  # a typical heart rate
    assert cal.hr_max_seen > cal.model.hr_at(cal.athlete.cp_w) - 5.0
    assert set(cal.summary) >= {"critical_power_paddle_w", "blade_efficiency_by_stroke_rate", "heart_rate_model"}


def test_reader_follows_the_guide(cal):
    cp = cal.athlete.cp_w
    hr = cal.model.hr_at(0.65 * cp)
    rows = [{"t": float(i), "heart_rate_bpm": hr, "rate_spm": 58.0, "paddle_w": 0.65 * cp} for i in range(60)]
    both = ht.read_zone(ht.Guide(mode="combined"), rows, cal, minutes_in=0.0)
    assert both["zone_hr"] == both["zone_pace"] == both["zone"] == zone_of_share(0.65)
    # pace and heart rate two zones apart: "agree" will not guess, "hr_first" takes heart rate
    hard = [dict(r, paddle_w=0.97 * cp) for r in rows]
    assert ht.read_zone(ht.Guide(combine="agree"), hard, cal, 0.0)["zone"] is None
    assert ht.read_zone(ht.Guide(combine="hr_first"), hard, cal, 0.0)["zone"] == zone_of_share(0.65)
    # a heart rate on the stroke rate is dropped by the plausibility check
    locked = [dict(r, heart_rate_bpm=58.5) for r in rows]
    rd = ht.read_zone(ht.Guide(plausibility_check=True), locked, cal, 0.0)
    assert rd.get("hr_dropped") and rd["zone"] == zone_of_share(0.65)
    # the day's offset is taken off before the model reads the heart rate
    hot = [dict(r, heart_rate_bpm=hr + 8.0) for r in rows]
    assert ht.read_zone(ht.Guide(mode="hr", hr_offset_bpm=8.0), hot, cal, 0.0)["zone"] == zone_of_share(0.65)
    # the pace route scales by today's critical power
    rd = ht.read_zone(ht.Guide(mode="pace", cp_scale=0.9), rows, cal, 0.0)
    assert rd["share_pace"] == pytest.approx(0.65 / 0.9, abs=1e-3)


def test_strap_faults_and_the_fix():
    f = StrapFault({"kind": "cadence_lock", "start_s": 600.0, "end_s": 1200.0}, seed=1)
    assert f.read(500.0, 150.0, 60.0) == 150.0
    assert abs(f.read(700.0, 150.0, 60.0) - 60.0) < 5.0
    assert f.fix(700.0) and f.active(759.0) and not f.active(761.0)
    p = StrapFault({"kind": "poor_contact", "start_s": 0.0, "end_s": 600.0, "factor": 0.6}, seed=2)
    assert p.read(100.0, 150.0, 60.0) < 120.0
    assert "fine" in answer(draw_block(1001), draw_block(1001).sessions[0], "strap", 100.0)["says"].lower()


def test_a_session_is_read_every_30_s_and_scored_on_steady_effort(block, cal):
    ctx = ht.run_session(block, cal, block.sessions[0], ht.PaceArm())
    end = sum(p[0] for p in block.sessions[0].plan)
    assert abs(len(ctx.reads) - end // 30) <= 1
    sc = ht.score_session(ctx)
    assert sc["n"] > 20 and sc["right"] + sc["wrong"] + sc["cannot_tell"] == sc["n"]
    assert all(r["zone_true"] is not None for r in ctx.reads if r["steady"])
    assert any(r["strap_fault"] for r in ctx.reads)        # the planted cadence lock happens
    summ = ht.session_summary(ctx)
    assert summ["stretches"] and "zone" not in json.dumps(summ)   # the devices' record carries no truth


def test_agent_is_woken_after_the_warm_up_and_by_a_locked_strap(block, cal):
    calls = []

    def brain(system, messages, tools):
        calls.append(json.loads(messages[0].text))
        if len(messages) == 1:
            ask = "fix_strap" if calls[-1]["wake"] == "anomaly" else "feel"
            return {"tool_calls": [{"name": "ask_paddler", "arguments": {"ask": ask}},
                                   {"name": "set_guide", "arguments": {"plausibility_check": True}}]}
        return "done"

    arm = AgentGuideArm(FakeProvider(brain, model="scripted"))
    arm.start_block(block, cal)
    ctx = ht.run_session(block, cal, block.sessions[0], arm)
    wakes = [c["wake"] for c in calls if "wake" in c]
    assert wakes[0] == "warm-up done" and "anomaly" in wakes
    assert any("stroke rate" in e for c in calls for e in c.get("events", []))
    assert any(a["ask"] == "fix_strap" for a in ctx.answers)
    lock = block.sessions[0].day.strap
    assert ctx.strap.s["end_s"] < lock["end_s"]               # the fix ended the fault early
    first = calls[0]
    assert "planned_changes_of_effort_at_min" in first["session"]
    assert "effort" not in json.dumps(first["session"]).replace("planned_changes_of_effort_at_min", "")
    assert math.isfinite(ht.score_session(ctx)["right_frac"])


def test_breezy_days_gust_and_the_refit_pools_sessions(block, cal):
    from paddling_expt.sim.training import session_twin
    from paddling_expt.heart_rate import fit_personal_model
    b = draw_block(1002)
    spec = next(s for s in b.sessions if s.day.kind == "breezy")
    w = session_twin(b, spec).su.water
    xs = [w.wind_at(t) for t in range(0, 2400, 30)]
    assert abs(xs[-1]) > abs(xs[0]) and max(xs) - min(xs) > 0.1           # builds, with gusts
    assert "breeze" in answer(b, spec, "conditions", 1200.0)["says"]
    # two recordings of the same person, each with its own clock, fit as one model
    rows = [dict(r, session=1) for r in cal.rows] + [dict(r, session=2, t=r["t"] + 5.0) for r in cal.rows]
    one = fit_personal_model(cal.rows, effort_key="paddle_w")
    two = fit_personal_model(rows, effort_key="paddle_w", segment_key="session")
    assert abs(one.hr_at(0.7 * cal.athlete.cp_w) - two.hr_at(0.7 * cal.athlete.cp_w)) < 2.0
    arm = ht.RefitArm()
    arm.start_block(block, cal)
    ctx = ht.run_session(block, cal, block.sessions[1], arm)
    arm.end_session(ctx, ht.score_session(ctx))
    assert arm.refits and arm.cal_now is not cal and arm.cal.model is cal.model     # the shared calibration is untouched


def test_both_routes_are_read_and_bad_readings_dropped_one_by_one(cal):
    cp = cal.athlete.cp_w
    hr = cal.model.hr_at(0.65 * cp)
    rows = [{"t": float(i), "heart_rate_bpm": hr, "rate_spm": 58.0, "paddle_w": 0.65 * cp} for i in range(60)]
    rd = ht.read_zone(ht.Guide(mode="pace"), rows, cal, 0.0)
    assert rd["share_hr"] is not None and rd["zone"] == rd["zone_pace"]        # heart rate still watched
    mixed = [dict(r, heart_rate_bpm=58.5) if i < 20 else r for i, r in enumerate(rows)]   # a third on the stroke rate
    rd = ht.read_zone(ht.Guide(mode="hr", plausibility_check=True), mixed, cal, 0.0)
    assert rd["hr_implausible_frac"] == pytest.approx(0.33, abs=0.01)
    assert rd["zone"] == zone_of_share(0.65)                                   # the good two thirds read
    mostly = [dict(r, heart_rate_bpm=58.5) if i < 40 else r for i, r in enumerate(rows)]
    assert ht.read_zone(ht.Guide(mode="hr", plausibility_check=True), mostly, cal, 0.0)["zone"] is None


def test_agent_can_ask_each_session_and_schedule_its_own_wake_up(block, cal):
    log = []

    def brain(system, messages, tools):
        pk = json.loads(messages[0].text)
        if len(messages) == 1:
            log.append(pk)
            calls = [{"name": "ask_paddler", "arguments": {"ask": "feel"}}]
            if pk["wake"] == "warm-up done":
                calls.append({"name": "wake_me", "arguments": {"minutes": 3, "why": "check the current"}})
                calls.append({"name": "session_data", "arguments": {}})
            return {"tool_calls": calls}
        log.append([json.loads(r.content) for r in messages[-1].tool_results])
        return "ok"

    arm = AgentGuideArm(FakeProvider(brain, model="scripted"))
    arm.start_block(block, cal)
    for spec in block.sessions[:2]:
        ctx = ht.run_session(block, cal, spec, arm)
        assert ctx.answers and ctx.answers[0]["t"] < spec.warmup_s + 60.0   # not blocked by the last session
    wakes = [p["wake"] for p in log if isinstance(p, dict)]
    assert wakes.count("you asked to be woken") == 2
    data = next(r for x in log if isinstance(x, list) for r in x if "minutes" in r)
    assert data["minutes"][0]["current_est"] is None                            # no turn yet at the warm-up's end


def test_turn_wake_route_gap_and_the_new_guide_levers(block, cal):
    log = []

    def brain(system, messages, tools):
        pk = json.loads(messages[0].text)
        if len(messages) == 1:
            log.append(pk)
            if pk["wake"] == "warm-up done":
                return {"tool_calls": [{"name": "wake_me", "arguments": {"after_next_turn": True, "why": "current"}},
                                       {"name": "set_guide", "arguments": {"current_before_first_turn_mps": 0.2}}]}
            if pk["wake"] == "you asked to be woken":
                return {"tool_calls": [{"name": "route_gap", "arguments": {}},
                                       {"name": "set_guide", "arguments": {"current_before_first_turn_mps": None,
                                                                           "hr_extra_creep_per_10min": 1.0}}]}
            return "ok"
        log.append([json.loads(r.content) for r in messages[-1].tool_results])
        return "ok"

    arm = AgentGuideArm(FakeProvider(brain, model="scripted"))
    arm.start_block(block, cal)
    ctx = ht.run_session(block, cal, block.sessions[1], arm)
    turn_wake = next(p for p in log if isinstance(p, dict) and p["wake"] == "you asked to be woken")
    assert turn_wake["course"]["turns_at_min"] and turn_wake["now_min"] - turn_wake["course"]["turns_at_min"][0] < 1.5
    gap = next(r for x in log if isinstance(x, list) for r in x if "if_heart_rate_is_right_cp_scale" in r)
    assert gap["ok"] and gap["minutes_used"] == 0        # a minute after the first turn nothing qualifies yet
    whole = arm.analyst.route_gap()                      # over the finished session
    assert whole["minutes_used"] >= 10 and 0.7 < whole["if_heart_rate_is_right_cp_scale"] < 1.4
    assert ctx.guide.current_before_first_turn_mps is None and ctx.guide.hr_extra_creep_per_10min == 1.0
    # the assumed current moves the pace route only before the computer knows the current
    rows = [r for r in ctx.rows if not r["push_known"] and r["t"] > 120.0][:60]
    g0, g1 = ht.Guide(mode="pace"), ht.Guide(mode="pace", current_before_first_turn_mps=0.3)
    a = ht.read_zone(g0, rows, cal, 3.0, boat_power=ctx.boat_power)["share_pace"]
    b = ht.read_zone(g1, rows, cal, 3.0, boat_power=ctx.boat_power)["share_pace"]
    assert b < a                       # water flowing with the boat on the way out: less effort for the speed
    later = [r for r in ctx.rows if r["push_known"]][-60:]
    assert ht.read_zone(g0, later, cal, 30.0, boat_power=ctx.boat_power)["share_pace"] == \
        ht.read_zone(g1, later, cal, 30.0, boat_power=ctx.boat_power)["share_pace"]
