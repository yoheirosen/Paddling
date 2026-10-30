"""The race panel on the platform's metric service, and the closed-loop trial."""
import json

import pytest

from agent_lib.providers.fake_provider import FakeProvider

from paddling_expt.coach import stand_in_coach
from paddling_expt.coaching_trial import (AgentArm, Autopilot, FixedPlan, NoCoach, Preparation,
                                          draw_preparation, holm, paired, run_episode)
from paddling_expt.race import Athlete, RaceState
from paddling_expt.race_panel import RacePanel, panel_values

from test_race import CRAFT, KG, _rows


@pytest.fixture(scope="module")
def prep():
    return draw_preparation(4)


def test_the_race_state_streams_into_the_platform_panel_and_the_screen_watches_fire():
    st = RaceState(CRAFT, KG, Athlete(150.0, 8000.0), distance_m=20_000.0, push=(0.0, 0.0),
                   reference_s=300.0)
    panel = RacePanel(st)
    alerts = []
    for r in _rows(minutes=15, fade_after_s=600, fade=0.08):
        st.update(r)
        alerts += panel.step(r["t"])
    assert [a["watch_id"] for a in alerts] == ["fading"]
    series = panel.service.call_tool("get_metric_series", {"metric": "fade_pct", "window_s": 60})
    assert series.get("ok", True) and "fade_pct" in json.dumps(series)
    assert "speed_water" in panel_values(st.last)


def test_the_coach_can_define_a_metric_and_a_watch_that_run_every_second():
    st = RaceState(CRAFT, KG, Athlete(150.0, 8000.0), distance_m=20_000.0, push=(0.0, 0.0))
    panel = RacePanel(st, watches=[], programs=[])
    tools = panel.tools()
    res = tools["define_derived_metric"].func(name="spm_per_ms", expression="rate_spm / speed_water")
    assert res["ok"]
    prog = tools["define_program_metric"].func(
        name="speed_drop_2min", tier="let",
        source="w = window('speed_water', 120)\n(w[0] - w[-1]) if len(w) > 1 else 0.0")
    assert prog["ok"]
    assert tools["add_watch"].func(watch_id="slow", condition="speed_water < 2.9", sustain_s=5)["ok"]
    alerts = []
    for r in _rows(minutes=15, fade_after_s=600, fade=0.08):
        st.update(r)
        alerts += panel.step(r["t"])
    derived = panel.service.buffer.series("spm_per_ms", 30.0)
    assert len(derived) > 0
    assert any(a["watch_id"] == "slow" for a in alerts)


def test_preparation_round_trips_through_json(prep):
    back = Preparation.from_dict(json.loads(json.dumps(prep.to_dict())))
    assert back.traits == prep.traits and back.setup == prep.setup
    assert back.athlete.cp_w == prep.athlete.cp_w


def test_every_arm_races_the_same_paddler_and_cues_reach_the_twin(prep):
    base = run_episode(prep, NoCoach())
    again = run_episode(prep, NoCoach())
    assert base["race_time_s"] == again["race_time_s"] and base["cues_spoken"] == 0
    fixed = run_episode(prep, FixedPlan(1.0, 300.0))
    assert fixed["cues_spoken"] > 0 and fixed["asks"] == {"hold": fixed["cues_spoken"]}
    assert fixed["race_time_s"] != base["race_time_s"]
    auto = run_episode(prep, Autopilot())
    assert sum(auto["alerts"].values()) >= auto["cues_spoken"]


def test_the_agent_arm_is_woken_by_alerts_and_its_cues_pass_the_gate(prep):
    r = run_episode(prep, AgentArm(stand_in_coach(), label="agent_stand_in"), latency_s=6.0)
    assert r["consultations"] > 0
    assert r["cues_spoken"] + r["cues_blocked"] == r["cue_calls"]      # the stand-in cues once per call
    silent = run_episode(prep, AgentArm(FakeProvider("no cue: fine"), label="agent_silent"))
    assert silent["cues_spoken"] == 0
    assert silent["race_time_s"] == run_episode(prep, NoCoach())["race_time_s"]


def test_paired_statistics_and_holm():
    p = paired([10.0, 11.0, 9.0, 12.0, 8.0], [12.0, 13.0, 10.0, 15.0, 9.0])
    assert p["mean_d"] == pytest.approx(-1.8) and p["n_a_faster"] == 5
    assert p["ci95"][1] < 0
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
