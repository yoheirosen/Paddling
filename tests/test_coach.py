"""The in-race coach: screen events, the cue gate, and the replay on the agentic
runtime (virtual clock, model latency, run record)."""
import json

import pytest

from agent_lib.providers.fake_provider import FakeProvider
from agent_lib.runtime.record import RunRecord

from paddling_expt.boats import SURFSKI_TYPICAL
from paddling_expt.coach import (CueSink, RaceCoach, RaceScreen, format_pace, make_cue_gate,
                                 replay_race, stand_in_coach)
from paddling_expt.race import Athlete, RaceState

from test_race import CRAFT, KG, _rows


def _snap(t, **kw):
    s = {"t": t, "elapsed_s": t, "distance_m": 10.0 * t, "power_w": 150.0, "rate_spm": 60.0,
         "w_bal_frac": 0.8, "remaining_m": 5000.0 - 10.0 * t}
    s.update(kw)
    return s


def test_screen_raises_once_after_the_hold_and_ignores_the_start():
    sc = RaceScreen()
    hot = dict(empties_in_m=500.0, sustainable_pace_s500=170.0, pace_water_s500=160.0)
    assert not any(sc.check(_snap(t, **hot), 5000.0) for t in range(0, 90))   # start sprint
    raised = [e for t in range(90, 200) for e in sc.check(_snap(t, **hot), 5000.0)]
    assert [e["kind"] for e in raised] == ["over_pace"] and raised[0]["t"] == 110
    fade = [e for t in range(200, 300) for e in sc.check(_snap(t, fade_pct=-6.0), 5000.0)]
    assert [e["kind"] for e in fade] == ["fading"]
    # re-armed after the condition clears for a minute
    for t in range(300, 400):
        sc.check(_snap(t, fade_pct=0.0), 5000.0)
    again = [e for t in range(400, 470) for e in sc.check(_snap(t, fade_pct=-6.0), 5000.0)]
    assert [e["kind"] for e in again] == ["fading"]


def test_cue_gate_limits_words_quiet_periods_cooldown_and_count():
    st = RaceState(CRAFT, KG, Athlete(150.0, 8000.0), distance_m=5000.0)
    sink = CueSink(st)
    gate = make_cue_gate(sink, st, cooldown_s=120.0, max_cues=2)
    st.last = {"t": 30.0, "elapsed_s": 30.0, "remaining_m": 4000.0}
    assert gate.deliver({"text": "go"}, now=30.0)["stage"] == "quiet"
    st.last = {"t": 200.0, "elapsed_s": 200.0, "remaining_m": 3000.0}
    assert gate.deliver({"text": "one two three four five six seven eight nine ten eleven twelve thirteen"},
                        now=200.0)["stage"] == "words"
    assert gate.deliver({"text": "Settle to 2:50."}, now=200.0)["delivered"]
    assert gate.deliver({"text": "Again."}, now=260.0)["stage"] == "cooldown"
    assert gate.deliver({"text": "Again."}, now=330.0)["delivered"]
    assert gate.deliver({"text": "Third."}, now=500.0)["stage"] == "max_count"
    st.last = {"t": 900.0, "elapsed_s": 900.0, "remaining_m": 50.0}
    assert len(sink.log) == 2


def test_replay_wakes_the_coach_and_the_cue_lands_one_latency_later(tmp_path):
    ath = Athlete(cp_w=150.0, w_prime_j=8000.0)
    st = RaceState(CRAFT, KG, ath, distance_m=20_000.0, push=(0.0, 0.0), reference_s=300.0)
    sink = CueSink(st)
    coach = RaceCoach(stand_in_coach(), st, make_cue_gate(sink, st))
    rec = tmp_path / "run.jsonl"
    res = replay_race(_rows(minutes=15, fade_after_s=600, fade=0.08), st, coach,
                      latency_s=6.0, periodic_s=None, record_path=str(rec))
    fades = [e for e in res["screen"] if e["kind"] == "fading"]
    assert fades and len(sink.log) == 1
    assert sink.log[0]["t"] - fades[0]["t"] == pytest.approx(6.0, abs=1.01)   # one model call
    assert "lengthen" in sink.log[0]["text"]
    events = RunRecord.load(str(rec))
    llm = [e for e in events if e["kind"] == "llm"]
    assert llm and all(e["caller"] == "race_coach" for e in llm)
    assert any(e["kind"] == "gate" and not e["blocked"] for e in events)
    assert res["consultations"][0]["cued"]


def test_a_coach_that_chooses_silence_speaks_nothing():
    st = RaceState(CRAFT, KG, Athlete(150.0, 8000.0), distance_m=20_000.0, push=(0.0, 0.0))
    sink = CueSink(st)
    coach = RaceCoach(FakeProvider("no cue: holding well"), st, make_cue_gate(sink, st))
    res = replay_race(_rows(minutes=15, fade_after_s=600, fade=0.08), st, coach, periodic_s=300.0)
    assert res["consultations"] and not sink.log
    assert all(not c["cued"] for c in res["consultations"])


def test_the_model_sees_the_compact_state_and_the_events():
    seen = []

    def answer(system, messages, tools):
        seen.append(json.loads(messages[-1].text))
        return "no cue: test"
    st = RaceState(CRAFT, KG, Athlete(150.0, 8000.0), distance_m=20_000.0, push=(0.0, 0.0))
    coach = RaceCoach(FakeProvider(answer), st, make_cue_gate(CueSink(st), st))
    replay_race(_rows(minutes=15, fade_after_s=600, fade=0.08), st, coach, periodic_s=None)
    assert seen and seen[0]["events"][0]["kind"] == "fading"
    assert "reserve_left_pct" in seen[0]["state"] and len(json.dumps(seen[0])) < 2000


def test_format_pace():
    assert format_pace(171.6) == "2:52" and format_pace(None) == "?"


def test_the_demo_reads_the_named_key_and_pins_the_api_address(monkeypatch):
    import importlib.util
    import os
    import types
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "demos" / "demo_race_coach.py"
    spec = importlib.util.spec_from_file_location("demo_race_coach", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    a = types.SimpleNamespace(provider="anthropic", model=None, api_key_env="COACH_KEY_FOR_TEST",
                              api_base_url="https://api.anthropic.com")
    monkeypatch.delenv("COACH_KEY_FOR_TEST", raising=False)
    with pytest.raises(SystemExit):
        demo._provider(a)
    monkeypatch.setenv("COACH_KEY_FOR_TEST", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://claude-code-own-connection")
    p = demo._provider(a)
    assert p.api_key == "sk-test" and os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
