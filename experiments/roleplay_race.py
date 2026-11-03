#!/usr/bin/env python3
"""Role-play the race coach: a person or an agent stands in for the model API.

The race runs exactly as in ``run_coaching_trial.py`` (same twin, panel, gate, coach
prompt and tools), but every model call is answered by the role-player.  The race is
deterministic, so the driver replays it from the start with the answers given so far
and stops at the next model call, printing what the model would see; the
role-player's answer is appended and the race continues.

    python experiments/roleplay_race.py start  --pack --seed 4012 --tag heldout_pack_rp
    python experiments/roleplay_race.py answer --pack --seed 4012 --tag heldout_pack_rp \\
        '{"tool_calls": [{"name": "get_metric_series", "arguments": {"metric": "fade_pct", "window_s": 120}}]}'
    python experiments/roleplay_race.py answer ... '{"text": "no cue: holding well"}'
    python experiments/roleplay_race.py collect --tag heldout_pack_rp
    python experiments/roleplay_race.py verify  --tag heldout_pack_rp [--pack]

An answer is JSON: ``{"text": "..."}`` (the reply, ending the wake-up) or
``{"tool_calls": [{"name": ..., "arguments": {...}}], "text": "optional"}`` (the tool
results come back in the next packet, within the same wake-up).  Each model call takes
a fixed ``--latency-s`` of race time.  When the race ends the driver prints RACE
FINISHED and saves the scored episode; ``collect`` appends the finished races to
``results/coaching_trial/episodes_<tag>.jsonl``.  The role-player's answers
(``answers.json``) are the whole record: ``verify`` replays every finished race of a
tag from its answers alone and checks the race time matches the saved result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agent_lib.providers.fake_provider import FakeProvider       # noqa: E402

from paddling_expt import coaching_trial as ct                     # noqa: E402

OUT = os.path.join(ROOT, "results", "coaching_trial")


def _dir(tag: str, seed: int) -> str:
    d = os.path.join(OUT, "roleplay", tag, str(seed))
    os.makedirs(d, exist_ok=True)
    return d


def _prep(a: argparse.Namespace) -> ct.Preparation:
    inc = getattr(a, "incidents", None)
    name = (f"preparations_open_{inc}.json" if inc else "preparations_pack.json" if a.pack
            else "preparations_coastal.json" if a.coastal else "preparations.json")
    path = os.path.join(OUT, name)
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if str(a.seed) in cache:
        return ct.Preparation.from_dict(cache[str(a.seed)])
    prep = ct.draw_preparation(a.seed, coastal=a.coastal, pack=a.pack, incidents=inc)
    if inc:                                  # open races are drawn once and kept (the draw takes seconds)
        cache[str(a.seed)] = prep.to_dict()
        with open(path, "w") as f:
            json.dump(cache, f)
    return prep


def _tool_line(t: Dict[str, Any]) -> str:
    """One tool as a compact line: name(param: type [one of ...], ...): description."""
    sch = t.get("input_schema", {}) or {}
    req = set(sch.get("required", []))
    parts = []
    for name, spec in (sch.get("properties") or {}).items():
        typ = spec.get("type", "any")
        if spec.get("enum"):
            typ += " one of " + "|".join(map(str, spec["enum"]))
        if spec.get("description"):
            typ += f" ({spec['description']})"
        parts.append(f"{name}{'' if name in req else '?'}: {typ}")
    return f"{t['name']}({', '.join(parts)}): {t.get('description', '')}"


def _packet(system: str, messages: List[Any], tools: List[Dict[str, Any]], first: bool,
            n_calls: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model_call": n_calls}
    if first:
        out["system_prompt"] = system
        out["tools"] = [_tool_line(t) for t in tools]
    last = messages[-1]
    rounds = sum(1 for m in messages if m.role == "assistant")
    out["round_in_this_wake_up"] = rounds + 1
    if len(messages) == 1:
        out["kind"] = "new wake-up (no memory of earlier wake-ups: decide from this alone)"
        try:
            out["update"] = json.loads(last.text)
        except (TypeError, ValueError):
            out["update"] = last.text
    else:
        names = {tc.id: tc.name for tc in messages[-2].tool_calls} if len(messages) >= 2 else {}
        res = []
        for tr in last.tool_results:
            try:
                content = json.loads(tr.content)
            except (TypeError, ValueError):
                content = tr.content
            res.append({"tool": names.get(tr.tool_call_id, "?"), "result": content, "is_error": tr.is_error})
        out["kind"] = "tool results (same wake-up)"
        out["tool_results"] = res
    return out


def _run(a: argparse.Namespace) -> None:
    d = _dir(a.tag, a.seed)
    ans_path = os.path.join(d, "answers.json")
    answers = json.load(open(ans_path)) if os.path.exists(ans_path) else []
    st = {"i": 0}

    def answer(system, messages, tools):
        if st["i"] < len(answers):
            v = answers[st["i"]]
            st["i"] += 1
            if isinstance(v, dict) and v.get("tool_calls"):
                return v
            return str(v.get("text", "") if isinstance(v, dict) else v)
        pk = _packet(system, messages, tools, first=st["i"] == 0, n_calls=st["i"] + 1)
        if st["i"] == 0:
            with open(os.path.join(d, "tools.txt"), "w") as f:
                f.write("\n".join(_tool_line(t) for t in tools) + "\n")
        with open(os.path.join(d, "packet.json"), "w") as f:
            json.dump(pk, f)
        sys.stdout.write(json.dumps(pk, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        os._exit(3)                      # paused: waiting for the role-player's answer

    prov = FakeProvider(answer, model=a.model_label)
    arm = ct.AgentArm(prov, label=a.label, coach_version=a.coach_version, periodic_s=a.periodic_s)
    r = ct.run_episode(_prep(a), arm, latency_s=a.latency_s)
    r["roleplay"] = {"answers": len(answers), "latency_s": a.latency_s, "stand_in": a.model_label}
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(r, f)
    print("RACE FINISHED: the result is saved. Thank you; nothing more to do for this race.")


def _replay(a: argparse.Namespace, answers: List[Any]) -> Dict[str, Any]:
    """Re-run one race from a complete list of answers (no pausing)."""
    it = iter(answers)

    def answer(system, messages, tools):
        v = next(it)
        if isinstance(v, dict) and v.get("tool_calls"):
            return v
        return str(v.get("text", "") if isinstance(v, dict) else v)

    arm = ct.AgentArm(FakeProvider(answer, model=a.model_label), label=a.label, coach_version=a.coach_version,
                      periodic_s=a.periodic_s)
    return ct.run_episode(_prep(a), arm, latency_s=a.latency_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["start", "answer", "show", "tools", "collect", "verify"])
    ap.add_argument("answer_json", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--coastal", action="store_true")
    ap.add_argument("--incidents", choices=["development", "held_out"], default=None,
                    help="an open race (incidents and a talking paddler); implies the open coach")
    ap.add_argument("--coach-version", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--periodic-s", type=float, default=None,
                    help="wake the coach at least this often (default 300 s; 180 s for the open coach)")
    ap.add_argument("--model-label", default="roleplay-subagent")
    ap.add_argument("--latency-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    if a.coach_version is None:
        a.coach_version = 8 if a.incidents else 7
    if a.periodic_s is None:
        a.periodic_s = 180.0 if a.coach_version >= 8 else 300.0
    if a.label is None:
        a.label = {8: "agent_open_roleplay", 9: "agent_open_v9_roleplay"}.get(a.coach_version, "agent_v7_roleplay")
    if a.cmd == "collect":
        base = os.path.join(OUT, "roleplay", a.tag)
        done = []
        for s in sorted(os.listdir(base), key=lambda x: int(x)) if os.path.isdir(base) else []:
            p = os.path.join(base, s, "result.json")
            if os.path.exists(p):
                done.append(json.load(open(p)))
        with open(os.path.join(OUT, f"episodes_{a.tag}.jsonl"), "a") as f:
            for r in done:
                f.write(json.dumps(r) + "\n")
        print(f"collected {len(done)} finished races into episodes_{a.tag}.jsonl")
        return 0
    if a.cmd == "verify":
        base = os.path.join(OUT, "roleplay", a.tag)
        bad = n = 0
        for s in sorted(os.listdir(base), key=lambda x: int(x)) if os.path.isdir(base) else []:
            res_p, ans_p = os.path.join(base, s, "result.json"), os.path.join(base, s, "answers.json")
            if not os.path.exists(ans_p):
                continue
            a.seed = int(s)
            r = _replay(a, json.load(open(ans_p)))
            n += 1
            saved = json.load(open(res_p))["race_time_s"] if os.path.exists(res_p) else None
            ok = saved is not None and abs(r["race_time_s"] - saved) < 1e-6
            bad += not ok
            print(f"seed {s}: replayed {r['race_time_s']:.3f} s, saved {saved if saved is None else round(saved, 3)}"
                  f" {'ok' if ok else 'MISMATCH'}")
        print(f"{n - bad} of {n} races reproduce from their answers")
        return 1 if bad else 0
    if a.seed is None:
        ap.error("--seed is required")
    d = _dir(a.tag, a.seed)
    if os.path.exists(os.path.join(d, "result.json")):
        print("RACE FINISHED: this race is already done.")
        return 0
    if a.cmd == "tools":
        p = os.path.join(d, "tools.txt")
        print(open(p).read() if os.path.exists(p) else "start the race first")
        return 0
    if a.cmd == "show":
        p = os.path.join(d, "packet.json")
        print(open(p).read() if os.path.exists(p) else "no packet yet: run start")
        return 0
    ans_path = os.path.join(d, "answers.json")
    answers = json.load(open(ans_path)) if os.path.exists(ans_path) else []
    if a.cmd == "start" and answers:
        print("already started: use show to see the current packet, then answer")
        return 0
    if a.cmd == "answer":
        if not a.answer_json:
            ap.error("answer needs a JSON answer")
        try:
            v = json.loads(a.answer_json)
        except ValueError as exc:
            print(f"not valid JSON: {exc}")
            return 2
        if not isinstance(v, dict) or not ("text" in v or "tool_calls" in v):
            print('an answer is {"text": "..."} or {"tool_calls": [{"name": ..., "arguments": {...}}]}')
            return 2
        answers.append(v)
        with open(ans_path, "w") as f:
            json.dump(answers, f)
    _run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
