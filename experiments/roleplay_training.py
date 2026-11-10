#!/usr/bin/env python3
"""Role-play the training-block heart-rate agent: a person or an agent stands in for the model API.

The block runs exactly as in ``training_hr.py`` (same paddler, calibration day, sessions,
reader, agent prompt and tools), but every model call is answered by the role-player.
The block is deterministic, so the driver replays it from the start with the answers given
so far and stops at the next model call, printing what the model would see; the
role-player's answer is appended and the block continues.

    python experiments/roleplay_training.py start  --seed 1001 --tag dev_rp
    python experiments/roleplay_training.py answer --seed 1001 --tag dev_rp \\
        '{"tool_calls": [{"name": "session_data", "arguments": {}}]}'
    python experiments/roleplay_training.py answer --seed 1001 --tag dev_rp '{"text": "guide kept"}'
    python experiments/roleplay_training.py collect --tag dev_rp
    python experiments/roleplay_training.py verify  --tag dev_rp

An answer is JSON: ``{"text": "..."}`` (the reply, ending the wake-up) or
``{"tool_calls": [{"name": ..., "arguments": {...}}], "text": "optional"}`` (the tool results
come back in the next packet, within the same wake-up).  Each model call takes a fixed
``--latency-s`` of session time.  When the block ends the driver prints BLOCK FINISHED and
saves the scored block; ``collect`` appends the finished blocks to
``results/training_hr/blocks_<tag>.jsonl``.  The answers (``answers.json``) are the whole
record: ``verify`` replays every finished block of a tag from its answers alone and checks
every session's score matches the saved result.
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
sys.path.insert(0, HERE)

from agent_lib.providers.fake_provider import FakeProvider       # noqa: E402

from paddling_expt import hr_training as ht                        # noqa: E402
from paddling_expt.hr_agent import AgentGuideArm                   # noqa: E402
from paddling_expt.sim.training import draw_block                  # noqa: E402

from roleplay_race import _packet, _tool_line                       # noqa: E402

OUT = os.path.join(ROOT, "results", "training_hr")


def _dir(tag: str, seed: int) -> str:
    d = os.path.join(OUT, "roleplay", tag, str(seed))
    os.makedirs(d, exist_ok=True)
    return d


def _block(a: argparse.Namespace):
    return draw_block(a.seed, pool=a.pool)


def _arm(a: argparse.Namespace, prov: Any) -> AgentGuideArm:
    return AgentGuideArm(prov, label=a.label)


def _scores(r: Dict[str, Any]) -> List[List[Any]]:
    return [[s["session"], s["right"], s["wrong"], s["cannot_tell"]] for s in r["sessions"]]


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
        pk = _packet(system, messages, tools, first=st["i"] == 0, n_calls=st["i"] + 1, inline_first=True)
        if len(messages) == 1:
            pk["kind"] = "new wake-up (your notes carry across wake-ups and sessions; nothing else does)"
        if st["i"] == 0:
            with open(os.path.join(d, "tools.txt"), "w") as f:
                f.write("\n".join(_tool_line(t) for t in tools) + "\n")
            with open(os.path.join(d, "system.txt"), "w") as f:
                f.write(system + "\n")
        with open(os.path.join(d, "packet.json"), "w") as f:
            json.dump(pk, f)
        sys.stdout.write(json.dumps(pk, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        os._exit(3)                      # paused: waiting for the role-player's answer

    arm = _arm(a, FakeProvider(answer, model=a.model_label))
    r = ht.run_block(_block(a), arm, latency_s=a.latency_s)
    r["roleplay"] = {"answers": len(answers), "latency_s": a.latency_s, "stand_in": a.model_label}
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(r, f)
    print("BLOCK FINISHED: the result is saved. Thank you; nothing more to do for this block.")


def _replay(a: argparse.Namespace, answers: List[Any]) -> Dict[str, Any]:
    """Re-run one block from a complete list of answers (no pausing)."""
    it = iter(answers)

    def answer(system, messages, tools):
        v = next(it)
        if isinstance(v, dict) and v.get("tool_calls"):
            return v
        return str(v.get("text", "") if isinstance(v, dict) else v)

    return ht.run_block(_block(a), _arm(a, FakeProvider(answer, model=a.model_label)), latency_s=a.latency_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["start", "answer", "show", "tools", "system", "collect", "verify"])
    ap.add_argument("answer_json", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool", choices=["development", "held_out"], default="development")
    ap.add_argument("--label", default="agent_roleplay")
    ap.add_argument("--model-label", default="roleplay-subagent")
    ap.add_argument("--latency-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    base = os.path.join(OUT, "roleplay", a.tag)
    seeds = sorted((s for s in os.listdir(base) if s.isdigit()), key=int) if os.path.isdir(base) else []
    if a.cmd == "collect":
        done = [json.load(open(os.path.join(base, s, "result.json"))) for s in seeds
                if os.path.exists(os.path.join(base, s, "result.json"))]
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"blocks_{a.tag}.jsonl"), "a") as f:
            for r in done:
                f.write(json.dumps(r) + "\n")
        print(f"collected {len(done)} finished blocks into blocks_{a.tag}.jsonl")
        return 0
    if a.cmd == "verify":
        bad = n = 0
        for s in seeds:
            res_p, ans_p = os.path.join(base, s, "result.json"), os.path.join(base, s, "answers.json")
            if not os.path.exists(ans_p):
                continue
            a.seed = int(s)
            saved = json.load(open(res_p)) if os.path.exists(res_p) else None
            if saved is not None:
                a.pool = saved.get("pool", a.pool)
            r = _replay(a, json.load(open(ans_p)))
            n += 1
            ok = saved is not None and _scores(r) == _scores(saved)
            bad += not ok
            print(f"seed {s}: replayed right {r['total']['right']}/{r['total']['n']}, saved "
                  f"{None if saved is None else saved['total']['right']} {'ok' if ok else 'MISMATCH'}")
        print(f"{n - bad} of {n} blocks reproduce from their answers")
        return 1 if bad else 0
    if a.seed is None:
        ap.error("--seed is required")
    d = _dir(a.tag, a.seed)
    if os.path.exists(os.path.join(d, "result.json")):
        print("BLOCK FINISHED: this block is already done.")
        return 0
    if a.cmd in ("tools", "system"):
        p = os.path.join(d, f"{a.cmd}.txt")
        print(open(p).read() if os.path.exists(p) else "start the block first")
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
