# CLAUDE.md — Conventions for this repo

A quick reference for AI assistants (or new contributors) working in this repository.

**What this is.** Real-time performance sensing for a solo outrigger canoe (OC1) or a
surf ski: a boat-mounted head unit (GNSS + motion sensor) and a paddle pod (strain
gauge + motion sensor) stream transport-agnostic records into a real-time pipeline
(strokes, rate, speed, the check, side, pull-force profile, roll, side-slip, wobble);
session and multi-session analyses run on it and on NK SpeedCoach exports; a
hull-shape resistance model links speed, effort and power. Design, results and
limits: **`PADDLING_PLAN.md`**; history: **`MAINTENANCE_LOG.md`**.

**Built on agentic_experiments.** The platform repository
`UCSC-S-Tier/agentic_experiments` (package `seizure-agent-control`) is a dependency
(`pyproject.toml`). Its `agent_lib` writes the figure artifacts now and is where the
coaching loop's agents come from (the runtime kernel `agent_lib/runtime/`: scheduler,
clock, `run_tool_loop`, `ActuatorGate`, run records; read that repository's
`CLAUDE.md` before building a loop). This project lived there as `paddling/` until
2026-10-24.
- Import shared machinery from the dependency; never copy it in. A change the
  platform needs goes to agentic_experiments as its own pull request.
- Working on both side by side: `pip install -e ../agentic_experiments`, then
  `pip install -e . --no-deps` (otherwise pip replaces the checkout with `main`).

---

## Standing instructions (repo owner's working preferences)

These apply to every task in this repo, on top of the conventions below.

### 1. Token and quota awareness (long-running autonomous work)
- My weekly quota usage is logged by my status line script to
  `~/.claude/quota-log.jsonl`. Each line has: time `t`, `session`,
  weekly percent used `wk`, the weekly reset time `wk_resets`, and this
  session's estimated spend `cost`.
- At the start of any long autonomous task, and at least hourly, read the
  newest line, and compare it with the line closest to 24 hours earlier.
- Hard stop: if weekly usage rose by more than 14 percentage points in
  the last 24 hours, save your state and ask my permission before going
  on. Report the current percent, the 24-hour rise, and a rough estimate
  of what the remaining work will cost.
- If `wk_resets` differs between the two lines, the weekly window reset
  in between. Measure from the first line of the current window instead,
  and say so.
- Soft warning: estimate this session's share as its own `cost` increase
  divided by the combined `cost` increase of all sessions in the log over
  the same period, times the weekly rise. If that estimate reaches about
  5 points, warn me and ask permission. Call it an estimate, because
  `cost` is a list-price figure and not the quota itself.
- If the log is missing or its newest line is over 30 minutes old, say so
  once and pause for a check-in at natural milestones instead.
- In cloud sessions the log will not exist. Skip the log check there, say
  so once, and use milestone check-ins.
- Do not read my credentials file or call undocumented usage endpoints to
  work around a missing log.

### 2. Project status summary: only at stopping points, or when I ask
Give a status summary of everything being worked on, including paused items, only:
- at the end of a long run of work, when you reach a stopping point: the work is finished, or you are handing it back to me because it needs my input or decision;
- when I ask for one. A short request is enough: "status", "status summary" or "where are we?" all mean give the full summary now.

Do not add it to messages in the middle of long-running or autonomous work. Waiting on tests, CI, a background job or a scheduled check-in is not a stopping point, and neither is finishing one step of a larger job. A summary repeated in every message makes the chat history hard to read.

The summary covers:
- Done: what is finished.
- Not yet done: what still needs doing.
- Paused: what is paused, and why.
- For experimental work: where results stand against the goals I stated.

### 3. Experiments for a paper
If the task involves experiments for a paper, say where we stand on those results compared with the paper's goals, including anything that is weaker than hoped:
- in the status summary at every stopping point (rule 2);
- in between, in one line, only when a result actually changes: a number moves, or a claim gets stronger or weaker. Do not repeat unchanged results in every message.

### 4. Stay on my goals
- Unless I say otherwise, all work serves the goals I state. I am usually building tools with a specific purpose, which may be broad, and the work is often exploratory.
- Optimize for my stated overall goal, never for a metric you chose yourself. A self-chosen metric may be irrelevant or only narrowly relevant to what I want.
- Do not drift into small technical details or side problems that do not move the stated goal. If something looks like a rabbit hole, note it in one line and continue.
- If you believe a different goal or metric is truly necessary, stop and ask my explicit permission first. Explain why.

### 5. Plain-language reporting
- Write reports the way they would appear in a paper for a semi-technical audience.
- Do not use internal jargon, shorthand, or names you invented during the work. Do not assume I know them. Define any necessary term in plain words the first time it appears.

### 6. Whenever you ask me to decide
This applies to every decision you put to me, however small and wherever it appears: in a chat reply, a multiple-choice question prompt, a plan, or a pull request comment. Never ask a bare question such as "A or B?". Give me, every time:
- Context: what the decision is, why it has come up now, and the background I need to answer without scrolling back or looking anything up.
- Options: every realistic option, normally at least two, each described in plain words. Include "do nothing" or "defer" when that is a real choice.
- Implications of each option: what it costs (time, quota, money, complexity), what it gains, what it rules out later, and what is hard or impossible to undo.
- Pros and cons of each option, listed side by side so the trade-off is easy to compare.
- Your recommendation: which option you would pick and why. Then leave the decision to me, and do not act on it until I answer.

---

## Naming: describe things, don't cite bare codes

Never refer to something by a bare code or an internal name invented during the work
(in commit messages, pull requests, comments, docs, or when talking to a human). Lead
with what the thing is; attach the code only as a trailing pointer to where it is
defined. Every name, heading and summary says what the thing does.

---

## Project rules

- **The feature test: does it bear on how fast the boat moves at race speed?** Build
  measures that do (stroke rate, speed through the water, the check, the pull-force
  curve, ama load, wobble, current and wind, pacing, power). Capsize detection was
  built once and removed at the owner's instruction (no bearing on training or
  racing): do not add it back. Remove only what the owner names.
- **Wherever speed, effort and power interact, use the hull model** (`hydro.py`), not
  a fitted curve; boats are typical (`boats.py`) until the owner's are measured.
- **Keep raw device exports out of git** (SpeedCoach CSVs and head-unit logs hold the
  paddler's GPS track; `.gitignore` covers `*.csv`, `*.jsonl`, `data/`). Figure data
  holds averages only, never positions.
- **Planted truth.** The simulator's physiology is chosen so questions have known
  answers; say so wherever a simulated result is reported, and keep its honesty
  ledger in `PADDLING_PLAN.md` current. Every timing names the machine it ran on.
- **Pull requests for new features**: develop on a branch, open a PR; keep
  `PADDLING_PLAN.md` (results, limits) and `MAINTENANCE_LOG.md` (dated entries,
  newest first) up to date. The owner lets Claude merge pull requests in this
  repository itself (2026-10-24), once the tests pass and nothing in the pull request
  is waiting on the owner. This covers this repository only: pull requests in
  agentic_experiments still wait for the owner.

## Figures

Every figure is a four-file artifact in `artifacts/` sharing one base name: `<name>.py`
(standalone, numpy + matplotlib only, redraws from the data), `<name>.data.json`,
`<name>.png`, `<name>.pdf`. Write them with `agent_lib.figure_artifacts.emit(name,
data=..., plot_func=..., artifacts_dir="artifacts")`; `plot_func` must be
self-contained (its source is copied into `<name>.py`). The `.py` and `.data.json` are
tracked, the `.png` / `.pdf` git-ignored. To deliver figures, zip one folder per figure
(its four files) and send the zip to the user directly.

## Running tests

```bash
pip install -e ".[test,plot]"
python -m pytest -q tests          # ~50 s, no network
```

`.github/workflows/tests.yml` runs the same when started by hand (Actions -> tests ->
Run workflow); it needs the `AGENTIC_EXPERIMENTS_TOKEN` secret described in the file.

## When to write what

| You want to… | Write a… |
|---|---|
| Feed device records (any uplink) and get strokes + a live panel | `paddling_expt/fusion.py` `OC1Engine` (`feed` in arrival order → `stroke_table()` / `panel()`; `HeadConfig(craft=, total_kg=)`); record schema `records.py` |
| Change head-unit or pod processing | `headunit.py` (clock, mount calibration, attitude, strokes, side, speed, check, roll / side-slip / wobble, runs) / `pod.py` (clock, strain zero, catch / exit, force profile, side) |
| Add a session analysis (one session) | `analysis.py` (rate–speed curve, % of full-effort power, profile change, fatigue, symmetry, splits, legs, current and wind) |
| Add an analysis across sessions | `sessions.py` (`prepare_session`, `best_efforts`, `critical_power` + `race_prediction`, `rate_envelope` + `percent_of_best`, `rate_effect`, `session_summary`) |
| Model a hull, an ama, drag or power | `hydro.py` (`HullForm`, Michell wave-making, ITTC friction, air, side-slip, `Outrigger` statics, cached tables), `boats.py` (typical boats), `boatspeed.py` (`stroke_physics`, ama load from roll, `power_at_speed` / `speed_at_power`, wobble cost) |
| Read NK SpeedCoach exports | `speedcoach.py` (`read_speedcoach`, `add_headings`) |
| Test against planted answers | `sim/` (water, paddler, hull, sensors; `scenarios.py` ready-made sessions, `kind="oc1"` / `"surfski"`) + `scoring.py` |
| Run an agent loop (coaching cues, a coach's questions) | the dependency's `agent_lib/runtime/` (scheduler, `runtime.llm.call` / `run_tool_loop`, cues behind an `ActuatorGate`) — never a hand-rolled driver |
| A report or demo | `demos/` (`demo_oc1_training.py`, `demo_boat_physics.py`, `speedcoach_report.py`, `speedcoach_sessions.py`) |
