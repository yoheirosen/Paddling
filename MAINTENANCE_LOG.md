# MAINTENANCE_LOG.md — change history for this repository

Dated entries, newest first. Entries before the move refer to paths under
`paddling/` in agentic_experiments, where this project began.

---

### A personal heart-rate model, and an analyst coach that works beside the rules (2026-11-09)

The owner, and several paddlers they know, have heart rates that do not follow the textbook:
high at moderate load, little rise at high load, with a low resting rate.

* `paddling_expt/heart_rate.py`: a personal heart-rate model fitted from one session (lag,
  a bending settled-level curve, a bounded slow creep). It reads a heart rate as an effort
  and a zone, and says where heart rate cannot tell efforts apart.
* `sim/twin.py`: `heart_rate_profile` (about 2 in 5 paddlers compressed) and
  `twin_step_test` (a step test out of order; can reuse a known person's physiology).
* `experiments/heart_rate_model.py`: 60 simulated paddlers, scored on held-out steady efforts
  against the textbook bands and against bands from the heart rate at threshold. Figure
  `heart_rate_personal_model`.
  * A first version gave the held-out day a different person's heart rate (the test
    session's own seed drew it); fixed by passing the same physiology.
  * Each held-out point is now plotted with the slow creep taken off.
* The heart-rate-learning coach (`RaceCoach(version=12)`) gets a `heart_rate_model` tool.
* The analyst coach (`RaceCoach(version=11)`, `AnalystArm`): the role-play driver had built
  the plain coach arm for it, so the first three practice races ran without the rule pacer.
  Those races moved to `roleplay/analyst_dev_without_pacer/`; new practice races use the
  right arm.

### A coach that runs experiments, and its held-out check (2026-11-07)

At the owner's direction ("what does an agent unlock if we give it the abilities it needs?";
fully open, no gate), the coach was given the means to experiment, and the check scored
what it learnt as well as race time.

* **The paddler twin:**
  * a short test piece (ask `piece`: up to three segments of effort or rate change, by
    the clock; a new piece replaces a running one, and a pace target pauses during a
    piece);
  * a heart-rate strap (`enable_heart_rate`, on its own random streams, so nothing else
    in a race changes).
* **The race state:** a pod gain correction (`RaceState.set_pod_gain`) and the pod's reserve
  at any critical power (`pod_reserve_at`).
* **The coach** (`RaceCoach` version 10, `OPEN_COACH_SYSTEM_V10`):
  * `measure` (stretches of the race; means with uncertainty, the heart-rate trend, a
    hull-model split of speed into the share of pod power reaching the boat and the push,
    or speed at a fixed share);
  * `test_cost`;
  * `set_belief` (the computer's critical power, reserve size and pod gain);
  * each piece's readings by segment; wake-ups cancelled one at a time; the next cue's
    time as a race second.
* **Scheduled tests** (`coaching_trial.ScheduledTests`): the same tests on a fixed schedule,
  read by fixed rules.
* **Learning measures** for every arm (`coaching_trial.score_learning`).
* **Runner:** `experiments/experimenter_race.py`.

Three practice races came first. They turned up a twin bug in overlapping pieces (+6 %
gave +25 %), now fixed. The check was then registered (`coaching_trial.HELDOUT_EXPERIMENTER`,
seeds 5060–5089) before any held-out race.

Results on 30 held-out races, role-played by Claude subagents, no API:
* race time: +5.6 s against the scheduled tests and +3.1 s against the incident rules
  (Holm p 0.86 for both); 14.3 s faster than no coach (p 0.08);
* critical-power error over the race: 2.5 % for the coach, 5.9 % for the scheduled tests
  (Holm p 0.007) and 2.0 % for rules that never change it (Holm p 0.03, against the coach);
* secondary:
  * the pod drift correction was 5 % off for the coach, 16 % for the scheduled tests and
    25 % uncorrected;
  * the reserve reading was 20 points from the truth for the coach, against 35 for every
    other arm.

The coach learnt the instruments well, and its critical-power experiments misled it: 24 of
its 39 raises went above the truth. Faults found during the run (left alone, as
registered):
* the prompt never says heart rate has a maximum;
* in agentic_experiments, a coach metric whose last line is an assignment never
  computes;
* the pod correction is one constant;
* message uptake is invisible to the coach.

The paddler took in only 6 of 17 first messages about a changed course. This was traced to
the paddler's random cue draws, not to a code change. Every earlier recorded race still
replays. One practice race no longer does, after the piece fix.

### The open coach's later version and its held-out check (2026-11-03)

The first open-coach check's role-players listed what they could not see. The later version
(`RaceCoach` version 9, `OPEN_COACH_SYSTEM_V9`) adds it. Its instructions are still goal,
means and facts.
* After each cue it sees what the instruments recorded: pod and boat power, rate and GPS
  speed before and after, whether a turn fell in between, and a back-stroke's speed dip.
* It has a pinned plan, wake-ups it sets and cancels (`wake_me`), a table of legs, series
  summarised in race seconds, alerts in race minutes, and a short list of streams.
* Corrections to critical power and to the reserve size recompute both reserves from the
  start (`RaceState.recalibrate`).
* The turn position is a stream (`race_panel.RACE_CATALOG`).
* The paddler's answers follow their state (`twin.rich_answers`, set by the coach arm for version 9), drawing the
  same random numbers as before.

Three practice races came first, then the check was registered before any held-out race
(`coaching_trial.HELDOUT_OPEN_V9`, seeds 5030–5059). The check ran on agentic_experiments
with its `maximum()` fix (that repository's pull request #234).

Results on 30 held-out races, role-played by Claude subagents, no API:
* +0.7 s against the incident rules (Holm p 0.95);
* 13.7 s faster than no coach (Holm p 0.0496);
* weed: 236 s raced with weed, against 737 s under the rules (20 of 22 cleared by a cue);
* a course changed by radio: the paddler was told in 18 of 19 races, and 1 of 19 sprinted
  for the old finish (10 of 19 under the rules);
* the coach left the paddler blown up longer than the rules did (178 s against 137 s).
  Its losses follow pushes and critical-power raises made on numbers it could not check.

Every race replays from its answers. The figure is `coaching_heldout_open_v9`: the summary
names the figure after the tag, so the first check's figure keeps its name.

### Open races and an open coach: what nobody wrote a rule for (2026-11-03)

At the owner's direction ("show the utility of agentic reasoning in real time, not an agent
pretending to be a deterministic algorithm"), the coach was rebuilt and the test changed.

Open races (`sim/incidents.py`, hooks in `sim/twin.py`), each drawn on its own random stream:
* weed on the rudder;
* an unforecast wind shift;
* the paddle pod drifting;
* the course changed by radio;
* a paddler who talks (honest, pessimist, optimist).

The paddler gained new actions: clear the rudder, reassure, and tell (with a distance).
Scoring of each incident from the simulator's truth is in `coaching_trial.score_incidents`. The
baseline `IncidentRules` is written knowing the practice incidents.

The open coach (`RaceCoach` version 8, `OPEN_COACH_SYSTEM`):
* its prompt gives the goal, the means and facts about the instruments, with no procedures;
* it keeps notes across wake-ups and can ask the paddler how they feel;
* it can correct the race distance;
* it sees its own recent cues and the cues it has left;
* it sets its own watches and starts with none.

Role play gained `--incidents`. The runner `experiments/open_race.py` races the baselines and
prints the summary and the figure. Three practice races led to input fixes, then the check
was registered (`coaching_trial.HELDOUT_OPEN`) before any held-out race.

Results on 30 held-out races, with the coach role-played by Claude subagents and no API:
* 24.8 s faster than no coach (Holm p 0.0007);
* 6.0 s faster than the incident rules (not significant);
* on a course changed by radio, the paddler was told in 21 of 22 races and never sprinted for
  the old finish (16 of 22 did without the coach); 12.1 s faster than the rules in those races;
* weed: 355 s raced with weed, against 618 s under the rules;
* the pod drift was found in its notes in 16 of 17 races, with no race-time gain.

Registered races and earlier role plays still reproduce. A platform bug was found and left
for agentic_experiments: `maximum()` fails in agent-defined metrics.

### Drafting re-checked with role-played coaching, no API (2026-11-02)

At the owner's instruction the model API is no longer used. `experiments/roleplay_race.py`
replays a deterministic race and stops at every model call, showing exactly what the model
would receive; a Claude subagent answers as the model. Commands `start` / `answer` / `show` /
`tools` / `collect`, and `verify`, which replays every finished race from its saved answers
(33 of 33 reproduce). Only the answers are tracked (`results/coaching_trial/roleplay/<tag>/
<seed>/answers.json`). Registered before any race as `coaching_trial.HELDOUT_PACK_ROLEPLAY`:
30 fresh pack races (4012–4041), plus a fidelity check on 3 of the 12 API races.

Result: the drafting gain did not repeat. The role-played coach was 0.2 s slower than the pack
rule (interval −13.6 to +13.8 s) and 3.5 s slower than no coach (−5.5 to +15.1 s), Holm p 1.0.
Pooled with the 12 API races: 3.0 s faster than the pack rule (p 0.25), level with no coach.
Exploratory: the role-players let go of the pack in the first 5 minutes in 16 of 30 races, the
API coach in 1 of 12, and those races were the costly ones. The fidelity check matched on one
race of three and diverged badly on another. Write-up, limits and next steps in
`PADDLING_PLAN.md`; figure `coaching_drafting_roleplay`; summary
`experiments/roleplay_summary.py [--figure]` → `results/coaching_trial/summary_heldout_pack_rp.json`.

Known flaws kept for this run's consistency, to fix next:
* the version-7 coach's new-leg note says "the rule layer sets the line from the forecast" in
  drafting races, which have no rule layer or line;
* a coach-written watch whose input is missing on a tick is silently false, and the coach is not
  told.

### Coaching: a 30-race held-out check, a tiered coach, drafting (2026-11-02)

Owner's choices A, B and C, each held-out check written down before it ran. Larger held-out check of
the version-4 coach (30 fresh coastal seeds): 21.6 s faster than no coach (Holm p 0.014), level
with the forecast line rule (−4.9 s, interval −14.5 to +4.4); pooled with the first 10: −22.2 s vs
no coach, −1.9 s vs the line rule. Tiered coach (`TieredAgentArm`, coach version 5: the rule layer
acts by default through a gate the coach can mute per lever; version 6 with `line_change_value`
was tried on development seeds and not kept): on the same 30 races 8.1 s faster than the line rule
and 3.2 s faster than the version-4 coach, neither detectable (Holm p 1.0); 24.8 s faster than no
coach. Drafting (`Pack` in the twin, `hang_on` / `let_go` cues, coach version 7, `PackRule`): a test
found the first pack model wrong (no slowing at its turns, gaps that could not close), corrected
before any coached held-out race and logged; on 12 fresh pack races the coach was 11.0 s faster than
the pack rule (Holm p 0.098) and 7.7 s faster than no coach (Holm p 0.12); both rules were slower
than no coach. Model spend about USD 35. Tests: 69.

### Coaching: the registered result, three improvement rounds, a held-out confirmation (2026-10-31 to 11-01)

Registered evaluation (30 seeds): the Opus 5.5 coach was 2.3 s faster than no coach and 6.1 s
slower than the best rule autopilot (not significant); the Haiku 5.5 coach 0.8 s faster than no
coach and 7.7 s slower than the autopilot. Cause, from the coach's stated reasons: it trusted the
boat-speed reserve, which misreads in both directions. Exploratory rounds on development seeds:
(1) the paddle pod's shaft power on the twin's rows and a reserve from it (`RaceState(athlete_shaft=)`,
`coaching_trial.shaft_athlete`), calibration evidence on the panel (`cp_implied_w`, steady powers,
time near empty), `RaceCoach(version=2)` with `set_critical_power`, the last five minutes in each
update and a revised prompt, `PodAutopilot`; (2) coastal water (`CoastalWater`, `draw_setup(seed,
coastal=True)`, `water_forecast`), line cues (`line_in` / `line_out` with `line_m`), GPS line and
course position, `push_local_mps`, coach version 3 with the forecast, `LineRule`; (3) coach version
4 with `stream_by_line` and a plan per leg. New random streams keep every registered race exactly
reproducible. Held-out confirmation on 10 fresh coastal seeds, written down before the run
(`HELDOUT_ROUND3`): the version-4 coach beat no coach in all 10 races (−24.2 s, Holm p 0.004) and
did not differ from the forecast line rule (+7.2 s, p 0.49). Model spend for all of it about
USD 17 (registered arms ~USD 7.5, rounds and confirmation ~USD 9.5). Tests: 66.

### Closed-loop coaching on a paddler twin, on the platform's streaming panel (2026-10-30)

The coach now runs on agentic_experiments' own streaming-metric machinery and is tested in a
closed loop. `race_panel.py` streams the race state every second into the platform's
`MetricService`, run in process on an in-memory stream: every number has a history; the coach
can define its own metrics (formula, several steps, or a program admitted by the platform's
latency check; streaming estimators) and standing watch rules; the screen's four events are the
starting watches. `RaceCoach(panel=...)` gets those tools, a structured cue (`ask` from a fixed
menu, optional target pace on the display) and its own prompt. `sim/twin.py` is a paddler twin
stepped stroke by stroke, whose paddler blows up, settles, and follows some cues (all planted).
`coaching_trial.py` + `experiments/run_coaching_trial.py`: arms (no coach, fixed plan, rule
autopilot, agent), paired statistics, the pre-registration (`PADDLING_PLAN.md`, closed-loop
coaching section), decided on development seeds before any evaluation run. Also: the race
state's sustainable pace uses a cached speed table (2.5x faster, same numbers); the platform's
`max(a, b)` is a reduction, so rules use `maximum(a, b)`; the gate checks the cue's ask.
Tests: `tests/test_twin.py`, `tests/test_coaching_trial.py`.

### Coach demo: the model's key from a named variable, sent only to the Anthropic API (2026-10-26)

Claude Code cloud sessions do not pass a variable named `ANTHROPIC_API_KEY` into the session, even
when the environment's settings hold one (checked in this session and in a fresh one). And inside a
Claude Code session `ANTHROPIC_BASE_URL` points at Claude Code's own connection, which the Anthropic
library would otherwise use. `demos/demo_race_coach.py` now reads the key from `--api-key-env`
(default `ANTHROPIC_API_KEY`) and pins the coach's requests to `--api-base-url` (default the
Anthropic API). Test: `tests/test_coach.py`.

### In-race coach: live race state, screen, reasoning coach and cue gate on the agent runtime (2026-10-24)

The loop the project was set up for. `paddling_expt/race.py` (`RaceState`) keeps, per stroke and
without look-ahead:
* speed through the water, with the push of current and wind fitted online from how speed depends on
  heading;
* power reaching the boat;
* the reserve above critical power (Skiba's model), from an `Athlete` calibrated from best efforts or
  a 3/12-minute test;
* the pace sustainable to the finish, and where the reserve runs out;
* fade at the same stroke rate.

`paddling_expt/coach.py`:
* `RaceScreen` raises over-pace, fading, reserve-conflict and finish-window events;
* `RaceCoach` (a language model through `run_tool_loop`) gives at most one 12-word cue through a
  `cue` tool behind an `ActuatorGate` (word limit, quiet start and finish, cooldown, cap);
* `replay_race` runs it on the runtime scheduler with a virtual clock and a fixed latency per model
  call;
* `stand_in_coach` is a scripted stand-in for tests and demos.

Simulator additions: `cp_test` (the 3/12-minute test) and `paced_race` (planted pacing, warm-up,
current). `sessions.prepare_rows` now takes any stroke rows (head unit or SpeedCoach), and
`critical_power` accepts two efforts.

Results (`PADDLING_PLAN.md`):
* An over-fast start is caught at once, and the fade cue comes 2–3 minutes before the planted reserve
  runs out.
* The reserve model drifts after the first minutes (power reaching the boat falls with fatigue), and
  the short test set critical power about 15 % low. That gave one false "too hot" in the evenly
  paced race, which the conflict flag later caught.
* On the owner's real 15 km session the projected finish was within 2 minutes from 17 minutes in.

Every cue so far comes from the stand-in: no model has coached a race yet (no API key in this
environment). New demo `demos/demo_race_coach.py` (figure `oc1_race_coach`); tests
`tests/test_race.py`, `tests/test_coach.py`.

### Moved out of agentic_experiments into its own repository (2026-10-24)

At the owner's request the project left `agentic_experiments/paddling/` for this
repository, which depends on agentic_experiments (package `seizure-agent-control`,
installed from git) for `agent_lib`. The history of the `paddling/` folder came
across with `git subtree split`, so its earlier commits are here. Changes for the new
home: `pyproject.toml` (package `paddling`, dependency on agentic_experiments, extras
`plot` and `test`); demos run from the repository root and import `agent_lib` from
the dependency instead of adding the agentic_experiments root to `sys.path`; tests'
`conftest.py` adds only this repository's root; README, plan and package paths
updated; new `CLAUDE.md` (the owner's standing instructions, project rules, figure
convention, when to write what); a hand-started test workflow that reads the private
dependency with a token. agentic_experiments keeps a pointer to this repository.

### Paddling: analyses across several SpeedCoach sessions (2026-10-24)

The owner shared nine more SpeedCoach sessions (30 September – 17 October; raw files
kept out of git). New `paddling/paddling_expt/sessions.py` compares sessions after
removing each one's push of current and wind and converting speed to power reaching
the boat with the hull model: best efforts over 1–40 minutes of continuous paddling,
a two-parameter critical-power fit with predicted race times in still water, the
best speed seen at each stroke rate with each session's share of it, and the speed
and distance-per-stroke change per extra stroke per minute within sessions (pooled,
interval by resampling whole sessions) next to the same slope between sessions. The
current-and-wind correction is now applied to every stroke, so single glitch strokes
(a missed stroke reads as half the rate, the next as double) no longer cut a
continuous effort; window stroke rates are medians for the same reason. Fixed
`hydro.water_resistance` for an array of speeds with one ama load (it returned one
value); added `boatspeed.power_at_speed` / `speed_at_power`. Report
`paddling/demos/speedcoach_sessions.py` (figure `oc1_speedcoach_sessions`, averages
only, no positions); tests `paddling/tests/test_sessions.py` (planted answers).
Results in `paddling/PADDLING_PLAN.md`: critical power 95 W (reserve 5.9 kJ, typical
OC1, 93 kg), predicted 10 km 57:11 in still water; best speed rises with rate with
no peak up to 70 strokes/min; within sessions +2.8 cm/s per extra stroke/min (95 %
+2.0 to +3.5), though rate moved with effort.

### Paddling: hull-shape boat physics, ama load from roll, surf ski support (2026-10-24)

At the owner's request, wherever the paddling pipeline relates speed, effort and
power it now uses a physical model of the boat in the water rather than a fitted
curve, and it supports a surf ski (no ama) as well as the OC1. New in `paddling/`:

- `hydro.py`: a parametric slender hull (waterline length, beam, draft, widest
  point, entry and run fullness, rocker, section shape; measured offsets can replace
  it) with closed-form sinkage; skin friction (ITTC-1957 line + Hoerner form factor);
  wave-making by Michell's thin-ship integral with hull and ama combined in the far
  field (so their wave interference counts); air drag; slender-body side-slip drag;
  outrigger statics (roll ↔ ama immersion ↔ load). Tables are cached on disk
  (`$PADDLING_HYDRO_CACHE`): ~3 s cold, ~1 ms cached. Checked on the standard Wigley
  hull; not yet checked against tow-tank or paddle-power data.
- `boats.py`: a typical OC1 and surf ski from published specs (to be replaced by
  the owner's measurements). `boatspeed.py`: power to the boat per stroke, the
  weight on the ama from side-balanced static roll and its cost in s/500 m,
  side-slip drag cost, and a data-driven estimate of what wobble costs the paddler.
- Head unit: an on-shore level reference (`head/level` record) for absolute roll,
  the boat's own accelerations removed from the gravity reference, the box's
  sideways offset estimated and corrected, a narrowband mount calibration (works
  for the surf ski's alternating strokes), an autocorrelation-based stroke period,
  alternating side detection, side-slip and roll / yaw wobble per stroke.
- Simulator: the same hull model as its physics; per-stroke ama load rising with
  fatigue; surf ski wobble; side-slip dynamics. Pod: gaps up to two packets bridged,
  strokes spanning a longer gap dropped. Water speed from pod force now uses the hull
  model with the current fitted as a vector (leg means within 0.012 m/s).
- Results (simulated): ama load from roll within 0.3–1.4 kg over 30 strokes,
  following the planted lean onto the ama (cost 1.2 → 2.9 s/500 m); surf ski strokes
  99.7–100 % found; wobble's cost to the paddler not measurable within one session.
  On the owner's real SpeedCoach session: median 82 W to the boat (typical OC1).
- `PADDLING_PLAN.md` adds where the computation runs and why a 5–10 s agent
  turnaround is real time for a race, the physics and roll sections, results, limits
  and running on a smartphone (external Bluetooth GNSS, side-loaded Android app). New
  demo `demos/demo_boat_physics.py` (figure `oc1_boat_physics`); `speedcoach_report.py`
  takes `--boat` / `--total-kg`; tests `test_hydro.py`, `test_boatspeed.py`. Root
  `CLAUDE.md` sibling note, layout and "when to write what" rows updated.

### Outrigger-canoe (OC1) performance sensing: new sibling project `paddling/` (2026-10-24)

A new self-contained project for a solo outrigger canoe, at the owner's request,
built from the hardware brief for two do-it-yourself devices: a boat-mounted head
unit (GNSS + accelerometer/gyroscope) and a paddle pod (strain gauge +
accelerometer/gyroscope, its own clock, Bluetooth packets). A planted-truth
simulator produces the records both devices would send (unknown box rotation,
vibration and wave slap, GNSS drop-outs, pod sleep/wake, packet loss and clock
drift); a transport-agnostic record format feeds a causal real-time pipeline
(stroke detection and rate, side, fused speed, the check, distance per stroke,
downwind runs; pod clock alignment, pull-force profile per stroke, side, left/right
balance); session analyses add the stroke-rate vs speed curve and best rate, % of
full-effort power at a rate, pull-force profile change, fatigue and pacing,
symmetry, splits, legs, speed through the water and the current or wind along the
course. On a simulated 37-minute training session every measure is scored against
the planted answer (`paddling/PADDLING_PLAN.md`, Results); a reader for NK
SpeedCoach exports runs the analyses on the owner's real 55-minute session (raw
file kept out of git). A capsize simulation and detector were built and then
removed at the owner's instruction (no bearing on training or racing). Packaging
(`paddling_expt`), the test command (root `CLAUDE.md`, `.github/workflows/tests.yml`)
and the root `CLAUDE.md` sibling note, layout and "when to write what" row were
updated.
