# MAINTENANCE_LOG.md — change history for this repository

Dated entries, newest first. Entries before the move refer to paths under
`paddling/` in agentic_experiments, where this project began.

---

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
