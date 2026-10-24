# Outrigger-canoe (OC1) performance sensing — design, status and next steps

A self-contained sibling project (like `icu/` and `cardiac/`): the repository's
real-time measurement and agent machinery pointed at a solo outrigger canoe in
training, flat-water racing and downwind (ocean swell) racing. Two do-it-yourself
devices — a boat-mounted **head unit** and a **paddle pod** on the shaft — stream
records; the pipeline fuses them in real time into per-stroke and per-second
measures a paddler or coach would act on.

## Why it fits the platform

The thesis of the repository is one real-time loop pointed at different signals:
measures computed continuously in the background, cheap checks on every tick, a
slower reviewer that decides, and any action behind a gate outside the model. A
race has the same shape: the moments that decide it (a current line, a fade
two-thirds of the way through, a set of rideable swells) pass while it is
happening. This project builds the measurement half — the panel — first, scored
against a simulator with known answers, and checks it on real SpeedCoach data.

## The two devices (from the hardware brief; nothing built yet)

| Device | Streams | Rates (targets) | Notes |
|---|---|---|---|
| Head unit (sealed box on the iako, the crossbeam to the float) | GNSS position, speed and course over ground, UTC; 3-axis accelerometer and gyroscope; buttons; battery / temperature | GNSS 5 Hz (10 Hz with two satellite systems); motion 50–100 Hz | Speed is over the **ground**; the box sits at an unknown rotation on a flexing beam; vibration and wave slap |
| Paddle pod (on the shaft) | One strain-gauge bridge (raw counts); 3-axis accelerometer and gyroscope | 20–80 Hz | Own clock (no UTC); sleeps when still, wakes on motion with a short delay; Bluetooth packets every 50 ms with loss, gaps and reconnects |

Out of scope by decision: a water-speed impeller (paused). No heart-rate sensor is
in the brief; the head unit could read a standard Bluetooth strap later.

## Record format — independent of the uplink

Both devices speak flat dict records (`paddling_expt/records.py` documents every
field). A phone relay, shore Wi-Fi or a file uploaded after the session all feed
the same `OC1Engine.feed(record)` in arrival order, so the uplink can be chosen
later. Malformed records are reported, never guessed at.

## What is measured, and how

**Head unit** (`headunit.py`, causal, runs as records arrive):

| Measure | Method |
|---|---|
| Clock | head-unit clock → UTC by a running line fit over GNSS fixes |
| Mount calibration | from ~40 s of paddling: gravity gives "up", the stroke surge's dominant horizontal direction gives fore-aft, its sign from matching the GNSS speed's within-stroke wobble; earlier samples are replayed once known |
| Attitude | complementary filter (gyro fast, gravity slow), with GNSS acceleration removed from the gravity reference and wave-slap / saturated samples left out |
| Strokes and rate | one peak per stroke in the band-passed fore-aft acceleration; refractory window scaled to the stroke period |
| Side (which side the paddle is on) | a one-sided stroke yaws the bow away from the paddle; two strokes agreeing confirm a switch |
| Speed and the within-stroke profile | Kalman filter: IMU at 100 Hz, corrected by GNSS speed at 5–10 Hz |
| **Check** | speed lost between strokes (max − min over the cycle, in m/s and as a share of the mean) and the strongest deceleration |
| Distance per stroke, course | integrated fused speed; GNSS course averaged over the cycle |
| Downwind runs | speed clearly above what the strokes hold (a low percentile of recent speed) **and** the boat nose-down and accelerating |

**Paddle pod** (`pod.py`): clock alignment from the lowest arrival delays (pod
time → head time within a few milliseconds); running zero of the strain from the
blade-out-of-water part of each stroke; catch and exit from thresholds, refined
by extending the edges to zero; per stroke: **impulse, peak force, where in the
drive the peak falls, share of impulse in the first half, drive time, and the
whole force profile**; side from the shaft's lean (gravity along the paddle's
sweep axis); short radio gaps bridged, strokes with longer gaps dropped. Counts
become newtons with a static-load calibration (known weights at a known lever).

**Joined** (`fusion.py`): one row per stroke from both devices (catch-to-surge
delay learned from the session — itself a measure of how quickly blade force
becomes boat speed), plus mean blade force (impulse × rate) and a power proxy
(mean blade force × speed). A live per-second panel summarises the last strokes
in the shape the platform's agent tiers read.

**Performance analyses** (`analysis.py`, on the stroke table — finished session or
trailing live window):

| Analysis | What it answers |
|---|---|
| Rate–speed curve | which stroke rate gives the best speed; run fresh and tired, how fatigue moves it |
| Full-effort capacity curve + % power | at each stroke's own rate, what share of full-effort boat power the paddler is producing: `100 × (speed / full-effort speed at this rate)³` |
| Efficiency by rate (pod) | boat power per unit paddler effort, by rate |
| Pull-force profile change | whether the force curve changed against a reference stretch (peak position, front-loading, impulse, drive time, shape), per side, with z-scores |
| Fatigue / pacing trends | distance per stroke at a matched rate, impulse, check and consistency, per 10 minutes |
| Left/right symmetry | impulse, peak position, speed and distance per stroke by side; cost of the first stroke after a switch |
| Splits | per distance or per labelled segment: pace per 500 m, rate, distance per stroke, check, % power |
| Speed through the water, current (pod) | `mean force = k × water speed²`, `k` fitted on out-and-back legs; current along the course = GNSS speed − water speed |
| Current and wind (GNSS only) | how speed depends on direction once rate is accounted for; reports only the along-line push when the course runs along one line, and says when direction is tied to the time in the session |
| Legs | splits a session into legs of steady direction |

## Results so far

**Simulated 37-minute training session** (`demos/demo_oc1_training.py`, seed 1):
a full-effort rate ladder at interleaved rates, a 15-minute steady piece, the
ladder again tired, out-and-back legs in a 0.25 m/s tidal current. Processing ran
111 times faster than real time on this sandbox (4 virtual CPUs).

| Measure | Result against the planted answer |
|---|---|
| Strokes found (hull) | 99.7 % found, no false strokes; stroke rate within 0.32 strokes/min (median) |
| Side | 99.9 % (hull alone), 100 % (pod) |
| Speed (fused) | 0.027 m/s error flat water; 0.05 m/s in swell with wave slap |
| Check | within 0.7 % of mean speed (median) |
| Distance per stroke | within 1.8 cm (median) |
| Pod stroke timing | catch within 5 ms, exit within 4 ms; clock alignment within 3 ms |
| Pod impulse | correlation 1.00 with the true impulse |
| Left/right asymmetry | 6.1 % estimated, 6.0 % planted |
| Current along the course (pod) | leg means within 0.013 m/s |
| Best stroke rate, fresh | 61.0 vs 60.8 true (on speed through the water); 57.3 on GNSS speed — the current biases it |
| Best stroke rate, tired | 51.9 vs 54.0 true |
| % of full-effort power (steady piece) | median 75.7 % vs 75.3 % true; 4.7 points per stroke (median) |
| Pull-force profile change | peak moved 0.035 of the drive earlier vs 0.028 planted, at z = −18.6 |
| Downwind runs | 80–83 % of runs found, 94–100 % of detections real; none in flat water |

**Real SpeedCoach session** (the owner's 55-minute, 9.1 km out-and-back off Santa
Cruz on 2026-10-17; GNSS speed, rate, distance per stroke and position only —
no heart rate, power or impeller; `demos/speedcoach_report.py`):

* Current and wind together pushed the boat roughly towards the north-east:
  0.16 m/s towards 55° if speed did not fade, 0.26 m/s towards 44° with a
  0.06 m/s per 10 min fade. On one out-and-back, direction and time are tied
  (correlation 0.88), so these bracket the answer.
* Once direction is accounted for, speed was flat from 47 to 68 strokes per
  minute (2.73–2.80 m/s), while distance per stroke fell 3.8 cm per extra stroke
  per minute: at this effort, extra rate shortened the stroke rather than adding
  speed. (Rate was not controlled, so this describes the session, not a test.)
* Slowest kilometre 3–4 km (196.5 s per 500 m, heading 252°, into the push);
  return kilometres 170–174 s per 500 m at 59 strokes per minute.

## Honesty ledger (planted assumptions and limits)

* The simulator's physiology is **phenomenological, chosen so the questions have
  known answers**: a critical-power fatigue reserve plus a slow fatigue term; a
  Gaussian efficiency-versus-rate curve whose best rate falls with fatigue; a
  force-profile peak that moves earlier with fatigue. None of these is a claim
  about real paddlers.
* The **side-from-yaw** sign (a left stroke yaws the bow right) is planted; it
  needs checking on a real hull. With the pod present, the pod's side labels can
  check it.
* **Speed is over the ground.** Current and wind bias every speed number unless
  corrected (pod force model, or the GNSS-only direction fit); out-and-back pieces
  cancel it.
* **% power** measures power that reaches the boat, relative to full-effort test
  pieces; a tired or off-rate paddler working as hard shows a lower number. It
  needs those test pieces (or a stated reference).
* The pod's **power proxy** uses blade-normal force and ground speed: relative,
  not watts.
* Rate tests must use **interleaved rates** (fatigue otherwise looks like a rate
  effect) and **bracket** the best rate (a fitted peak outside the tested range is
  flagged, not trusted).
* All timings are on this cloud sandbox (4 virtual CPUs, shared).
* All rates, battery lives and ranges in the brief are targets; replace with
  measurements once hardware exists.

## Next steps

1. **Real devices**: replace simulated records with the first head-unit and pod
   logs; check the mount calibration, stroke detection, side-from-yaw and pod
   clock alignment on real data.
2. **More SpeedCoach sessions**, ideally with planned interleaved rate pieces in
   both directions (and an impeller or heart-rate strap if available), to test
   the rate–speed and % power analyses on real paddling.
3. **Coaching loop**: the per-second panel into the platform's tiered agents
   (`agent_lib/runtime/`), with cues to the paddler behind an `ActuatorGate`
   (at most one cue every few minutes, quiet during starts, turns and runs, a
   mute the paddler controls), and the shore/escort coach asking questions
   through `ask_agent`.
4. **Uplink**: choose phone relay, shore Wi-Fi or batch upload; the record format
   is already independent of it.

## Layout

```
paddling/
  PADDLING_PLAN.md   README.md
  paddling_expt/
    records.py       record schema, validation, JSON lines
    headunit.py      head-unit processing (clock, mount, attitude, strokes, side, speed, check, runs)
    pod.py           paddle-pod processing (clock, zero, strokes, force profile, side)
    fusion.py        OC1Engine: both devices -> stroke table + live panel
    analysis.py      performance analyses
    speedcoach.py    NK SpeedCoach export reader
    scoring.py       scores against the simulator's planted truth
    geo.py           frames, rotations, coordinates
    sim/             water, paddler, hull, sensors, ready-made sessions
  demos/             demo_oc1_training.py, speedcoach_report.py
  tests/             pytest suite (no network; ~25 s)
  artifacts/         four-file figure artifacts (.py + .data.json tracked)
```
