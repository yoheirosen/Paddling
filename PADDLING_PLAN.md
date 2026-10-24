# Outrigger-canoe (OC1) and surf ski performance sensing — design, status and next steps

A project built on the agentic_experiments platform (`UCSC-S-Tier/agentic_experiments`,
installed as a dependency; it lived there as the `paddling/` folder until
2026-10-24): that platform's real-time measurement and agent machinery pointed at a solo outrigger canoe or a
surf ski in training, flat-water racing and downwind (ocean swell) racing. Two
do-it-yourself devices — a boat-mounted **head unit** and a **paddle pod** on the
shaft — stream records; the pipeline fuses them in real time into per-stroke and
per-second measures a paddler or coach would act on.

The test for every feature: **does it bear on how fast the boat moves at race
speed?** Measures that do (stroke rate, speed through the water, the check, the
pull-force curve, ama load, wobble, current and wind) are built; things that do
not (capsize detection was built once and removed) are not.

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

## Where the computation runs, and why it is real time

Everything that turns raw sensor data into per-stroke and per-second numbers runs
**locally** — on the head unit, a phone in the boat, or a laptop on the escort
boat — never on the language model. The Python reference pipeline processes a
37-minute two-device session about **100 times faster than real time on this cloud
sandbox (4 virtual CPUs, shared)**, i.e. roughly 1 % of one core for one paddler;
a phone, or the same algorithms ported to the head unit's processor, has ample
margin. The hull-model tables (resistance against speed and ama load) are built
once per boat and cached (about 3 s cold, 1 ms cached).

The reasoning agent works on the compact per-second panel, not on raw data. A race
unfolds over minutes — a current line, a fade, a set of bumps — so a reasoning
turnaround of **5–10 seconds is real time for a race** and still far faster than
asking a model to compute the numbers itself. That split — continuous local
numerics, slower deliberate reasoning on their outputs — is the same design
principle as the rest of the repository, and it is what the write-up should
argue: the local compute is what keeps the loop fast enough to matter.

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

## Boat physics: resistance from the hull's shape (`hydro.py`, `boats.py`)

Wherever speed, effort and power interact, the pipeline uses a physical model of
the boat moving through the water rather than a fitted curve:

| Component | Model |
|---|---|
| Hull and ama shape | waterline length, beam, draft, where the widest point is, entry and run fullness, keel rocker, section shape (V to rounded to wall-sided); sections flare so sinking deeper widens the waterline; measured offsets can replace the parametric form |
| Hydrostatics | sinkage for a given load (closed-form displacement), wetted surface |
| Skin friction | ITTC-1957 line `C_F = 0.075 / (log10 Re - 2)^2` x Hoerner's streamlined-body form factor |
| Wave-making | Michell's thin-ship integral (1898), the standard theory for hulls as slender as canoes and kayaks; hull and ama summed in the far field with their spacing, so wave interference between them is included |
| Air | `0.5 rho CdA (speed + head wind)^2` (CdA about 0.4 m² for a seated paddler) |
| Side-slip | slender-body (low aspect ratio wing) induced drag with the free surface as a mirror: `0.5 pi rho U^2 T^2 beta^2`; side-slip builds when heading swings faster than the sway time constant `m / (pi rho U T^2)` (about 0.6 s at race pace) |
| Outrigger statics | ama at the iako span (1.8 m) and rigged height: the boat's roll sets how deep the ama sits, hence its load, hence its drag |

**Typical boats** (`boats.py`; replace with your boat's measurements): an OC1 race
hull (6.3 m waterline, 0.34 m waterline beam, 11 kg) with a 3.8 m ama, and a surf
ski (5.9 m waterline, 0.40 m beam, 11 kg), from published overall specs (OC1 hulls
6.2–6.7 m x 0.38–0.42 m; Epic V10 Gen 4 6.1 m x 0.46 m; no ama standard exists).

At 3.3 m/s (11.9 km/h), with an 89–93 kg total:

| Boat | Resistance | Power to the boat |
|---|---|---|
| Surf ski | 41.0 N (friction 31.7, waves 6.6, air 2.7) | 135 W |
| OC1, 3 kg on the ama | 42.3 N | 140 W |
| OC1, 10 kg on the ama | 45.4 N | 150 W |
| OC1, 20 kg on the ama | 48.5 N | 160 W |

On the standard Wigley test hull the wave-making integral has the expected shape
(hump near Froude number 0.3, hollow at 0.35, peak near 0.5; peak coefficient
4.6 x 10⁻³). Thin-ship theory is known to over-predict that peak for some hulls,
and none of this has yet been checked against tow-tank or paddle-power data for
these boats.

## Roll, ama load and wobble (`headunit.py`, `boatspeed.py`)

* **Ama load (OC1).** About **0.13° of roll per kilogram** on the ama in this rig,
  so roll must be absolute: a *level reference* (a button press with the boat
  level on shore; the same accelerometer's bias is captured with it and cancels)
  and removal of the boat's own accelerations from the gravity reference
  (fore-aft from GNSS speed changes, sideways from speed x yaw rate in turns). The
  static roll is the average of the left-side and right-side mean roll over 30
  strokes (each stroke leans the boat towards its own side). An optional
  *zero-load roll* (paddler seated, ama just touching the water) absorbs any
  level-reference error and the real rig height. The ama load then gives its
  drag from the hull model and the **speed it costs** at the same power.
* **Side-slip and wobble.** Side-slip from the gyro, high-passed with the sway
  time constant; roll and yaw wobble as what is left after subtracting each
  side's running average stroke pattern. The hull model says side-slip drag is
  small (about 0.03 N at 1°; it matters only at several degrees), so most of
  what a wobbly boat costs is the paddler's power — bracing and shortened strokes.
  That is estimated from the data: distance per stroke against wobble, in blocks
  of strokes, at a matched rate, with a steady trend taken out.
* **Box placement.** The box sits off the centreline, so each stroke's yaw kick
  adds a fore-aft acceleration at the box; its sideways offset is estimated from
  the session (the yaw kick changes sign with the side, the surge does not) and
  removed. This halved to quartered the surf ski's stroke-rate error.
* **Surf ski.** Strokes alternate every stroke (side labels anchored on the last
  eight strokes); no ama; the mount calibration takes the fore-aft axis in a
  narrow band around the stroke frequency, where the alternating sideways motion
  (at half that frequency) does not reach. A single-bridge pod on a double-bladed
  kayak paddle is not modelled yet.

## Results so far

**Simulated 37-minute OC1 training session** (`demos/demo_oc1_training.py`,
seed 1): a full-effort rate ladder at interleaved rates, a 15-minute steady piece,
the ladder again tired, out-and-back legs in a 0.25 m/s tidal current, hull-model
resistance. Processing ran 99 times faster than real time on this sandbox.

| Measure | Result against the planted answer |
|---|---|
| Strokes found (hull) | 97.7 % found, no false strokes (all misses are light recovery strokes straight after hard pieces); stroke rate within 0.32 strokes/min (median) |
| Side | 100 % (hull alone and pod) |
| Speed (fused) | 0.04 m/s error; 0.05 m/s in swell with wave slap |
| Check | within 0.7 % of mean speed (median) |
| Distance per stroke | within 1.7 cm (median) |
| Pod stroke timing | catch within 5 ms, exit within 4 ms; clock alignment within 3 ms |
| Pod impulse | correlation 1.00 with the true impulse |
| Left/right asymmetry | 5.2 % estimated, 6.0 % planted |
| Current along the course (pod force + hull model) | leg means within 0.012 m/s |
| Best stroke rate, fresh | 60.8 vs 60.9 true (on speed through the water); 56.5 on GNSS speed — the current biases it |
| Best stroke rate, tired | 52.7 vs 54.0 true |
| % of full-effort power (steady piece) | median 75.2 % vs 74.3 % true; 4.8 points per stroke (median) |
| Pull-force profile change | peak moved 0.036 of the drive earlier vs 0.028 planted, at z = −22.9 |
| Downwind runs | 80–83 % of runs found, 94–100 % of detections real; none in flat water |

**Boat physics** (`demos/demo_boat_physics.py`, 10-minute simulated races):

| Measure | Result against the planted answer |
|---|---|
| Power to the boat per stroke | within 1–3 % (median) |
| Ama load from roll (30 strokes) | within 0.3–1.4 kg (median) over three seeds, correlation 0.86–0.90; follows the planted lean onto the ama as the paddler tires (about 6 to 8 kg) |
| Cost of that ama load | 1.2 → 2.9 s per 500 m against a light (3 kg) ama |
| Side-slip from the gyro | median 0.99° vs 0.88° true (surf ski); its drag costs 0.06 s per 500 m |
| Wobble cost measured from one session | not conclusive (the planted cost is under 1 % of speed, within the estimate's uncertainty) |
| Surf ski strokes, sides, rate | 99.7–100 % found, sides 100 %, rate within about 1 stroke/min |

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
* Power to the boat from the hull model (typical OC1, 93 kg, 5 kg assumed on the
  ama, on speed with current and wind removed): median 82 W, 76–99 W by leg.

**Nine real SpeedCoach sessions, 30 September – 17 October 2026**
(`demos/speedcoach_sessions.py`, `sessions.py`; eight on the owner's home water,
one 15 km session 73 km away on 3 October; GNSS only — no heart rate, impeller or
power; typical OC1 hull, 93 kg, 5 kg assumed on the ama). Each session's push of
current and wind is removed from every stroke before anything is compared.

* **Conditions matter as much as fitness.** The push of current and wind ranged
  from 0.01 to 0.34 m/s between sessions, in every direction; at a typical 2.7 m/s,
  0.34 m/s into it costs about 27 s per 500 m. Raw GPS pace is not comparable
  between sessions without this correction.
* **Best efforts** (continuous paddling, power reaching the boat): 147 W for
  1 min, 136 W for 2 min, 116 W for 5 min, 107 W for 10 min, 100 W for 20 min and
  99 W for 40 min (the 20- and 40-minute bests both from the 15 km session).
* **Critical power** (the standard fit `work = CP x time + W'` to the 2–20 minute
  bests): **CP 95 W**, reserve above it **W' 5.9 kJ**, r² 0.999. With the hull
  model this predicts, in still water: 1 km 5:26, 5 km 28:26, 10 km 57:11 (171.6 s
  per 500 m). The model also says 40 minutes at about 98 W, close to the 99 W
  actually held for 40 minutes on 3 October. Training best efforts are not
  all-out, so these are conservative (the 1- and 2-minute bests sit well under the
  fitted curve).
* **Best speed at each stroke rate** (top 5 % of one-minute windows): rises
  steadily from 2.50 m/s at 41 strokes/min to 3.20 m/s at 70, about 2.5 cm/s per
  stroke/min, with **no peak** inside the rates paddled.
* **% of best power at the rate paddled** (the goal "% power from speed and rate";
  100 % = the best speed seen at that rate, hull-model power): session medians 70 %
  (30 September, an easy session) to 86 %.
* **Going up in rate within a session** (each session compared with itself,
  407 one-minute windows): **+2.8 cm/s per extra stroke/min** (95 % interval +2.0
  to +3.5, from resampling whole sessions); distance per stroke −2.3 cm. Above
  58 strokes/min the gain was larger (+3.3 cm/s per stroke/min) and the stroke
  shortened less (−1.1 cm) than below (+1.8 cm/s; −4.1 cm). Every session showed a
  gain, from +0.8 cm/s (17 October, which is why that session alone looked flat)
  to +4.1 cm/s. Rate moved with effort in these unplanned sessions, so this is
  what going up in rate gave, not what rate alone buys.

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
* The **hull model** uses typical dimensions until the paddler's boat is
  measured, has not been checked against tow-tank or paddle-power data, and
  thin-ship theory can over-predict the wave-drag peak. The simulator uses the
  same model as its physics, so the simulation tests the measurement chain, not
  the model itself.
* **Ama load** needs an accurate level reference (a 0.1° error is about 0.8 kg);
  iako flex under load (not modelled) adds roll that is not ama immersion; the
  zero-load roll capture is the remedy for both.
* **Wobble's cost to the paddler** is a planted assumption in the simulator and
  could not be measured within one session; it needs sessions with different
  conditions.
* **Real-session analyses** use GNSS speed with the push of current and wind
  fitted per session as a constant (no change within the session beyond a
  straight-line fade); a wind shift mid-session stays in the numbers. Absolute
  watts depend on the hull, the total mass and the assumed ama load (all typical
  until measured); comparisons between sessions do not. The rate effect cannot
  separate rate from effort without a force or power measurement.
* Light recovery strokes right after hard pieces can be missed (2 % of strokes in
  the training session, none in race paddling).
* All timings are on this cloud sandbox (4 virtual CPUs, shared).
* All rates, battery lives and ranges in the brief are targets; replace with
  measurements once hardware exists.

## Running it on a smartphone

The computation fits easily (about 1 % of a core); the question is sensors and
connectivity.

* **Sensors.** A phone's own GNSS usually reaches apps once a second with
  metre-level noise — too coarse for the check or per-stroke speed, which need
  5–10 Hz. Its motion sensors are adequate if the phone is rigidly mounted. The
  better arrangement is the head unit (5–10 Hz GNSS + motion sensor in a sealed
  box) and the pod talking Bluetooth LE to the phone, which then does the
  computing, the display or audio, and the internet.
* **External GPS.** Yes. Consumer Bluetooth GNSS receivers run at 10–25 Hz on
  Android (RaceChrono's Android tests: RaceBox Mini 25 Hz with its own motion
  sensor; Qstarz BL-1000GT a steady 10 Hz; Garmin GLO 2 rated 10 Hz but 7–8 Hz
  in their tests). The app should read the receiver directly over Bluetooth, not
  through Android's location service, so it keeps the full rate and the receiver's
  own timestamps. A receiver with a motion sensor, fixed rigidly to the boat, is
  close to a ready-made head unit: it feeds the same `head/gnss` and `head/imu`
  records (the mount calibration handles any orientation).
* **App.** An Android app installed from an APK ("side-loaded") is
  straightforward: a foreground service reads the devices, the pipeline runs on
  the phone (ported to Kotlin, or the Python reference via an embedded Python
  runtime), and cues go out by text-to-speech. The Claude API can be called from
  Kotlin with Anthropic's official Java SDK; the key should be entered by the
  user and stored in Android's encrypted key store, not built into the APK.
* **Connectivity.** The pipeline runs fully offline; agent calls happen when
  there is signal and queue otherwise, so a dropped connection never stops the
  measurements.

## Next steps

1. **Your boat's measurements**: waterline length and beam, widest point, section
   shape, and for the OC1 the iako span, the ama's dimensions and its rigged
   height — they replace the typical boats.
2. **Real devices**: replace simulated records with the first head-unit and pod
   logs; check the mount calibration, stroke detection, side-from-yaw, the level
   and zero-load roll captures, and pod clock alignment on real data.
3. **A planned rate test** (to separate rate from effort, which unplanned sessions
   cannot): pieces at fixed rates (e.g. 56, 60, 64, 68, 72) in random order, each
   at the same steady effort and in both directions, ideally with a heart-rate
   strap; and one all-out 3-minute and one all-out 12-minute piece to pin down
   critical power. The pod's force measurement would make this direct.
4. **Coaching loop**: the per-second panel into the platform's tiered agents
   (agentic_experiments' `agent_lib/runtime/`), with cues to the paddler behind an `ActuatorGate`
   (at most one cue every few minutes, quiet during starts, turns and runs, a
   mute the paddler controls), and the shore/escort coach asking questions
   through `ask_agent`.
5. **Phone app**: the head unit and pod over Bluetooth to an Android app that runs
   the pipeline, speaks cues and calls the agent when there is signal.

## Layout

```
paddling (this repository)
  PADDLING_PLAN.md   README.md   CLAUDE.md   MAINTENANCE_LOG.md   pyproject.toml
  paddling_expt/
    records.py       record schema, validation, JSON lines
    headunit.py      head-unit processing (clock, mount, attitude, strokes, side, speed, check, runs)
    pod.py           paddle-pod processing (clock, zero, strokes, force profile, side)
    fusion.py        OC1Engine: both devices -> stroke table + live panel
    analysis.py      performance analyses
    sessions.py      several sessions: best efforts, critical power, best speed at each rate, rate effect
    hydro.py         resistance from hull shape (friction, Michell waves, air, side-slip), outrigger statics
    boats.py         typical OC1 and surf ski (replace with measurements)
    boatspeed.py     power to the boat, ama load and its cost, side-slip and wobble costs
    speedcoach.py    NK SpeedCoach export reader
    scoring.py       scores against the simulator's planted truth
    geo.py           frames, rotations, coordinates
    sim/             water, paddler, hull, sensors, ready-made sessions
  demos/             demo_oc1_training.py, demo_boat_physics.py, speedcoach_report.py,
                     speedcoach_sessions.py
  tests/             pytest suite (no network; ~50 s)
  artifacts/         four-file figure artifacts (.py + .data.json tracked)
```
