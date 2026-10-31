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

## The in-race coach (`race.py`, `coach.py`)

The loop the project exists for: an agent that knows, during a race, whether the
pace can be held to the finish, whether the paddler is fading, and when to go, and
says it in a few words. It runs on agentic_experiments' runtime, in three tiers:

1. **Race state** (`race.RaceState`), updated per stroke, never looking ahead:
   * speed through the water (the push of current and wind fitted online from how
     speed depends on heading, forgetting over ~10 minutes; unknown, and shrunk to
     zero, until the course has turned);
   * power reaching the boat (hull model);
   * the reserve above critical power, the "W′ balance" from endurance sport.
     Work above critical power drains it; below critical power it refills in
     proportion to how far below (Skiba's model). Critical power and the reserve
     are calibrated in the same units, from a test (an all-out 3 and 12 minutes)
     or from training best efforts (`race.calibrate`);
   * for a known race distance, the steady pace that just empties the reserve at
     the finish, the gap to the pace being paddled, and how far before the finish
     the reserve runs out at the current power;
   * **fade**: speed through the water at the stroke rate being paddled against
     the same rate earlier in the race (held for 45 s after each turn). It needs no
     calibration.
2. **Screen** (`coach.RaceScreen`), deterministic, on every stroke. It raises an
   event only when something may be worth saying:
   * `over_pace` (early in the race, after the start sprint, the reserve runs out
     well before the finish);
   * `fading` (speed at the same rate down 4 % for a minute);
   * `reserve_conflict` (the reserve reads empty but the speed holds: the
     calibration is probably low);
   * `finish_window` (reserve left near the finish, the speed known to hold, no
     fade in the last five minutes).

   The screen wakes the coach on an event or every five minutes. An unknown
   reading freezes an event's timers rather than clearing it.
3. **Coach** (`coach.RaceCoach`), a language model through `run_tool_loop`. It
   sees the compact race state and the events, can read recent history, and gives
   at most one cue of at most 12 words through the `cue` tool. Every cue passes an
   `ActuatorGate`: word limit, quiet start and last metres, a two-minute
   cooldown, a cap per race.

`coach.replay_race` drives the loop on the platform's scheduler with a virtual
clock, every model call taking a fixed latency (6 s by default). A replay
therefore shows what a live coach could have said and when, with the reasoning
delay of a real model. The cue lands 6 s after the event, which is real time for a
race. The default coach is a scripted stand-in (`coach.stand_in_coach`, one fixed
cue per event): it tests the loop, not a model's judgement. With an API key,
`--provider anthropic` puts a real model in the seat. The race state costs about
1 ms per stroke on this cloud sandbox.

## Closed-loop coaching on a paddler twin (pre-registered 2026-10-30)

**Question.** Does a reasoning coach in the loop make the boat faster to the line
than no coach, a fixed pacing plan, or a rule autopilot, when the paddler hears the
cues and may act on them?

**Why a twin.** A replay of a recorded race (above) shows what a coach would have
said and when, but nothing said can change the finish. The paddler twin
(`sim/twin.py`) steps a race stroke by stroke on the coach's clock, and its paddler
changes what they do when a cue reaches them, so the finish time is the outcome.

**The twin (planted; see the honesty ledger).** It reuses the simulator's
physiology (the reserve above critical power, slow fatigue, the efficiency hill in
stroke rate whose best rate falls as the paddler tires) and the hull model, and adds:
* *blowing up*: once the reserve is spent the paddler holds only 75–90 % of critical
  power until 10–20 % of the reserve has come back; afterwards their own effort
  settles to 95–100 % of their even pace;
* *own pacing*: a start sprint (1.15–1.5 times critical power for 60–120 s); a body
  effort 0.94–1.08 times their even pace (the highest constant effort that reaches
  the line without blowing up, at their own rate, ignoring the current); a held
  habitual rate that drifts down only 0–4 spm as they tire (the best rate falls
  3–9 spm); slow wander of effort by feel; a kick 250–700 m from the line;
* *cue response*: a cue is followed with probability 0.55–0.95, after 4–15 s.
  "Ease" and "push" change effort by 3–6 % of critical power; "rate up" and "rate
  down" change the rate by 2 spm; a target pace on the display is chased by feel
  within −15 %/+12 % of the effort of the moment, and given up on blowing up;
  "hold" keeps the effort; "finish" starts the kick (as a push further than 1.5 km
  out). A followed change fades back to the paddler's own pacing over 2–7 minutes
  unless it is repeated;
* *measurement*: per-stroke GPS speed (s.d. 0.06 m/s), rate (0.4 spm) and
  heading (4°), as the SpeedCoach or head unit reports them;
* *the race*: 6, 8 or 10 km on out-and-back legs of 1 or 1.25 km, a current of up
  to 0.35 m/s along the course, a 5-minute warm-up on a short stretch;
* *calibration*: an all-out 3- and 12-minute test on the same twin in flat water.
  It reads critical power 10–25 % low against the paddler at their best, so the
  reserve figure reads empty early for most paddlers, as a real short test can.

**The panel.** Every second the race state's numbers stream into agentic_experiments'
own metric service (`race_panel.py`: `agent_lib.streaming_metrics.service.MetricService`,
in-process on an in-memory stream, as a phone would run it). The four screen events
are its starting watch rules (`over_pace`, `fading`, `reserve_conflict`,
`finish_window`), checked every second with no model call.

**Arms.** Every arm races the same paddler in the same race for a given seed, and
every cue passes the same gate: at most 12 words, none in the first minute or the
last 100 m, at least 60 s apart, at most 20 per race.
* `no_coach`: the paddler's own pacing.
* `fixed_plan`: the usual race plan. Hold a target pace on the display, computed
  before the race from the calibration (the flat-water pace that spends the reserve
  at the line) times 0.97, 1.0 or 1.03, cued after 90 s and every 120 or 300 s.
* `autopilot`: each watch alert mapped to a fixed cue (over-pace → ease to the
  sustainable pace on the display; fading → rate down; reserve conflict → hold;
  finish window → finish), with the over-pace margin 150, 300 or 600 m and the fade
  threshold −3 or −5 %. No model.
* `agent`: the reasoning coach (`coach.RaceCoach` with the panel). It is woken by
  any watch alert and every 5 minutes; it can read series, define its own metrics
  (a formula; several steps; a program admitted after the platform's latency check;
  streaming estimators) and its own watch rules, and cue. At most 5 tool rounds per
  wake-up. Each model call takes its measured wall-clock time in race time.
* reference: the perfect-knowledge even pacer, which knows the paddler and the
  current and holds the best rate. Not an arm, and not a bound (a coach can pace
  with the current or time a kick).

**Outcomes.** Primary: race time, start to the line. Secondary: time spent blown
up; true reserve left at the line; cues spoken, blocked and followed; the share of
the no-coach-to-reference gap closed (seeds where that gap is at least 10 s).

**Statistics.** Paired per seed, d = agent − comparator. Wilcoxon signed-rank
(two-sided) and a bootstrap 95 % interval of the mean d (10 000 resamples). Holm
correction over the primary family: race time against no coach, the best fixed plan
and the best autopilot. For the two baseline families, "best" is the setting with
the lowest mean race time on the evaluation seeds themselves (optimistic for the
baselines) and "typical" the median-ranked one.

**Seeds.** Development 0–19 (design checks, the scripted stand-in); evaluation
1000–1029 (the real model, and the baselines' setting selection). Nothing may be
tuned on the evaluation seeds.

**Fixed before registration, on development seeds only.** The first twin let an
over-paced paddler hold critical power once the reserve was spent (pacing barely
mattered; the perfect-knowledge pacer sprinted from the gun), so blowing up was
added, with settling after a blow-up. The drawn rate hill was too steep (a tired
paddler at their habit rate lost half their efficiency), so its ranges were
narrowed to the ones above. Pace chasing escalated without limit while blown up,
so it was bounded. The fade reading is held through turns on the panel, as the
screen did. The platform's expression language reads `max(a, b)` numpy-style (a
reduction), so the over-pace rule uses `maximum(a, b)`; the coach's prompt says so.

**What a result can say.** What in-the-loop coaching is worth *if paddlers respond as
modelled*: whether a model reasoning over the live panel, under sensor noise, a
miscalibrated reserve, partial compliance and its own latency, gives better cues at
better moments than the same panel with fixed rules or a pre-race plan. It is not
evidence that coaching helps real paddlers. The scripted stand-in
(`coach.stand_in_coach`) only checks the harness.

Runner: `experiments/run_coaching_trial.py` (`run`, `summarize`, `selftest`); the
design in code: `coaching_trial.PREREGISTERED`.

**Results so far (2026-10-30).** Baselines on the 30 evaluation seeds (races of
31–58 minutes; mean no-coach race 43.5 minutes):

| arm | mean race time vs no coach | time blown up (mean) | cues (followed) |
|---|---|---|---|
| perfect-knowledge even pacer (reference) | −31.5 s | — | — |
| rule autopilot, best setting (fade −3 %; the over-pace margin made no difference) | −8.5 s | 130 s | 5.9 (73 %) |
| fixed pacing plan, best setting (0.97 x planned pace, every 5 min) | −0.1 s | 101 s | 8.6 (72 %) |
| fixed pacing plan, typical setting | +7.8 s | 74 s | 18.7 (76 %) |
| no coach | 0 | 156 s | — |

So a perfectly informed pacer would gain about half a minute per race on this twin;
the rule autopilot gets about a quarter of that. A fixed plan prevents blow-ups but
paces too conservatively to gain time. The autopilot loses time on paddlers who were
already pacing well: the short test reads critical power low, so its over-pace rule
tells them to ease. The scripted stand-in coach (the autopilot's mapping after a 6-s
latency; a harness check) gained 3.1 s. The language-model coach has not run yet: it
waits for the API key and the choice of model. Development seeds (20): reference
−52 s, autopilot −11.5 s, best fixed plan −0.5 s.

### Held-out confirmation of the improved coach (written down 2026-11-01, before the run)

After the registered result (above), three exploratory rounds on development seeds
improved the coach: the paddle pod's shaft power and a reserve from it, live
evidence about the calibration and a tool to revise it, a prompt that says how far to
trust each number, the last five minutes in each update (round 1); water that varies
across the course, with eddies, a stream that can turn, a forecast with errors, a
local current reading, and line cues (round 2); a forecast calculator, a plan per leg
and decisive line targets (round 3). The confirmation runs on 10 fresh coastal seeds
(3000–3009) never used before: no coach, the registered autopilot, the same rules on
the pod's reserve, the pod autopilot plus "inside against the forecast stream", and
the improved coach (Opus 5.5), every baseline setting fixed in advance. Primary: race
time, coach minus line rule and coach minus no coach, paired per seed (Wilcoxon,
bootstrap interval, Holm over the two). Ten seeds detect only a large difference.
Design in code: `coaching_trial.HELDOUT_ROUND3`.

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

**In-race coach, simulated** (`demos/demo_race_coach.py`, seeds 3 and 7; a
5 km race on out-and-back legs in a 0.2 m/s current after a warm-up; critical power
from the simulated 3/12-minute test; scripted stand-in coach, 6 s per model call):

| | Over-paced race (planted reserve runs out at 23 min) | Evenly paced race (reserve never runs out) |
|---|---|---|
| Estimated vs planted reserve, first 5 min | mean error 0.05–0.06 | mean error 0.13–0.15 |
| "Too hot for the distance" cue | at 1.9 min, and again at 9.2 min (seed 3) | at 1.9 min: **a false alarm** |
| Fade cue ("drop two beats, lengthen") | 3.1 and 2.2 min before the planted reserve runs out | none |
| Reserve conflict flagged (calibration low) | at 12.4 min | at 12.5 min |
| Coach turnaround | 6 s (one model call) | 6 s |

* What works: an over-fast start shows at once, and the fade separates the two
  races cleanly from mid-race (speed at the same rate falls 5–9 % in the
  over-paced race and stays at about +2 % in the even one). The coach speaks
  rarely (2–3 cues in a 28-minute race).
* What does not, yet: the reserve model goes wrong after the first few minutes.
  * The simulated 3/12-minute test set critical power about 15 % low, because the
    paddler's blade efficiency falls during the 12-minute piece. That caused the
    even race's false alarm.
  * Later, power reaching the boat falls with fatigue at the same effort. The
    simulated efficiency drops from 0.72 to 0.45 over a race, as the best stroke
    rate drifts down while the paddler holds 62. So the model reads a tiring
    paddler as easing off, and the reserve appears to refill.

  The screen leans on the fade and the conflict flag for these reasons, and the
  coach is told to trust the speed over the reserve later in a race. The paddle
  pod's force would measure effort itself.

**In-race coach on the owner's real 15 km session of 3 October** (calibrated from
the other eight sessions: critical power 93 W, reserve 6.4 kJ; stand-in coach):
* The projected finish was within 2 minutes of the actual 86.4 minutes from
  17 minutes in: 84.5, then 88.6, 86.6 and 85.9 minutes.
* The screen raised a fast start, a reserve conflict, two fades (−11 % and −7 %)
  and two late over-pace flags. The coach spoke four cues.
* The current and wind on that course changed by leg (the online estimate reached
  0.39 m/s). That makes the live speed through the water noisy, so some of the
  fade and reserve events reflect estimation error rather than the paddler. Not
  a race and no ground truth: an illustration, not a score.

## Honesty ledger (planted assumptions and limits)

* **The paddler twin's coaching answers are planted** (`sim/twin.py`): how much
  blowing up costs (75–90 % of critical power until 10–20 % of the reserve is back),
  that a paddler settles after one blow-up, how far below their best rate a tired
  paddler drifts, and how a paddler responds to a cue (follows 55–95 % of them,
  after 4–15 s, by a fixed step, fading back over minutes). A coaching result on the
  twin is a statement about this model of a paddler, not about real paddlers.
* The twin steps **once per stroke with a steady-state hull model** and a 6-s
  speed lag, not the full 100-Hz simulator; turns cost 40 % efficiency for 6 s.
  Measurement noise is drawn per stroke, not produced by the head-unit pipeline.
* The twin's **critical-power test reads 10–25 % low** against the paddler at their
  best rate and fresh; this is what the test measures on this model (efficiency
  falls as the reserve drains inside each piece), and it is left in on purpose.

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
* **The race coach's reserve model** works on power reaching the boat, which is
  effort times blade efficiency. It is right early in a race and drifts as
  efficiency falls, and a short test can misjudge critical power (see Results).
  Its live current estimate needs the course to turn, and on a single heading
  (a point-to-point race) it cannot separate current from speed. A known push (a
  tide model or the pod's water speed) can be passed in instead. Every cue so far
  comes from the scripted stand-in; no model has coached a race yet.
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
4. **Coaching loop, next**: built, replayed, and put in a closed loop on the paddler
   twin (above). Still to do:
   * the pre-registered evaluation with a real model in the coach's seat (waits
     for the key and the choice of model);
   * the replay demo on the race panel's watches instead of the old screen;
   * effort from the pod (force) so the reserve does not drift;
   * the current from a tide model when the course does not turn;
   * the shore or escort coach asking questions through `ask_agent`;
   * a mute the paddler controls.
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
    race.py          live race state: reserve above critical power, sustainable pace, fade, push
    coach.py         in-race coach: screen, reasoning coach, cue gate, replay on the agent runtime
    race_panel.py    the race state as a live panel on the platform's metric service (watches, agent metrics)
    coaching_trial.py  closed-loop coaching on the paddler twin: arms, episodes, paired statistics
    hydro.py         resistance from hull shape (friction, Michell waves, air, side-slip), outrigger statics
    boats.py         typical OC1 and surf ski (replace with measurements)
    boatspeed.py     power to the boat, ama load and its cost, side-slip and wobble costs
    speedcoach.py    NK SpeedCoach export reader
    scoring.py       scores against the simulator's planted truth
    geo.py           frames, rotations, coordinates
    sim/             water, paddler, hull, sensors, ready-made sessions; twin.py = the paddler twin
  demos/             demo_oc1_training.py, demo_boat_physics.py, speedcoach_report.py,
                     speedcoach_sessions.py, demo_race_coach.py
  experiments/       run_coaching_trial.py (closed-loop coaching experiment runner)
  results/           coaching_trial/: preparations, per-episode tables, summaries
  tests/             pytest suite (no network; ~55 s)
  artifacts/         four-file figure artifacts (.py + .data.json tracked)
```
