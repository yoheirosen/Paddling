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
latency; a harness check) gained 3.1 s. Development seeds (20): reference −52 s,
autopilot −11.5 s, best fixed plan −0.5 s.

**The registered result: the language-model coach did not beat the rules.** All 30
evaluation races, no failed model calls:

| coach | vs no coach (mean, 95 % interval) | vs best rule autopilot | model calls |
|---|---|---|---|
| Opus 5.5 (primary) | −2.3 s (−11.7 to +6.6) | +6.1 s (−0.3 to +12.7), faster on 9/30 | median 3.6 s each, ~USD 7 for 30 races |
| Haiku 5.5 (secondary) | −0.8 s (−9.1 to +7.0) | +7.7 s (+3.4 to +12.0), faster on 7/30 | median 1.4–2 s each, ~USD 0.3 |

Holm-adjusted p for the primary family: 1.0 (no coach), 1.0 (fixed plan), 0.23
(autopilot). Why, from the coach's own stated reasons: the boat-speed reserve is
wrong in both directions (it reads empty early because the short test reads critical
power low, and refills late because tired, less efficient blade work looks like
easing off), and the coach believed it. It told paddlers who were already pacing well
to ease (182 ease cues; +15.8 s on paddlers who held back, against −16.2 s on
paddlers who started too hot), spoke more often than the autopilot (9.7 cues a race
against 5.9), and rarely wrote rules of its own (0.2 a race).

**Exploratory rounds (development seeds only).** Round 1 gave the coach the paddle
pod's shaft power and a reserve from it, evidence about the calibration and a tool to
revise it, a prompt that says how far to trust each number, and the last five minutes
in each update: on 8 development seeds it was 18 s faster than no coach and level
with the autopilot, no longer cost paddlers who hold back (+2.3 s), and revised the
calibration 8 times, each time citing a reserve that read empty while power held with
no fade. Round 2 made the water vary across the course (above): on 6 coastal
development seeds the coach was 31 s faster than no coach and about 25 s faster than
both autopilots, but 7 s behind the rule "inside against the forecast stream": it
moved in 100-m steps, late. Round 3 added a forecast calculator, a plan per leg and
decisive line targets.

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

**Result (10 held-out races, no failed model calls, ~USD 4.2):**

| arm | vs no coach (mean) | improved coach minus this arm (95 % interval) | coach faster in |
|---|---|---|---|
| improved coach | −24.2 s | — | — |
| no coach | 0 | −24.2 s (−36.1 to −12.9); Holm p 0.004 | 10 / 10 |
| registered rule autopilot | −6.4 s | −17.8 s (−30.8 to −6.7) | 8 / 10 |
| autopilot on the pod's reserve | −11.8 s | −12.4 s (−29.0 to +3.6) | 5 / 10 |
| line rule (pod autopilot + inside against the forecast) | −31.4 s | +7.2 s (−10.4 to +23.4); Holm p 0.49 | 5 / 10 |
| perfect-knowledge even pacer (reference) | −46.0 s | +21.8 s | 4 / 10 |

The improved coach is useful on this twin: faster than no coach in every held-out
race, and faster than the pacing-only rules. It is not better than a simple rule that
steers by the forecast; ten races cannot tell them apart. It used two line cues a race
with direct targets (50–70 m inshore, 300–400 m out). Figure:
`artifacts/coaching_heldout_coastal`.

### A larger held-out check (written down 2026-11-02, before the run)

Thirty more fresh coastal seeds (3010–3039), identical design (`coaching_trial.
HELDOUT_ROUND3_EXTENSION`); primary on the new 30, also pooled with the first 10. The
same 30 races then serve as the held-out set for the tiered coach (below), whose
design is frozen on development seeds before these results are read.

**Result of the larger check (no failed model calls, ~USD 12.4).** On the 30 new races
the improved coach was faster than no coach by 21.6 s (95 % interval 6.9–41.0 s;
faster in 20 of 30; Holm p 0.014) and 4.9 s faster than the line rule, which is no
detectable difference (interval −14.5 to +4.4 s; p 0.42). Pooled over all 40 held-out
races: 22.2 s faster than no coach (10.8–37.6 s; faster in 30 of 40; p 0.0001), 1.9 s
faster than the line rule (−10.3 to +6.5 s; p 0.72), and 12.0 s faster than the
registered autopilot. So the 10-race result (7.2 s behind the line rule) was noise:
the improved coach and the forecast line rule are level, and both clearly beat no
coach and the pacing-only rules.

### The tiered coach and its held-out check (written down 2026-11-02, before the run)

The rule layer (pod autopilot plus "inside against the forecast stream") coaches by
default; the reasoning coach reviews each rule action, is woken at each leg start and
every five minutes, and steps in only on exceptions, muting a lever of the rule when
it overrules it (`coaching_trial.TieredAgentArm`, coach version 5). On development
seeds 14–19 it was 22.7 s faster than no coach and 2.0 s faster than the line rule
(faster on 3 of 6); a variant with a line-move calculator and corrected water advice
(version 6) was 2.8 s slower than the rule, so version 5 was kept, as decided
beforehand. Held-out check on the same 30 fresh seeds as the larger check (3010–3039),
written down before those results were read: tiered coach minus line rule and minus
the version-4 coach, paired, Holm over the two (`coaching_trial.HELDOUT_TIERED`).

**Result of the tiered coach's check (30 held-out races, no failed model calls, ~USD
10.9).** Against the line rule: 8.1 s faster (95 % interval −19.9 to +3.2 s; Holm p
1.0; faster in 16 of 30). Against the version-4 coach: 3.2 s faster (−10.0 to +3.4 s;
Holm p 1.0). Against no coach (secondary): 24.8 s faster (8.1–45.1 s; faster in 22 of
30). On these races the order of the means is tiered coach (−24.8 s), version-4 coach
(−21.6 s), line rule (−16.7 s) against no coach, but none of the differences between
those three is detectable with 30 races. The tiered coach mostly let the rule act: it
overruled 0.3 rule cues a race and added 3.7 cues of its own.

### Drafting a pack, and its held-out check (written down 2026-11-02, before the run)

The twin can carry a pack of boats (`sim/twin.py` `Pack`, planted): a little faster
than the paddler's even pace (1.00–1.07 times), and sitting on its wash, 6–10 m
behind, cuts drag by 8–16 %. The paddler hangs on while the effort feels bearable
(a planted willingness) and lets go otherwise; more than about 25 m back, the wash is
lost. Cues "hang on" and "let go". The coach (version 7) is told a pack exists but
cannot see it: a wash shows as speed above what the shaft power gives. Baseline: the
pod autopilot plus "hang on at the start, let go when the pod's over-pace alarm
fires". Development (pack seeds 0–5): coach 14.8 s faster than no coach and 11.8 s
faster than the pack rule (faster on 3 of 6). Held-out check on 12 fresh pack seeds
(4000–4011): coach minus pack rule and minus no coach, Holm over the two
(`coaching_trial.HELDOUT_PACK`).

**Result of the drafting check (12 held-out races, no failed model calls, ~USD 4.9).**
Against the pack rule: 11.0 s faster (95 % interval −19.1 to −4.0 s; Wilcoxon p 0.049,
Holm p 0.098; faster in 7 of 12). Against no coach: 7.7 s faster (−17.0 to +0.5 s;
Holm p 0.12). Against the pod autopilot (secondary): 8.2 s faster (−14.8 to −2.6 s).
Both fixed rules were slower than no coach here (pack rule +3.4 s, pod autopilot
+0.5 s): a rule cannot tell when a wash is worth keeping. The coach mostly cued
pushes (46) to keep the paddler on the wash and let go once. This looked like the first
lever where the reasoning coach beats the rules, but with 12 races it falls just short
of significance after the correction for two comparisons, and the role-played check
on 30 further races (below) did not repeat it.

### Confirming drafting with role-played coaching (written down 2026-11-02, before any race)

At the owner's instruction the model API is not used any more. Claude subagents stand
in for the coach's model: each coaches one race through `experiments/roleplay_race.py`,
which replays the deterministic race and stops at every model call to show exactly the
model's input (the same prompt, tools and live state), taking each answer as the
model's. Thirty fresh pack races (4012–4041) against the pack rule and no coach,
primary as before (Holm over the two); pooled with the 12 API races as a secondary,
labelled as mixed; a fidelity check role-plays 3 of those 12 races. A role-player is
not the deployed model (it remembers earlier wake-ups of its race; latency fixed at
5 s). Design in code: `coaching_trial.HELDOUT_PACK_ROLEPLAY`.

**Result of the role-played check (30 fresh pack races, every race finished; no API
calls).** The drafting gain did not repeat. Against the pack rule: 0.2 s slower (95 %
interval −13.6 to +13.8 s; coach faster in 14 of 30). Against no coach: 3.5 s slower
(−5.5 to +15.1 s; faster in 13 of 30). Holm p 1.0 for both. Against the pod autopilot
(secondary): 7.5 s slower (−0.6 to +18.1 s). Pooled with the 12 API-coached races
(42 races, two different coaches, secondary): 3.0 s faster than the pack rule (−13.1
to +7.2 s, p 0.25) and level with no coach (+0.3 s). On these 30 races the pack rule
was 3.3 s slower than no coach and the pod autopilot 4.0 s faster. Each race replays
exactly from the role-player's saved answers (`roleplay_race.py verify`: 33 of 33).
Figure `coaching_drafting_roleplay`.

Why it differs from the API run (exploratory, not part of the registered test):
* **The role-played coach let go of the pack at once; the API coach almost never
  did.** The first wake-up comes about a minute in, when riding the wash takes more
  shaft power than the paddler can sustain alone. The role-players cued "let go" in
  the first 5 minutes in 16 of 30 races (14 of them at 1.1–1.3 minutes); the API
  coach did so in 1 of 12. Races where the role-player let go early averaged 6.8 s
  slower than no coach; the rest −0.2 s. The two worst races (+129 s and +64 s) both
  began with an early let-go, followed by pace targets the paddler chased into a
  blow-up after the current changed.
* **The fidelity check agrees only in part.** On the 3 races both coaches ran, the
  role play matched the API coach on one (−45.4 against −45.1 s against no coach),
  came close on one (−5.7 against −1.4 s), and lost badly on the third (+85.6
  against −17.1 s, after two let-go cues and ten ease cues). So the role play is a
  different coach, not a copy of the API one, and this run cannot tell whether the
  API result was luck on 12 races or a real difference between the two coaches.
* **The coach cannot see whether the wash is worth keeping.** Nearly every
  role-player said so. The only sign of a wash is the gap between the measured local
  push and the current along the heading, which is noisy and also moves when the
  current changes. The reserve and the fade are both measured against minutes spent
  on the wash.
* **The coach cannot see whether a cue was followed.** The twin paddler followed
  165 of 210 role-played cues, but by a fixed step that fades back over minutes, and
  nothing in the coach's input says so. Coaches repeated ease cues that looked
  ignored, and several were blocked for coming within 60 s of the last one, which
  the coach could not see either.

What this does to the drafting claim: the result is weaker. With the API coach on
12 races, the coach beat the pack rule by 11 s, just short of significance. With a
role-played coach on 30 further races it did not, and the pooled 42 races show no
reliable gain. The parts of the coach's input that the role-players found misleading
are listed under next steps; they are the same ones that make the early let-go
look right.

### Open races: what nobody wrote a rule for (written down 2026-11-03, before any held-out race)

**Why.** Every coaching test so far asked the coach to pace a race whose best answer a
pacing algorithm can compute, and the coach's prompt had grown into that algorithm written
in English. A model following such rules can at best equal the rules. Reasoning should
pay off when something happens that nobody planned for, the evidence is ambiguous, or the
paddler says something. So these races add incidents, a paddler who talks, and an open
coach.

**The races** (`sim/incidents.py`, planted, drawn per race on their own random stream):
* **weed on the rudder**: 6–12 % more drag until a back-stroke clears it. Half the
  paddlers say the boat feels heavy; a third clear it on their own after 2–7 minutes.
  Left alone it cost 39 s on average in 18 development races (6 to 91 s).
* **an unforecast wind shift** of 0.12–0.30 m/s along the course; half the time the
  escort boat radios it;
* **the paddle pod drifting** to 72–82 % or 120–132 % of the true effort (held out);
* **the race committee adding or removing a leg** by radio (held out). The paddler
  learns of it only if told; otherwise they sprint for the old finish.
* **the paddler's talk**:
  * an honest paddler says when the legs are going;
  * a pessimist says they are cooked while they are not, then eases off unless
    reassured. That often *helps*, because many paddlers start too hard, so the right
    reply depends on the numbers;
  * an optimist says they feel great while the reserve is low, then pushes unless held.

**The open coach** (`OPEN_COACH_SYSTEM`):
* Its prompt gives the goal, the means and facts about the instruments ("any instrument
  can fail or drift"; "paddlers misjudge themselves"), and no procedures.
* It keeps notes across wake-ups, can ask the paddler how they feel, and sets its own
  watches; it starts with none.
* It is woken by its watches, every 3 minutes, and by what the paddler or the radio says.
* Each wake-up shows its recent cues and how many cues it has left.
* Paddler actions on its menu: ease, push, hold, rate up or down, finish, clear the
  rudder, reassure, and tell (with a distance).

**Practice** (three development races, role-played, before this registration):
* The coach was 24.7 s faster than no coach.
* It was 16.5 s slower than rules written knowing the weed. The rules spot the drag step
  in about a minute; the coach took about four, because the wind hid the signal.
* In two races it first judged the weed to be the wind. It then found it by comparing
  speed on matching legs and had it cleared.
* It corrected the critical power from the race.
* The practice coaches' input gaps were fixed: the cue cap, the cap on corrections,
  note length and the names that watches can use.

**The check** (`coaching_trial.HELDOUT_OPEN`):
* 30 fresh races (seeds 5000–5029). Each holds at least one held-out kind (pod drift or a
  course change), never used in development.
* Arms: no coach; the pod autopilot; the incident rules (threshold 0.06 m/s, the best on
  24 development races); the role-played open coach.
* Primary: coach minus incident rules and coach minus no coach, Holm over the two.
* Secondary: the races with each held-out kind, and what happened to each incident (from
  the simulator's truth).
* What would show the value of reasoning in the race: faster than the incident rules
  overall, or clearly better on the kinds no rule anticipated. On the kinds the rules
  were written for, they are expected to react faster.
* The coach's menu holds the actions the held-out kinds need (tell with a distance,
  correcting the race distance). What is held out is the incident and its diagnosis,
  not the action.

**Result of the open-race check.** 30 held-out races; every race finished, and every race
replays exactly from the role-player's answers. The coach answered 57 model calls a race
on average (37 to 78). Figure `coaching_open_race`.

| Open coach minus … | Mean | 95 % interval | Holm p | Coach faster in |
|---|---|---|---|---|
| no coach (primary) | −24.8 s | −37.1 to −12.3 | 0.0007 | 24 of 30 |
| rules written for the practice incidents (primary) | −6.0 s | −16.5 to +4.3 | 0.27 | 18 of 30 |
| fixed pod rules (secondary) | −9.7 s | −19.5 to +0.6 | p 0.047, uncorrected | 21 of 30 |

Against the incident rules at their other two detection thresholds: −8.6 s and −6.3 s.

* **A course changed by radio** (held out; 22 races).
  * The coach corrected the race distance and told the paddler in 21 races.
  * No coached paddler sprinted for the old finish. Under the rules and with no coach,
    16 of 22 did: on a lengthened course they started their sprint 1.5–1.9 km early.
  * The coached sprint started 300–700 m from the true finish in 20 of the 21 races
    with a recorded start.
  * In these races the coach was 12.1 s faster than the incident rules (−23.9 to −1.1 s).
* **The paddle pod drifting** (held out; 17 races).
  * Race time was no different from the rules (−2.0 s, −16.5 to +11.4 s).
  * The pre-registered cue count does not separate the arms. It counts every cue in the
    direction the faulty reading points: 45 such cues and 21 against for the coach, 3
    and 0 for the rules, which cue far less. Many of the coach's cues had other reasons,
    such as finish pushes.
  * Exploratory, from its notes: the coach named the pod as faulty in 16 of 17 drift
    races. It did so by checking the pod against GPS speed on matching legs. It also
    suspected the pod in 5 of the 13 races without drift.
* **Weed** (a practice kind; 24 races).

  | | Open coach | Incident rules | No coach |
  |---|---|---|---|
  | Mean seconds raced with weed | 355 | 618 | 1058 |
  | Weed cleared by a cue | 18 | 12 | 0 |
  | Weed never cleared | 3 | 8 | 14 |
  | Median time to the first clearing cue | 89 s | 63 s | — |
  | Back-strokes with no weed to clear (each costs about 3 s) | 14 | 3 | 0 |

  The rules react faster when their detector fires. The coach found weed the detector
  missed, using the paddler's words or by comparing legs. It also cleared "weed" that
  was the wind, a blow-up or a tired blade.
* **The paddler's talk:** little difference. Self-misjudgements caught in time: 9 for
  the coach, 8 for the rules. Most paddler remarks were fixed phrases.
* **Blowing up:** an average of 182 s blown up per race with the coach, 223 s with the
  rules and 237 s with no coach.
* **Where the coach lost** (exploratory). In the 8 races without a course change it was
  10.6 s slower than the rules (−6.9 to +28.8 s). It lost 12 of the 30 races to the
  rules, by up to 58 s. The role-players' own reports name two causes:
  * pushes made on reserve figures that were wrong, which led to blow-ups;
  * corrections to critical power made on a few minutes of evidence.

**What it shows.** This is the first check in this project where a reasoning coach
clearly beats no coach on races it was not tuned on, and is ahead of every fixed rule.
It is not significantly ahead of rules written for the practice incidents.

The clearest value comes on the incident no rule anticipated: the course changed by
radio. A rule could be written for it, but only by someone who foresaw it. The coach
also found instrument faults it was never told about, but that did not turn into race
time. Its weakest point is acting too confidently on numbers it had no way to check.

Limits:
* the role-player is a stand-in, not the deployed model;
* the incidents and the paddler's responses are planted;
* the rules do not read the radio, as registered.

### The open coach's later version, and its held-out check (written down 2026-11-03, before any race)

The first open-coach check left one comparison unsettled: against rules written for the
practice incidents, the coach was 6.0 s faster (Holm p 0.27). The role-played coaches
had listed what they could not see, and the later version (`RaceCoach` version 9) adds
it. Its instructions stay goal, means and facts, with no procedures.

What the new version adds:
* **What a cue did:** after each cue, what the instruments saw:
  * pod power, boat-model power, stroke rate and GPS speed in the minute before and
    after;
  * whether a turn fell inside that window;
  * for a rudder clear, whether the boat's speed dipped as a back-stroke does.
* **Memory and timing:** a pinned race plan; wake-ups it sets for itself and can cancel.
* **Easier reading:**
  * a table of legs, so legs in the same direction can be compared;
  * series summarised in race seconds and alerts given in race minutes;
  * a short list of stream names;
  * the turn position as a stream, with the finish not counted as a turn.
* **Calibration:** corrections to critical power and to the reserve size, which
  recompute both reserves over the race so far.
* **The paddler:** answers and remarks that follow their state.
* **One fact:** an emptied reserve forces the paddler well below critical power.

**Practice** (three development races; inputs fixed afterwards: the short stream list, a
way to cancel wake-ups, the finish no longer counted as a turn):
* The coaches used the new inputs. They confirmed that a back-stroke had happened (a 33 %
  dip when the paddler did it, 1 % when not), and found weed by comparing legs in the
  same direction.
* In two of the three races they still raised critical power on a few minutes of
  evidence, and pushed the paddler into a blow-up.

**The check** (`coaching_trial.HELDOUT_OPEN_V9`):
* 30 fresh races (seeds 5030–5059), drawn as before: every race holds pod drift or a
  course change.
* The same arms as before.
* Primary: coach minus the incident rules, and coach minus no coach, Holm over the two.
* Run on agentic_experiments with the `maximum()` fix. Every race replays from its
  answers.

**Result of the later version's check.** 30 held-out races; every race finished and every
race replays exactly from the role-player's answers. The coach answered 59 model calls a
race on average (34 to 84). Figure `coaching_heldout_open_v9`.

| Later open coach minus … | Mean | 95 % interval | Holm p | Coach faster in |
|---|---|---|---|---|
| rules written for the practice incidents (primary) | +0.7 s | −8.7 to +10.4 | 0.95 | 14 of 30 |
| no coach (primary) | −13.7 s | −27.6 to +0.1 | 0.0496 | 22 of 30 |
| fixed pod rules (secondary) | −6.4 s | −17.8 to +4.7 | p 0.25, uncorrected | 17 of 30 |

The comparison with no coach passes Holm's correction just barely, and its bootstrap
interval just reaches zero. Against the incident rules at their other two detection
thresholds the coach was +0.2 s and +0.7 s.

* **Weed** (a practice kind; 22 races).

  | | Later open coach | Incident rules | No coach |
  |---|---|---|---|
  | Mean seconds raced with weed | 236 | 737 | 1048 |
  | Weed cleared by a cue | 20 | 7 | 0 |
  | Weed never cleared | 1 | 10 | 15 |
  | Median time to the first clearing cue | 82 s | 55 s | — |
  | Back-strokes with no weed to clear | 9 | 2 | 0 |

  The coach found and cleared the weed far more often than the rules did. In these
  races it was still only 2.2 s faster than the rules (−13.8 to +10.1 s).
* **A course changed by radio** (held out; 19 races).
  * The coach told the paddler the new distance in 18 races. One coached paddler
    sprinted for the old finish, against 10 of 19 under the rules and with no coach.
  * The coached sprint started within 700 m of the true finish in 17 of the 19 races.
  * The coach was 6.6 s faster than the rules in these races (−18.0 to +6.3 s).
* **The paddle pod drifting** (held out; 23 races). The coach was 3.9 s slower than the
  rules (−6.4 to +14.9 s). It gave 60 cues in the direction the faulty reading points
  and 37 against it. The exploratory reading of its notes no longer separates the
  races: the notes call the pod suspect in all 23 drift races, and also in all 7 races
  without drift.
* **The paddler's talk.** The rules caught more of the paddler's own misjudgements in
  time: 21, against 14 for the coach, which caught another 10 late. Any ease or hold
  catches a paddler about to push too hard, and the rules ease often.
* **Blowing up.** The coach left the paddler blown up for 178 s a race on average,
  against 137 s for the rules and 196 s for no coach. Across races, how much more time
  the coach spent blown up than the rules goes with how much slower it raced
  (correlation 0.6). 13 of the 54 blow-ups under the coach began within 3 minutes of
  one of its pushes. At 20 of its 99 pushes the paddler's true reserve was below a
  fifth.
* **Critical-power corrections** (exploratory). The coach corrected critical power in
  23 of 30 races (18 of 30 for the first version). In those races it left the paddler
  blown up 59 s longer than the rules did. In the 7 races without a correction it was
  16 s shorter.

**Against the first version** (descriptive only: different races, so not a paired
comparison):
* Weed time fell (236 s against 355 s), and fewer back-strokes were wasted (9 against 14).
* Course changes were handled as well as before.
* Overall, the later version was not faster against the rules (+0.7 s against −6.0 s),
  and its lead over no coach was smaller (−13.7 s against −24.8 s).
* Its races were shorter on average (41 minutes against 48), and the rules blew up less
  in them (137 s against 223 s). That left the coach less room to gain.

**What it shows.** The new inputs did what they were built for: the coach confirmed its
back-strokes, cleared weed sooner, and kept every course change under control. Race time
did not follow.

The coach loses its gains by spending the paddler's reserve on figures it cannot check.
It raises critical power after a few minutes that look strong, or pushes on a reserve
reading that a drifting pod or the wind has bent. The rules never touch critical power
and never push on a reserve, so they cannot make that mistake. Better diagnosis does
not fix it, because the error is in what the coach does with an uncertain number.

What the coaches said they still lacked, most often first:
* a way to tell, after a blow-up, a pod reading high from a tired blade or weed;
* reserve figures that stay put: every correction recomputes both reserves from the
  start;
* the wind as a live input; the fitted current lags, and boat power reads wrong in
  opposite directions on alternate legs;
* a pod offset it can set once it has diagnosed drift;
* the time until the next cue counted at the moment of the cue. Within a wake-up the
  value goes stale, and cues were refused by a second.

Two faults in the experiment were seen during the run and left alone, as registered:
* the role-player's terminal cut off the first input, so the role-players read the
  tool list separately;
* a wake-up set in the same batch as a change of race distance uses the old
  distance.

Limits:
* the role-player is a stand-in, not the deployed model;
* the incidents and the paddler's responses are planted;
* the rules do not read the radio, as registered.

### A coach that runs experiments, and its held-out check (written down 2026-11-07, before any held-out race)

**Why.** In both open-coach checks the coach found problems well: weed, a course changed by
radio, a drifting pod. It lost race time when it pushed the paddler, or raised its estimate
of their critical power (the effort they can hold for a long time), on numbers it had no way
to check. A reasoning coach should not have to guess: it can test. This version gives the
coach the means to run small experiments during the race, and to keep what it has learned.

**What the coach can now do** (`RaceCoach` version 10, `OPEN_COACH_SYSTEM_V10`; goal, means and
facts only, no procedures):
* **Ask for a test piece**: up to three segments, each a change of effort or stroke rate held
  by the clock (for example 150 s at +6 %, then 120 s back). A piece is one cue.
* **Measure any stretch of the race** (`measure`): means with their uncertainty, the
  heart-rate trend, and a fit through the hull model. The fit splits speed over the ground
  into two parts:
  * the share of pod power that reaches the boat, which drag, the pod's gain and blade
    efficiency all scale;
  * a push of current and wind, which flips sign between the out and back legs.
* **Price a piece** (`test_cost`): metres gained while it lasts, the reserve it spends, and
  the net seconds at the finish once the reserve is repaid.
* **Keep beliefs** (`set_belief`): a value, a range and the evidence, shown at every wake-up.
  Beliefs about critical power, the reserve's size and the pod's gain set the boat
  computer, which recomputes both reserves over the race so far. Other beliefs (the
  paddler's honesty, their best stroke rate) are the coach's own notes.

**A new instrument: a heart-rate strap.** Heart rate gives a test something to read:
* below critical power it levels off at a steady effort;
* above it, it keeps climbing as the reserve drains;
* it falls back as the reserve refills.

This is the textbook shape, and it is planted. Its level differs between paddlers and
creeps up over a race. The strap is drawn on its own random streams, so it changes nothing
else in a race, and every arm races with it.

**What the coach is scored on.**
* Race time.
* How well the boat computer knew the paddler (the learning measures,
  `coaching_trial.score_learning`, the same for every arm, because every arm's corrections
  go through the boat computer):
  * the mean error of its critical power over the race, against the true critical power,
    which falls slowly as the paddler tires;
  * the mean error of the reserve read from the pod;
  * the error of the pod's gain in races where it drifts.

  The reserve error turned out to depend on the pacing as well: a paddler kept below
  critical power keeps both the reading and the truth near full. So the critical-power
  error is the co-primary learning measure.

**A new comparison: the same tests, on a schedule, without reasoning** (`ScheduledTests`).
These are the incident rules plus two tests:
* a three-minute steady hold once per leg, read by fixed thresholds on the heart-rate
  trend;
* an out-leg against back-leg hull fit after every turn, with a rudder clear and then a pod
  correction when the share moves.

What the coach can show over this arm is what reasoning adds: choosing which test, when, and
what its result means.

**Tuning the scheduled tests** (24 development races, seeds 101–124):
* With the pod check's threshold at 7 %, the scheduled tests were 4.3–4.8 s faster than the
  incident rules at every heart-rate threshold (0.4, 0.6, 0.9 beats per minute per
  minute). At 10 % they gained nothing.
* They left the paddler blown up longer than the incident rules did: 145 s a race against
  116 s.
* Primary setting: 0.6 and 7 %, the middle of a flat range.

**Practice** (three development races, role-played, seeds 21–23):

| Race | Experimenting coach | Incident rules | Scheduled tests | No coach |
|---|---|---|---|---|
| 21 | 3229 s | 3308 s | 3263 s | 3323 s |
| 22 | 1855 s | 1852 s | 1852 s | 1890 s |
| 23 | 2658 s | 2665 s | 2684 s | 2678 s |

* The coaches read heart rate as intended. In one race: falling at 203–208 W, flat at
  212 W, climbing at 216–219 W. They used the hull fit to correct the boat computer's
  critical power, which disagreed with the pod's.
* They found weed by speed per pod watt and confirmed the clear.
* Their critical-power estimates were no better than the pre-race test's (about 2 % off).
* The fixed heart-rate rule moved critical power on noise: 3–5 % off.

Fixed after practice, before registration:
* The paddler's handling of test pieces (a bug): a pace target kept raising the effort
  under a piece, and a piece started during another took that inflated effort as its
  base (+6 % gave +25 %). Now a pace target pauses during a piece, and a new piece
  replaces a running one from the effort being held. One practice race (23) used a piece
  started during another, so it no longer replays to the same time.
* Inputs the coaches asked for:
  * each piece's readings by segment;
  * the pod's reserve at both ends of the coach's critical-power belief;
  * measuring at a fixed share of pod power;
  * wake-ups cancelled one at a time;
  * the distance since the last turn as a stream;
  * facts on pace targets, on which metrics can read past values, and on current and
    wind flipping with direction.

**The check** (`coaching_trial.HELDOUT_EXPERIMENTER`):
* 30 fresh races (seeds 5060–5089), drawn as before: every race holds pod drift or a
  course change, kinds never used in practice.
* Arms: no coach; the incident rules (threshold 0.06 m/s); the scheduled tests (primary
  setting); the experimenting coach, role-played by Claude subagents at 5 s per model call.
* Co-primary outcomes, each tested paired (Wilcoxon, bootstrap 95 % interval) with Holm's
  correction over its two comparisons:
  * race time, coach minus scheduled tests and coach minus incident rules;
  * the mean critical-power error, coach minus scheduled tests and coach minus incident
    rules.
* Secondary:
  * coach minus no coach;
  * the reserve error;
  * critical-power error at halfway and three quarters;
  * pod-gain error in drift races;
  * races with each incident kind and the incident table;
  * the scheduled tests at their other heart-rate thresholds.

**Result of the experimenting coach's check.**
* 30 held-out races. Every race finished, and every race replays exactly from the
  role-player's answers.
* Five races were interrupted by a container restart and resumed from their saved
  answers.
* The coach answered 76 model calls a race on average. It ran 4 test pieces a race, in 28
  of the 30 races (8 of them stroke-rate tests), and set 7.6 beliefs a race.
* Figure `coaching_heldout_experimenter`.

Race time (co-primary):

| Experimenting coach minus … | Mean | 95 % interval | Holm p | Coach faster in |
|---|---|---|---|---|
| the same tests on a schedule | +5.6 s | −12.2 to +22.5 | 0.86 | 13 of 30 |
| rules written for the practice incidents | +3.1 s | −16.0 to +20.6 | 0.86 | 13 of 30 |
| no coach (secondary) | −14.3 s | −32.4 to +2.7 | p 0.08, uncorrected | 20 of 30 |

Against the scheduled tests at their other heart-rate thresholds the coach was +5.8 s
and +5.1 s.

How well each arm knew the paddler's critical power (co-primary): the mean error of the
critical power the boat computer used, over the race.

| | Mean error | Coach minus it | Holm p |
|---|---|---|---|
| experimenting coach | 2.5 % | | |
| the same tests on a schedule | 5.9 % | −3.4 points (−5.5 to −1.6) | 0.007 |
| rules (never change it) | 2.0 % | +0.5 points (+0.1 to +1.0) | 0.03 |

* The coach knew critical power better than fixed rules running the same tests, but
  slightly worse than leaving the pre-race test alone.
* Leaving it alone gains over a race in this simulator: the true critical power falls as
  the paddler tires, towards the test's low estimate (4.3 % off at the start, 1.1 % at
  three quarters).
* Of the coach's 75 changes to critical power, 50 moved it towards the truth and 25 away.
* Of its 39 raises, 24 went above the truth.
* The paddler was blown up for 204 s a race with the coach, 189 s with the scheduled
  tests, 180 s with the incident rules and 192 s with no coach.

What the coach learnt well (secondary):
* **The paddle sensor's drift** (25 races): its correction was 5 % off the true gain on
  average. The scheduled tests' correction was 16 % off, and with no correction (rules, no
  coach) the reading was 25 % off. The coach found the drift by comparing the sensor with
  GPS speed, the hull model and heart rate. In most races it confirmed the correction on
  legs in both directions.
* **The reserve:** the reading of the paddler's reserve was 20 points from the truth on
  average with the coach, against 35 for every other arm.
* **Weed** (23 races): 346 s raced with weed, against 386 s for the scheduled tests (whose
  sensor check also clears weed), 807 s for the incident rules and 1236 s with no coach.
  Weed was cleared by a cue in 20 races, and 20 back-strokes had no weed to clear.
* **A course changed by radio** (17 races):
  * The coach corrected the distance and told the paddler in all 17.
  * The paddler took in the first message in only 6 of the 17. In the earlier checks they
    took it in 19 of 22 and 16 of 19 times.
  * Each miss was traced to the paddler's random draw for following a cue. It is chance,
    not a code change, but it decided the outcome: 7 coached paddlers still sprinted for
    the old finish, against 11 of 17 under every other arm.
  * The coach was 6.2 s faster than the incident rules in these races (not significant).
* By race kind, coach minus scheduled tests: +9.4 s with pod drift, −3.9 s with a course
  change, +1.1 s with weed. None was significant.

**What it shows.**
* The coach used its abilities to learn about the instruments and the water. It found and
  corrected a drifting sensor far better than any fixed rule, and read the reserve far
  closer to the truth.
* That knowledge did not turn into race time.
* Critical power is where the coach most needed to experiment, and its experiments misled
  it there. A heart rate that climbs slowly just above critical power, or that sits at its
  maximum once the reserve is empty, looks flat. The coach read flat as "below critical
  power", raised its estimate, pushed, and the paddler blew up.
* The same fixed rules running the same tests did worse still on critical power (5.9 %),
  so reading a noisy test is hard for rules and coach alike. The coach was only better at
  it.

Faults found during the run (left alone, as registered):
* The instructions never say that heart rate stops at a maximum; at least five coaches
  read a plateau at the maximum as a steady state below critical power.
* In agentic_experiments, a metric the coach writes over past values whose last line is an
  assignment (`hr_slope = slope(w) * 60`) is accepted but never produces a value. The
  warning says only "produced NO value". Several coaches lost their own metrics this way.
* The sensor correction takes one constant from one race second, so a drift that ramps
  cannot be entered.
* The coach cannot see whether the paddler took in a message.
* Every input repeats all the coach's notes, and long inputs were cut off in the
  role-player's terminal.

Limits:
* the role-player stands in for the deployed model;
* the incidents, the heart rate and the paddler are planted;
* the rules do not read the radio, as registered.

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
* The **paddle pod's shaft power** on the twin is the true shaft power with 5 %
  noise per stroke; a real pod adds calibration and temperature drift.
* The **coastal water is planted** (`sim/twin.py` `CoastalWater`): the inshore
  profile (25 % of the channel stream at the shore, rising over ~150 m), eddies that
  reverse the inshore water behind points, a stream that changes linearly and can
  turn, crossing at 0.8 m/s sideways. The forecast's errors are drawn too (stream
  ±0.1 m/s, its change ±0.15 m/s, eddies misplaced by ~120 m, 30 % missing, 30 %
  chance of a false one, the inshore width ±25 %). How much line choice is worth on
  the twin follows from these choices.
* **Drafting is planted** (`sim/twin.py` `Pack`): one pack, steady through the water
  at 1.00–1.07 times the paddler's even-pace speed, slowing 15 % at its turns; on its
  wash (6–10 m behind) the paddler's drag falls by 8–16 %; off it, holding on takes the
  full effort plus a margin to close, and more than ~25 m back the wash is lost; the
  paddler's willingness to work for the wash is a drawn trait. Real packs surge,
  split and race each other, and wash riding depends on position and sea state.
* **Incidents and the paddler's talk are planted** (`sim/incidents.py`, `sim/twin.py`):
  how much weed drags and what clearing it costs (half the boat's speed for a moment),
  how often paddlers notice or clear it themselves, the wind shift's size and onset, the
  pod's drift, the course change and the paddler's belief about the finish, and how each
  kind of paddler talks and acts on their own words. The paddler's words are fixed
  phrases, and they act only on a cue's request, never its words.
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
  tide model or the pod's water speed) can be passed in instead.
* **The heart-rate strap is planted physiology.** Heart rate follows effort with a lag of
  30-50 s, levels off below critical power, and above it climbs with the share of the
  reserve spent (reaching the maximum as it empties). It also creeps up 1-3 beats per 10
  minutes and reads with 1.5 beats of noise. This is the textbook shape, not measured on
  the paddler. Real heart rate also moves with heat, hydration and nerves, which the twin
  leaves out, so a heart-rate test in the twin is cleaner than on the water.
* **Role-played coaching is a stand-in, not the deployed model.** In the drafting
  check (above), Claude subagents answered every model call, with the same prompt,
  tools and live state, and a fixed 5 s of race time per call. Each role-player
  coached one whole race and was told to decide each wake-up from its input alone,
  but it could still remember earlier wake-ups. In the first 5 role-played races
  (4000, 4004, 4008, 4012, 4014) the tool list in the first input was cut short.
  On the 3 races both ran, the role play matched the API coach on one and lost
  badly on another, so a role-played result says what this kind of reasoner does
  with the coach's inputs, not what the API model would do.
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
   twin (above), and evaluated with real models and with role play. Still to do,
   from what the role-played coaches found misleading in the drafting races:
   * show the coach its own recent cues, whether the paddler followed them, and when
     the next cue is allowed;
   * a direct estimate of how much the wash is giving (speed above what the shaft
     power gives in still water, with the current removed), with its uncertainty;
   * measure the fade and the reserve against the paddler's own power off the wash,
     and put the paddle pod's numbers (not only the boat's) in the race history;
   * the distance to the next turn; pace targets as effort rather than speed over the
     ground, which goes wrong when the current changes;
   * tell the coach when a watch it wrote cannot be evaluated: a missing input makes
     the watch silently false;
   * drop the "check the line" note at each new leg in races with no line to choose;
   * from the open-race coaches, built in the later open coach: whether a cue was
     carried out, wake-ups it sets itself, summarised series on one clock, a pinned
     plan, legs to compare, turn position as a stream, critical-power corrections that
     recompute the reserves, and a paddler who says more than fixed phrases.
   * built in the experimenting coach: test pieces, measuring stretches through the hull
     model, a heart-rate strap, beliefs that set the boat computer (including a pod
     correction), the reserve across a belief's range, the next cue's time, the first
     input read on request, wake-ups that use a new race distance.
   * still open, from the experimenting coach's held-out races (most often raised first):
     * critical power from heart rate: say that heart rate stops at a maximum, and give
       a heart-rate trend corrected for the slow creep, so a plateau at the maximum and
       a slow climb just above critical power are not read as "below it";
     * in agentic_experiments: a coach's own metric whose last line is an assignment is
       accepted but never computes (make it take the last assigned value, or refuse it
       with a clear reason);
     * a pod correction that can follow a drift as it grows;
     * whether the paddler took in a message (where they think the finish is);
     * shorter inputs: the notes repeated in full at every wake-up crowd out the latest
       cue results;
     * whether a cue or piece was taken up, sooner than the 30-second means show it.
   * in agentic_experiments: `maximum()` in an agent-defined metric was taken for a
     missing stream name, so the metric never computed. Fixed in that repository's pull
     request #234, which waits for the owner.
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
