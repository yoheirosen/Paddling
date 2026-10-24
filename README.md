# paddling — outrigger-canoe (OC1) and surf ski performance sensing

Real-time measures for a solo outrigger canoe or a surf ski from two do-it-yourself devices — a
boat-mounted **head unit** (GNSS + motion sensor) and a **paddle pod** (strain gauge
+ motion sensor) — plus a reader for **NK SpeedCoach** exports. Built and scored on a
simulator with planted answers; checked on real SpeedCoach sessions. Built on the
**agentic_experiments** platform (`UCSC-S-Tier/agentic_experiments`, installed as a
dependency): `agent_lib` writes the figure artifacts today and will run the coaching
agents. Design, results and limits: **`PADDLING_PLAN.md`**; conventions for
contributors and AI assistants: **`CLAUDE.md`**.

Per stroke: stroke rate, speed, **check** (speed lost between strokes), distance per
stroke, side, and with the pod the **pull-force profile** (impulse, peak, where the
peak falls in the drive, front-loading, drive time). Per session: the stroke-rate vs
speed curve and best rate, **% of full-effort power at a given rate**, changes in the
pull-force profile, fatigue and pacing trends, left/right symmetry, splits, legs,
speed through the water and the current or wind along the course, downwind runs.
Across sessions: best efforts from 1 to 40 minutes, critical power and predicted race
times, the best speed seen at each stroke rate and each session's share of it, and
whether going up in rate bought speed (`sessions.py`).
In a race, a **coach** in the paddler's ear (`race.py`, `coach.py`):
* **The live race state:** the reserve above critical power, the pace that can be held
  to the finish, and fade at the same stroke rate.
* **A screen on every stroke** decides when something may be worth saying.
* **A language model** gives at most one short cue, behind a gate (word limit, quiet
  start, cooldown), on agentic_experiments' runtime.

Where speed and effort interact it uses a **hull-shape model** (`hydro.py`): skin
friction, wave-making (Michell's thin-ship integral, hull and ama together) and air
drag, for an OC1 with its ama or a surf ski (`boats.py`: typical dimensions until
your boat is measured). From the head unit's roll it reads **how much weight sits on
the ama** and what that costs in seconds per 500 m; for the surf ski it reads **bow
wobble** (side-slip, roll and yaw wobble) and its cost (`boatspeed.py`).

## Install

```bash
# needs read access to the private agentic_experiments repository (your git credentials)
pip install -e ".[test,plot]"

# or, working on both repositories side by side: your local agentic_experiments checkout,
# then this package without its dependencies (else pip replaces the checkout with main)
pip install -e ../agentic_experiments
pip install -e . --no-deps && pip install pytest matplotlib
```

## Run

```bash
# simulated 37-min training session through both devices, scored against the truth
python "demos/demo_oc1_training.py" --figure oc1_sim_training

# hull-shape physics: resistance, ama load from roll and its cost, surf ski wobble
python "demos/demo_boat_physics.py" --figure oc1_boat_physics

# report on a real SpeedCoach export (keep the raw file out of git: it holds a GPS track)
python "demos/speedcoach_report.py" SESSION.csv --boat oc1 --total-kg 93 \
    --figure oc1_speedcoach_session

# across several SpeedCoach sessions: best efforts, critical power and predicted race
# times, best speed at each stroke rate, % of best, what going up in rate gave
python "demos/speedcoach_sessions.py" EXPORT_FOLDER --figure oc1_speedcoach_sessions

# the in-race coach replayed: a 3/12-minute test, then an over-paced and an even race
# (scripted stand-in coach; --provider anthropic with ANTHROPIC_API_KEY for a model)
python "demos/demo_race_coach.py" --figure oc1_race_coach
# ... or one of your SpeedCoach sessions, calibrated from the others
python "demos/demo_race_coach.py" --speedcoach SESSION.csv --calibrate-from EXPORT_FOLDER

# tests (no network, ~55 s)
python -m pytest -q tests
```

In code:

```python
from paddling_expt.fusion import OC1Engine
from paddling_expt import analysis as an

from paddling_expt.boats import OC1_TYPICAL            # or SURFSKI_TYPICAL
from paddling_expt.boatspeed import stroke_physics
from paddling_expt.headunit import HeadConfig

engine = OC1Engine(head_cfg=HeadConfig(craft=OC1_TYPICAL, total_kg=93.0))
engine.feed_all(records)                   # records from any uplink, in arrival order
rows = engine.stroke_table()               # one row per stroke, both devices joined
stroke_physics(rows, OC1_TYPICAL, 93.0)    # power to the boat, ama load, ama / wobble cost
an.rate_speed_curve(rows)                  # best stroke rate
an.profile_change(rows, ref=(t0, t1), now=(t2, t3))
engine.panel()                             # live once-a-second summary
```

The hull-model tables are cached on disk (`$PADDLING_HYDRO_CACHE`, default
`~/.cache/paddling_hydro`): about 3 s to build per boat, 1 ms afterwards.
