# `paddling/` — outrigger-canoe (OC1) and surf ski performance sensing (standalone sibling project)

Real-time measures for a solo outrigger canoe or a surf ski from two do-it-yourself devices — a
boat-mounted **head unit** (GNSS + motion sensor) and a **paddle pod** (strain gauge
+ motion sensor) — plus a reader for **NK SpeedCoach** exports. Built and scored on a
simulator with planted answers; checked on a real SpeedCoach session. It reuses
nothing from the ephys, cardiac, calcium or ICU trees (`agent_lib` only for figure
artifacts). Design, results and limits: **`PADDLING_PLAN.md`**.

Per stroke: stroke rate, speed, **check** (speed lost between strokes), distance per
stroke, side, and with the pod the **pull-force profile** (impulse, peak, where the
peak falls in the drive, front-loading, drive time). Per session: the stroke-rate vs
speed curve and best rate, **% of full-effort power at a given rate**, changes in the
pull-force profile, fatigue and pacing trends, left/right symmetry, splits, legs,
speed through the water and the current or wind along the course, downwind runs.
Across sessions: best efforts from 1 to 40 minutes, critical power and predicted race
times, the best speed seen at each stroke rate and each session's share of it, and
whether going up in rate bought speed (`sessions.py`).

Where speed and effort interact it uses a **hull-shape model** (`hydro.py`): skin
friction, wave-making (Michell's thin-ship integral, hull and ama together) and air
drag, for an OC1 with its ama or a surf ski (`boats.py`: typical dimensions until
your boat is measured). From the head unit's roll it reads **how much weight sits on
the ama** and what that costs in seconds per 500 m; for the surf ski it reads **bow
wobble** (side-slip, roll and yaw wobble) and its cost (`boatspeed.py`).

## Run

```bash
# simulated 37-min training session through both devices, scored against the truth
python "paddling/demos/demo_oc1_training.py" --figure oc1_sim_training

# hull-shape physics: resistance, ama load from roll and its cost, surf ski wobble
python "paddling/demos/demo_boat_physics.py" --figure oc1_boat_physics

# report on a real SpeedCoach export (keep the raw file out of git: it holds a GPS track)
python "paddling/demos/speedcoach_report.py" SESSION.csv --boat oc1 --total-kg 93 \
    --figure oc1_speedcoach_session

# across several SpeedCoach sessions: best efforts, critical power and predicted race
# times, best speed at each stroke rate, % of best, what going up in rate gave
python "paddling/demos/speedcoach_sessions.py" EXPORT_FOLDER --figure oc1_speedcoach_sessions

# tests (no network, ~50 s)
python -m pytest -q paddling/tests
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
