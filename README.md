# `paddling/` — outrigger-canoe (OC1) performance sensing (standalone sibling project)

Real-time measures for a solo outrigger canoe from two do-it-yourself devices — a
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

## Run

```bash
# simulated 37-min training session through both devices, scored against the truth
python "paddling/demos/demo_oc1_training.py" --figure oc1_sim_training

# report on a real SpeedCoach export (keep the raw file out of git: it holds a GPS track)
python "paddling/demos/speedcoach_report.py" SESSION.csv --figure oc1_speedcoach_session

# tests (no network, ~25 s)
python -m pytest -q paddling/tests
```

In code:

```python
from paddling_expt.fusion import OC1Engine
from paddling_expt import analysis as an

engine = OC1Engine().feed_all(records)     # records from any uplink, in arrival order
rows = engine.stroke_table()               # one row per stroke, both devices joined
an.rate_speed_curve(rows)                  # best stroke rate
an.profile_change(rows, ref=(t0, t1), now=(t2, t3))
engine.panel()                             # live once-a-second summary
```
