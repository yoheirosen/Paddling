"""Read an NK SpeedCoach session export (the CSV from NK LiNK Logbook).

The export has four blocks — ``Session Information``, ``Session Summary``,
``Interval Summaries`` and ``Per-Stroke Data`` — each a header row, a units row,
then data.  Missing values are ``---``.  This reader returns the per-stroke rows in
SI units, with the same keys as :meth:`.fusion.OC1Engine.stroke_table` where the
quantity is the same (``t``, ``rate_spm``, ``v_mean``, ``dps``), so every analysis
in :mod:`.analysis` runs on SpeedCoach data unchanged.

What a SpeedCoach records per stroke: elapsed time, GPS distance and speed,
stroke rate, distance per stroke, position, and — only with the matching
accessories — impeller (water) speed, heart rate and oarlock power/angles.  It
does not record the within-stroke speed, so the check cannot be measured from it.

Units: the export follows the device's unit setting (miles or km, feet or
metres); the units row is read and every value converted.  The start time is
the device's local clock with no time zone.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

__all__ = ["read_speedcoach", "parse_speedcoach", "add_headings"]

_DIST = {"(miles)": 1609.344, "(km)": 1000.0, "(kilometers)": 1000.0, "(meters)": 1.0,
         "(m)": 1.0, "(feet)": 0.3048, "(ft)": 0.3048}
_SPEED = {"(mph)": 0.44704, "(km/h)": 1 / 3.6, "(kph)": 1 / 3.6, "(m/s)": 1.0}


def _num(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x or x.startswith("---"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _hms(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x or x.startswith("---"):
        return None
    parts = x.split(":")
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    except ValueError:
        return None


def _block(lines: List[List[str]], title: str) -> Optional[tuple]:
    """(header, units, data rows) of the block whose first cell is ``title``."""
    for i, row in enumerate(lines):
        if row and row[0].strip() == title:
            j = i + 1
            while j < len(lines) and not any(c.strip() for c in lines[j]):
                j += 1
            if j + 1 >= len(lines):
                return None
            header, units = lines[j], lines[j + 1]
            data = []
            for row2 in lines[j + 2:]:
                if not any(c.strip() for c in row2):
                    break
                data.append(row2)
            return header, units, data
    return None


def parse_speedcoach(text: str) -> Dict[str, Any]:
    lines = list(csv.reader(io.StringIO(text)))
    info: Dict[str, str] = {}
    for row in lines[:12]:
        for k in range(0, len(row) - 1, 4):
            key = row[k].strip().rstrip(":")
            if key and row[k + 1].strip():
                info[key] = row[k + 1].strip()
    blk = _block(lines, "Per-Stroke Data:")
    if blk is None:
        raise ValueError("no 'Per-Stroke Data:' block — not a SpeedCoach session export")
    header, units, data = blk
    col = {h.strip(): i for i, h in enumerate(header)}
    unit = {h.strip(): (units[i].strip().lower() if i < len(units) else "")
            for i, h in enumerate(header)}

    def get(row, name, kind=None):
        i = col.get(name)
        if i is None or i >= len(row):
            return None
        if kind == "time":
            return _hms(row[i])
        v = _num(row[i])
        if v is None:
            return None
        u = unit.get(name, "")
        if kind == "dist":
            return v * _DIST.get(u, 1.0)
        if kind == "speed":
            return v * _SPEED.get(u, 1.0)
        return v

    start = None
    if info.get("Start Time"):
        for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                start = datetime.strptime(info["Start Time"], fmt)
                break
            except ValueError:
                pass
    strokes: List[Dict[str, Any]] = []
    prev_t = None
    for row in data:
        t = get(row, "Elapsed Time", "time")
        if t is None:
            continue
        r = {"t": t, "interval": get(row, "Interval"),
             "distance_m": get(row, "Distance (GPS)", "dist"),
             "v_mean": get(row, "Speed (GPS)", "speed"),
             "rate_spm": get(row, "Stroke Rate"),
             "stroke": get(row, "Total Strokes"),
             "dps": get(row, "Distance/Stroke (GPS)", "dist"),
             "lat": get(row, "GPS Lat."), "lon": get(row, "GPS Lon."),
             "hr": get(row, "Heart Rate"), "power_w": get(row, "Power"),
             "water_speed_imp": get(row, "Speed (IMP)", "speed")}
        if r["water_speed_imp"] == 0.0:
            r["water_speed_imp"] = None       # no impeller fitted
        r["cycle_s"] = (t - prev_t) if prev_t is not None else None
        prev_t = t
        strokes.append(r)
    return {"info": info, "start_local": start, "strokes": strokes,
            "has": {k: any(s.get(k) is not None for s in strokes)
                    for k in ("hr", "power_w", "water_speed_imp", "lat")}}


def read_speedcoach(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return parse_speedcoach(fh.read())


def add_headings(strokes: List[Dict[str, Any]], half_window: int = 4) -> List[Dict[str, Any]]:
    """Direction of travel (compass degrees) at each stroke, from the positions
    ``half_window`` strokes before and after (about +/- 4 s)."""
    pts = [(s.get("lat"), s.get("lon")) for s in strokes]
    for i, s in enumerate(strokes):
        a, b = max(0, i - half_window), min(len(strokes) - 1, i + half_window)
        (la0, lo0), (la1, lo1) = pts[a], pts[b]
        if None in (la0, lo0, la1, lo1):
            continue
        dy = math.radians(la1 - la0) * 6_371_008.8
        dx = math.radians(lo1 - lo0) * 6_371_008.8 * math.cos(math.radians(la0))
        if math.hypot(dx, dy) > 3.0:
            s["heading_deg"] = math.degrees(math.atan2(dx, dy)) % 360.0
    return strokes
