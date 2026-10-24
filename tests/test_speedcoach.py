"""Reading an NK SpeedCoach session export (synthetic file in the same layout)."""
import math

import pytest

from paddling_expt import analysis as an
from paddling_expt.speedcoach import add_headings, parse_speedcoach

HEAD = "Interval,Distance (GPS),Distance (IMP),Elapsed Time,Split (GPS),Speed (GPS),Split (IMP),Speed (IMP),Stroke Rate,Total Strokes,Distance/Stroke (GPS),Distance/Stroke (IMP),Heart Rate,Power,Catch,Slip,Finish,Wash,Force Avg,Work,Force Max,Max Force Angle,GPS Lat.,GPS Lon."
UNITS = "(Interval),(Miles),(Miles),(HH:MM:SS.tenths),(/Mile),(MPH),(/Mile),(MPH),(SPM),(Strokes),(Feet),(Feet),(BPM),(Watts),(Degrees),(Degrees),(Degrees),(Degrees),(Newtons),(Joules),(Newtons),(Degrees),(Degrees),(Degrees)"


def _export(n=80):
    lines = ["Session Information:,,,,Device Information:,,,,Oarlock Information:,,,,Oarlock Settings:",
             "", "Name:,Test,,,Name:,SpdCoach 1,,,Firmware:,---,,,Boat ID:,---",
             "Start Time:,10/17/2026 11:00:38,,,Model:,SpeedCoach GPS Pro,,,,,,,Seat Number:,---",
             "", "Per-Stroke Data:", "", HEAD, UNITS]
    lat, lon, dist = 36.95, -122.0, 0.0
    for i in range(n):
        mph = 6.0
        dps_ft = 9.8
        dist += dps_ft * 0.3048 / 1609.344
        lat += 3.0 / 6_371_008.8 * 180 / math.pi          # 3 m north per stroke
        t = 1.0 + i * 1.05
        el = f"00:{int(t // 60):02d}:{t % 60:04.1f}"
        hr = "---" if i % 2 else "150"
        lines.append(f"1,{dist:.3f},0.000,{el},00:10:00.0,{mph:.3f},00:00:00.0,0.000,57.0,{i + 1},"
                     f"{dps_ft},0.0,{hr},---,---,---,---,---,---,---,---,---,{lat:.7f},{lon:.7f}")
    return "\n".join(lines) + "\n"


def test_parse_converts_units_and_missing_values():
    d = parse_speedcoach(_export())
    s = d["strokes"]
    assert len(s) == 80 and d["info"]["Model"] == "SpeedCoach GPS Pro"
    assert s[0]["v_mean"] == pytest.approx(6.0 * 0.44704)
    assert s[0]["dps"] == pytest.approx(9.8 * 0.3048)
    assert s[0]["rate_spm"] == 57.0 and s[0]["t"] == pytest.approx(1.0)
    assert s[0]["water_speed_imp"] is None                 # 0.000 = no impeller fitted
    assert s[0]["hr"] == 150 and s[1]["hr"] is None        # "---" = missing
    assert d["has"]["hr"] and not d["has"]["power_w"]
    assert d["start_local"].hour == 11


def test_headings_and_analyses_run_on_speedcoach_rows():
    s = add_headings(parse_speedcoach(_export())["strokes"])
    hs = [r["heading_deg"] for r in s if r.get("heading_deg") is not None]
    assert hs and all(min(h, 360 - h) < 2.0 for h in hs)    # due north
    sp = an.splits(s, every_m=50)
    assert sp and sp[0]["pace_500m_s"] > 0


def test_not_a_session_export():
    with pytest.raises(ValueError):
        parse_speedcoach("hello,world\n")
