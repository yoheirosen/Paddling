"""Performance analyses: unit checks on planted rows, and one scored session."""
import math

import numpy as np
import pytest

from paddling_expt import analysis as an
from paddling_expt.fusion import OC1Engine
from paddling_expt.pod import StrainCalibration
from paddling_expt.scoring import score_pod, score_training
from paddling_expt.sim.paddler import PaddlerParams
from paddling_expt.sim.scenarios import training_session
from paddling_expt.sim.sensors import strain_calibration_counts


def _rows(n=600, seed=0, push=(0.15, -0.1), headings=(30.0, 210.0, 120.0)):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rate = 50 + 20 * rng.random()
        h = headings[(i // 100) % len(headings)] + rng.normal(0, 5)
        hr = math.radians(h)
        v = 3.0 - 0.004 * (rate - 60) ** 2 + push[0] * math.sin(hr) + push[1] * math.cos(hr)
        v += rng.normal(0, 0.03)
        rows.append({"t": float(i), "rate_spm": rate, "v_mean": v, "heading_deg": h % 360,
                     "dps": v * 60 / rate, "cycle_s": 60 / rate, "side": 1 if (i // 12) % 2 else -1})
    return rows


def test_rate_speed_curve_finds_the_best_rate():
    rows = _rows(push=(0.0, 0.0))
    fit = an.rate_speed_curve(rows)["fit"]
    assert abs(fit["best_rate"] - 60.0) < 1.0 and fit["best_inside_range"]


def test_heading_effect_recovers_the_push_and_corrects_speed():
    rows = _rows()
    he = an.heading_effect(rows, time_trend=False)
    assert he["heading_coverage"].startswith("two")
    assert abs(he["push_east"]["value"] - 0.15) < 0.02
    assert abs(he["push_north"]["value"] + 0.10) < 0.02
    fit = an.rate_speed_curve(rows, speed_key="v_corrected")["fit"]
    assert abs(fit["best_rate"] - 60.0) < 1.0


def test_heading_effect_on_one_line_reports_only_the_along_line_push():
    rows = _rows(headings=(0.0, 180.0))
    he = an.heading_effect(rows, time_trend=False)
    assert he["heading_coverage"].startswith("one line")
    assert abs(he["push_along_axis"]["value"]) == pytest.approx(0.10, abs=0.02)


def test_profile_change_detects_an_earlier_peak():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 41)

    def stroke(t, g):
        prof = np.sin(np.pi * x ** g) * (1 + rng.normal(0, 0.03))
        pk = 0.5 ** (1 / g)
        return {"t": t, "profile": prof.tolist(), "peak_pos": pk + rng.normal(0, 0.01),
                "impulse": prof.mean(), "peak": prof.max(), "drive_s": 0.5 + rng.normal(0, 0.01),
                "first_half_share": float(prof[x <= 0.5].sum() / prof.sum()),
                "side": 1 if t % 2 else -1}
    rows = [stroke(t, 1.0) for t in range(60)] + [stroke(t, 0.8) for t in range(60, 120)]
    g = an.profile_change(rows, (0, 59), (60, 119))["both"]
    assert g["peak_pos"]["change"] < -0.04 and g["peak_pos"]["z"] < -5
    assert "peak_pos" in g["flags"] and "first_half_share" in g["flags"]


def test_symmetry_and_splits():
    rows = _rows(push=(0.0, 0.0))
    for r in rows:
        r["impulse"] = 40.0 * (1.03 if r["side"] == 1 else 0.97)
    OC1Engine._side_runs(rows)
    sy = an.symmetry(rows)
    assert sy["impulse"]["rel"] == pytest.approx(0.06, abs=1e-6)
    assert sy["strokes_per_side_median"] == 12
    sp = an.splits(rows, every_m=500)
    assert sp and abs(sum(s["distance_m"] for s in sp) - sum(r["dps"] for r in rows)) < 1e-6


@pytest.fixture(scope="module")
def session():
    s = training_session(seed=5, piece_s=45, rest_s=40, steady_s=360,
                         paddler=PaddlerParams(work_capacity_j=3.0e5, rate_width=12.0))
    loads = [0, 50, 100, 150]
    cal = StrainCalibration.from_loads(loads, strain_calibration_counts(s.sensors, loads))
    e = OC1Engine(calibration=cal).feed_all(s.records)
    return s, e, e.stroke_table()


def test_session_current_best_rate_power_and_profile(session):
    s, e, rows = session
    sc = score_training(s, rows, settle=6)
    assert sc["current_leg_mean_err"] < 0.05                        # m/s along the course
    assert abs(sc["best_rate_water_fresh"] - sc["best_rate_true_fresh"]) < 2.5
    assert sc["pct_power_err_median"] < 10.0                        # percentage points
    assert sc["peak_pos_shift_true"] < 0
    assert sc["peak_pos_shift_est"] < 0 and sc["peak_pos_shift_z"] < -3
    p = score_pod(e.pod, s.truth, s.sensors)
    assert abs(p["asymmetry_est"] - p["asymmetry_true"]) < 0.03
