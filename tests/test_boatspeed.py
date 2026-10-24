"""Roll -> ama load -> speed cost; side-slip; surf ski; power to the boat."""
import math

import numpy as np
import pytest

from paddling_expt.boatspeed import physics_summary, stroke_physics, wobble_cost_from_data
from paddling_expt.fusion import OC1Engine
from paddling_expt.headunit import HeadConfig
from paddling_expt.scoring import score_head, true_time
from paddling_expt.sim.paddler import PaddlerParams
from paddling_expt.sim.scenarios import steady_race


def _run(kind, seed, dur, **kw):
    s = steady_race(seed=seed, duration_s=dur, kind=kind, **kw)
    e = OC1Engine(head_cfg=HeadConfig(craft=s.craft, total_kg=s.total_kg)).feed_all(s.records)
    rows = e.stroke_table()
    stroke_physics(rows, s.craft, s.total_kg)
    return s, e, rows


def _truth_index(s, rows):
    tt = true_time(s.sensors, [r["t"] for r in rows])
    return np.array([int(np.argmin(np.abs(s.truth.strokes["catch_t"] - (x - 0.34)))) for x in tt])


@pytest.fixture(scope="module")
def oc1():
    return _run("oc1", 7, 480, effort=1.0, paddler=PaddlerParams(work_capacity_j=3.5e5))


def test_oc1_ama_load_from_roll_tracks_the_lean(oc1):
    s, e, rows = oc1
    R = [r for r in rows if r.get("ama_load_kg") is not None][40:]
    est = np.array([r["ama_load_kg"] for r in R])
    tru = s.truth.strokes["ama_load"][_truth_index(s, R)]
    k = 30
    roll_est = np.array([np.median(est[max(0, i - k + 1):i + 1]) for i in range(len(est))])[k:]
    roll_tru = np.convolve(tru, np.ones(k) / k, mode="full")[:len(tru)][k:]
    assert np.median(np.abs(roll_est - roll_tru)) < 2.5                  # kg, 30 strokes
    # the paddler leans onto the ama as they tire, and the estimate sees it
    assert (np.median(est[-100:]) - np.median(est[:100])) > 0.5 * (tru[-100:].mean() - tru[:100].mean())
    costs = [r["ama_cost_s500"] for r in R]
    assert np.median(costs[-100:]) > np.median(costs[:100]) >= 0


def test_boat_power_and_side_slip(oc1):
    s, e, rows = oc1
    R = [r for r in rows if r.get("boat_power_w") is not None][40:]
    idx = _truth_index(s, R)
    pw = np.array([r["boat_power_w"] for r in R])
    assert np.median(np.abs(pw / s.truth.strokes["boat_power"][idx] - 1)) < 0.06
    b = np.array([r["beta_rms_deg"] for r in R])
    tb = np.degrees(s.truth.strokes["beta_rms"][idx])
    assert np.corrcoef(b, tb)[0, 1] > 0.4
    summ = physics_summary(R)
    assert summ["speed_basis"] == "v_mean" and summ["yaw_cost_s500"] < 0.5


def test_surfski_alternating_sides_rate_and_wobble():
    s, e, rows = _run("surfski", 8, 300)
    sc = score_head(e.head, s.truth, s.sensors)
    assert sc["stroke_recall"] > 0.98 and sc["side_accuracy"] > 0.95
    assert sc["rate_err_spm_median"] < 1.5 and sc["speed_rmse"] < 0.06
    R = [r for r in rows if r.get("boat_power_w") is not None]
    assert all(r.get("ama_load_kg") is None for r in R)
    assert np.median([r["roll_wobble_deg"] for r in R]) > 0.4
    w = wobble_cost_from_data(R)
    assert "conclusive" in w and w["se"] > 0
