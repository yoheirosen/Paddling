"""Head unit and paddle pod processing against the simulator's planted truth."""
import numpy as np
import pytest

from paddling_expt.fusion import OC1Engine
from paddling_expt.pod import StrainCalibration
from paddling_expt.records import read_jsonl, write_jsonl
from paddling_expt.scoring import score_head, score_pod
from paddling_expt.sim.scenarios import downwind_race, steady_race
from paddling_expt.sim.sensors import PodConfig, strain_calibration_counts


@pytest.fixture(scope="module")
def flat():
    s = steady_race(seed=11, duration_s=180, current_mps=0.2)
    loads = [0, 50, 100, 150]
    cal = StrainCalibration.from_loads(loads, strain_calibration_counts(s.sensors, loads))
    e = OC1Engine(calibration=cal).feed_all(s.records)
    return s, e


def test_head_unit_strokes_rate_side_speed_check(flat):
    s, e = flat
    sc = score_head(e.head, s.truth, s.sensors)
    assert sc["stroke_recall"] > 0.98 and sc["stroke_precision"] > 0.98
    assert sc["rate_err_spm_median"] < 0.8
    assert sc["side_accuracy"] > 0.95
    assert sc["speed_rmse"] < 0.06                       # m/s, fused IMU + GNSS
    assert sc["check_ratio_err_median"] < 0.02           # speed lost between strokes
    assert sc["dps_err_m_median"] < 0.06
    assert sc["mount_err_deg"] < 6.0                     # unknown box rotation recovered
    assert e.head.runs == []                             # a hard start is not a surf run


def test_pod_timing_force_side_balance(flat):
    s, e = flat
    sc = score_pod(e.pod, s.truth, s.sensors)
    assert sc["clock_err_ms_median"] < 6.0
    assert sc["stroke_recall"] > 0.85 and sc["stroke_precision"] > 0.97
    assert sc["catch_err_ms_median"] < 10 and sc["exit_err_ms_median"] < 10
    assert sc["impulse_corr"] > 0.98
    assert 1.0 < sc["impulse_ratio_median"] < 1.15       # blade-normal vs propulsive impulse
    assert sc["side_accuracy"] > 0.97
    assert abs(sc["asymmetry_est"] - sc["asymmetry_true"]) < 0.03


def test_stroke_table_joins_devices(flat):
    s, e = flat
    rows = e.stroke_table()
    with_pod = [r for r in rows if r.get("impulse") is not None]
    assert len(with_pod) > 0.8 * len(rows)
    lag = e.catch_to_surge_s()
    assert 0.2 < lag < 0.5
    r = with_pod[len(with_pod) // 2]
    for k in ("rate_spm", "v_mean", "check_ratio", "dps", "peak_pos", "profile",
              "mean_force", "power_proxy", "heading_deg", "stroke_on_side"):
        assert r.get(k) is not None, k
    p = e.panel()
    assert p["rate_spm"] and p["speed"] > 2 and p["pod_connected"]


def test_head_unit_alone_and_log_replay(tmp_path):
    s = steady_race(seed=12, duration_s=120, pod=PodConfig(present=False))
    assert not any(r["dev"] == "pod" for r in s.records)
    path = tmp_path / "log.jsonl"
    write_jsonl(s.records, str(path))
    e = OC1Engine().feed_all(read_jsonl(str(path)))       # same path as a live relay
    rows = e.stroke_table()
    assert len(rows) > 100 and all(r.get("impulse") is None for r in rows)
    assert score_head(e.head, s.truth, s.sensors)["stroke_recall"] > 0.98


def test_downwind_speed_and_runs():
    s = downwind_race(seed=3, duration_s=300)
    e = OC1Engine()
    for r in s.records:
        if r["dev"] == "head":
            e.feed(r)
    e.finish()
    sc = score_head(e.head, s.truth, s.sensors)
    assert sc["speed_rmse"] < 0.1                        # swell + wave slap
    assert sc["stroke_recall"] > 0.97
    assert sc["runs_recall"] >= 0.6 and sc["runs_precision"] >= 0.7
