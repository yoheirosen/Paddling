"""The record format: validation, JSON lines round trip, arrival order."""
import numpy as np

from paddling_expt.fusion import OC1Engine
from paddling_expt.records import arrival_order, read_jsonl, validate, write_jsonl


def test_validate_flags_bad_records():
    assert validate({"dev": "head", "stream": "imu", "t": [0.0], "acc": [[0, 0, 9.8]],
                     "gyr": [[0, 0, 0]]}) is None
    assert "missing" in validate({"dev": "head", "stream": "gnss", "t": 1.0})
    assert "arrays" in validate({"dev": "head", "stream": "imu", "t": [0.0, 0.01],
                                 "acc": [[0, 0, 9.8]], "gyr": [[0, 0, 0]]})
    assert "lat and lon" in validate({"dev": "head", "stream": "gnss", "t": 1.0, "utc": 2.0,
                                      "fix": 3, "lat": None, "lon": None})
    assert "unknown" in validate({"dev": "boat", "stream": "x"})


def test_engine_reports_invalid_records_instead_of_guessing():
    e = OC1Engine()
    e.feed({"dev": "pod", "stream": "pkt", "rx": 1.0, "seq": 1, "t0": 0.0, "fs": 0.0,
            "strain": [1, 2]})
    assert e.invalid and "fs" in e.invalid[0]


def test_jsonl_round_trip_and_arrival_order(tmp_path):
    recs = [{"dev": "pod", "stream": "pkt", "rx": 2.0, "seq": 1, "t0": 0.0, "fs": 50.0,
             "strain": np.array([1, 2], dtype=np.int64), "acc": np.zeros((2, 3), np.float32),
             "gyr": np.zeros((2, 3), np.float32)},
            {"dev": "head", "stream": "gnss", "t": 1.0, "utc": 5.0, "fix": 0, "lat": None,
             "lon": None, "arrival": 1.03}]
    path = tmp_path / "s.jsonl"
    assert write_jsonl(recs, str(path)) == 2
    back = list(read_jsonl(str(path)))
    assert back[0]["strain"] == [1, 2] and back[1]["fix"] == 0
    assert [r["dev"] for r in arrival_order(back)] == ["head", "pod"]
