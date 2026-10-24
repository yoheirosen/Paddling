"""The record format both devices speak, independent of how records travel.

A *record* is a flat dict.  Whatever carries it — a phone relaying Bluetooth, shore
Wi-Fi, a file uploaded after the session — the pipeline only ever sees records, fed
one at a time in the order they arrived (:meth:`OC1Engine.feed`).  So the uplink can
be chosen later without touching the analysis.

Every record has ``dev`` (``"head"`` or ``"pod"``) and ``stream``.  Times are in the
**sending device's own clock**, in seconds; only GNSS records carry UTC.  An optional
``arrival`` (head-unit clock, set by whatever relays the record) gives the order to
replay a log in.  Arrays may be lists (as in JSON) or numpy arrays.

==========  ==========  =====================================================
dev         stream      fields
==========  ==========  =====================================================
head        meta        t, imu_hz, gnss_hz, acc_range (m/s^2), gyr_range (rad/s)
head        imu         t[n] (head clock), acc[n][3] (m/s^2), gyr[n][3] (rad/s),
                        in the box's own axes
head        gnss        t (head clock at the fix), utc (s), fix (0 = none, 2/3),
                        lat, lon (deg), sog (m/s), cog (deg true), optional
                        vel_e / vel_n (m/s), nsat, hdop — position fields are
                        null when fix == 0
head        house       t, vbat (V), charging, temp_c, state
head        button      t, button, action
head        level       t, up[3] — gravity direction in the box's axes captured with
                        the boat level on shore (a button press); the reference
                        for absolute roll and pitch
head        link        t, state ("up" / "down") — the pod's Bluetooth link
pod         meta        rx (head clock), fs (Hz), fw
pod         pkt         rx (head clock at receipt), seq (16-bit, wraps), t0 (pod
                        clock of the first sample), fs (Hz), strain[n] (raw
                        counts), acc[n][3] (m/s^2), gyr[n][3] (rad/s), pod axes
==========  ==========  =====================================================

The schema is the interface contract with the hardware: a record that fails
:func:`validate` is reported, never silently dropped or guessed at.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

__all__ = ["REQUIRED", "validate", "to_jsonable", "write_jsonl", "read_jsonl",
           "arrival_order"]

REQUIRED: Dict[tuple, tuple] = {
    ("head", "meta"): ("t",),
    ("head", "imu"): ("t", "acc", "gyr"),
    ("head", "gnss"): ("t", "utc", "fix"),
    ("head", "house"): ("t",),
    ("head", "button"): ("t", "button"),
    ("head", "level"): ("t", "up"),
    ("head", "link"): ("t", "state"),
    ("pod", "meta"): ("fs",),
    ("pod", "pkt"): ("rx", "seq", "t0", "fs", "strain"),
}


def validate(rec: Dict[str, Any]) -> Optional[str]:
    """``None`` if the record is well formed, else what is wrong with it."""
    key = (rec.get("dev"), rec.get("stream"))
    need = REQUIRED.get(key)
    if need is None:
        return f"unknown record type {key}"
    missing = [f for f in need if f not in rec]
    if missing:
        return f"{key[0]}/{key[1]} record missing {missing}"
    if key == ("head", "imu"):
        n = len(rec["t"])
        if np.shape(rec["acc"]) != (n, 3) or np.shape(rec["gyr"]) != (n, 3):
            return "head/imu arrays must be t[n], acc[n][3], gyr[n][3]"
    if key == ("pod", "pkt"):
        n = len(rec["strain"])
        for f in ("acc", "gyr"):
            if f in rec and rec[f] is not None and np.shape(rec[f]) != (n, 3):
                return f"pod/pkt {f} must be [n][3] with n = len(strain)"
        if not rec["fs"] or float(rec["fs"]) <= 0:
            return "pod/pkt fs must be positive"
    if key == ("head", "gnss") and rec.get("fix"):
        if rec.get("lat") is None or rec.get("lon") is None:
            return "head/gnss with a fix must carry lat and lon"
    return None


def to_jsonable(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(to_jsonable(r), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _arrival(rec: Dict[str, Any]) -> float:
    for k in ("arrival", "rx"):
        if rec.get(k) is not None:
            return float(rec[k])
    t = rec.get("t")
    if isinstance(t, (list, tuple, np.ndarray)):
        return float(t[-1])
    return float(t) if t is not None else 0.0


def arrival_order(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort a batch (e.g. a log uploaded after the session) into arrival order."""
    return sorted(records, key=_arrival)
