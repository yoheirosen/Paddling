"""Frames, rotations and local coordinates shared by the simulator and the pipeline.

Conventions (used everywhere in this package):

* **World frame** is local East-North-Up (ENU) metres around an origin
  latitude/longitude.  Angles in the world frame are *math* angles (radians,
  0 = east, counter-clockwise positive); GNSS course over ground is reported as a
  *compass* bearing in degrees (0 = north, clockwise) and converted at the edge.
* **Hull frame**: x forward (bow), y left (towards the ama — the outrigger float —
  of a standard left-ama OC1), z up.  Hull attitude is ZYX Euler angles
  ``R_hull_to_world = Rz(heading) @ Ry(pitch) @ Rx(roll)``; with this convention a
  positive pitch is **nose down** and a positive roll lifts the left side.
* **Specific force** (what an accelerometer reads) is ``R^T (a_world + g_up)``:
  at rest it reads +9.81 m/s^2 along the frame's up direction.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

G = 9.80665
EARTH_R = 6_371_008.8

__all__ = ["G", "rx", "ry", "rz", "euler_zyx", "body_rates_zyx", "random_rotation",
           "rotation_angle_deg", "enu_to_latlon", "latlon_to_enu", "compass_to_math",
           "math_to_compass", "wrap_pi"]


def rx(a):
    """Rotation about x by angle(s) ``a`` (radians); shape (..., 3, 3)."""
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1),
                     np.stack([z, s, c], -1)], -2)


def ry(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def rz(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def euler_zyx(yaw, pitch, roll):
    """``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (vectorised over leading axes)."""
    return rz(yaw) @ ry(pitch) @ rx(roll)


def body_rates_zyx(yaw_rate, pitch_rate, roll_rate, pitch, roll):
    """Body-frame angular velocity (p, q, r) for ZYX Euler angle rates."""
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)
    p = roll_rate - yaw_rate * sp
    q = pitch_rate * cr + yaw_rate * sr * cp
    r = -pitch_rate * sr + yaw_rate * cr * cp
    return np.stack([p, q, r], -1)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (degrees) of the rotation taking ``b`` to ``a``."""
    m = a @ b.T
    c = (np.trace(m) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def enu_to_latlon(x, y, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Local east/north metres to latitude/longitude (equirectangular; fine over a
    race course of a few tens of kilometres)."""
    lat = lat0 + np.degrees(np.asarray(y, float) / EARTH_R)
    lon = lon0 + np.degrees(np.asarray(x, float) / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def latlon_to_enu(lat, lon, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    y = np.radians(np.asarray(lat, float) - lat0) * EARTH_R
    x = np.radians(np.asarray(lon, float) - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return x, y


def compass_to_math(deg):
    """Compass bearing (degrees, 0 = north, clockwise) to math angle (radians)."""
    return np.radians(90.0 - np.asarray(deg, float))


def math_to_compass(rad):
    return np.mod(90.0 - np.degrees(np.asarray(rad, float)), 360.0)


def wrap_pi(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi
