"""The hull-shape resistance model: magnitudes, consistency, statics."""
import math

import numpy as np
import pytest

from paddling_expt.boats import OC1_TYPICAL, SURFSKI_TYPICAL
from paddling_expt.geo import G
from paddling_expt.hydro import (RHO_SEA, HullGrid, ama_load_from_roll, friction_resistance,
                                 hull_at_load, michell_resistance, resistance_table, roll_table,
                                 yaw_drag)


def _wigley(L=10.0, B=1.0, T=0.625, nx=301, nz=30):
    x = np.linspace(0, L, nx)
    z = -T * np.linspace(0, 1, nz) ** 1.5
    xi = 2 * x / L - 1
    Y = (B / 2) * (1 - xi[:, None] ** 2) * (1 - (z[None, :] / T) ** 2)
    return HullGrid(x, z, Y, 4 / 9 * L * B * T, 0.1484 * L * L, T, L, B)


def test_michell_wigley_hull_in_the_published_range():
    """Thin-ship wave resistance of the standard Wigley test hull: hump near
    Froude 0.3, hollow near 0.35, peak of a few times 1e-3 near 0.5."""
    g = _wigley()
    cw = {}
    for fn in (0.3, 0.35, 0.5):
        U = fn * math.sqrt(G * 10.0)
        cw[fn] = michell_resistance(U, [(g, 0, 0)]) / (0.5 * RHO_SEA * U * U * 14.84)
    assert cw[0.35] < cw[0.3] < cw[0.5]
    assert 1.5e-3 < cw[0.5] < 6e-3


def test_two_far_apart_hulls_approach_twice_one():
    g = hull_at_load(SURFSKI_TYPICAL.hull, 89.0)
    U = 3.3
    one = michell_resistance(U, [(g, 0, 0)])
    two = michell_resistance(U, [(g, 0, -25.0), (g, 0, 25.0)])
    assert two == pytest.approx(2 * one, rel=0.08)


def test_hydrostatics_and_friction():
    g = hull_at_load(OC1_TYPICAL.hull, 88.0)
    assert g.volume_m3 * RHO_SEA == pytest.approx(88.0, rel=0.01)
    assert 1.4 < g.wetted_m2 < 2.4 and 0.07 < g.draft_m < 0.15
    re = 3.3 * g.length_m / 1.19e-6
    cf = 0.075 / (math.log10(re) - 2) ** 2
    assert friction_resistance(3.3, g) > 0.5 * RHO_SEA * 3.3 ** 2 * g.wetted_m2 * cf


def test_race_pace_power_is_plausible_and_the_ama_costs():
    ski = SURFSKI_TYPICAL.resistance(3.3, 89.0)
    assert 30 < ski["total"] < 55                       # N at 11.9 km/h
    assert 100 < ski["total"] * 3.3 < 180               # W to the boat
    light = OC1_TYPICAL.resistance(3.3, 93.0, ama_load_kg=3.0)["total"]
    heavy = OC1_TYPICAL.resistance(3.3, 93.0, ama_load_kg=15.0)["total"]
    assert heavy > light * 1.08


def test_roll_and_ama_load_are_inverse():
    rt = roll_table(OC1_TYPICAL, 93.0)
    for L in (2.0, 6.0, 12.0, 20.0):
        roll = OC1_TYPICAL.roll_for_ama_load(93.0, L)
        assert float(ama_load_from_roll(rt, roll)) == pytest.approx(L, abs=0.2)
    # rolling towards the ama loads it
    assert OC1_TYPICAL.roll_for_ama_load(93.0, 15.0) < OC1_TYPICAL.roll_for_ama_load(93.0, 3.0)


def test_yaw_drag_is_slender_body_and_small_for_small_slip():
    d = yaw_drag(3.3, math.radians(1.0), 0.10)
    assert d == pytest.approx(0.5 * math.pi * RHO_SEA * 3.3 ** 2 * 0.01 * math.radians(1.0) ** 2)
    assert d < 0.1


def test_resistance_table_lookup_matches_direct():
    tab = resistance_table(OC1_TYPICAL, 93.0)
    j = list(tab["ama_loads"]).index(6.0)
    r = OC1_TYPICAL.resistance(float(tab["speeds"][12]), 93.0, ama_load_kg=6.0)
    assert tab["R"][12, j] == pytest.approx(r["hull_friction"] + r["ama_friction"] + r["waves"],
                                           rel=1e-9)
