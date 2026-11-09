"""The personal heart-rate model on simulated paddlers whose heart-rate shape is planted."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from paddling_expt.heart_rate import fit_personal_model, textbook_zones, zones   # noqa: E402
from paddling_expt.sim.twin import draw_traits, heart_rate_profile, twin_step_test   # noqa: E402

import heart_rate_model as hrm                                                    # noqa: E402


def _seed(shape):
    return next(s for s in range(1, 200) if heart_rate_profile(s).get("shape") == shape)


def test_typical_response_is_read_back_from_one_step_test():
    r = hrm.one(_seed("typical"))
    assert r["shape"] == "typical" and r["model"] is not None
    assert r["model"]["heart_rate_separates_efforts_up_to_share_of_cp"] >= 1.0
    for p in r["points"]:                       # held-out steady efforts on another day
        assert p["personal_zone"] == p["true_zone"]
        assert abs(p["personal_effort_error_pct"]) < 8.0


def test_compressed_response_says_where_heart_rate_cannot_tell():
    r = hrm.one(_seed("compressed"))
    m = r["model"]
    assert r["shape"] == "compressed" and m is not None
    assert m["heart_rate_separates_efforts_up_to_share_of_cp"] < 0.75
    assert any("compressed" in n for n in m["notes"])
    hard = [p for p in r["points"] if p["share_of_cp"] >= 0.84]
    assert hard and all(p["personal_zone"] is None for p in hard)    # it does not guess
    assert all(p["personal_zone"] in (None, p["true_zone"]) for p in r["points"])
    assert all(p["textbook_zone"] != p["true_zone"] for p in hard)


def test_zones_flag_the_ones_heart_rate_cannot_hold():
    seed = _seed("compressed")
    tr = draw_traits(seed)
    model = fit_personal_model(twin_step_test(tr, seed=seed)[0])
    z = zones(model, tr.paddler.critical_power_w)
    assert [d["zone"] for d in z][0] == "recovery"
    assert not all(d["heart_rate_can_hold_it"] for d in z)
    assert len(textbook_zones(180.0)) == 5


def test_too_little_data_gives_no_model():
    assert fit_personal_model([{"t": float(i), "shaft_power_w": 100.0, "heart_rate_bpm": 120.0}
                               for i in range(20)]) is None


def test_effort_at_inverts_hr_at():
    seed = _seed("typical")
    tr = draw_traits(seed)
    model = fit_personal_model(twin_step_test(tr, seed=seed)[0])
    cp = tr.paddler.critical_power_w
    for share in (0.6, 0.8, 0.95):
        assert model.effort_at(model.hr_at(share * cp)) == pytest.approx(share * cp, rel=1e-6)


def test_read_names_a_zone_or_says_heart_rate_cannot_tell():
    seed = _seed("compressed")
    tr = draw_traits(seed)
    model = fit_personal_model(twin_step_test(tr, seed=seed)[0])
    cp = tr.paddler.critical_power_w
    easy = model.read(model.hr_at(0.45 * cp), cp)
    assert easy["zone"] == "recovery" and abs(easy["share_of_cp"] - 0.45) < 0.02
    hard = model.read(model.hr_at(0.95 * cp), cp)
    assert hard["zone"] is None and "cannot tell" in hard["note"]
