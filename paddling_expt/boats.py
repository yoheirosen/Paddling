"""Typical craft, as starting points until a paddler's own boat is measured.

Dimensions are waterline values taken from published overall specs (OC1 race
hulls about 6.2-6.7 m long and 0.38-0.42 m wide at the gunwale, about 11 kg in
carbon; surf skis such as the Epic V10 Gen 4 at 6.1 m x 0.46 m and the V12 at
6.2 m x 0.43 m, about 11 kg).  Waterline beam is narrower than the overall beam,
and no manufacturer publishes an ama standard, so the ama here is a typical 3.8 m
float.  Replace every number with measurements of the actual boat: length and
beam at the waterline, the widest point, the section shape, and for an OC1 the
iako span and the ama's rigged height.
"""
from __future__ import annotations

from .hydro import Craft, HullForm, Outrigger

__all__ = ["OC1_TYPICAL", "SURFSKI_TYPICAL", "CRAFT"]

OC1_TYPICAL = Craft(
    name="OC1 (typical race hull, 3.8 m ama)",
    hull=HullForm("oc1 hull", length_m=6.30, beam_m=0.34, draft_m=0.10,
                  max_beam_at=0.56, entry_exp=1.7, run_exp=2.1, rocker_exp=2.6,
                  section_exp=2.0),
    hull_mass_kg=11.0,
    outrigger=Outrigger(
        ama=HullForm("oc1 ama", length_m=3.80, beam_m=0.15, draft_m=0.09,
                     max_beam_at=0.45, entry_exp=1.8, run_exp=1.8, rocker_exp=2.2,
                     section_exp=1.5),
        span_m=1.80, rig_height_m=0.06, x_m=0.0, mass_kg=4.0, side=1),
    cda_m2=0.42)

SURFSKI_TYPICAL = Craft(
    name="Surf ski (typical intermediate, 6.1 m)",
    hull=HullForm("surfski hull", length_m=5.90, beam_m=0.40, draft_m=0.10,
                  max_beam_at=0.55, entry_exp=1.8, run_exp=2.2, rocker_exp=2.8,
                  section_exp=2.0),
    hull_mass_kg=11.0, outrigger=None, cda_m2=0.40)

CRAFT = {"oc1": OC1_TYPICAL, "surfski": SURFSKI_TYPICAL}
