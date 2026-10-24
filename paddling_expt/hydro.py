"""How a slender paddle craft moves through the water: resistance from hull shape.

Used by the simulator (as its planted physics) and by the analyses (to turn
measured speed, roll and yaw into power and into the speed each loss costs).
Every piece is a standard approximation for very slender hulls; the numbers it
gives are only as good as the hull shape fed to it, so measure your boat.

* **Hull shape** (:class:`HullForm`): waterline length, beam and draft, where the
  widest point sits, how fine the entry and run are, how much keel rocker, and the
  section shape (V to rounded to wall-sided).  Sections flare: sinking deeper
  widens the waterline.  Or supply measured offsets directly.
* **Hydrostatics**: how deep the hull sits for a given load, its wetted surface.
* **Skin friction**: ITTC-1957 friction line, ``C_F = 0.075 / (log10 Re - 2)^2``,
  times a streamlined-body form factor ``1 + 2 t/L + 60 (t/L)^4`` (Hoerner).
* **Wave-making**: Michell's thin-ship integral (1898), the standard theory for
  hulls as slender as canoes, kayaks and rowing shells.  For an outrigger the hull
  and the ama are summed in the far field with their lateral and fore-aft spacing,
  so their wave interference is included.
* **Air drag** of paddler and boat: ``0.5 rho_air CdA (boat speed + head wind)^2``.
* **Yaw drag**: a hull moving at a drift angle beta to its own centreline makes a
  side force and an induced drag; slender-body (low aspect ratio wing) theory with
  the free surface as a mirror gives ``0.5 * pi * rho * U^2 * T^2 * beta^2``.  This
  is what bow wobble and one-sided-stroke yaw cost.

Outrigger statics (:class:`Outrigger`): the ama hangs at ``span`` from the hull's
centreline at a rigged height; the boat's roll angle fixes how deep the ama sits,
hence how much weight it carries, hence its drag.  Measured roll (against a level
reference) therefore gives the ama load and what it costs.
"""
from __future__ import annotations

import functools
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .geo import G

__all__ = ["RHO_SEA", "NU_SEA", "RHO_AIR", "HullForm", "HullGrid", "michell_resistance",
           "friction_resistance", "yaw_drag", "air_drag", "Outrigger", "Craft",
           "resistance_table", "water_resistance", "roll_table", "ama_load_from_roll",
           "speed_for_power", "sway_time_constant", "hull_at_load"]

RHO_SEA = 1025.0      # kg/m^3
NU_SEA = 1.19e-6      # m^2/s, sea water ~15 C
RHO_AIR = 1.225
_MODEL_VERSION = 1          # bump when the physics changes (invalidates cached tables)


@dataclass(frozen=True)
class HullForm:
    """A slender hull from principal dimensions (all at the design waterline).

    ``section_exp`` q sets the section: half-breadth grows as (height above the
    keel)^(1/q) — q = 1 a V, q = 2 a rounded V/U, large q wall-sided.  The keel
    rises towards the ends as ``1 - |2 xi - 1|^rocker_exp`` (higher = flatter middle).
    """

    name: str
    length_m: float
    beam_m: float
    draft_m: float
    max_beam_at: float = 0.55       # share of length from the bow
    entry_exp: float = 1.8          # waterline fullness forward (higher = fuller)
    run_exp: float = 2.2            # and aft
    rocker_exp: float = 3.0
    section_exp: float = 2.0

    def stations(self, nx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """x from the bow (m), design waterline half-beam and design draft per station."""
        xi = np.linspace(0.0, 1.0, nx)
        m = self.max_beam_at
        g = np.where(xi <= m, 1 - (np.clip(m - xi, 0, None) / m) ** self.entry_exp,
                     1 - (np.clip(xi - m, 0, None) / (1 - m)) ** self.run_exp)
        b = 0.5 * self.beam_m * np.clip(g, 0.0, 1.0)
        t = self.draft_m * np.clip(1 - np.abs(2 * xi - 1) ** self.rocker_exp, 0.0, 1.0)
        return xi * self.length_m, b, np.maximum(t, 0.02 * self.draft_m)


@dataclass
class HullGrid:
    """The immersed hull at one load: offsets on a grid (x from the bow, z <= 0)."""

    x: np.ndarray            # (nx,)
    z: np.ndarray            # (nz,) from 0 (waterline) down to -max immersion
    Y: np.ndarray            # (nx, nz) half-breadth
    volume_m3: float
    wetted_m2: float
    draft_m: float           # deepest immersion
    length_m: float
    beam_m: float            # widest waterline beam


def _grid(form: HullForm, depth_ref: float, nx: int = 201, nz: int = 20) -> HullGrid:
    x, b, t = form.stations(nx)
    q = form.section_exp
    h = np.clip(t + (depth_ref - form.draft_m), 0.0, None)        # immersion per station
    hmax = float(h.max())
    s = np.linspace(0.0, 1.0, nz)
    z = -hmax * s ** 1.5                                        # finer near the surface
    eta = h[:, None] + z[None, :]                               # height above the keel
    Y = b[:, None] * (np.clip(eta, 0.0, None) / t[:, None]) ** (1.0 / q)
    wz = np.abs(np.gradient(z))                                 # trapezoid-ish weights
    wz[0] *= 0.5
    wz[-1] *= 0.5
    area = 2.0 * (Y * wz[None, :]).sum(1)
    vol = float(np.trapezoid(area, x))
    # girth: polyline length of each immersed section (both sides)
    ee = h[:, None] * np.linspace(0, 1, 40)[None, :] ** 2
    yy = b[:, None] * (ee / t[:, None]) ** (1.0 / q)
    girth = 2.0 * np.sum(np.hypot(np.diff(yy, axis=1), np.diff(ee, axis=1)), axis=1)
    wet = float(np.trapezoid(girth, x))
    return HullGrid(x, z, Y, vol, wet, hmax, form.length_m, float(2 * Y[:, 0].max()))


def _volume(form: HullForm, depth_ref: float, nx: int = 301) -> float:
    """Displaced volume at a sinkage, in closed form per station:
    int_0^h 2 b (eta / T)^(1/q) d eta = 2 b T q / (q + 1) (h / T)^((q + 1) / q)."""
    x, b, t = form.stations(nx)
    q = form.section_exp
    h = np.clip(t + (depth_ref - form.draft_m), 0.0, None)
    area = 2.0 * b * t * q / (q + 1) * (h / t) ** ((q + 1) / q)
    return float(np.trapezoid(area, x))


@functools.lru_cache(maxsize=4096)
def _grid_for_volume(form: HullForm, volume_m3: float) -> HullGrid:
    """The immersed hull carrying ``volume_m3`` (bisection on sinkage)."""
    lo, hi = 0.0, 4.0 * form.draft_m
    if volume_m3 <= 0:
        return _grid(form, 0.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _volume(form, mid) < volume_m3:
            lo = mid
        else:
            hi = mid
    return _grid(form, 0.5 * (lo + hi))


def hull_at_load(form: HullForm, mass_kg: float, rho: float = RHO_SEA) -> HullGrid:
    return _grid_for_volume(form, round(max(0.0, mass_kg) / rho, 7))


def friction_resistance(U: float, grid: HullGrid, *, rho: float = RHO_SEA,
                        nu: float = NU_SEA) -> float:
    if U <= 0 or grid.wetted_m2 <= 0:
        return 0.0
    re = U * grid.length_m / nu
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    t = grid.beam_m / grid.length_m
    form = 1.0 + 2.0 * t + 60.0 * t ** 4
    return 0.5 * rho * U * U * grid.wetted_m2 * cf * form


def _kochin(grid: HullGrid, k0: float, lam: np.ndarray) -> np.ndarray:
    """A(lambda) = sum over the centreplane of dY/dx exp(k0 lam^2 z + i k0 lam x)."""
    x, z, Y = grid.x, grid.z, grid.Y
    Yx = np.gradient(Y, x, axis=0)
    wx = np.gradient(x)
    wz = np.abs(np.gradient(z))
    wz[0] *= 0.5
    wz[-1] *= 0.5
    W = Yx * wx[:, None] * wz[None, :]                              # (nx, nz)
    Ez = np.exp(k0 * np.outer(lam ** 2, z))                         # (nl, nz)
    inner = Ez @ W.T                                                # (nl, nx)
    return np.sum(inner * np.exp(1j * k0 * np.outer(lam, x - x.mean())), axis=1)


def michell_resistance(U: float, hulls: Sequence[Tuple[HullGrid, float, float]], *,
                       rho: float = RHO_SEA, n_lambda: int = 600) -> float:
    """Michell wave resistance (N) of one or more thin hulls at speed ``U``.

    ``hulls``: (immersed grid, fore-aft position of its middle, lateral position)
    for each hull.  One centreline hull reduces to the classical
    ``R = 4 rho g^2 / (pi U^2) * int_1^inf |A|^2 lam^2 / sqrt(lam^2 - 1) dlam``;
    several are summed in the far field with their phase offsets.
    """
    if U <= 0.05 or not hulls:
        return 0.0
    k0 = G / (U * U)
    dx = min(float(np.mean(np.diff(h.x))) for h, _, _ in hulls)
    lam_max = max(1.05, min(40.0, 0.5 / (k0 * dx)))
    u = np.linspace(0.0, math.acosh(lam_max), n_lambda)
    lam = np.cosh(u)
    ky = k0 * lam * np.sqrt(np.maximum(lam * lam - 1.0, 0.0))
    Ap = np.zeros(n_lambda, complex)
    Am = np.zeros(n_lambda, complex)
    for grid, xc, yc in hulls:
        if grid.volume_m3 <= 0:
            continue
        a = _kochin(grid, k0, lam) * np.exp(1j * k0 * lam * xc)
        Ap += a * np.exp(1j * ky * yc)
        Am += a * np.exp(-1j * ky * yc)
    integrand = (np.abs(Ap) ** 2 + np.abs(Am) ** 2) * np.cosh(u) ** 2    # dlam = sinh u du
    return float(2.0 * rho * G * G / (math.pi * U * U) * np.trapezoid(integrand, u))


def yaw_drag(U: float, beta_rad, draft_m: float, rho: float = RHO_SEA):
    """Induced drag of a hull at drift angle beta (slender-body theory, free surface
    as a mirror): 0.5 pi rho U^2 T^2 beta^2."""
    return 0.5 * math.pi * rho * U * U * draft_m ** 2 * np.square(beta_rad)


def air_drag(U: float, head_wind: float = 0.0, cda_m2: float = 0.40) -> float:
    v = U + head_wind
    return 0.5 * RHO_AIR * cda_m2 * v * abs(v)


@dataclass(frozen=True)
class Outrigger:
    """The ama and how it is rigged.  ``span_m``: hull centreline to ama
    centreline; ``rig_height_m``: ama keel above the hull keel with the boat level;
    ``x_m``: ama middle ahead (+) of the hull middle; ``side``: +1 ama on the left."""

    ama: HullForm
    span_m: float = 1.80
    rig_height_m: float = 0.06
    x_m: float = 0.0
    mass_kg: float = 4.0
    side: int = 1


@dataclass(frozen=True)
class Craft:
    """A paddle craft: hull, optional outrigger, empty mass, air drag area."""

    name: str
    hull: HullForm
    hull_mass_kg: float
    outrigger: Optional[Outrigger] = None
    cda_m2: float = 0.40

    @property
    def kind(self) -> str:
        return "oc1" if self.outrigger else "surfski"

    # ── statics ───────────────────────────────────────────────────────────
    def ama_immersion(self, ama_load_kg: float) -> float:
        if not self.outrigger or ama_load_kg <= 0:
            return 0.0
        return hull_at_load(self.outrigger.ama, ama_load_kg).draft_m

    def hull_draft(self, total_kg: float, ama_load_kg: float = 0.0) -> float:
        return hull_at_load(self.hull, total_kg - ama_load_kg).draft_m

    def roll_for_ama_load(self, total_kg: float, ama_load_kg: float) -> float:
        """Static roll (rad; + lifts the ama side) that puts ``ama_load_kg`` on the
        ama.  At zero load the ama just touches the water."""
        o = self.outrigger
        if o is None:
            return 0.0
        d = self.hull_draft(total_kg, ama_load_kg)
        h = self.ama_immersion(ama_load_kg)
        s = (d - o.rig_height_m - h) / o.span_m
        return o.side * math.asin(max(-1.0, min(1.0, s)))

    def ama_load_for_roll(self, total_kg: float, roll_rad: float) -> float:
        """Inverse of :meth:`roll_for_ama_load` (kg on the ama for a measured roll)."""
        if self.outrigger is None:
            return 0.0
        target = roll_rad
        f = lambda L: self.roll_for_ama_load(total_kg, L)
        if f(0.0) * self.outrigger.side <= target * self.outrigger.side:
            return 0.0                                  # ama flying (out of the water)
        lo, hi = 0.0, 0.6 * total_kg
        if f(hi) * self.outrigger.side > target * self.outrigger.side:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if f(mid) * self.outrigger.side > target * self.outrigger.side:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ── resistance ───────────────────────────────────────────────────────
    def resistance(self, U: float, total_kg: float, *, ama_load_kg: float = 0.0,
                   beta_rms_rad: float = 0.0, head_wind: float = 0.0) -> Dict[str, float]:
        """Resistance components (N) at speed ``U`` (through the water)."""
        hl = hull_at_load(self.hull, total_kg - ama_load_kg)
        out = {"hull_friction": friction_resistance(U, hl)}
        parts = [(hl, 0.0, 0.0)]
        out["ama_friction"] = 0.0
        if self.outrigger is not None and ama_load_kg > 0:
            o = self.outrigger
            al = hull_at_load(o.ama, ama_load_kg)
            out["ama_friction"] = friction_resistance(U, al)
            parts.append((al, o.x_m, o.side * o.span_m))
        out["waves"] = michell_resistance(U, parts)
        out["air"] = air_drag(U, head_wind, self.cda_m2)
        out["yaw"] = float(yaw_drag(U, beta_rms_rad, hl.draft_m))
        out["total"] = (out["hull_friction"] + out["ama_friction"] + out["waves"]
                        + out["air"] + out["yaw"])
        return out


_CACHE_DIR = os.environ.get("PADDLING_HYDRO_CACHE",
                            os.path.join(os.path.expanduser("~"), ".cache", "paddling_hydro"))


@functools.lru_cache(maxsize=32)
def resistance_table(craft: Craft, total_kg: float,
                     speeds: Tuple[float, ...] = tuple(np.round(np.linspace(0.25, 7.0, 28), 4)),
                     ama_loads: Tuple[float, ...] = (0.0, 3.0, 6.0, 10.0, 15.0, 22.0, 30.0)
                     ) -> Dict[str, np.ndarray]:
    """Water resistance (friction + waves; no air, no yaw) on a speed x ama-load
    grid, for fast lookup in the simulator and the analyses.  Cached in memory and
    on disk (``$PADDLING_HYDRO_CACHE``), keyed by every input."""
    loads = ama_loads if craft.outrigger else (0.0,)
    key = hashlib.sha1(repr((craft, round(total_kg, 3), speeds, loads, _MODEL_VERSION))
                       .encode()).hexdigest()[:16]
    path = os.path.join(_CACHE_DIR, f"rt_{key}.npz")
    try:
        with np.load(path) as z:
            return {"speeds": z["speeds"], "ama_loads": z["ama_loads"], "R": z["R"]}
    except Exception:
        pass
    R = np.zeros((len(speeds), len(loads)))
    for j, L in enumerate(loads):
        for i, U in enumerate(speeds):
            r = craft.resistance(U, total_kg, ama_load_kg=L)
            R[i, j] = r["hull_friction"] + r["ama_friction"] + r["waves"]
    out = {"speeds": np.array(speeds), "ama_loads": np.array(loads), "R": R}
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.savez(path, **out)
    except OSError:
        pass
    return out


def water_resistance(table: Dict[str, np.ndarray], U, ama_load_kg=0.0) -> np.ndarray:
    """Bilinear lookup in a :func:`resistance_table` (quadratic below its first speed)."""
    s, L, R = table["speeds"], table["ama_loads"], table["R"]
    U = np.asarray(U, float)
    if len(L) == 1:
        r = np.interp(np.abs(U), s, R[:, 0])
    else:
        Ub, Lc = np.broadcast_arrays(np.abs(U), np.clip(np.asarray(ama_load_kg, float), L[0], L[-1]))
        j = np.clip(np.searchsorted(L, Lc) - 1, 0, len(L) - 2)
        w = (Lc - L[j]) / (L[j + 1] - L[j])
        # interpolate in speed on every load column, then linearly between columns
        cols = np.stack([np.interp(Ub.ravel(), s, R[:, k]) for k in range(len(L))], -1)
        jf, wf = j.ravel(), w.ravel()
        idx = np.arange(len(jf))
        r = ((1 - wf) * cols[idx, jf] + wf * cols[idx, jf + 1]).reshape(Ub.shape)
        r = r if np.ndim(r) else float(r)
    low = np.abs(U) < s[0]
    if np.any(low):
        r = np.where(low, R[0, 0] * (np.abs(U) / s[0]) ** 2, r)
    return r * np.sign(U) if np.ndim(U) else float(r) * (1 if U >= 0 else -1)


@functools.lru_cache(maxsize=32)
def roll_table(craft: Craft, total_kg: float,
               loads: Tuple[float, ...] = tuple(float(x) for x in np.arange(0.0, 40.5, 1.0))
               ) -> Dict[str, np.ndarray]:
    """Static roll (rad) against ama load (kg) for this craft and total mass."""
    roll = np.array([craft.roll_for_ama_load(total_kg, L) for L in loads])
    return {"loads": np.array(loads), "roll": roll}


def ama_load_from_roll(table: Dict[str, np.ndarray], roll_rad, side: int = 1) -> np.ndarray:
    """Ama load for a measured static roll (0 when the ama is out of the water;
    clipped at the table's largest load).  Roll falls monotonically as load rises."""
    r = table["roll"] * side
    x = np.asarray(roll_rad, float) * side
    # np.interp needs increasing x: roll decreases with load, so flip
    return np.interp(-x, -r, table["loads"], left=0.0, right=float(table["loads"][-1]))


def sway_time_constant(U: float, total_kg: float, draft_m: float, rho: float = RHO_SEA) -> float:
    """How quickly side-slip dies out after a heading change: the hull's side
    force per unit slip (slender-body theory) against its mass, m / (pi rho U T^2)."""
    return total_kg / (math.pi * rho * max(0.3, U) * draft_m ** 2)


def speed_for_power(curve_speeds: np.ndarray, curve_R: np.ndarray, power_w: float, *,
                    cda_m2: float = 0.0, head_wind: float = 0.0, extra_n: float = 0.0,
                    extra_coef: float = 0.0) -> float:
    """Steady speed at which ``(R_water(U) + air + extra_n + extra_coef * U^2) * U``
    equals ``power_w``."""
    if power_w <= 0:
        return 0.0
    f = lambda U: (np.interp(U, curve_speeds, curve_R) + air_drag(U, head_wind, cda_m2)
                   + extra_n + extra_coef * U * U) * U - power_w
    lo, hi = 0.05, float(curve_speeds[-1])
    if f(hi) < 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
