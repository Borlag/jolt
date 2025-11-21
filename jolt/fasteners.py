"""Fastener compliance helper functions."""
from __future__ import annotations

import math
from typing import Literal


def boeing69_compliance(
    ti: float,
    Ei: float,
    tj: float,
    Ej: float,
    Eb: float,
    nu_b: float,
    diameter: float,
    *,
    shear_planes: int = 1,
    shear_ti: float | None = None,
    shear_tj: float | None = None,
    bending_ti: float | None = None,
    bending_tj: float | None = None,
    bearing_ti: float | None = None,
    bearing_tj: float | None = None,
) -> float:
    """Return fastener compliance according to Boeing (1969).

    The formulation follows D6-29942 where the constant is expressed as a
    compliance value ``C_F`` [in/lb].  The stiffness is the reciprocal of the
    returned value.  ``shear_t*`` and ``bending_t*`` allow the caller to
    override the grip length used in the shear and bending terms, while
    ``bearing_t*`` overrides the thickness used in the bearing contribution.
    """
    if diameter <= 0:
        raise ValueError("Fastener diameter must be positive")
    if Eb <= 0:
        raise ValueError("Fastener modulus Eb must be positive")


    area_bolt = math.pi * diameter**2 / 4.0
    inertia_bolt = math.pi * diameter**4 / 64.0
    shear_modulus = Eb / (2.0 * (1.0 + nu_b))

    ti_shear = shear_ti if shear_ti is not None else ti
    tj_shear = shear_tj if shear_tj is not None else tj
    term_shear = 4.0 * (ti_shear + tj_shear) / (9.0 * shear_modulus * area_bolt)

    ti_bending = bending_ti if bending_ti is not None else ti
    tj_bending = bending_tj if bending_tj is not None else tj
    term_bending = (
        ti_bending**3
        + 5.0 * ti_bending**2 * tj_bending
        + 5.0 * ti_bending * tj_bending**2
        + tj_bending**3
    ) / (40.0 * Eb * inertia_bolt)
    bearing_divisor = float(1 if shear_planes <= 1 else 2 * shear_planes)
    ti_bearing = bearing_ti if bearing_ti is not None else ti
    tj_bearing = bearing_tj if bearing_tj is not None else tj
    term_bearing = (
        (1.0 / ti_bearing) * (1.0 / Eb + 1.0 / Ei)
        + (1.0 / tj_bearing) * (1.0 / Eb + 1.0 / Ej)
    ) / bearing_divisor
    return term_shear + term_bending + term_bearing


def huth_compliance(
    ti: float,
    Ei: float,
    tj: float,
    Ej: float,
    Ef: float,
    diameter: float,
    shear: Literal["single", "double"] = "single",
    joint_type: Literal["bolted_metal", "riveted_metal", "bolted_graphite"] = "bolted_metal",
) -> float:
    """Return the Huth fastener compliance (ASTM STP 927).

    Parameters follow the notation from Soderberg.  ``shear`` denotes single or
    double shear, and ``joint_type`` selects a and b factors.
    """

    if joint_type == "bolted_metal":
        a_factor, b_factor = 2.0 / 3.0, 3.0
    elif joint_type == "riveted_metal":
        a_factor, b_factor = 2.0 / 5.0, 2.2
    else:
        a_factor, b_factor = 2.0 / 3.0, 4.2

    shear_planes = 1.0 if shear == "single" else 2.0
    geometry = ((ti + tj) / (2.0 * diameter)) ** a_factor
    compliance_core = (
        1.0 / (ti * Ei)
        + 1.0 / (shear_planes * tj * Ej)
        + 1.0 / (2.0 * ti * Ef)
        + 1.0 / (2.0 * shear_planes * tj * Ef)
    )
    return geometry * (b_factor / shear_planes) * compliance_core


def grumman_compliance(ti: float, Ei: float, tj: float, Ej: float, Ef: float, diameter: float) -> float:
    """Return the empirical Grumman fastener compliance."""

    return (ti + tj) ** 2 / (Ef * diameter) + 3.72 * (1.0 / (Ei * ti) + 1.0 / (Ej * tj))


__all__ = ["boeing69_compliance", "huth_compliance", "grumman_compliance"]
