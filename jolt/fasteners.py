"""
Fastener compliance helper functions.

Formula Selection Guide:
* Huth: Best for general purpose, mixed metal/composite, and riveted joints (empirically tuned).
* Grumman: Best for legacy single-shear metallic joints (e.g., Saab 37 Viggen data).
* Swift: Best for Damage Tolerance/Fatigue Crack growth analysis (conservative). Also known as Vought/Douglas.
* Boeing 69: Best for detailed geometric stack-ups where bending/bearing separation is required.
"""
from __future__ import annotations

import math
from typing import Literal

# -----------------------------------------------------------------------------
# COMPONENT METHODS (Calculate parts of compliance)
# -----------------------------------------------------------------------------

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
    """Return fastener compliance according to Boeing (1969) (D6-29942)."""
    if diameter <= 0: raise ValueError("Fastener diameter must be positive")
    if Eb <= 0: raise ValueError("Fastener modulus Eb must be positive")

    area_bolt = math.pi * diameter**2 / 4.0
    inertia_bolt = math.pi * diameter**4 / 64.0
    shear_modulus = Eb / (2.0 * (1.0 + nu_b))

    # Shear Term
    ti_s = shear_ti if shear_ti is not None else ti
    tj_s = shear_tj if shear_tj is not None else tj
    term_shear = 4.0 * (ti_s + tj_s) / (9.0 * shear_modulus * area_bolt)

    # Bending Term
    ti_b = bending_ti if bending_ti is not None else ti
    tj_b = bending_tj if bending_tj is not None else tj
    term_bending = (
        ti_b**3 + 5.0*ti_b**2*tj_b + 5.0*ti_b*tj_b**2 + tj_b**3
    ) / (40.0 * Eb * inertia_bolt)

    # Bearing Term
    div = float(1 if shear_planes <= 1 else 2 * shear_planes)
    ti_brg = bearing_ti if bearing_ti is not None else ti
    tj_brg = bearing_tj if bearing_tj is not None else tj
    
    term_bearing = (
        (1.0 / ti_brg) * (1.0/Eb + 1.0/Ei) + 
        (1.0 / tj_brg) * (1.0/Eb + 1.0/Ej)
    ) / div

    return term_shear + term_bending + term_bearing

def boeing69_eq1_compliance(
    t1: float, E1: float,
    t2: float, E2: float,
    Ef: float, Gf: float,
    d: float,
) -> float:
    """
    Boeing 1 (Eq.6 Eremin) – single shear compliance.
    
    Difference from Legacy:
    - Shear term denominator uses 5*G*A instead of 9*G*A.
    """
    if d <= 0: raise ValueError("Fastener diameter must be positive")
    if Ef <= 0: raise ValueError("Fastener modulus Ef must be positive")
    
    area_bolt = math.pi * d**2 / 4.0
    inertia_bolt = math.pi * d**4 / 64.0
    
    # 1. Bending Term (Same as Legacy / Eq.6 first term)
    # (t2^3 + 5t2^2t1 + 5t2t1^2 + t1^3) / (40 * Ef * I)
    term_bending = (
        t2**3 + 5.0*t2**2*t1 + 5.0*t2*t1**2 + t1**3
    ) / (40.0 * Ef * inertia_bolt)
    
    # 2. Shear Term (Eq.6 second term)
    # 4(t2 + t1) / (5 * G3 * A3)
    # Legacy uses 9 instead of 5.
    term_shear = 4.0 * (t2 + t1) / (5.0 * Gf * area_bolt)
    
    # 3. Bearing Term (Eq.6 last terms)
    # (t2+t1)/(t2*t1*E3) + 1/(t2*E2) + 1/(t1*E1)
    # Rewrite: 1/(t1*E3) + 1/(t2*E3) + 1/(t1*E1) + 1/(t2*E2)
    #        = 1/t1 * (1/E1 + 1/E3) + 1/t2 * (1/E2 + 1/E3)
    term_bearing = (1.0 / t1) * (1.0/E1 + 1.0/Ef) + (1.0 / t2) * (1.0/E2 + 1.0/Ef)
    
    return term_bending + term_shear + term_bearing

def boeing69_eq2_compliance(
    t1: float, E1: float,
    t2: float, E2: float,
    Ef: float,
    d: float,
) -> float:
    """
    Boeing 2 (Eq.7 Eremin / Eq.2 Methodology) – single shear compliance.
    
    C = (2^((t1/d)^0.85) / t1) * (1/E1 + 3/(8Ef)) + ... (symmetric for t2)
    """
    if d <= 0: raise ValueError("Fastener diameter must be positive")
    
    # Exponent terms
    exp1 = (t1 / d) ** 0.85
    exp2 = (t2 / d) ** 0.85
    
    # Term 1
    # Note: 2^(x) can be computed as pow(2, x) or 2**x
    term1_factor = (2.0 ** exp1) / t1
    term1_compliance = term1_factor * (1.0/E1 + 3.0/(8.0*Ef))
    
    # Term 2
    term2_factor = (2.0 ** exp2) / t2
    term2_compliance = term2_factor * (1.0/E2 + 3.0/(8.0*Ef))
    
    return term1_compliance + term2_compliance

def boeing_pair_compliance(
    t1: float, E1: float,
    t2: float, E2: float,
    Ef: float, Gf: float,
    d: float,
    variant: Literal["legacy", "eq1", "eq2"] = "legacy",
) -> float:
    """Router for varying Boeing compliance formulas."""
    
    if variant == "eq1":
        return boeing69_eq1_compliance(t1, E1, t2, E2, Ef, Gf, d)
    elif variant == "eq2":
        return boeing69_eq2_compliance(t1, E1, t2, E2, Ef, d)
    else:
        # Legacy Boeing 69 (matches original code)
        # Note: Legacy function signature is slightly different (takes nu_b), 
        # but here we can compute nu_b from Ef, Gf if needed, OR just call legacy.
        # Legacy function: boeing69_compliance(ti, Ei, tj, Ej, Eb, nu_b, diameter...)
        # We need nu_b. G = E / (2(1+nu)) -> 1+nu = E/2G -> nu = E/2G - 1
        
        # Guard against Gf=0 or weird values
        if Gf > 0:
            nu_b = (Ef / (2.0 * Gf)) - 1.0
        else:
            nu_b = 0.3 # Fallback
            
        return boeing69_compliance(t1, E1, t2, E2, Ef, nu_b, d)


# -----------------------------------------------------------------------------
# EMPIRICAL METHODS (Calculate TOTAL compliance of the joint)
# -----------------------------------------------------------------------------

def huth_compliance(
    t1: float, E1: float,
    t2: float, E2: float,
    Ef: float, diameter: float,
    shear: Literal["single", "double"] = "single",
    joint_type: Literal["bolted_metal", "riveted_metal", "bolted_graphite"] = "bolted_metal",
) -> float:
    """
    Huth Formula (ASTM STP 927).
    Source: Soderberg, Huth
    """
    if joint_type == "riveted_metal":
        a, b = 0.4, 2.2
    elif joint_type == "bolted_graphite":
        a, b = 2.0/3.0, 4.2
    else: # bolted_metal
        a, b = 2.0/3.0, 3.0

    n = 1.0 if shear == "single" else 2.0

    # Eq 2.14 Soderberg
    term1 = ((t1 + t2) / (2.0 * diameter)) ** a
    term2 = (b / n)
    term3 = (1.0 / (t1 * E1)) + (1.0 / (n * t2 * E2)) + \
            (1.0 / (2.0 * t1 * Ef)) + (1.0 / (2.0 * n * t2 * Ef))
    
    return term1 * term2 * term3

def grumman_compliance(
    t1: float, E1: float, 
    t2: float, E2: float, 
    Ef: float, diameter: float
) -> float:
    """
    Grumman Formula.
    Source: Soderberg Eq 2.13
    """
    term1 = (t1 + t2)**2 / (Ef * diameter)
    term2 = 3.72 * (1.0/(E1 * t1) + 1.0/(E2 * t2))
    return term1 + term2

def swift_douglas_compliance(
    t1: float, E1: float, 
    t2: float, E2: float, 
    Ef: float, diameter: float
) -> float:
    """
    Swift (Douglas) Formula.
    Source: Eremin Eq 5
    """
    A = 5.0
    B = 0.8
    return (1.0 / (Ef * diameter)) * (A + B * diameter * (1.0/t1 + 1.0/t2))

def tate_rosenfeld_compliance(
    t1: float, E1: float, 
    t2: float, E2: float, 
    Ef: float, diameter: float,
    nu_f: float = 0.33
) -> float:
    """
    Tate and Rosenfeld Formula.
    Source: ICAS 2018 Eq 5
    """
    term1 = 1.0/(Ef*t1) + 1.0/(Ef*t2) + 1.0/(E1*t1) + 1.0/(E2*t2)
    
    term2 = (32.0 * (1.0 + nu_f) * (t1 + t2)) / (9.0 * Ef * math.pi * diameter**2)
    
    cubic_part = t1**3 + 5.0*t1**2*t2 + 5.0*t1*t2**2 + t2**3
    term3 = (8.0 * cubic_part) / (5.0 * Ef * math.pi * diameter**4)
    
    return term1 + term2 + term3

def morris_compliance(
    t1: float, E1: float, 
    t2: float, E2: float, 
    Ef: float, diameter: float
) -> float:
    """
    Morris (TU Delft) Formula approx.
    """
    # Fallback to Tate-Rosenfeld as per plan/request for now
    return tate_rosenfeld_compliance(t1, E1, t2, E2, Ef, diameter)

__all__ = [
    "boeing69_compliance", 
    "boeing69_eq1_compliance",
    "boeing69_eq2_compliance",
    "boeing_pair_compliance",
    "huth_compliance", 
    "grumman_compliance",
    "swift_douglas_compliance",
    "tate_rosenfeld_compliance",
    "morris_compliance"
]
