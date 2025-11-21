
import math

def boeing69_compliance_debug(
    ti: float,
    Ei: float,
    tj: float,
    Ej: float,
    Eb: float,
    nu_b: float,
    diameter: float,
    shear_planes: int = 1,
    cap_bending: bool = False,
    cap_shear: bool = False,
):
    area_bolt = math.pi * diameter**2 / 4.0
    inertia_bolt = math.pi * diameter**4 / 64.0
    shear_modulus = Eb / (2.0 * (1.0 + nu_b))

    ti_bending = ti
    tj_bending = tj
    if cap_bending:
        if ti > diameter: ti_bending = diameter
        if tj > diameter: tj_bending = diameter

    ti_shear = ti
    tj_shear = tj
    if cap_shear:
        if ti > diameter: ti_shear = diameter
        if tj > diameter: tj_shear = diameter

    term_shear = 4.0 * (ti_shear + tj_shear) / (9.0 * shear_modulus * area_bolt)

    term_bending = (
        ti_bending**3
        + 5.0 * ti_bending**2 * tj_bending
        + 5.0 * ti_bending * tj_bending**2
        + tj_bending**3
    ) / (40.0 * Eb * inertia_bolt)
    
    bearing_divisor = float(1 if shear_planes <= 1 else 2 * shear_planes)
    
    term_bearing = (
        (1.0 / ti) * (1.0 / Eb + 1.0 / Ei)
        + (1.0 / tj) * (1.0 / Eb + 1.0 / Ej)
    ) / bearing_divisor
    
    compliance = term_shear + term_bending + term_bearing
    
    print(f"ti={ti}, tj={tj}")
    print(f"Shear Term:   {term_shear:.4e}")
    print(f"Bending Term: {term_bending:.4e}")
    print(f"Bearing Term: {term_bearing:.4e}")
    print(f"Total Comp:   {compliance:.4e}")
    print(f"Stiffness:    {1.0/compliance:.4e}")

Ei = 1.030e7
Ej = 1.030e7
Eb = 1.600e7
D = 0.344
nu_b = 0.33

print("--- Pair 2 (0.130 - 0.429) Capped Bending ONLY ---")
boeing69_compliance_debug(0.130, Ei, 0.429, Ej, Eb, nu_b, D, cap_bending=True, cap_shear=False)

print("\n--- Pair 2 (0.130 - 0.429) Capped Bending AND Shear ---")
boeing69_compliance_debug(0.130, Ei, 0.429, Ej, Eb, nu_b, D, cap_bending=True, cap_shear=True)
