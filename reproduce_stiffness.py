
import math

def boeing69_compliance(
    ti: float,
    Ei: float,
    tj: float,
    Ej: float,
    Eb: float,
    nu_b: float,
    diameter: float,
    shear_planes: int = 1,
) -> float:
    area_bolt = math.pi * diameter**2 / 4.0
    inertia_bolt = math.pi * diameter**4 / 64.0
    shear_modulus = Eb / (2.0 * (1.0 + nu_b))

    term_shear = 4.0 * (ti + tj) / (9.0 * shear_modulus * area_bolt)

    term_bending = (
        ti**3
        + 5.0 * ti**2 * tj
        + 5.0 * ti * tj**2
        + tj**3
    ) / (40.0 * Eb * inertia_bolt)
    
    bearing_divisor = float(1 if shear_planes <= 1 else 2 * shear_planes)
    
    term_bearing = (
        (1.0 / ti) * (1.0 / Eb + 1.0 / Ei)
        + (1.0 / tj) * (1.0 / Eb + 1.0 / Ej)
    ) / bearing_divisor
    
    compliance = term_shear + term_bending + term_bearing
    return compliance

Ei = 1.030e7
Ej = 1.030e7
Eb = 1.600e7
D = 0.344
nu_b = 0.33

# Target: 3.887e5 (The incorrect value user sees)
# Expected: 4.27e5 (With 0.130 and 0.429)

print(f"Target Incorrect Value: 3.887e5")

# Hypothesis 1: Skin thickness is wrong. Maybe it's using 0.130 (Doubler thickness) instead of 0.429?
k_hyp1 = 1.0 / boeing69_compliance(0.130, Ei, 0.130, Ej, Eb, nu_b, D)
print(f"Hypothesis 1 (0.130 - 0.130): {k_hyp1:.4e}")

# Hypothesis 2: Skin thickness is 0.140 (Tripler thickness)?
k_hyp2 = 1.0 / boeing69_compliance(0.130, Ei, 0.140, Ej, Eb, nu_b, D)
print(f"Hypothesis 2 (0.130 - 0.140): {k_hyp2:.4e}")

# Hypothesis 3: Maybe Modulus is wrong?
# Hypothesis 4: Maybe Diameter is wrong?
