
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
ti = 0.130

target_k = 3.887e5

print(f"Target Stiffness: {target_k:.4e}")
print(f"{'tj':<10} | {'Stiffness':<15} | {'Diff':<15}")
print("-" * 45)

for t_val in [0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.429]:
    c = boeing69_compliance(ti, Ei, t_val, Ej, Eb, nu_b, D)
    k = 1.0 / c
    diff = k - target_k
    print(f"{t_val:<10.3f} | {k:<15.4e} | {diff:<15.4e}")
