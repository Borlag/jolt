import math
from jolt.fasteners import boeing69_compliance

def calculate_single_stiffness(t, E, D, Eb, nu_b):
    # Copied from my implementation (updated)
    A_b = math.pi * (D / 2.0) ** 2
    I_b = math.pi * (D / 2.0) ** 4 / 4.0
    G_b = Eb / (2.0 * (1.0 + nu_b))

    C_shear = (4.0 * t) / (9.0 * G_b * A_b)
    C_bending = (6.0 * t**3) / (40.0 * fastener.Eb * I_b) if 'fastener' in globals() else (6.0 * t**3) / (40.0 * Eb * I_b)
    C_bearing = (1.0 / t) * (1.0 / Eb + 1.0 / E)
            
    compliance = C_shear + C_bending + C_bearing
    return 1.0 / compliance

def compare():
    # Parameters from Boeing Screenshot (Case 2?)
    # Node 2003 (Top) t=0.140. Node 3003 (Middle) t=0.130.
    t1 = 0.140
    t2 = 0.130
    E = 1.6e7 # From screenshot "Modulus 1.600E7"
    D = 0.344 # From screenshot "Diameter 0.344"
    Eb = 1.6e7 # Assuming bolt modulus same as plate? Or standard?
    # Screenshot doesn't show Eb directly, but typically steel/Ti.
    # Let's assume typical values if not given.
    # Wait, screenshot says "Modulus 1.600E7". This is likely Plate E.
    # Bolt E? If not specified, maybe same?
    # Let's try to match 3.78e5.
    
    # If I use the values from the screenshot:
    # t1=0.140, t2=0.130.
    # Stiffness = 3.78e5.
    
    # Let's assume Eb = 1.6e7 for now (Titanium?).
    nu_b = 0.3 # Standard
    
    K_single_1 = calculate_single_stiffness(t1, E, D, Eb, nu_b)
    K_single_2 = calculate_single_stiffness(t2, E, D, Eb, nu_b)
    
    # Check Mixed Case (Tripler-Doubler)
    # t1=0.063, t2=0.040.
    # Boeing Target: 1.21e5.
    
    K_single_1 = calculate_single_stiffness(t1, E, D, Eb, nu_b)
    K_single_2 = calculate_single_stiffness(t2, E, D, Eb, nu_b)
    
    # Virtual Pair Stiffness
    K_eff_mixed = 1.0 / (1.0/K_single_1 + 1.0/K_single_2)
    
    print(f"Boeing Target (Mixed): 1.21e5")
    print(f"My Reported (Mixed): {K_eff_mixed:.2e}")

if __name__ == "__main__":
    compare()
