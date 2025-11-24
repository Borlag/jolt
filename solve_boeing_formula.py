from jolt.fasteners import boeing69_compliance
import math

def calculate_boeing_stiffness():
    # Case 2 Parameters
    # Tripler
    t1 = 0.14
    E1 = 10.3e6
    # Doubler
    t2 = 0.13
    E2 = 10.3e6
    # Skin
    t3 = 0.429
    E3 = 10.3e6
    
    # Fastener
    D = 0.344
    Eb = 16.0e6
    nu_b = 0.3
    
    # Pair 1: Tripler-Doubler (t1-t2)
    # Assuming single shear for this pair in isolation?
    # Or does Boeing treat it as part of a stack?
    # The boeing69_compliance function takes shear_planes argument.
    # For a 3-plate stack, the top interface is single shear relative to the top plate?
    # But the bolt goes through 3 plates.
    
    # Let's try calculating as if they were independent pairs first.
    
    print("--- Independent Pair Calculation ---")
    
    # Pair 1: Tripler (t1) - Doubler (t2)
    c12 = boeing69_compliance(
        ti=t1, Ei=E1, tj=t2, Ej=E2,
        Eb=Eb, nu_b=nu_b, diameter=D,
        shear_planes=1 # Single shear interface
    )
    k12 = 1.0 / c12
    print(f"Tripler-Doubler (Independent): K = {k12:.4e}")
    
    # Pair 2: Doubler (t2) - Skin (t3)
    c23 = boeing69_compliance(
        ti=t2, Ei=E2, tj=t3, Ej=E3,
        Eb=Eb, nu_b=nu_b, diameter=D,
        shear_planes=1 # Single shear interface
    )
    k23 = 1.0 / c23
    print(f"Doubler-Skin (Independent): K = {k23:.4e}")

if __name__ == "__main__":
    calculate_boeing_stiffness()
