from jolt.model import Joint1D, Plate, FastenerRow
from jolt.fasteners import boeing69_compliance, huth_compliance

def verify_methods():
    # Setup a simple 2-plate joint
    plates = [
        Plate(name="P1", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0, Fx_right=0.0),
        Plate(name="P2", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0, Fx_right=0.0)
    ]
    pitches = [1.0]
    supports = [(0, 0, 0.0)] # Fix P1 left
    
    # Boeing
    f_boeing = FastenerRow(row=1, D=0.2, Eb=10e6, nu_b=0.3, method="Boeing69")
from jolt.model import Joint1D, Plate, FastenerRow
from jolt.fasteners import boeing69_compliance, huth_compliance

def verify_methods():
    # Setup a simple 2-plate joint
    plates = [
        Plate(name="P1", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0, Fx_right=0.0),
        Plate(name="P2", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0, Fx_right=0.0)
    ]
    pitches = [1.0]
    supports = [(0, 0, 0.0)] # Fix P1 left
    
    # Boeing
    f_boeing = FastenerRow(row=1, D=0.2, Eb=10e6, nu_b=0.3, method="Boeing69")
    model_b = Joint1D(pitches, plates, [f_boeing])
from jolt.model import Joint1D, Plate, FastenerRow
from jolt.fasteners import boeing69_compliance, huth_compliance

def verify_methods():
    # Setup a simple 2-plate joint
    plates = [
        Plate(name="P1", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0, Fx_right=0.0),
        Plate(name="P2", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0, Fx_right=0.0)
    ]
    pitches = [1.0]
    supports = [(0, 0, 0.0)] # Fix P1 left
    
    # Boeing
    f_boeing = FastenerRow(row=1, D=0.2, Eb=10e6, nu_b=0.3, method="Boeing69")
    model_b = Joint1D(pitches, plates, [f_boeing])
    sol_b = model_b.solve(supports)
    k_boeing = sol_b.fasteners[0].stiffness
    print(f"Boeing Stiffness: {k_boeing:.2e}")
    
    # Huth
    f_huth = FastenerRow(row=1, D=0.2, Eb=10e6, nu_b=0.3, method="Huth")
    model_h = Joint1D(pitches, plates, [f_huth])
    sol_h = model_h.solve(supports)
    k_huth = sol_h.fasteners[0].stiffness
    print(f"Huth Stiffness:   {k_huth:.2e}")
    
    # Grumman
    f_grumman = FastenerRow(row=1, D=0.2, Eb=10e6, nu_b=0.3, method="Grumman")
    model_g = Joint1D(pitches, plates, [f_grumman])
    sol_g = model_g.solve(supports)
    k_grumman = sol_g.fasteners[0].stiffness
    print(f"Grumman Stiffness: {k_grumman:.2e}")

    # if k_boeing != k_huth:
    #     print("PASS: Stiffness values are different.")
    # else:
    #     print("FAIL: Stiffness values are identical.")

if __name__ == "__main__":
    verify_methods()
