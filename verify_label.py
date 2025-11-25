
from jolt import Joint1D, Plate, FastenerRow

def test_fastener_label():
    pitches = [1.0]
    plates = [
        Plate(name="P1", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1]),
        Plate(name="P2", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1]),
    ]
    fasteners = [
        FastenerRow(row=1, D=0.2, Eb=1e7, nu_b=0.3)
    ]
    
    joint = Joint1D(pitches, plates, fasteners)
    supports = [(0, 0, 0.0)] # Fix P1 left
    solution = joint.solve(supports)
    
    fastener_dicts = solution.fasteners_as_dicts()
    print("Fastener Dicts:", fastener_dicts)
    
    label = fastener_dicts[0]["Fastener"]
    expected = "F1 (0-1)"
    if label == expected:
        print("PASS: Label format is correct.")
    else:
        print(f"FAIL: Expected '{expected}', got '{label}'")

if __name__ == "__main__":
    test_fastener_label()
