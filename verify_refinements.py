
from jolt import Joint1D, Plate, FastenerRow
import math

def verify_refinements():
    print("Verifying Refinements...")
    
    # Define a simple case: 2 plates, 1 fastener
    p0 = Plate(name="Tripler", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1])
    p1 = Plate(name="Skin", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1])
    
    # Fastener at row 1
    f1 = FastenerRow(row=1, D=0.2, Eb=1e7, nu_b=0.3)
    
    model = Joint1D(pitches=[1.0, 1.0], plates=[p0, p1], fasteners=[f1])
    
    supports = [(1, 1, 0.0)] # Plate 1, Row 2
    forces = [(0, 0, -1000.0)] # Negative force to test abs() logic
    
    solution = model.solve(supports=supports, point_forces=forces)
    
    # 1. Verify Classic Results Node ID
    classic = solution.classic_results_as_dicts()
    c_res = classic[0]
    
    print(f"Classic Result Keys: {c_res.keys()}")
    assert "Node ID" in c_res, "Node ID column missing in Classic Results"
    assert "Row" not in c_res, "Row column should be removed/replaced in Classic Results"
    
    # Check ID value. Plate 0, Row 1 -> 1001.
    # But wait, classic results are per fastener row.
    # Row 1 has fasteners.
    # Plate 0 at Row 1 is Node 1001.
    # Plate 1 at Row 1 is Node 2001.
    # Let's check values.
    ids = [c["Node ID"] for c in classic]
    print(f"Classic IDs: {ids}")
    assert 1001 in ids
    assert 2001 in ids
    
    print("Classic Results Node ID verified.")
    
    # 2. Verify Min/Max Absolute Values
    # Applied force is -1000.
    # Bars should have compression (negative force).
    bars = solution.bars_as_dicts()
    forces = [b["Force"] for b in bars]
    print(f"Bar Forces: {forces}")
    
    # Max Plate Load should be max(abs(forces)) -> 1000 (approx)
    # Min Plate Load should be min(abs(forces)) -> 0 (or close to 0 if all loaded)
    
    # In UI logic:
    max_plate_load = max([abs(f) for f in forces])
    min_plate_load = min([abs(f) for f in forces])
    
    print(f"Max Plate Load (Abs): {max_plate_load}")
    print(f"Min Plate Load (Abs): {min_plate_load}")
    
    assert max_plate_load > 900.0, "Max load should capture magnitude of negative force"
    
    print("Min/Max Absolute Values verified.")

if __name__ == "__main__":
    verify_refinements()
