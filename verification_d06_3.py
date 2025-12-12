import json
import sys
from pathlib import Path
from jolt.model import Joint1D, Plate, FastenerRow

def verify_d06_3():
    config_path = "test_values/D06_3_config.json"
    ref_path = "test_values/D06_3_reference.json"

    with open(config_path) as f:
        config = json.load(f)
    
    with open(ref_path) as f:
        reference = json.load(f)
    
    # Create plates
    plates = []
    for p in config["plates"]:
        plates.append(Plate(
            name=p["name"],
            E=p["E"],
            t=p["t"],
            first_row=p["first_row"],
            last_row=p["last_row"],
            A_strip=p["A_strip"],
            Fx_left=p.get("Fx_left", 0.0),
            Fx_right=p.get("Fx_right", 0.0),
            widths=p.get("widths"),
            thicknesses=p.get("thicknesses"),
        ))
    
    # Create fasteners
    fasteners = []
    for f_data in config["fasteners"]:
        fasteners.append(FastenerRow(
            row=f_data["row"],
            D=f_data["D"],
            Eb=f_data["Eb"],
            nu_b=f_data.get("nu_b", 0.3),
            method=f_data.get("method", "Boeing69"),
            connections=f_data.get("connections"),
            topology=f_data.get("topology"),
        ))

    # Create joint
    joint = Joint1D(config["pitches"], plates, fasteners)
    
    # Parse supports
    supports = [(int(s[0]), int(s[1]), float(s[2])) for s in config["supports"]]
    
    # Solve
    print("Solving D06_3...")
    solution = joint.solve(supports)
    
    # Get reference fasteners
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    
    # Create lookup
    ref_lookup = {}
    for rf in ref_fasteners:
        key = (rf["row"], rf["plate_i"], rf["plate_j"])
        ref_lookup[key] = rf
        # Also map reverse for convenience
        key2 = (rf["row"], rf["plate_j"], rf["plate_i"])
        ref_lookup[key2] = rf
    
    # Plate name lookup
    plate_names = {i: p.name for i, p in enumerate(solution.plates)}
    
    print("\n=== VERIFICATION RESULTS (All Rows) ===")
    print(f"{'Row':<4} {'Interface':<20} {'Ref (lb)':<10} {'Calc (lb)':<10} {'Diff %':<8} {'Status'}")
    print("-" * 75)
    
    max_error = 0.0
    for f in solution.fasteners:
        plate_i_name = plate_names.get(f.plate_i)
        plate_j_name = plate_names.get(f.plate_j)
        
        key = (f.row, plate_i_name, plate_j_name)
        ref = ref_lookup.get(key)
        
        if ref:
            ref_force = abs(ref.get("force", 0.0))
            model_force = abs(f.force)
            delta = model_force - ref_force
            rel_error = 100 * abs(delta) / ref_force if ref_force > 0.1 else 0.0
            max_error = max(max_error, rel_error)
            
            flag = "PASS" if rel_error < 1.0 else "FAIL"
            if rel_error > 5.0: flag = "FAIL!!"
            
            interface_str = f"{plate_i_name[:9]}-{plate_j_name[:9]}"
            print(f"{f.row:<4} {interface_str:<20} {ref_force:<10.2f} {model_force:<10.2f} {rel_error:<8.2f} {flag}")
    
    print("-" * 75)
    print(f"Max Error: {max_error:.2f}%")
    if max_error < 1.0:
        print("OVERALL RESULT: SUCCESS (<1%)")
    else:
        print("OVERALL RESULT: FAILURE (>1%)")

if __name__ == "__main__":
    verify_d06_3()
