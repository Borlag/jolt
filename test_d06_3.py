"""Quick test for D06_3 case to verify the mixed-shear fix."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jolt.model import Joint1D, Plate, FastenerRow


def main():
    config_path = Path(__file__).parent / "test_values" / "D06_3_config.json"
    ref_path = Path(__file__).parent / "test_values" / "D06_3_reference.json"
    
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
    
    # Diagnostic: Check what code path will be taken
    print("=== DIAGNOSTIC ===")
    f = fasteners[0]  # Row 2 fastener
    n_connections = len(f.connections) if f.connections else 0
    n_branches = n_connections + 1
    is_boeing = "boeing" in f.method.lower()
    print(f"Fastener row 2: {n_connections} connections = {n_branches} branches")
    print(f"is_boeing={is_boeing}, n_branches >= 3 = {n_branches >= 3}")
    if is_boeing and n_branches >= 3:
        print("  -> Direct Mixed-Shear Assembly (JOLT logic)")
    elif n_branches == 2:
        print("  -> 2-layer symmetric split")
    else:
        print("  -> WLS path (non-Boeing)")
    
    # Create joint
    joint = Joint1D(config["pitches"], plates, fasteners)
    
    # Parse supports
    supports = [(int(s[0]), int(s[1]), float(s[2])) for s in config["supports"]]
    
    # Solve
    solution = joint.solve(supports)
    
    # Get reference fasteners
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    
    # Create lookup
    ref_lookup = {}
    for rf in ref_fasteners:
        key = (rf["row"], rf["plate_i"], rf["plate_j"])
        ref_lookup[key] = rf
    
    # Plate name lookup
    plate_names = {i: p.name for i, p in enumerate(solution.plates)}
    
    print("\n=== ROW 2 RESULTS ===")
    
    max_error = 0.0
    for f in solution.fasteners:
        if f.row != 2:
            continue
        plate_i_name = plate_names.get(f.plate_i)
        plate_j_name = plate_names.get(f.plate_j)
        
        key1 = (f.row, plate_i_name, plate_j_name)
        key2 = (f.row, plate_j_name, plate_i_name)
        
        ref = ref_lookup.get(key1) or ref_lookup.get(key2)
        
        if ref:
            ref_force = abs(ref.get("force", 0.0))
            model_force = abs(f.force)
            delta = model_force - ref_force
            rel_error = 100 * abs(delta) / ref_force if ref_force > 0.1 else 0.0
            max_error = max(max_error, rel_error)
            
            flag = "OK" if rel_error < 5 else "FAIL"
            short_i = plate_i_name[:8] if plate_i_name else "?"
            short_j = plate_j_name[:8] if plate_j_name else "?"
            print(f"R{f.row} {short_i:8}-{short_j:8}: Ref={ref_force:6.1f} Model={model_force:6.1f} Err={rel_error:5.1f}% {flag}")
    
    print(f"\nMax Error on Row 2: {max_error:.1f}%")
    if max_error < 5:
        print("PASS: Error within 5% target")
    else:
        print("FAIL: Error exceeds 5% target")
    
    # Print displacement comparison
    print("\n=== DISPLACEMENT CHECK ===")
    # Find sheartie node at row 2
    for n in solution.nodes:
        if "shear" in n.plate_name.lower() and n.row == 2:
            ref_disp = 0.0016224  # From reference
            print(f"sheartie Row 2 disp: Model={abs(n.displacement):.7f} Ref={ref_disp:.7f}")
            break


if __name__ == "__main__":
    main()
