import json
import os
import glob
from jolt.model import Joint1D, Plate, FastenerRow

def verify_single_case(config_path, ref_path):
    import copy
    print(f"Testing {os.path.basename(config_path)}...")
    
    with open(config_path) as f:
        config = json.load(f)
    with open(ref_path) as f:
        reference = json.load(f)

    # Build base objects
    plates_base = []
    for p in config["plates"]:
        plates_base.append(Plate(
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
    
    # Fasteners base
    fasteners_data = config["fasteners"]
    pitches = config["pitches"]
    supports_data = config["supports"]
    supports = [(int(s[0]), int(s[1]), float(s[2])) for s in supports_data]

    # Ref map
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    ref_map = {}
    for rf in ref_fasteners:
        r = rf["row"]
        p1 = rf.get("plate_i", "UNKNOWN")
        p2 = rf.get("plate_j", "UNKNOWN")
        f_val = rf.get("force", 0.0)
        ref_map[(r, p1, p2)] = abs(f_val)
        ref_map[(r, p2, p1)] = abs(f_val)

    topologies_to_test = [
        ("Default (Star Raw)", None),
        ("Star Raw", "boeing_star_raw"),
        ("Star Scaled", "boeing_star_scaled"),
        ("Star Eq1", "boeing_star_eq1"),
        ("Star Eq2", "boeing_star_eq2"),
        ("Boeing Chain", "boeing_chain"),
        ("Boeing Beam (Ladder)", "boeing_beam"),
    ]

    best_topo = None
    best_max_err = 1e9
    best_results = []
    
    for topo_name, topo_key in topologies_to_test:
        # Rebuild fasteners with forced topology
        fasteners = []
        for f_data in fasteners_data:
            # We must override topology if specified, else keep config's (which is likely None)
            # But wait, config might have none.
            # We want to FORCE topology.
            f_new = FastenerRow(
                row=f_data["row"],
                D=f_data["D"],
                Eb=f_data["Eb"],
                nu_b=f_data.get("nu_b", 0.3),
                method=f_data.get("method", "Boeing69"),
                connections=f_data.get("connections"),
                topology=topo_key, 
            )
            fasteners.append(f_new)

        joint = Joint1D(pitches, plates_base, fasteners)
        
        try:
            solution = joint.solve(supports)
        except Exception as e:
            # print(f"  {topo_name}: SOLVER ERROR: {e}")
            continue

        current_max_err = 0.0
        # Calculate error
        plate_names = {i: p.name for i, p in enumerate(solution.plates)}
        
        for f in solution.fasteners:
            p1 = plate_names[f.plate_i]
            p2 = plate_names[f.plate_j]
            ref_val = ref_map.get((f.row, p1, p2))
            
            if ref_val is not None:
                calc = abs(f.force)
                # Only count large violations or significant loads?
                # D06 failure criteria is <1%
                if ref_val > 0.01:
                    err = abs(calc - ref_val) / ref_val * 100.0
                else:
                    err = 0.0 if calc < 0.01 else 100.0
                current_max_err = max(current_max_err, err)
        
        if current_max_err < best_max_err:
            best_max_err = current_max_err
            best_topo = topo_name
        
        print(f"  {topo_name:<20}: Max Error = {current_max_err:.2f}%")
        if "D06_3" in config_path:
             print("    Detailed D06_3 Results:")
             for f in solution.fasteners:
                p1 = plate_names[f.plate_i]
                p2 = plate_names[f.plate_j]
                ref_val = ref_map.get((f.row, p1, p2))
                if ref_val is not None:
                     calc = abs(f.force)
                     err = (abs(calc - ref_val) / ref_val * 100.0) if ref_val > 0.01 else 0.0
                     
                     # Get Stiffness Ref
                     k_ref = rf.get("stiffness", 0.0)
                     k_calc = f.stiffness
                     k_err = (abs(k_calc - k_ref) / k_ref * 100.0) if k_ref > 1.0 else 0.0
                     
                     print(f"    R{f.row} {p1[:6]}-{p2[:6]}: Force Ref={ref_val:.1f} Calc={calc:.1f} (Err={err:.1f}%) | Stiff Ref={k_ref:.0f} Calc={k_calc:.0f} (Err={k_err:.1f}%)")


    print(f"  -> BEST MATCH: {best_topo} with {best_max_err:.2f}% error")
    
    if best_max_err < 1.05: # Allow rounding
        return True
    return False

def main():
    cases = [
        ("test_values/D06_2_config.json", "test_values/D06_2_reference.json"),
        ("test_values/D06_3_config.json", "test_values/D06_3_reference.json"),
        ("test_values/D06_4_config.json", "test_values/D06_4_reference.json"),
        ("test_values/D06_5_config.json", "test_values/D06_5_reference.json"),
    ]
    
    results = {}
    for cfg, ref in cases:
        if os.path.exists(cfg) and os.path.exists(ref):
            results[cfg] = verify_single_case(cfg, ref)
            print("-" * 60)
        else:
            print(f"Skipping {cfg} (file missing)")
    
    print("\nSUMMARY:")
    for cfg, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{os.path.basename(cfg)}: {status}")

if __name__ == "__main__":
    main()
