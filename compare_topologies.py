"""
Test: Does Boeing JOLT use chain or star topology?
This script compares results for Boeing chain vs star topologies.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jolt.model import Joint1D, Plate, FastenerRow


def load_config(config_path: Path):
    """Load a configuration file and create a Joint1D instance."""
    with open(config_path) as f:
        config = json.load(f)
    
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
        ))
    
    return config, plates


def load_reference(ref_path: Path):
    with open(ref_path) as f:
        return json.load(f)


def solve_with_topology(config, plates, topology_name):
    """Solve with a specific topology."""
    fasteners = []
    for f_data in config["fasteners"]:
        fasteners.append(FastenerRow(
            row=f_data["row"],
            D=f_data["D"],
            Eb=f_data["Eb"],
            nu_b=f_data.get("nu_b", 0.3),
            method=f_data.get("method", "Boeing69"),
            connections=f_data.get("connections"),
            topology=topology_name,  # Force specific topology
        ))
    
    joint = Joint1D(config["pitches"], plates, fasteners)
    supports = [(int(s[0]), int(s[1]), float(s[2])) for s in config["supports"]]
    
    return joint.solve(supports)


def compare_topologies(config_path, ref_path):
    """Compare chain vs star topologies against reference."""
    config, plates = load_config(config_path)
    reference = load_reference(ref_path)
    label = config.get("label", config_path.stem)
    
    # Try both topologies
    topologies = ["boeing_chain", "boeing_star_raw"]
    
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    
    # Create lookup
    ref_lookup = {}
    for rf in ref_fasteners:
        key = (rf["row"], tuple(sorted([rf["plate_i"], rf["plate_j"]])))
        ref_lookup[key] = rf
    
    plate_names = [p.name for p in plates]
    
    print(f"\n{'='*100}")
    print(f"TOPOLOGY COMPARISON: {label}")
    print(f"{'='*100}")
    
    for topo in topologies:
        try:
            solution = solve_with_topology(config, plates, topo)
            
            print(f"\n--- {topo.upper()} ---")
            print(f"{'Row':>4} {'Plate_i':>12} {'Plate_j':>12} {'Ref Force':>12} {'Model Force':>12} {'Δ Force':>10} {'Rel%':>8}")
            print("-" * 80)
            
            total_error = 0.0
            matches = 0
            
            for f in solution.fasteners:
                plate_i_name = plate_names[f.plate_i] if f.plate_i < len(plate_names) else f"P{f.plate_i}"
                plate_j_name = plate_names[f.plate_j] if f.plate_j < len(plate_names) else f"P{f.plate_j}"
                
                key = (f.row, tuple(sorted([plate_i_name, plate_j_name])))
                ref = ref_lookup.get(key)
                
                if ref:
                    matches += 1
                    ref_force = abs(ref.get("force", 0.0))
                    model_force = abs(f.force)
                    delta = model_force - ref_force
                    rel_error = 100 * abs(delta) / ref_force if ref_force > 0.1 else 0.0
                    
                    total_error += rel_error
                    
                    flag = " " if rel_error < 1.0 else ("!" if rel_error < 5.0 else "!!!")
                    print(f"{f.row:>4} {plate_i_name:>12} {plate_j_name:>12} {ref_force:>12.2f} {model_force:>12.2f} {delta:>+10.2f} {rel_error:>7.1f}%{flag}")
                else:
                    print(f"{f.row:>4} {plate_i_name:>12} {plate_j_name:>12} {'N/A':>12} {abs(f.force):>12.2f}")
            
            if matches > 0:
                avg_error = total_error / matches
                print(f"\n  Matched: {matches}, Avg Error: {avg_error:.2f}%")
                
        except Exception as e:
            print(f"\n--- {topo.upper()} ---")
            print(f"  ERROR: {e}")


def main():
    test_values_dir = Path(__file__).parent / "test_values"
    
    # Test all D06 variants
    configs = sorted(test_values_dir.glob("D06*_config.json"))
    
    for config_path in configs:
        ref_path = config_path.with_name(config_path.name.replace("_config.json", "_reference.json"))
        
        if ref_path.exists():
            compare_topologies(config_path, ref_path)


if __name__ == "__main__":
    main()
