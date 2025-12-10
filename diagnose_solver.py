"""
Diagnostic script to compare solver output against Boeing JOLT references.
This script loads all D06 test configurations and compares with reference values.
"""
import json
import sys
from pathlib import Path

# Add jolt to path
sys.path.insert(0, str(Path(__file__).parent))

from jolt.model import Joint1D, Plate, FastenerRow


def load_config(config_path: Path):
    """Load a configuration file and create a Joint1D instance."""
    with open(config_path) as f:
        config = json.load(f)
    
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
    
    return joint, supports, config.get("label", config_path.stem)


def load_reference(ref_path: Path):
    """Load reference data."""
    with open(ref_path) as f:
        return json.load(f)


def compare_with_reference(solution, reference, label):
    """Compare solver output with reference data and print discrepancies."""
    results = []
    
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    
    # Create lookup: (row, plate_i_name, plate_j_name) -> ref_data
    ref_lookup = {}
    for rf in ref_fasteners:
        key = (rf["row"], rf["plate_i"], rf["plate_j"])
        ref_lookup[key] = rf
    
    # Plate name lookup
    plate_names = {i: p.name for i, p in enumerate(solution.plates)}
    
    # Compare fasteners
    print(f"\n{'='*80}")
    print(f"FASTENER COMPARISON: {label}")
    print(f"{'='*80}")
    print(f"{'Row':>4} {'Plate_i':>12} {'Plate_j':>12} {'Ref Force':>12} {'Model Force':>12} {'Δ Force':>10} {'Rel%':>8} {'Ref K':>10} {'Model K':>10}")
    print("-" * 104)
    
    total_abs_error = 0.0
    max_rel_error = 0.0
    matches = 0
    
    for f in solution.fasteners:
        plate_i_name = plate_names.get(f.plate_i, f"P{f.plate_i}")
        plate_j_name = plate_names.get(f.plate_j, f"P{f.plate_j}")
        
        # Try both orderings
        key1 = (f.row, plate_i_name, plate_j_name)
        key2 = (f.row, plate_j_name, plate_i_name)
        
        ref = ref_lookup.get(key1) or ref_lookup.get(key2)
        
        if ref:
            matches += 1
            ref_force = abs(ref.get("force", 0.0))
            model_force = abs(f.force)
            delta = model_force - ref_force
            rel_error = 100 * abs(delta) / ref_force if ref_force > 0.1 else 0.0
            
            ref_k = ref.get("stiffness", 0)
            model_k = f.stiffness
            
            total_abs_error += abs(delta)
            max_rel_error = max(max_rel_error, rel_error)
            
            flag = " " if rel_error < 1.0 else ("!" if rel_error < 5.0 else "!!!")
            
            print(f"{f.row:>4} {plate_i_name:>12} {plate_j_name:>12} {ref_force:>12.2f} {model_force:>12.2f} {delta:>+10.2f} {rel_error:>7.1f}% {ref_k:>10.0f} {model_k:>10.0f} {flag}")
            
            results.append({
                "row": f.row,
                "plate_i": plate_i_name,
                "plate_j": plate_j_name,
                "ref_force": ref_force,
                "model_force": model_force,
                "delta": delta,
                "rel_error_pct": rel_error,
                "ref_stiffness": ref_k,
                "model_stiffness": model_k,
            })
        else:
            print(f"{f.row:>4} {plate_i_name:>12} {plate_j_name:>12} {'N/A':>12} {abs(f.force):>12.2f} {'N/A':>10} {'N/A':>8} {'N/A':>10} {f.stiffness:>10.0f}")
    
    print("-" * 104)
    print(f"Matched: {matches}   Total Abs Error: {total_abs_error:.2f}   Max Rel Error: {max_rel_error:.1f}%")
    
    return results


def diagnose_all_cases():
    """Run diagnosis on all D06 test cases."""
    test_values_dir = Path(__file__).parent / "test_values"
    
    configs = sorted(test_values_dir.glob("D06*_config.json"))
    
    all_results = {}
    
    for config_path in configs:
        ref_path = config_path.with_name(config_path.name.replace("_config.json", "_reference.json"))
        
        if not ref_path.exists():
            print(f"Skipping {config_path.name}: no reference file found")
            continue
        
        try:
            joint, supports, label = load_config(config_path)
            solution = joint.solve(supports)
            reference = load_reference(ref_path)
            
            results = compare_with_reference(solution, reference, label)
            all_results[label] = results
            
        except Exception as e:
            print(f"ERROR processing {config_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for label, results in all_results.items():
        if results:
            avg_error = sum(r["rel_error_pct"] for r in results) / len(results)
            max_error = max(r["rel_error_pct"] for r in results)
            print(f"{label:>10}: Avg Error = {avg_error:6.2f}%  Max Error = {max_error:6.2f}%  ({len(results)} interfaces)")
        else:
            print(f"{label:>10}: No matches")


if __name__ == "__main__":
    diagnose_all_cases()
