"""Investigate Boeing Star (Scaled Double Shear) Topology.

Issue: Similar load values but significantly different fastener stiffnesses.
"""
import copy
from pathlib import Path
import json

from jolt import load_joint_from_json
from jolt.model import Joint1D, Plate, FastenerRow

def investigate_3_plate_case():
    """Investigate with a 3-plate double shear case."""
    print("=" * 60)
    print("3 PLATE DOUBLE SHEAR INVESTIGATION")
    print("=" * 60)
    
    # Create a simple 3-plate joint (Plate A | Plate B | Plate A)
    # This is double shear configuration
    pitches = [1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_left=0.0, Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.10, first_row=1, last_row=2, A_strip=[0.10], Fx_left=0.0, Fx_right=0.0),
        Plate(name="C", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_left=0.0, Fx_right=100.0),
    ]
    fastener = FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69")
    supports = [(1, 0, 0.0)]

    # Test all star-based topologies
    topologies = ["boeing_star_raw", "boeing_star_scaled", "boeing_chain", "empirical_star"]
    
    print("\n" + "-" * 60)
    print(f"{'Topology':<25} | {'Loads':<20} | {'Stiffnesses':<30}")
    print("-" * 60)
    
    for topo in topologies:
        model = Joint1D(pitches=pitches, plates=copy.deepcopy(plates), fasteners=[copy.deepcopy(fastener)])
        model.fasteners[0].topology = topo
        sol = model.solve(supports=supports)
        
        loads = [abs(f.force) for f in sol.fasteners]
        stiffnesses = [f.stiffness for f in sol.fasteners]
        
        print(f"{topo:<25} | {str(loads):<20} | {str([f'{s:.2e}' for s in stiffnesses]):<30}")
        
        # Print detailed fastener info
        print(f"  Fasteners: {len(sol.fasteners)}")
        for i, f in enumerate(sol.fasteners):
            print(f"    [{i}] plates ({f.plate_i}->{f.plate_j}): force={f.force:.2f}, k={f.stiffness:.2e}, C={f.compliance:.4e}")
        print()

def investigate_case_5():
    """Investigate with the actual Case_5 file if available."""
    print("\n" + "=" * 60)
    print("CASE 5 INVESTIGATION (3-element, Row A)")
    print("=" * 60)
    
    case_path = Path(__file__).resolve().parent / "Case_5_3_elements_row_a.json"
    if not case_path.exists():
        print(f"Case file not found: {case_path}")
        return
    
    model, supports, point_forces, _ = load_joint_from_json(case_path)
    
    topologies = ["boeing_star_raw", "boeing_star_scaled", "boeing_chain"]
    
    print("\n" + "-" * 80)
    print(f"{'Topology':<25} | {'Total Load Sum':<15} | Number of Fasteners")
    print("-" * 80)
    
    for topo in topologies:
        test_model = copy.deepcopy(model)
        for fast in test_model.fasteners:
            fast.topology = topo
        sol = test_model.solve(supports=supports, point_forces=point_forces)
        
        loads = [abs(f.force) for f in sol.fasteners]
        stiffnesses = [f.stiffness for f in sol.fasteners]
        
        print(f"\n{topo}")
        print(f"  Loads:      {loads}")
        print(f"  Stiffness:  {[f'{s:.2e}' for s in stiffnesses]}")
        
        # Check for stiffness variation
        if stiffnesses:
            min_k = min(stiffnesses)
            max_k = max(stiffnesses)
            ratio = max_k / min_k if min_k > 0 else float('inf')
            print(f"  Stiffness range: {min_k:.2e} - {max_k:.2e} (ratio: {ratio:.2f})")

def analyze_star_scaled_assembly():
    """Deep dive into _assemble_boeing_star_scaled logic."""
    print("\n" + "=" * 60)
    print("DEEP DIVE: _assemble_boeing_star_scaled")
    print("=" * 60)
    
    # Same 3-plate case
    pitches = [1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_left=0.0, Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.10, first_row=1, last_row=2, A_strip=[0.10], Fx_left=0.0, Fx_right=0.0),
        Plate(name="C", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_left=0.0, Fx_right=100.0),
    ]
    fastener = FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69")
    
    model = Joint1D(pitches=pitches, plates=copy.deepcopy(plates), fasteners=[copy.deepcopy(fastener)])
    
    # Manually trace the _assemble_boeing_star_scaled logic
    plate_lookup = {i: p for i, p in enumerate(plates)}
    ordered_plates = [0, 1, 2]  # A, B, C
    row_index = 1
    
    # Calculate chain compliance
    total_comp = 0.0
    for i in range(len(ordered_plates) - 1):
        idx_i = ordered_plates[i]
        idx_j = ordered_plates[i + 1]
        plate_i = plate_lookup[idx_i]
        plate_j = plate_lookup[idx_j]
        t_i = plate_i.t
        t_j = plate_j.t
        
        c_ij = model._calculate_compliance_pairwise(plate_i, plate_j, fastener, t_i, t_j)
        print(f"  Pairwise compliance C({plate_i.name}-{plate_j.name}): {c_ij:.4e}")
        total_comp += c_ij
    
    print(f"\n  Total chain compliance: {total_comp:.4e}")
    
    # Boeing star scaled logic:
    effective_compliance = total_comp * 2.0  # Double for "double shear"
    print(f"  Effective compliance (x2): {effective_compliance:.4e}")
    
    total_stiffness = 1.0 / effective_compliance
    print(f"  Total stiffness: {total_stiffness:.2e}")
    
    k_branch = total_stiffness
    print(f"  k_branch (same as total): {k_branch:.2e}")
    
    c_branch = 1.0 / k_branch
    print(f"  c_branch: {c_branch:.4e}")
    
    n_pairs = len(ordered_plates) - 1
    k_elem = total_stiffness / max(n_pairs, 1)
    print(f"  k_elem (total / n_pairs={n_pairs}): {k_elem:.2e}")
    
    print("\n  NOTE: branch_compliances = [c_branch for _ in ordered_plates]")
    print(f"  So all {len(ordered_plates)} branches get: c_branch = {c_branch:.4e}")
    print(f"  Which means k_branch = {k_branch:.2e} for ALL branches")
    
    print("\n  ISSUE IDENTIFIED:")
    print("  The _interface_properties stores (1/k_elem, k_elem) for reporting,")
    print("  but the actual star assembly uses uniform c_branch for ALL plates.")
    print("  This creates a MISMATCH between reported interface stiffness")
    print("  and actual branch stiffness used in the FEM model.")

def main():
    investigate_3_plate_case()
    investigate_case_5()
    analyze_star_scaled_assembly()

if __name__ == "__main__":
    main()
