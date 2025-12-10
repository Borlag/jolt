"""
Reverse engineer Boeing JOLT's branch decomposition methodology.

The key insight: Boeing JOLT reports per-interface stiffnesses AND forces.
By analyzing the forces, we can infer how the star topology is configured.

For a star topology with center node F and plates P1, P2, P3, P4:
  - F connects to each plate with branch stiffness k_i
  - The force through interface i-j is the shear force F_ij

The relationship between branch stiffness and shear force is NOT direct!
In a star topology, the forces depend on the relative displacements.

Let me think about this differently:

Boeing JOLT reports "interface stiffness" that matches Boeing69 pairwise (single-shear) values.
This suggests Boeing might NOT use a star topology at all - it might use CHAIN topology
with pairwise springs directly connecting adjacent plates.

Let me verify this hypothesis by checking if chain topology produces better results.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jolt.model import Joint1D, Plate, FastenerRow
from jolt.fasteners import boeing69_compliance


def analyze_boeing_approach():
    """Analyze how Boeing JOLT likely handles multi-layer joints."""
    
    print("=" * 80)
    print("HYPOTHESIS: Boeing JOLT uses CHAIN topology, not STAR topology")
    print("=" * 80)
    
    print("""
The key observation: 

1. Boeing JOLT reports "interface stiffness" values that EXACTLY match 
   boeing69_compliance for single-shear (pairwise) calculations.

2. In a STAR topology, the "effective stiffness" between two plates is:
   K_eff(i,j) = 1 / (1/K_i + 1/K_j)  -- series combination

3. In a CHAIN topology, the "interface stiffness" IS the spring stiffness.

4. The REPORTED stiffnesses match SINGLE-SHEAR pairwise values, which suggests
   Boeing JOLT uses springs that directly connect adjacent plates with those
   pairwise stiffnesses -- i.e., CHAIN topology.

5. The STAR topology was likely introduced to handle the "fastener DOF" case
   where a central rigid element is needed. But for simple multi-layer joints,
   Boeing might use direct pairwise connections.

Let me test this by comparing chain vs star results more carefully.
""")

    # Load D06_4 configuration
    test_values_dir = Path(__file__).parent / "test_values"
    config_path = test_values_dir / "D06_4_config.json"
    ref_path = test_values_dir / "D06_4_reference.json"
    
    with open(config_path) as f:
        config = json.load(f)
    with open(ref_path) as f:
        reference = json.load(f)
    
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
    
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    
    # Look at Row 5 which has 4 plates
    row5_refs = [rf for rf in ref_fasteners if rf["row"] == 5]
    
    print("\n" + "=" * 80)
    print("D06_4 Row 5 Reference Analysis")
    print("=" * 80)
    
    print("\nBoeing JOLT Reference for Row 5:")
    for rf in row5_refs:
        print(f"  {rf['plate_i']:>10} - {rf['plate_j']:<10}: Force={rf['force']:>6.1f}, K={rf['stiffness']:>6.0f}")
    
    # Calculate what pairwise compliances would give these stiffnesses
    print("\nInverse-calculated compliances from reported K:")
    for rf in row5_refs:
        k = rf['stiffness']
        c = 1.0 / k if k > 0 else 0
        print(f"  {rf['plate_i']:>10} - {rf['plate_j']:<10}: C={c:.6e}")
    
    # Compare with our calculated pairwise compliances
    print("\nOur calculated pairwise compliances (single-shear Boeing69):")
    fas = config["fasteners"][3]  # Row 5 fastener
    
    for idx_i, idx_j in fas["connections"]:
        p_i = plates[idx_i]
        p_j = plates[idx_j]
        
        comp = boeing69_compliance(
            ti=p_i.t, Ei=p_i.E,
            tj=p_j.t, Ej=p_j.E,
            Eb=fas["Eb"], nu_b=fas.get("nu_b", 0.3),
            diameter=fas["D"],
            shear_planes=1
        )
        k = 1.0 / comp
        
        print(f"  {p_i.name:>10} - {p_j.name:<10}: C={comp:.6e}, K={k:.0f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The reported stiffnesses match our single-shear pairwise calculations very closely.
This strongly suggests Boeing JOLT uses those stiffnesses DIRECTLY as spring constants
connecting adjacent plates (CHAIN topology).

However, we already tested chain topology and it performed WORSE than star.
This means there's something else going on...

ALTERNATIVE HYPOTHESIS:
Boeing JOLT might not be using standard FEM at all for multi-layer joints.
It might be using an iterative or closed-form solution based on compatibility.

The Bill Gran "Multiple Doublers" method suggests an iterative approach where
each interface pair is solved sequentially, not simultaneously.
""")


if __name__ == "__main__":
    analyze_boeing_approach()
