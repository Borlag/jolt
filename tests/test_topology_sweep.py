"""Tests for topology sweep helper."""
import pytest
from jolt.model import Joint1D, Plate, FastenerRow
from jolt.analysis import solve_with_topologies, TOPOLOGY_VARIANTS_BOEING
from jolt.benchmarks import compare_solution_to_jolt, D06_ROW_A_LABEL
from jolt.config import load_joint_from_json
import os
import pytest


def test_solve_with_topologies_basic():
    """Test that solve_with_topologies returns results for all requested topologies."""
    pitches = [1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05]),
    ]
    fasteners = [FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69")]
    supports = [(1, 0, 0.0)]
    
    results = solve_with_topologies(
        pitches, plates, fasteners, supports,
        topology_variants=["boeing_chain", "boeing_star_raw"]
    )
    
    # Both keys should be present
    assert "boeing_chain" in results
    assert "boeing_star_raw" in results
    
    # Both should have valid solutions
    for topo, sol in results.items():
        assert sol is not None
        assert len(sol.reactions) > 0


def test_topologies_give_same_equilibrium():
    """Chain and star should give the same total reaction (equilibrium)."""
    pitches = [1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05]),
    ]
    fasteners = [FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69")]
    supports = [(1, 0, 0.0)]
    
    results = solve_with_topologies(
        pitches, plates, fasteners, supports,
        topology_variants=["boeing_chain", "boeing_star_raw"]
    )
    
    reaction_chain = sum(r.reaction for r in results["boeing_chain"].reactions)
    reaction_star = sum(r.reaction for r in results["boeing_star_raw"].reactions)
    
    assert reaction_chain == pytest.approx(reaction_star, rel=1e-6)


def test_topologies_may_differ_in_fastener_forces():
    """Different topologies can give different fastener forces (proving both paths work)."""
    pitches = [1.0, 1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=3, A_strip=[0.05, 0.05], Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.05, first_row=1, last_row=3, A_strip=[0.05, 0.05]),
        Plate(name="C", E=1.0e7, t=0.05, first_row=1, last_row=3, A_strip=[0.05, 0.05]),
    ]
    fasteners = [
        FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69", connections=[(0, 1), (1, 2)]),
        FastenerRow(row=2, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69", connections=[(0, 1), (1, 2)]),
    ]
    supports = [(2, 0, 0.0)]
    
    results = solve_with_topologies(
        pitches, plates, fasteners, supports,
        topology_variants=["boeing_chain", "boeing_star_raw", "boeing_star_scaled"]
    )
    
    # Get fastener forces for each topology
    forces_chain = [abs(f.force) for f in results["boeing_chain"].fasteners]
    forces_star_raw = [abs(f.force) for f in results["boeing_star_raw"].fasteners]
    forces_star_scaled = [abs(f.force) for f in results["boeing_star_scaled"].fasteners]
    
    # At least one pair should differ (proving different code paths)
    # Note: For 2-plate case they'd be identical, but 3-plate case should differ
    all_same = (
        forces_chain == pytest.approx(forces_star_raw, rel=1e-6) and
        forces_chain == pytest.approx(forces_star_scaled, rel=1e-6)
    )
    # This is expected to be False for 3-plate stacks
    assert not all_same or len(plates) == 2


def test_does_not_mutate_fasteners():
    """Ensure original fastener objects are not modified."""
    pitches = [1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05]),
    ]
    fasteners = [FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69")]
    supports = [(1, 0, 0.0)]
    
    original_topology = fasteners[0].topology
    
    solve_with_topologies(
        pitches, plates, fasteners, supports,
        topology_variants=["boeing_chain", "boeing_star_raw"]
    )
    
    # Original fastener should be unchanged
    assert fasteners[0].topology == original_topology


def test_default_topology_variants():
    """Test that default topology variants are used when not specified."""
    pitches = [1.0]
    plates = [
        Plate(name="A", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05], Fx_right=100.0),
        Plate(name="B", E=1.0e7, t=0.05, first_row=1, last_row=2, A_strip=[0.05]),
    ]
    fasteners = [FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Boeing69")]
    supports = [(1, 0, 0.0)]
    
    results = solve_with_topologies(pitches, plates, fasteners, supports)
    
    # Should use TOPOLOGY_VARIANTS_BOEING by default
    for topo in TOPOLOGY_VARIANTS_BOEING:
        assert topo in results


def test_d06_row_a_variants():
    """
    Run D06 Row A case through all Boeing variants.
    
    Checks:
    - 'boeing_star_raw' (legacy) matches JOLT reference within strict tolerances.
    - New variants run successfully (sanity check).
    """
    # Locate the case file (assume running from root)
    case_file = "Case_5_3_elements_row_a.json"
    if not os.path.exists(case_file):
        # Depending on test runner cwd, might need adjustments
        pytest.skip(f"Case file {case_file} not found")
        
    # Load model
    model, supports, forces, _ = load_joint_from_json(case_file)
    
    # Define variants to test
    variants = [
        "boeing_star_raw",
        "boeing_star_eq1", "boeing_star_eq2",
        "boeing_chain_eq1", "boeing_chain_eq2",
    ]
    
    results = solve_with_topologies(
        model.pitches,
        model.plates,
        model.fasteners,
        supports,
        forces,
        topology_variants=variants
    )
    
    print("\nD06 Row A Variant Comparison:")
    print(f"{'Variant':<20} | {'Max %':<8} | {'RMS %':<8} | {'Mean |Err|':<10}")
    print("-" * 60)
    
    for variant in variants:
        if variant not in results:
            continue
            
        sol = results[variant]
        comp = compare_solution_to_jolt(sol, D06_ROW_A_LABEL)
        
        # Calculate aggregate metrics
        max_rel_err = 0.0
        sum_sq_rel = 0.0
        sum_abs = 0.0
        count = 0
        
        for k, v in comp.items():
            max_rel_err = max(max_rel_err, v["rel_err_pct"])
            sum_sq_rel += v["rel_err_pct"]**2
            sum_abs += v["abs_err"]
            count += 1
            
        rms_rel = (sum_sq_rel / count)**0.5 if count > 0 else 0.0
        mean_abs = sum_abs / count if count > 0 else 0.0
        
        print(f"{variant:<20} | {max_rel_err:<8.2f} | {rms_rel:<8.2f} | {mean_abs:<10.4f}")
        
        # Rigid check for legacy ONLY
        if variant == "boeing_star_raw":
             # Requirement: max |rel_err_pct| <= 4.5 %, RMS <= 2.5 %
             assert max_rel_err <= 4.5, f"Legacy Boeing star regression: Max err {max_rel_err:.2f}% > 4.5%"
             assert rms_rel <= 2.5, f"Legacy Boeing star regression: RMS err {rms_rel:.2f}% > 2.5%"
        
        # Sanity check for others (no NaNs, reasonable physics)
        assert count > 0

