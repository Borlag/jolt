"""Boeing JOLT benchmark reference data.

This module contains reference fastener loads from Boeing JOLT for
validation purposes. It should NOT be imported by the core solver.

To remove benchmark functionality:
1. Delete this file
2. Remove the ENABLE_JOLT_BENCHMARKS flag from jolt/ui/__init__.py
3. Remove the benchmark import and expander from jolt/ui/comparison.py
"""
from typing import Dict, Tuple, Optional, Any

# Key: (case_label, row, plate_i_name, plate_j_name) - names sorted alphabetically
Key = Tuple[str, int, str, str]

D06_ROW_A_LABEL = "Case_5_3_elements_row_a"

# Reference fastener loads in pounds from Boeing JOLT D06 Row A
# Plate pair names are stored in sorted order for consistent lookup
JOLT_FASTENER_REFS: Dict[Key, float] = {
    (D06_ROW_A_LABEL, 2, "Doubler", "Skin"): 364.8,
    (D06_ROW_A_LABEL, 2, "Skin", "Strap"): 538.4,
    (D06_ROW_A_LABEL, 3, "Doubler", "Skin"): 371.0,
    (D06_ROW_A_LABEL, 3, "Skin", "Strap"): 461.6,
    (D06_ROW_A_LABEL, 4, "Doubler", "Skin"): 264.3,
}


def _normalize_plate_pair(plate_i: str, plate_j: str) -> Tuple[str, str]:
    """Return plate names in sorted order for consistent keys."""
    return tuple(sorted([plate_i, plate_j]))


def compare_solution_to_jolt(
    solution: Any,  # JointSolution - using Any to avoid circular import
    case_label: str,
    jolt_refs: Optional[Dict[Key, float]] = None,
) -> Dict[Tuple[int, str, str], Dict[str, float]]:
    """
    Compare solution fastener loads to JOLT reference values.
    
    For each fastener interface that exists both in solution and in jolt_refs,
    returns a dict:
        (row, plate_i, plate_j) -> {
            "ref": ref_load,
            "model": model_load,
            "abs_err": model_load - ref_load,
            "rel_err_pct": 100 * (model_load - ref_load) / ref_load,
        }
    
    Plate names are normalized (sorted pair) for consistent lookup.
    Interfaces not present in jolt_refs are quietly skipped.
    
    Args:
        solution: JointSolution object from solver
        case_label: Label identifying the benchmark case (e.g., D06_ROW_A_LABEL)
        jolt_refs: Optional dict of reference values; defaults to JOLT_FASTENER_REFS
    
    Returns:
        Dict mapping (row, plate_i, plate_j) to comparison metrics
    """
    if jolt_refs is None:
        jolt_refs = JOLT_FASTENER_REFS
    
    results: Dict[Tuple[int, str, str], Dict[str, float]] = {}
    
    for f in solution.fasteners:
        # Get plate names from solution
        plate_i_name = solution.plates[f.plate_i].name
        plate_j_name = solution.plates[f.plate_j].name
        
        # Normalize plate pair for lookup (alphabetically sorted)
        sorted_pair = _normalize_plate_pair(plate_i_name, plate_j_name)
        key = (case_label, f.row, sorted_pair[0], sorted_pair[1])
        
        ref_load = jolt_refs.get(key)
        if ref_load is None:
            continue  # Skip interfaces not in reference data
        
        model_load = abs(f.force)
        abs_err = model_load - ref_load
        rel_err_pct = 100.0 * abs_err / ref_load if abs(ref_load) > 1e-9 else 0.0
        
        result_key = (f.row, sorted_pair[0], sorted_pair[1])
        results[result_key] = {
            "ref": ref_load,
            "model": model_load,
            "abs_err": abs_err,
            "rel_err_pct": rel_err_pct,
        }
    
    return results


__all__ = [
    "D06_ROW_A_LABEL",
    "JOLT_FASTENER_REFS",
    "compare_solution_to_jolt",
]
