"""
Error Comparator for Verification Module.

Computes error statistics between solver outputs and reference values.
Provides per-element and aggregate metrics for validation.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FieldComparison:
    """
    Comparison results for a single field across all elements.
    
    Attributes:
        field_name: Name of the compared field (e.g., 'force', 'displacement')
        model_values: List of values from solver
        ref_values: List of reference values
        abs_errors: Absolute errors per element
        rel_errors: Relative errors (%) per element
        signs: Error signs ('+' or '-') per element
        max_abs: Maximum absolute error
        mean_abs: Mean absolute error
        max_rel: Maximum relative error (%)
        mean_rel: Mean relative error (%)
        rms: Root mean square error
        count: Number of elements compared
    """
    field_name: str
    model_values: List[float] = field(default_factory=list)
    ref_values: List[float] = field(default_factory=list)
    abs_errors: List[float] = field(default_factory=list)
    rel_errors: List[float] = field(default_factory=list)
    signs: List[str] = field(default_factory=list)
    max_abs: float = 0.0
    mean_abs: float = 0.0
    max_rel: float = 0.0
    mean_rel: float = 0.0
    rms: float = 0.0
    count: int = 0
    
    @property
    def within_tolerance(self) -> bool:
        """Check if all errors are within default tolerances."""
        return self.max_rel <= 1.0  # 1% default tolerance


@dataclass 
class CategoryComparison:
    """
    Comparison results for a category (nodes, fasteners, plates, loads).
    
    Attributes:
        category: Category name (e.g., 'fasteners', 'nodes')
        fields: Dict mapping field name to FieldComparison
        element_keys: Keys identifying each element (for table rows)
        matched_count: Number of elements matched between model and reference
        unmatched_model: Elements in model but not in reference
        unmatched_ref: Elements in reference but not in model
    """
    category: str
    fields: Dict[str, FieldComparison] = field(default_factory=dict)
    element_keys: List[str] = field(default_factory=list)
    matched_count: int = 0
    unmatched_model: List[str] = field(default_factory=list)
    unmatched_ref: List[str] = field(default_factory=list)


@dataclass
class ModelComparison:
    """
    Complete comparison results for a model/formula combination.
    
    Attributes:
        model_id: Test case identifier
        formula: Formula used
        categories: Dict mapping category name to CategoryComparison
        overall_max_rel_error: Maximum relative error across all fields
        overall_pass: True if all fields within tolerance
    """
    model_id: str
    formula: str
    categories: Dict[str, CategoryComparison] = field(default_factory=dict)
    overall_max_rel_error: float = 0.0
    overall_pass: bool = True


# =============================================================================
# Default Tolerances
# =============================================================================

DEFAULT_TOLERANCES = {
    "force": {"abs": 10.0, "rel": 10.0},       # ±10 lb or ±10%
    "load": {"abs": 20.0, "rel": 15.0},        # ±20 lb or ±15% for incoming/bypass/transfer
    "stiffness": {"abs": 2000.0, "rel": 2.0},  # ±2000 lb/in or ±2.0%
    "displacement": {"abs": 1e-4, "rel": 200.0}, # High rel tol for near-zero values
    "stress": {"abs": 100.0, "rel": 10.0},     # ±100 psi or ±10.0%
    "bypass": {"abs": 20.0, "rel": 15.0},      # Same as load
    "default": {"abs": 10.0, "rel": 15.0},     # Default tolerance
}


def get_tolerance(field_name: str, metric: str = "rel") -> float:
    """
    Get the tolerance for a field.
    
    Args:
        field_name: Name of the field
        metric: 'abs' for absolute, 'rel' for relative (%)
        
    Returns:
        Tolerance value
    """
    # Normalize field name
    normalized = field_name.lower()
    
    for key, tols in DEFAULT_TOLERANCES.items():
        if key in normalized:
            return tols.get(metric, tols.get("rel", 1.0))
            
    return DEFAULT_TOLERANCES["default"].get(metric, 1.0)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_field(
    field_name: str,
    model_values: List[float],
    ref_values: List[float],
) -> FieldComparison:
    """
    Compare a single field across all elements.
    
    Args:
        field_name: Name of the field being compared
        model_values: Values from solver
        ref_values: Reference values
        
    Returns:
        FieldComparison with all error metrics
    """
    comparison = FieldComparison(
        field_name=field_name,
        model_values=list(model_values),
        ref_values=list(ref_values),
        count=min(len(model_values), len(ref_values)),
    )
    
    if comparison.count == 0:
        return comparison
        
    abs_errors = []
    rel_errors = []
    signs = []
    abs_model_values = []
    abs_ref_values = []
    
    for i in range(comparison.count):
        # Use absolute values for comparison (handles sign differences)
        model_val = abs(model_values[i])
        ref_val = abs(ref_values[i])
        
        abs_model_values.append(model_val)
        abs_ref_values.append(ref_val)
        
        # Absolute error
        abs_err = model_val - ref_val
        abs_errors.append(abs(abs_err))
        
        # Relative error (%)
        if ref_val > 1e-12:
            rel_err = 100.0 * abs(abs_err) / ref_val
        else:
            rel_err = 0.0 if abs(abs_err) < 1e-12 else 100.0
        rel_errors.append(rel_err)
        
        # Error sign
        signs.append("+" if abs_err >= 0 else "-")
    
    # Store absolute values for display
    comparison.model_values = abs_model_values
    comparison.ref_values = abs_ref_values
    comparison.abs_errors = abs_errors
    comparison.rel_errors = rel_errors
    comparison.signs = signs
    
    # Aggregate metrics
    comparison.max_abs = max(abs_errors) if abs_errors else 0.0
    comparison.mean_abs = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    comparison.max_rel = max(rel_errors) if rel_errors else 0.0
    comparison.mean_rel = sum(rel_errors) / len(rel_errors) if rel_errors else 0.0
    
    # RMS error
    if abs_errors:
        comparison.rms = math.sqrt(sum(e**2 for e in abs_errors) / len(abs_errors))
        
    return comparison


def compare_fasteners(
    model_fasteners: List[Dict[str, Any]],
    ref_fasteners: List[Dict[str, Any]],
) -> CategoryComparison:
    """
    Compare fastener results between model and reference.
    
    Matches fasteners by (row, plate_i, plate_j) key.
    """
    comparison = CategoryComparison(category="fasteners")
    
    # Build lookup by key
    def make_key(f: Dict) -> str:
        plates = sorted([f.get("plate_i", ""), f.get("plate_j", "")])
        return f"{f.get('row', 0)}-{plates[0]}-{plates[1]}"
    
    model_by_key = {make_key(f): f for f in model_fasteners}
    ref_by_key = {make_key(f): f for f in ref_fasteners}
    
    # Find matched keys
    model_keys = set(model_by_key.keys())
    ref_keys = set(ref_by_key.keys())
    matched_keys = sorted(model_keys & ref_keys)
    
    comparison.element_keys = matched_keys
    comparison.matched_count = len(matched_keys)
    comparison.unmatched_model = sorted(model_keys - ref_keys)
    comparison.unmatched_ref = sorted(ref_keys - model_keys)
    
    # Compare each field
    fields_to_compare = ["force", "stiffness", "bearing_upper", "bearing_lower"]
    
    for field_name in fields_to_compare:
        model_vals = []
        ref_vals = []
        
        for key in matched_keys:
            model_f = model_by_key[key]
            ref_f = ref_by_key[key]
            
            model_val = model_f.get(field_name, 0.0)
            ref_val = ref_f.get(field_name)
            
            if ref_val is not None:
                model_vals.append(float(model_val) if model_val else 0.0)
                ref_vals.append(float(ref_val))
                
        if model_vals and ref_vals:
            comparison.fields[field_name] = compare_field(field_name, model_vals, ref_vals)
            
    return comparison


def compare_nodes(
    model_nodes: List[Dict[str, Any]],
    ref_nodes: List[Dict[str, Any]],
) -> CategoryComparison:
    """
    Compare node results between model and reference.
    
    Matches nodes by (plate_name, row) key.
    """
    comparison = CategoryComparison(category="nodes")
    
    def make_key(n: Dict) -> str:
        return f"{n.get('plate_name', '')}-{n.get('row', 0)}"
    
    model_by_key = {make_key(n): n for n in model_nodes}
    ref_by_key = {make_key(n): n for n in ref_nodes}
    
    model_keys = set(model_by_key.keys())
    ref_keys = set(ref_by_key.keys())
    matched_keys = sorted(model_keys & ref_keys)
    
    comparison.element_keys = matched_keys
    comparison.matched_count = len(matched_keys)
    comparison.unmatched_model = sorted(model_keys - ref_keys)
    comparison.unmatched_ref = sorted(ref_keys - model_keys)
    
    fields_to_compare = ["displacement"]  # net_bypass is covered in loads table
    
    for field_name in fields_to_compare:
        model_vals = []
        ref_vals = []
        
        for key in matched_keys:
            model_n = model_by_key[key]
            ref_n = ref_by_key[key]
            
            model_val = model_n.get(field_name, 0.0)
            ref_val = ref_n.get(field_name)
            
            if ref_val is not None:
                model_vals.append(float(model_val) if model_val else 0.0)
                ref_vals.append(float(ref_val))
                
        if model_vals and ref_vals:
            comparison.fields[field_name] = compare_field(field_name, model_vals, ref_vals)
            
    return comparison


def compare_plates(
    model_plates: List[Dict[str, Any]],
    ref_plates: List[Dict[str, Any]],
) -> CategoryComparison:
    """
    Compare plate/bar results between model and reference.
    
    Matches by (plate_name, segment) key.
    """
    comparison = CategoryComparison(category="plates")
    
    def make_key(p: Dict) -> str:
        return f"{p.get('plate_name', '')}-{p.get('segment', 0)}"
    
    model_by_key = {make_key(p): p for p in model_plates}
    ref_by_key = {make_key(p): p for p in ref_plates}
    
    model_keys = set(model_by_key.keys())
    ref_keys = set(ref_by_key.keys())
    matched_keys = sorted(model_keys & ref_keys)
    
    comparison.element_keys = matched_keys
    comparison.matched_count = len(matched_keys)
    comparison.unmatched_model = sorted(model_keys - ref_keys)
    comparison.unmatched_ref = sorted(ref_keys - model_keys)
    
    fields_to_compare = ["axial_force", "stiffness"]
    
    for field_name in fields_to_compare:
        model_vals = []
        ref_vals = []
        
        for key in matched_keys:
            model_p = model_by_key[key]
            ref_p = ref_by_key[key]
            
            model_val = model_p.get(field_name, 0.0)
            ref_val = ref_p.get(field_name)
            
            if ref_val is not None:
                model_vals.append(float(model_val) if model_val else 0.0)
                ref_vals.append(float(ref_val))
                
        if model_vals and ref_vals:
            comparison.fields[field_name] = compare_field(field_name, model_vals, ref_vals)
            
    return comparison


def compare_loads(
    model_loads: List[Dict[str, Any]],
    ref_loads: List[Dict[str, Any]],
) -> CategoryComparison:
    """
    Compare classic results (loads table) between model and reference.
    
    Matches by (element, row) key.
    """
    comparison = CategoryComparison(category="loads")
    
    def make_key(ld: Dict) -> str:
        return f"{ld.get('element', '')}-{ld.get('row', 0)}"
    
    model_by_key = {make_key(ld): ld for ld in model_loads}
    ref_by_key = {make_key(ld): ld for ld in ref_loads}
    
    model_keys = set(model_by_key.keys())
    ref_keys = set(ref_by_key.keys())
    matched_keys = sorted(model_keys & ref_keys)
    
    comparison.element_keys = matched_keys
    comparison.matched_count = len(matched_keys)
    comparison.unmatched_model = sorted(model_keys - ref_keys)
    comparison.unmatched_ref = sorted(ref_keys - model_keys)
    
    fields_to_compare = ["incoming_load", "bypass_load", "load_transfer", "detail_stress", "bearing_stress"]
    
    for field_name in fields_to_compare:
        model_vals = []
        ref_vals = []
        
        for key in matched_keys:
            model_ld = model_by_key[key]
            ref_ld = ref_by_key[key]
            
            model_val = model_ld.get(field_name, 0.0)
            ref_val = ref_ld.get(field_name)
            
            if ref_val is not None:
                model_vals.append(float(model_val) if model_val else 0.0)
                ref_vals.append(float(ref_val))
                
        if model_vals and ref_vals:
            comparison.fields[field_name] = compare_field(field_name, model_vals, ref_vals)
            
    return comparison


def compare_results(
    model_id: str,
    formula: str,
    solver_results: Dict[str, List[Dict[str, Any]]],
    reference_data: Dict[str, List[Dict[str, Any]]],
) -> ModelComparison:
    """
    Complete comparison of solver results against reference.
    
    Args:
        model_id: Test case identifier
        formula: Formula name
        solver_results: Dict with keys 'nodes', 'plates', 'fasteners', 'loads'
        reference_data: Reference data dict with same structure
        
    Returns:
        ModelComparison with all category comparisons
    """
    comparison = ModelComparison(model_id=model_id, formula=formula)
    
    # Compare each category
    if "fasteners" in reference_data and reference_data["fasteners"]:
        comparison.categories["fasteners"] = compare_fasteners(
            solver_results.get("fasteners", []),
            reference_data["fasteners"],
        )
        
    if "nodes" in reference_data and reference_data["nodes"]:
        comparison.categories["nodes"] = compare_nodes(
            solver_results.get("nodes", []),
            reference_data["nodes"],
        )
        
    if "plates" in reference_data and reference_data["plates"]:
        comparison.categories["plates"] = compare_plates(
            solver_results.get("plates", []),
            reference_data["plates"],
        )
        
    if "loads" in reference_data and reference_data["loads"]:
        comparison.categories["loads"] = compare_loads(
            solver_results.get("loads", []),
            reference_data["loads"],
        )
    
    # Calculate overall metrics
    max_rel = 0.0
    all_pass = True
    has_critical_error = False
    
    for cat in comparison.categories.values():
        for field_comp in cat.fields.values():
            if field_comp.max_rel > max_rel:
                max_rel = field_comp.max_rel
            # Check tolerance-based pass/fail
            tolerance = get_tolerance(field_comp.field_name, "rel")
            if field_comp.max_rel > tolerance:
                all_pass = False
            # Critical threshold: 100% error always fails
            if field_comp.max_rel >= 99.0:
                has_critical_error = True
                all_pass = False
                
    comparison.overall_max_rel_error = max_rel
    comparison.overall_pass = all_pass and not has_critical_error
    
    return comparison


__all__ = [
    "FieldComparison",
    "CategoryComparison", 
    "ModelComparison",
    "DEFAULT_TOLERANCES",
    "get_tolerance",
    "compare_field",
    "compare_fasteners",
    "compare_nodes",
    "compare_plates",
    "compare_loads",
    "compare_results",
]
