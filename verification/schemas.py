"""
JSON Schema Definitions for Verification Module.

This module defines the strict JSON schemas for:
1. Reference result files (Boeing JOLT outputs)
2. Configuration validation

Units Convention (all values stored as):
    - Stiffness: lb/in
    - Displacement: in
    - Loads/Forces: lb
    - Stress: psi
    - Thickness: in
    - Area: in²
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Reference Schema Definition
# =============================================================================

REFERENCE_SCHEMA = {
    "type": "object",
    "required": ["model_id", "formulas"],
    "properties": {
        "model_id": {"type": "string", "description": "Must match config file model_id"},
        "description": {"type": "string"},
        "source": {"type": "string", "description": "Source of reference data (e.g., 'Boeing JOLT v2.1')"},
        "formulas": {
            "type": "object",
            "description": "Results keyed by formula name (boeing, huth, etc.)",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["plate_name", "row"],
                            "properties": {
                                "plate_name": {"type": "string"},
                                "row": {"type": "integer"},
                                "displacement": {"type": "number", "description": "in"},
                                "net_bypass": {"type": "number", "description": "lb"},
                            }
                        }
                    },
                    "plates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["plate_name", "segment"],
                            "properties": {
                                "plate_name": {"type": "string"},
                                "segment": {"type": "integer"},
                                "axial_force": {"type": "number", "description": "lb"},
                                "stiffness": {"type": "number", "description": "lb/in"},
                            }
                        }
                    },
                    "fasteners": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["row", "plate_i", "plate_j", "force"],
                            "properties": {
                                "row": {"type": "integer"},
                                "plate_i": {"type": "string"},
                                "plate_j": {"type": "string"},
                                "force": {"type": "number", "description": "lb"},
                                "stiffness": {"type": "number", "description": "lb/in"},
                                "bearing_upper": {"type": "number", "description": "lb"},
                                "bearing_lower": {"type": "number", "description": "lb"},
                            }
                        }
                    },
                    "loads": {
                        "type": "array",
                        "description": "Classic Results table entries",
                        "items": {
                            "type": "object",
                            "required": ["element", "row"],
                            "properties": {
                                "element": {"type": "string"},
                                "row": {"type": "integer"},
                                "incoming_load": {"type": "number", "description": "lb"},
                                "bypass_load": {"type": "number", "description": "lb"},
                                "load_transfer": {"type": "number", "description": "lb"},
                                "detail_stress": {"type": "number", "description": "psi"},
                                "bearing_stress": {"type": "number", "description": "psi"},
                            }
                        }
                    },
                    "reactions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["plate_name", "local_node", "reaction"],
                            "properties": {
                                "plate_name": {"type": "string"},
                                "local_node": {"type": "integer"},
                                "reaction": {"type": "number", "description": "lb"},
                            }
                        }
                    }
                }
            }
        }
    }
}


CONFIG_SCHEMA = {
    "type": "object",
    "description": "Standard JOLT configuration schema (already defined in jolt.config)",
}


# =============================================================================
# Validation Functions
# =============================================================================

@dataclass
class ValidationError:
    """Represents a schema validation error."""
    path: str
    message: str
    value: Any = None


def validate_reference(data: Dict[str, Any]) -> List[ValidationError]:
    """
    Validate reference data against the schema.
    
    Args:
        data: Reference JSON data dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors: List[ValidationError] = []
    
    # Check required fields
    if "model_id" not in data:
        errors.append(ValidationError("model_id", "Missing required field 'model_id'"))
    elif not isinstance(data["model_id"], str):
        errors.append(ValidationError("model_id", "model_id must be a string", data["model_id"]))
        
    if "formulas" not in data:
        errors.append(ValidationError("formulas", "Missing required field 'formulas'"))
    elif not isinstance(data["formulas"], dict):
        errors.append(ValidationError("formulas", "formulas must be an object", type(data["formulas"])))
    else:
        # Validate each formula
        for formula_name, formula_data in data["formulas"].items():
            if not isinstance(formula_data, dict):
                errors.append(ValidationError(
                    f"formulas.{formula_name}", 
                    f"Formula data must be an object",
                    type(formula_data)
                ))
                continue
                
            # Validate fasteners array
            if "fasteners" in formula_data:
                fasteners = formula_data["fasteners"]
                if not isinstance(fasteners, list):
                    errors.append(ValidationError(
                        f"formulas.{formula_name}.fasteners",
                        "fasteners must be an array",
                        type(fasteners)
                    ))
                else:
                    for i, f in enumerate(fasteners):
                        if not isinstance(f, dict):
                            errors.append(ValidationError(
                                f"formulas.{formula_name}.fasteners[{i}]",
                                "Fastener entry must be an object"
                            ))
                            continue
                        for req in ["row", "plate_i", "plate_j", "force"]:
                            if req not in f:
                                errors.append(ValidationError(
                                    f"formulas.{formula_name}.fasteners[{i}].{req}",
                                    f"Missing required field '{req}'"
                                ))
                                
            # Validate nodes array
            if "nodes" in formula_data:
                nodes = formula_data["nodes"]
                if not isinstance(nodes, list):
                    errors.append(ValidationError(
                        f"formulas.{formula_name}.nodes",
                        "nodes must be an array",
                        type(nodes)
                    ))
                else:
                    for i, n in enumerate(nodes):
                        if not isinstance(n, dict):
                            errors.append(ValidationError(
                                f"formulas.{formula_name}.nodes[{i}]",
                                "Node entry must be an object"
                            ))
                            continue
                        for req in ["plate_name", "row"]:
                            if req not in n:
                                errors.append(ValidationError(
                                    f"formulas.{formula_name}.nodes[{i}].{req}",
                                    f"Missing required field '{req}'"
                                ))
                                
            # Validate loads array
            if "loads" in formula_data:
                loads = formula_data["loads"]
                if not isinstance(loads, list):
                    errors.append(ValidationError(
                        f"formulas.{formula_name}.loads",
                        "loads must be an array",
                        type(loads)
                    ))
                else:
                    for i, ld in enumerate(loads):
                        if not isinstance(ld, dict):
                            errors.append(ValidationError(
                                f"formulas.{formula_name}.loads[{i}]",
                                "Load entry must be an object"
                            ))
                            continue
                        for req in ["element", "row"]:
                            if req not in ld:
                                errors.append(ValidationError(
                                    f"formulas.{formula_name}.loads[{i}].{req}",
                                    f"Missing required field '{req}'"
                                ))
    
    return errors


def validate_config(data: Dict[str, Any]) -> List[ValidationError]:
    """
    Validate configuration data against the schema.
    
    Args:
        data: Configuration JSON data dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors: List[ValidationError] = []
    
    # Check required fields for config
    required_fields = ["pitches", "plates", "fasteners", "supports"]
    for field in required_fields:
        if field not in data:
            errors.append(ValidationError(field, f"Missing required field '{field}'"))
            
    # Validate pitches
    if "pitches" in data and not isinstance(data["pitches"], list):
        errors.append(ValidationError("pitches", "pitches must be an array"))
        
    # Validate plates
    if "plates" in data:
        if not isinstance(data["plates"], list):
            errors.append(ValidationError("plates", "plates must be an array"))
        else:
            for i, p in enumerate(data["plates"]):
                if not isinstance(p, dict):
                    errors.append(ValidationError(f"plates[{i}]", "Plate must be an object"))
                elif "name" not in p:
                    errors.append(ValidationError(f"plates[{i}].name", "Missing plate name"))
                    
    # Validate fasteners
    if "fasteners" in data:
        if not isinstance(data["fasteners"], list):
            errors.append(ValidationError("fasteners", "fasteners must be an array"))
        else:
            for i, f in enumerate(data["fasteners"]):
                if not isinstance(f, dict):
                    errors.append(ValidationError(f"fasteners[{i}]", "Fastener must be an object"))
                elif "row" not in f:
                    errors.append(ValidationError(f"fasteners[{i}].row", "Missing fastener row"))
    
    return errors


# =============================================================================
# Formula Name Normalization
# =============================================================================

FORMULA_ALIASES = {
    "boeing": "Boeing",
    "boeing69": "Boeing",          # Deprecated - backward compat
    "boeing_69": "Boeing",         # Deprecated
    "huth": "Huth_metal",
    "huth_metal": "Huth_metal",
    "huth_metallic": "Huth_metal",
    "huth_graphite": "Huth_graphite",
    "grumman": "Grumman",
    "swift": "Swift_Douglas",
    "swift_douglas": "Swift_Douglas",
    "tate": "Tate_Rosenfeld",
    "tate_rosenfeld": "Tate_Rosenfeld",
    "morris": "Morris",
    "rutman": "Rutman",
}


def normalize_formula_name(name: str) -> str:
    """
    Normalize a formula name to the solver's expected format.
    
    Args:
        name: Formula name from reference file (e.g., 'boeing', 'huth')
        
    Returns:
        Normalized formula name for solver (e.g., 'Boeing', 'Huth_metal')
    """
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return FORMULA_ALIASES.get(normalized, name)


__all__ = [
    "REFERENCE_SCHEMA",
    "CONFIG_SCHEMA",
    "ValidationError",
    "validate_reference",
    "validate_config",
    "normalize_formula_name",
    "FORMULA_ALIASES",
]
