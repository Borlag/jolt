"""
Test Case Loader for Verification Module.

Discovers and loads test cases from the test_values/ directory.
Automatically pairs {model_id}_config.json with {model_id}_reference.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import validate_config, validate_reference, ValidationError


@dataclass
class TestCase:
    """
    Represents a single test case with configuration and reference data.
    
    Attributes:
        model_id: Unique identifier for this test case
        config_path: Path to the configuration JSON file
        reference_path: Path to the reference JSON file
        config_data: Loaded configuration dictionary
        reference_data: Loaded reference dictionary
        formulas: List of formula names available in reference data
        errors: Any validation errors encountered during loading
    """
    model_id: str
    config_path: Path
    reference_path: Path
    config_data: Dict[str, Any] = field(default_factory=dict)
    reference_data: Dict[str, Any] = field(default_factory=dict)
    formulas: List[str] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Return True if the test case has no validation errors."""
        return len(self.errors) == 0
    
    @property
    def label(self) -> str:
        """Return a human-readable label for the test case."""
        return self.config_data.get("label", self.model_id)


class TestLoader:
    """
    Discovers and loads test cases from a directory.
    
    Usage:
        loader = TestLoader("test_values")
        cases = loader.discover()
        
        for case in cases:
            if case.is_valid:
                print(f"Loaded {case.model_id} with formulas: {case.formulas}")
    """
    
    def __init__(self, test_dir: str = "test_values"):
        """
        Initialize the loader.
        
        Args:
            test_dir: Path to directory containing test JSON files
        """
        self.test_dir = Path(test_dir)
        
    def discover(self) -> List[TestCase]:
        """
        Discover and load all test cases from the test directory.
        
        Returns:
            List of TestCase objects (may include invalid cases with errors)
        """
        cases: List[TestCase] = []
        
        if not self.test_dir.exists():
            return cases
            
        # Find all config files
        config_files = list(self.test_dir.glob("*_config.json"))
        
        for config_path in config_files:
            # Extract model_id from filename
            model_id = config_path.stem.replace("_config", "")
            
            # Look for matching reference file
            reference_path = self.test_dir / f"{model_id}_reference.json"
            
            if not reference_path.exists():
                # Create case with error about missing reference
                case = TestCase(
                    model_id=model_id,
                    config_path=config_path,
                    reference_path=reference_path,
                    errors=[ValidationError(
                        "reference_path",
                        f"Reference file not found: {reference_path}"
                    )]
                )
                cases.append(case)
                continue
                
            # Load and validate both files
            case = self._load_case(model_id, config_path, reference_path)
            cases.append(case)
            
        return cases
    
    def load_single(self, model_id: str) -> Optional[TestCase]:
        """
        Load a single test case by model ID.
        
        Args:
            model_id: The model identifier (e.g., "D06")
            
        Returns:
            TestCase if found, None otherwise
        """
        config_path = self.test_dir / f"{model_id}_config.json"
        reference_path = self.test_dir / f"{model_id}_reference.json"
        
        if not config_path.exists():
            return None
            
        if not reference_path.exists():
            return TestCase(
                model_id=model_id,
                config_path=config_path,
                reference_path=reference_path,
                errors=[ValidationError(
                    "reference_path",
                    f"Reference file not found: {reference_path}"
                )]
            )
            
        return self._load_case(model_id, config_path, reference_path)
    
    def _load_case(
        self, 
        model_id: str, 
        config_path: Path, 
        reference_path: Path
    ) -> TestCase:
        """
        Load and validate a test case from files.
        
        Args:
            model_id: Model identifier
            config_path: Path to configuration file
            reference_path: Path to reference file
            
        Returns:
            TestCase with loaded data and any validation errors
        """
        errors: List[ValidationError] = []
        config_data: Dict[str, Any] = {}
        reference_data: Dict[str, Any] = {}
        formulas: List[str] = []
        
        # Load config file
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            config_errors = validate_config(config_data)
            errors.extend(config_errors)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                "config_path",
                f"Invalid JSON in config file: {e}"
            ))
        except Exception as e:
            errors.append(ValidationError(
                "config_path",
                f"Error reading config file: {e}"
            ))
            
        # Load reference file
        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                reference_data = json.load(f)
            ref_errors = validate_reference(reference_data)
            errors.extend(ref_errors)
            
            # Extract available formulas
            if "formulas" in reference_data and isinstance(reference_data["formulas"], dict):
                formulas = list(reference_data["formulas"].keys())
                
            # Validate model_id match
            ref_model_id = reference_data.get("model_id", "")
            if ref_model_id and ref_model_id != model_id:
                errors.append(ValidationError(
                    "model_id",
                    f"model_id mismatch: file={model_id}, reference={ref_model_id}"
                ))
                
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                "reference_path",
                f"Invalid JSON in reference file: {e}"
            ))
        except Exception as e:
            errors.append(ValidationError(
                "reference_path",
                f"Error reading reference file: {e}"
            ))
            
        return TestCase(
            model_id=model_id,
            config_path=config_path,
            reference_path=reference_path,
            config_data=config_data,
            reference_data=reference_data,
            formulas=formulas,
            errors=errors,
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model IDs without fully loading them.
        
        Returns:
            List of model ID strings
        """
        if not self.test_dir.exists():
            return []
            
        config_files = list(self.test_dir.glob("*_config.json"))
        return [f.stem.replace("_config", "") for f in config_files]


def create_sample_reference(model_id: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create a sample reference JSON structure for a new test case.
    
    Args:
        model_id: Model identifier for the new case
        output_path: Optional path to write the sample file
        
    Returns:
        Sample reference dictionary
    """
    sample = {
        "model_id": model_id,
        "description": f"Boeing JOLT reference results for {model_id}",
        "source": "Boeing JOLT v2.x",
        "formulas": {
            "boeing": {
                "nodes": [
                    {
                        "plate_name": "Plate1",
                        "row": 1,
                        "displacement": 0.0,
                        "net_bypass": 0.0
                    }
                ],
                "plates": [
                    {
                        "plate_name": "Plate1",
                        "segment": 0,
                        "axial_force": 0.0,
                        "stiffness": 0.0
                    }
                ],
                "fasteners": [
                    {
                        "row": 1,
                        "plate_i": "Plate1",
                        "plate_j": "Plate2",
                        "force": 0.0,
                        "stiffness": 0.0,
                        "bearing_upper": 0.0,
                        "bearing_lower": 0.0
                    }
                ],
                "loads": [
                    {
                        "element": "Plate1",
                        "row": 1,
                        "incoming_load": 0.0,
                        "bypass_load": 0.0,
                        "load_transfer": 0.0
                    }
                ],
                "reactions": [
                    {
                        "plate_name": "Plate1",
                        "local_node": 0,
                        "reaction": 0.0
                    }
                ]
            }
        }
    }
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)
            
    return sample


__all__ = [
    "TestCase",
    "TestLoader",
    "create_sample_reference",
]
