"""
Test Runner for Verification Module.

Executes the joint solver for each test case and formula combination,
collecting results in a structured format for comparison.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Import from main jolt package
from jolt import JointConfiguration, Joint1D, Plate, FastenerRow, JointSolution
from jolt.config import load_joint_from_json

from .loader import TestCase
from .schemas import normalize_formula_name


@dataclass
class SolverResults:
    """
    Contains solver output structured for comparison with reference data.
    
    Attributes:
        model_id: Test case identifier
        formula: Formula used for this run
        success: Whether the solver completed successfully
        error_message: Error message if solver failed
        nodes: List of node result dictionaries
        plates: List of plate/bar result dictionaries
        fasteners: List of fastener result dictionaries
        loads: Classic results table data
        reactions: Reaction force data
        solution: Full JointSolution object (for additional analysis)
    """
    model_id: str
    formula: str
    success: bool = True
    error_message: str = ""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    plates: List[Dict[str, Any]] = field(default_factory=list)
    fasteners: List[Dict[str, Any]] = field(default_factory=list)
    loads: List[Dict[str, Any]] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    solution: Optional[JointSolution] = None


@dataclass
class TestResult:
    """
    Complete result for a single test case across all formulas.
    
    Attributes:
        case: The test case that was executed
        results: Dict mapping formula name to SolverResults
    """
    case: TestCase
    results: Dict[str, SolverResults] = field(default_factory=dict)


class TestRunner:
    """
    Executes the joint solver for verification test cases.
    
    Usage:
        runner = TestRunner()
        result = runner.run_case(test_case, formula="boeing")
        
        # Or run all formulas for a case
        test_result = runner.run_all_formulas(test_case)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the test runner.
        
        Args:
            verbose: If True, print progress messages
        """
        self.verbose = verbose
        
    def run_case(self, case: TestCase, formula: str) -> SolverResults:
        """
        Run the solver for a single test case with a specific formula.
        
        Args:
            case: TestCase containing configuration data
            formula: Formula name to use (e.g., 'boeing', 'huth')
            
        Returns:
            SolverResults containing solver outputs
        """
        result = SolverResults(
            model_id=case.model_id,
            formula=formula,
        )
        
        if not case.is_valid:
            result.success = False
            result.error_message = "; ".join(e.message for e in case.errors)
            return result
            
        try:
            # Build configuration with formula override
            config = self._build_config_with_formula(case.config_data, formula)
            
            # Build and solve the model
            model = config.build_model()
            solution = model.solve(
                supports=config.supports,
                point_forces=config.point_forces if config.point_forces else None,
            )
            
            # Extract results in comparison format
            result.nodes = self._extract_nodes(solution)
            result.plates = self._extract_plates(solution)
            result.fasteners = self._extract_fasteners(solution, config)
            result.loads = self._extract_loads(solution)
            result.reactions = self._extract_reactions(solution)
            result.solution = solution
            result.success = True
            
            if self.verbose:
                print(f"  ✓ {case.model_id}/{formula}: Solved successfully")
                
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if self.verbose:
                print(f"  ✗ {case.model_id}/{formula}: {e}")
                
        return result
    
    def run_all_formulas(self, case: TestCase) -> TestResult:
        """
        Run the solver for all formulas defined in the reference data.
        
        Args:
            case: TestCase to execute
            
        Returns:
            TestResult containing results for all formulas
        """
        test_result = TestResult(case=case)
        
        if self.verbose:
            print(f"\nRunning test case: {case.model_id}")
            
        for formula in case.formulas:
            solver_result = self.run_case(case, formula)
            test_result.results[formula] = solver_result
            
        return test_result
    
    def run_all(self, cases: List[TestCase]) -> List[TestResult]:
        """
        Run all test cases with all their formulas.
        
        Args:
            cases: List of test cases to execute
            
        Returns:
            List of TestResult objects
        """
        results: List[TestResult] = []
        
        for case in cases:
            if case.is_valid:
                result = self.run_all_formulas(case)
                results.append(result)
            elif self.verbose:
                print(f"\n⚠ Skipping invalid case: {case.model_id}")
                for error in case.errors:
                    print(f"  - {error.path}: {error.message}")
                    
        return results
    
    def _build_config_with_formula(
        self, 
        config_data: Dict[str, Any], 
        formula: str
    ) -> JointConfiguration:
        """
        Create a JointConfiguration with all fasteners using the specified formula.
        
        Args:
            config_data: Original configuration dictionary
            formula: Formula name to apply to all fasteners
            
        Returns:
            JointConfiguration with formula override
        """
        # Deep copy to avoid modifying original
        data = copy.deepcopy(config_data)
        
        # Normalize formula name to solver's expected format
        solver_formula = normalize_formula_name(formula)
        
        # Override formula for all fasteners
        if "fasteners" in data:
            for fastener in data["fasteners"]:
                fastener["method"] = solver_formula
                
        return JointConfiguration.from_dict(data)
    
    def _extract_nodes(self, solution: JointSolution) -> List[Dict[str, Any]]:
        """Extract node results in comparison format."""
        nodes = []
        for node in solution.nodes:
            nodes.append({
                "plate_name": node.plate_name,
                "row": node.row,
                "local_node": node.local_node,
                "displacement": node.displacement,
                "net_bypass": node.net_bypass,
                "x": node.x,
            })
        return nodes
    
    def _extract_plates(self, solution: JointSolution) -> List[Dict[str, Any]]:
        """Extract plate/bar results in comparison format."""
        plates = []
        for bar in solution.bars:
            plates.append({
                "plate_name": bar.plate_name,
                "segment": bar.segment,
                "axial_force": bar.axial_force,
                "stiffness": bar.stiffness,
                "modulus": bar.modulus,
            })
        return plates
    
    def _extract_fasteners(
        self, 
        solution: JointSolution, 
        config: JointConfiguration
    ) -> List[Dict[str, Any]]:
        """Extract fastener results in comparison format."""
        fasteners = []
        
        # Build plate name lookup
        plate_names = [p.name for p in config.plates]
        
        for f in solution.fasteners:
            plate_i_name = plate_names[f.plate_i] if f.plate_i < len(plate_names) else str(f.plate_i)
            plate_j_name = plate_names[f.plate_j] if f.plate_j < len(plate_names) else str(f.plate_j)
            
            fasteners.append({
                "row": f.row,
                "plate_i": plate_i_name,
                "plate_j": plate_j_name,
                "force": abs(f.force),
                "stiffness": f.stiffness,
                "bearing_upper": f.bearing_force_upper,
                "bearing_lower": f.bearing_force_lower,
                "compliance": f.compliance,
            })
        return fasteners
    
    def _extract_loads(self, solution: JointSolution) -> List[Dict[str, Any]]:
        """Extract classic results table data."""
        try:
            classic = solution.classic_results_as_dicts()
            loads = []
            for entry in classic:
                loads.append({
                    "element": entry.get("Element", ""),
                    "row": entry.get("Row", 0),
                    "incoming_load": entry.get("Incoming Load", 0.0),
                    "bypass_load": entry.get("Bypass Load", 0.0),
                    "load_transfer": entry.get("Load Transfer", 0.0),
                    "detail_stress": entry.get("Detail Stress", 0.0),
                    "bearing_stress": entry.get("Bearing Stress", 0.0),
                })
            return loads
        except Exception:
            return []
    
    def _extract_reactions(self, solution: JointSolution) -> List[Dict[str, Any]]:
        """Extract reaction force data."""
        reactions = []
        for r in solution.reactions:
            reactions.append({
                "plate_name": r.plate_name,
                "local_node": r.local_node,
                "reaction": r.reaction,
            })
        return reactions


__all__ = [
    "SolverResults",
    "TestResult", 
    "TestRunner",
]
