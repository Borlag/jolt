"""
Command-Line Interface for Verification Module.

Provides CLI commands for running verification tests and generating reports.

Usage:
    python run_verification.py [OPTIONS]
    
Options:
    --model MODEL       Run only specified model(s)
    --formula FORMULA   Run only specified formula (boeing, huth, etc.)
    --output PATH       Output Excel report path
    --verbose           Print detailed progress
    --list              List available test cases
    --create-sample ID  Create sample reference file for model ID
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .loader import TestLoader, TestCase, create_sample_reference
from .runner import TestRunner, TestResult
from .comparator import compare_results, ModelComparison
from .reporter import VerificationReporter, generate_markdown_summary


def run_verification(
    test_dir: str = "test_values",
    models: Optional[List[str]] = None,
    formula: Optional[str] = None,
    output_path: str = "reports/verification_report.xlsx",
    verbose: bool = False,
) -> List[ModelComparison]:
    """
    Run the verification suite.
    
    Args:
        test_dir: Path to test_values directory
        models: Optional list of model IDs to run (None = all)
        formula: Optional formula to use (None = all from reference)
        output_path: Path for Excel report
        verbose: Print progress messages
        
    Returns:
        List of ModelComparison results
    """
    # Load test cases
    loader = TestLoader(test_dir)
    
    if models:
        cases = [loader.load_single(m) for m in models]
        cases = [c for c in cases if c is not None]
    else:
        cases = loader.discover()
        
    if not cases:
        print(f"No test cases found in {test_dir}/")
        return []
        
    if verbose:
        print(f"Found {len(cases)} test case(s)")
        
    # Filter valid cases
    valid_cases = [c for c in cases if c.is_valid]
    invalid_cases = [c for c in cases if not c.is_valid]
    
    if invalid_cases and verbose:
        print(f"\nSkipping {len(invalid_cases)} invalid case(s):")
        for case in invalid_cases:
            print(f"  - {case.model_id}: {case.errors[0].message if case.errors else 'Unknown error'}")
            
    if not valid_cases:
        print("No valid test cases to run")
        return []
        
    # Run tests
    runner = TestRunner(verbose=verbose)
    comparisons: List[ModelComparison] = []
    reporter = VerificationReporter(output_path)
    
    for case in valid_cases:
        # Determine formulas to run
        formulas_to_run = [formula] if formula else case.formulas
        
        if not formulas_to_run:
            if verbose:
                print(f"No formulas defined for {case.model_id}")
            continue
            
        for f in formulas_to_run:
            if f not in case.formulas:
                if verbose:
                    print(f"Formula '{f}' not in reference for {case.model_id}")
                continue
                
            # Run solver
            solver_result = runner.run_case(case, f)
            
            if not solver_result.success:
                if verbose:
                    print(f"  ✗ Solver failed: {solver_result.error_message}")
                continue
                
            # Get reference data for this formula
            ref_data = case.reference_data.get("formulas", {}).get(f, {})
            
            # Compare results
            comparison = compare_results(
                model_id=case.model_id,
                formula=f,
                solver_results={
                    "nodes": solver_result.nodes,
                    "plates": solver_result.plates,
                    "fasteners": solver_result.fasteners,
                    "loads": solver_result.loads,
                },
                reference_data=ref_data,
            )
            
            comparisons.append(comparison)
            reporter.add_model_comparison(comparison, solver_result, ref_data)
            
            if verbose:
                status = "PASS" if comparison.overall_pass else "FAIL"
                print(f"  {case.model_id}/{f}: {status} (max error: {comparison.overall_max_rel_error:.3f}%)")
                
    # Generate report
    if comparisons:
        reporter.save(output_path)
        print(f"\nReport saved to: {output_path}")
        
        # Print summary
        passed = sum(1 for c in comparisons if c.overall_pass)
        print(f"\nSummary: {passed}/{len(comparisons)} passed")
        
    return comparisons


def list_test_cases(test_dir: str = "test_values"):
    """List available test cases."""
    loader = TestLoader(test_dir)
    cases = loader.discover()
    
    if not cases:
        print(f"No test cases found in {test_dir}/")
        return
        
    print(f"\nAvailable test cases in {test_dir}/:")
    print("-" * 60)
    
    for case in cases:
        status = "✓" if case.is_valid else "✗"
        formulas = ", ".join(case.formulas) if case.formulas else "None"
        print(f"  {status} {case.model_id}")
        print(f"      Config: {case.config_path.name}")
        print(f"      Reference: {case.reference_path.name}")
        print(f"      Formulas: {formulas}")
        if not case.is_valid:
            for error in case.errors[:2]:
                print(f"      Error: {error.message}")
        print()


def create_sample(model_id: str, test_dir: str = "test_values"):
    """Create a sample reference file for a model."""
    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    
    output_path = test_path / f"{model_id}_reference.json"
    create_sample_reference(model_id, output_path)
    
    print(f"Created sample reference file: {output_path}")
    print("Edit this file with actual Boeing JOLT reference values.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JOLT Verification Suite - Compare solver results against Boeing JOLT reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_verification.py                     # Run all tests
  python run_verification.py --model D06         # Run specific model
  python run_verification.py --formula boeing    # Run with Boeing formula only
  python run_verification.py --list              # List available tests
  python run_verification.py --create-sample D07 # Create sample reference file
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        action="append",
        help="Model ID(s) to run (can specify multiple times)"
    )
    parser.add_argument(
        "--formula", "-f",
        help="Formula to use (boeing, huth, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        default="reports/verification_report.xlsx",
        help="Output Excel report path (default: reports/verification_report.xlsx)"
    )
    parser.add_argument(
        "--test-dir", "-d",
        default="test_values",
        help="Directory containing test files (default: test_values)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available test cases"
    )
    parser.add_argument(
        "--create-sample",
        metavar="MODEL_ID",
        help="Create a sample reference file for the specified model ID"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also output Markdown summary to console"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_test_cases(args.test_dir)
        return 0
        
    # Handle create-sample command
    if args.create_sample:
        create_sample(args.create_sample, args.test_dir)
        return 0
        
    # Run verification
    comparisons = run_verification(
        test_dir=args.test_dir,
        models=args.model,
        formula=args.formula,
        output_path=args.output,
        verbose=args.verbose,
    )
    
    # Output Markdown if requested
    if args.markdown and comparisons:
        print("\n" + generate_markdown_summary(comparisons))
        
    # Return exit code based on results
    if not comparisons:
        return 1
    if all(c.overall_pass for c in comparisons):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
