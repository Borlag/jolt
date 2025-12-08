"""
Standalone Verification Module for JOLT Joint Solver.

This module provides automated testing and validation of the joint solver
against Boeing JOLT reference results. It is isolated from the main codebase
and imports the solver as a dependency.

Key Components:
- schemas: JSON schema definitions for reference data
- loader: Discovers and loads test cases from test_values/
- runner: Executes solver with specified formulas
- comparator: Computes error statistics
- reporter: Generates Excel validation reports
- cli: Command-line interface

Usage:
    python run_verification.py [--model MODEL] [--formula FORMULA] [--output PATH]
"""

from .schemas import REFERENCE_SCHEMA, CONFIG_SCHEMA, validate_reference, validate_config
from .loader import TestCase, TestLoader
from .runner import TestRunner, SolverResults
from .comparator import FieldComparison, ModelComparison, compare_results
from .reporter import VerificationReporter

__all__ = [
    "REFERENCE_SCHEMA",
    "CONFIG_SCHEMA",
    "validate_reference",
    "validate_config",
    "TestCase",
    "TestLoader",
    "TestRunner",
    "SolverResults",
    "FieldComparison",
    "ModelComparison",
    "compare_results",
    "VerificationReporter",
]
