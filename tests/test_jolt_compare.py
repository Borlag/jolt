"""Tests for JOLT comparison helper."""
import pytest
from dataclasses import dataclass, field
from typing import List

from jolt.benchmarks import compare_solution_to_jolt, D06_ROW_A_LABEL, JOLT_FASTENER_REFS


# Minimal mock objects for testing comparison logic without running the full solver
@dataclass
class MockPlate:
    name: str


@dataclass
class MockFastener:
    row: int
    plate_i: int
    plate_j: int
    force: float


@dataclass
class MockSolution:
    fasteners: List[MockFastener] = field(default_factory=list)
    plates: List[MockPlate] = field(default_factory=list)


def test_compare_exact_match():
    """Test comparison when model exactly matches reference."""
    plates = [MockPlate("Doubler"), MockPlate("Skin"), MockPlate("Strap")]
    fasteners = [
        MockFastener(row=2, plate_i=0, plate_j=1, force=364.8),  # Exact match
    ]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, D06_ROW_A_LABEL)
    
    assert (2, "Doubler", "Skin") in results
    comparison = results[(2, "Doubler", "Skin")]
    assert comparison["ref"] == pytest.approx(364.8)
    assert comparison["model"] == pytest.approx(364.8)
    assert comparison["abs_err"] == pytest.approx(0.0)
    assert comparison["rel_err_pct"] == pytest.approx(0.0)


def test_compare_with_difference():
    """Test comparison when model differs from reference."""
    plates = [MockPlate("Doubler"), MockPlate("Skin"), MockPlate("Strap")]
    fasteners = [
        MockFastener(row=2, plate_i=0, plate_j=1, force=400.0),  # Different from 364.8
    ]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, D06_ROW_A_LABEL)
    
    comparison = results[(2, "Doubler", "Skin")]
    assert comparison["ref"] == pytest.approx(364.8)
    assert comparison["model"] == pytest.approx(400.0)
    assert comparison["abs_err"] == pytest.approx(35.2)
    expected_rel = 100 * 35.2 / 364.8
    assert comparison["rel_err_pct"] == pytest.approx(expected_rel, rel=0.01)


def test_compare_negative_force():
    """Test that negative forces are handled correctly (absolute value)."""
    plates = [MockPlate("Doubler"), MockPlate("Skin"), MockPlate("Strap")]
    fasteners = [
        MockFastener(row=2, plate_i=0, plate_j=1, force=-364.8),  # Negative force
    ]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, D06_ROW_A_LABEL)
    
    comparison = results[(2, "Doubler", "Skin")]
    # Should use abs(force)
    assert comparison["model"] == pytest.approx(364.8)
    assert comparison["abs_err"] == pytest.approx(0.0)


def test_compare_skips_missing_refs():
    """Test that interfaces not in reference data are skipped."""
    plates = [MockPlate("Unknown1"), MockPlate("Unknown2")]
    fasteners = [MockFastener(row=99, plate_i=0, plate_j=1, force=100.0)]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, D06_ROW_A_LABEL)
    
    assert len(results) == 0  # No matching references


def test_plate_order_insensitive():
    """Test that plate order doesn't affect lookup (normalization works)."""
    # Plate indices are reversed compared to the reference order
    plates = [MockPlate("Skin"), MockPlate("Doubler")]  # Skin at index 0, Doubler at index 1
    fasteners = [
        MockFastener(row=2, plate_i=0, plate_j=1, force=364.8),  # Skin-Doubler order
    ]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, D06_ROW_A_LABEL)
    
    # Should still match because we normalize plate order alphabetically
    assert (2, "Doubler", "Skin") in results
    assert results[(2, "Doubler", "Skin")]["ref"] == pytest.approx(364.8)


def test_multiple_interfaces():
    """Test comparison with multiple fastener interfaces."""
    plates = [MockPlate("Doubler"), MockPlate("Skin"), MockPlate("Strap")]
    fasteners = [
        MockFastener(row=2, plate_i=0, plate_j=1, force=370.0),   # Doubler-Skin
        MockFastener(row=2, plate_i=1, plate_j=2, force=540.0),   # Skin-Strap
        MockFastener(row=3, plate_i=0, plate_j=1, force=375.0),   # Doubler-Skin
    ]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, D06_ROW_A_LABEL)
    
    # All three should be present
    assert len(results) == 3
    assert (2, "Doubler", "Skin") in results
    assert (2, "Skin", "Strap") in results
    assert (3, "Doubler", "Skin") in results
    
    # Check specific values
    assert results[(2, "Doubler", "Skin")]["ref"] == pytest.approx(364.8)
    assert results[(2, "Skin", "Strap")]["ref"] == pytest.approx(538.4)
    assert results[(3, "Doubler", "Skin")]["ref"] == pytest.approx(371.0)


def test_custom_refs():
    """Test that custom reference dict can be passed."""
    custom_refs = {
        ("custom_case", 1, "A", "B"): 100.0,
    }
    
    plates = [MockPlate("A"), MockPlate("B")]
    fasteners = [MockFastener(row=1, plate_i=0, plate_j=1, force=110.0)]
    solution = MockSolution(fasteners=fasteners, plates=plates)
    
    results = compare_solution_to_jolt(solution, "custom_case", jolt_refs=custom_refs)
    
    assert (1, "A", "B") in results
    assert results[(1, "A", "B")]["ref"] == pytest.approx(100.0)
    assert results[(1, "A", "B")]["model"] == pytest.approx(110.0)
    assert results[(1, "A", "B")]["abs_err"] == pytest.approx(10.0)


def test_jolt_refs_structure():
    """Verify JOLT_FASTENER_REFS has expected structure and values."""
    # Check that all expected keys are present
    expected_keys = [
        (D06_ROW_A_LABEL, 2, "Doubler", "Skin"),
        (D06_ROW_A_LABEL, 2, "Skin", "Strap"),
        (D06_ROW_A_LABEL, 3, "Doubler", "Skin"),
        (D06_ROW_A_LABEL, 3, "Skin", "Strap"),
        (D06_ROW_A_LABEL, 4, "Doubler", "Skin"),
    ]
    
    for key in expected_keys:
        assert key in JOLT_FASTENER_REFS
        assert isinstance(JOLT_FASTENER_REFS[key], float)
        assert JOLT_FASTENER_REFS[key] > 0
