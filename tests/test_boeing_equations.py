"""Unit tests for Boeing Eq1/Eq2 compliance formulas."""
import math
import pytest
from jolt.fasteners import (
    boeing69_eq1_compliance,
    boeing69_eq2_compliance,
    boeing_pair_compliance,
    boeing69_compliance
)

def test_boeing_eq1_vs_legacy():
    """Boeing Eq1 should be similar to Legacy but slightly different in shear."""
    t1, t2 = 0.1, 0.1
    E1, E2 = 1e7, 1e7
    Ef = 1e7
    nu_b = 0.3
    Gf = Ef / (2 * (1 + nu_b))
    d = 0.25
    
    c_legacy = boeing69_compliance(t1, E1, t2, E2, Ef, nu_b, d)
    c_eq1 = boeing69_eq1_compliance(t1, E1, t2, E2, Ef, Gf, d)
    
    # Eq 1 uses 5GA in shear denominator, Legacy uses 9GA.
    # 1/5 > 1/9, so Eq1 shear compliance > Legacy shear compliance.
    # Total compliance Eq1 > Legacy.
    assert c_eq1 > c_legacy
    
    # Check difference magnitude (should be small for this geometry where bending dominates)
    # But let's verify it works.
    assert c_eq1 > 0

def test_boeing_eq2_value():
    """Test Boeing Eq2 returns positive finite value."""
    t1, t2 = 0.1, 0.15
    E1, E2 = 1e7, 7e6
    Ef = 1e7
    d = 0.25
    
    c_eq2 = boeing69_eq2_compliance(t1, E1, t2, E2, Ef, d)
    assert c_eq2 > 0
    assert math.isfinite(c_eq2)

def test_boeing_pair_compliance_routing():
    """Test the router function."""
    args = {
        "t1": 0.1, "E1": 1e7,
        "t2": 0.1, "E2": 1e7,
        "Ef": 1e7, "Gf": 3.8e6,
        "d": 0.25
    }
    
    c_legacy = boeing_pair_compliance(variant="legacy", **args)
    c_eq1 = boeing_pair_compliance(variant="eq1", **args)
    c_eq2 = boeing_pair_compliance(variant="eq2", **args)
    
    assert c_legacy != c_eq1
    assert c_legacy != c_eq2
    assert c_eq1 != c_eq2
    
    # Check Router Default
    c_def = boeing_pair_compliance(**args)
    assert c_def == c_legacy

def test_symmetry():
    """Formulas should be symmetric wrt indices 1 and 2."""
    t1, t2 = 0.1, 0.2
    E1, E2 = 1e7, 7e6
    Ef = 1e7
    Gf = 3.8e6
    d = 0.25
    
    # Eq 1
    c1_a = boeing69_eq1_compliance(t1, E1, t2, E2, Ef, Gf, d)
    c1_b = boeing69_eq1_compliance(t2, E2, t1, E1, Ef, Gf, d)
    assert c1_a == pytest.approx(c1_b)
    
    # Eq 2
    c2_a = boeing69_eq2_compliance(t1, E1, t2, E2, Ef, d)
    c2_b = boeing69_eq2_compliance(t2, E2, t1, E1, Ef, d)
    assert c2_a == pytest.approx(c2_b)
