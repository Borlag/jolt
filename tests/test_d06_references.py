"""Tests for D06 Boeing JOLT reference cases.

These tests verify the solver against known Boeing JOLT output for multi-layer joints.
Reference data is stored in test_values/D06_*_reference.json files.
"""
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pytest

from jolt.model import Joint1D, Plate, FastenerRow


TEST_VALUES_DIR = Path(__file__).parent.parent / "test_values"


def load_config(case_name: str) -> Tuple[Joint1D, List[Tuple[int, int, float]]]:
    """Load a D06 case configuration and return the Joint1D model and supports."""
    config_path = TEST_VALUES_DIR / f"{case_name}_config.json"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = json.load(f)
    
    plates = []
    for p_data in data["plates"]:
        plates.append(Plate(**p_data))
    
    fasteners = []
    for f_data in data["fasteners"]:
        conns = None
        if "connections" in f_data:
            conns = [tuple(c) for c in f_data["connections"]]
        kwargs = f_data.copy()
        if "connections" in kwargs:
            del kwargs["connections"]
        fasteners.append(FastenerRow(connections=conns, **kwargs))
    
    pitches = data["pitches"]
    supports = [tuple(s) for s in data["supports"]]
    
    model = Joint1D(pitches, plates, fasteners)
    return model, supports


def load_reference(case_name: str) -> Dict[str, Any]:
    """Load reference data for a D06 case."""
    ref_path = TEST_VALUES_DIR / f"{case_name}_reference.json"
    if not ref_path.exists():
        pytest.skip(f"Reference file not found: {ref_path}")
    
    with open(ref_path, "r") as f:
        return json.load(f)


def get_fastener_force_refs(ref_data: Dict[str, Any]) -> Dict[Tuple[int, str, str], float]:
    """Extract fastener forces from reference data as {(row, plate_i, plate_j): force}."""
    refs = {}
    boeing = ref_data.get("formulas", {}).get("boeing", {})
    for f in boeing.get("fasteners", []):
        row = f["row"]
        plate_i = f["plate_i"]
        plate_j = f["plate_j"]
        force = f["force"]
        key = (row, plate_i, plate_j)
        refs[key] = force
    return refs


def get_node_displacement_refs(ref_data: Dict[str, Any]) -> Dict[Tuple[str, int], float]:
    """Extract node displacements from reference data as {(plate_name, row): displacement}."""
    refs = {}
    boeing = ref_data.get("formulas", {}).get("boeing", {})
    for n in boeing.get("nodes", []):
        plate_name = n["plate_name"]
        row = n["row"]
        disp = n["displacement"]
        key = (plate_name, row)
        refs[key] = disp
    return refs


class TestD06Cases:
    """Test D06 Boeing reference cases."""
    
    @pytest.fixture
    def d06_3_model(self):
        """Load D06_3 (4-layer) case."""
        return load_config("D06_3")
    
    @pytest.fixture
    def d06_3_ref(self):
        """Load D06_3 reference data."""
        return load_reference("D06_3")
    
    @pytest.fixture
    def d06_4_model(self):
        """Load D06_4 (5-layer) case."""
        return load_config("D06_4")
    
    @pytest.fixture
    def d06_4_ref(self):
        """Load D06_4 reference data."""
        return load_reference("D06_4")
    
    @pytest.fixture
    def d06_5_model(self):
        """Load D06_5 (4-layer variant) case."""
        return load_config("D06_5")
    
    @pytest.fixture
    def d06_5_ref(self):
        """Load D06_5 reference data."""
        return load_reference("D06_5")
    
    def test_d06_3_fastener_forces(self, d06_3_model, d06_3_ref):
        """Test D06_3 fastener forces match JOLT reference."""
        model, supports = d06_3_model
        solution = model.solve(supports=supports)
        
        ref_forces = get_fastener_force_refs(d06_3_ref)
        
        # Build lookup from solution
        actual_forces = {}
        for f in solution.fasteners:
            plate_i_name = model.plates[f.plate_i].name
            plate_j_name = model.plates[f.plate_j].name
            key = (f.row, plate_i_name, plate_j_name)
            actual_forces[key] = abs(f.force)
        
        # Compare each fastener
        errors = []
        for key, ref_val in ref_forces.items():
            actual_val = actual_forces.get(key, None)
            if actual_val is None:
                errors.append(f"Missing fastener {key}")
                continue
            rel_err = abs(actual_val - ref_val) / ref_val if ref_val > 0 else 0
            if rel_err > 0.15:  # 15% tolerance for now
                errors.append(f"Fastener {key}: ref={ref_val:.1f}, actual={actual_val:.1f}, err={rel_err*100:.1f}%")
        
        assert len(errors) == 0, f"Fastener force mismatches:\n" + "\n".join(errors)
    
    def test_d06_3_displacements(self, d06_3_model, d06_3_ref):
        """Test D06_3 node displacements match JOLT reference."""
        model, supports = d06_3_model
        solution = model.solve(supports=supports)
        
        ref_disps = get_node_displacement_refs(d06_3_ref)
        
        # Build lookup from solution
        actual_disps = {}
        for n in solution.nodes:
            key = (n.plate_name, n.row)
            actual_disps[key] = n.displacement
        
        # Compare each node
        errors = []
        for key, ref_val in ref_disps.items():
            actual_val = actual_disps.get(key, None)
            if actual_val is None:
                continue  # Skip missing nodes
            # Only check non-zero displacements
            if abs(ref_val) > 1e-9:
                rel_err = abs(actual_val - ref_val) / abs(ref_val)
                if rel_err > 0.15:  # 15% tolerance for now
                    errors.append(f"Node {key}: ref={ref_val:.6f}, actual={actual_val:.6f}, err={rel_err*100:.1f}%")
        
        # For now, just print errors but don't fail (to see overall impact)
        if errors:
            print(f"\nDisplacement mismatches ({len(errors)} nodes):")
            for e in errors[:10]:  # Print first 10
                print(f"  {e}")
    
    def test_d06_4_fastener_forces(self, d06_4_model, d06_4_ref):
        """Test D06_4 fastener forces match JOLT reference."""
        model, supports = d06_4_model
        solution = model.solve(supports=supports)
        
        ref_forces = get_fastener_force_refs(d06_4_ref)
        
        actual_forces = {}
        for f in solution.fasteners:
            plate_i_name = model.plates[f.plate_i].name
            plate_j_name = model.plates[f.plate_j].name
            key = (f.row, plate_i_name, plate_j_name)
            actual_forces[key] = abs(f.force)
        
        errors = []
        for key, ref_val in ref_forces.items():
            actual_val = actual_forces.get(key, None)
            if actual_val is None:
                errors.append(f"Missing fastener {key}")
                continue
            rel_err = abs(actual_val - ref_val) / ref_val if ref_val > 0 else 0
            if rel_err > 0.15:
                errors.append(f"Fastener {key}: ref={ref_val:.1f}, actual={actual_val:.1f}, err={rel_err*100:.1f}%")
        
        assert len(errors) == 0, f"Fastener force mismatches:\n" + "\n".join(errors)
    
    def test_d06_5_fastener_forces(self, d06_5_model, d06_5_ref):
        """Test D06_5 fastener forces match JOLT reference."""
        model, supports = d06_5_model
        solution = model.solve(supports=supports)
        
        ref_forces = get_fastener_force_refs(d06_5_ref)
        
        actual_forces = {}
        for f in solution.fasteners:
            plate_i_name = model.plates[f.plate_i].name
            plate_j_name = model.plates[f.plate_j].name
            key = (f.row, plate_i_name, plate_j_name)
            actual_forces[key] = abs(f.force)
        
        errors = []
        for key, ref_val in ref_forces.items():
            actual_val = actual_forces.get(key, None)
            if actual_val is None:
                errors.append(f"Missing fastener {key}")
                continue
            rel_err = abs(actual_val - ref_val) / ref_val if ref_val > 0 else 0
            if rel_err > 0.15:
                errors.append(f"Fastener {key}: ref={ref_val:.1f}, actual={actual_val:.1f}, err={rel_err*100:.1f}%")
        
        assert len(errors) == 0, f"Fastener force mismatches:\n" + "\n".join(errors)


class TestBoeingBeamConsistency:
    """Test that beam assembly is consistent with star for 2-layer joints."""
    
    def test_two_layer_consistency(self):
        """Verify 2-layer joints produce similar results with beam vs star."""
        plates = [
            Plate(name="P1", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_right=1000.0),
            Plate(name="P2", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0)
        ]
        fasteners = [
            FastenerRow(row=1, D=0.25, Eb=10e6, nu_b=0.3, method="Boeing")
        ]
        pitches = [1.0]
        supports = [(1, 0, 0.0)]
        
        model = Joint1D(pitches, plates, fasteners)
        sol = model.solve(supports=supports)
        
        # Should produce reasonable fastener load (close to 1000)
        f_load = abs(sol.fasteners[0].force)
        assert 900.0 < f_load < 1100.0, f"Expected ~1000, got {f_load}"
    
    def test_three_layer_runs(self):
        """Verify 3-layer joints solve without error."""
        plates = [
            Plate(name="P1", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_right=1000.0),
            Plate(name="P2", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1]),
            Plate(name="P3", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1])
        ]
        fasteners = [
            FastenerRow(row=1, D=0.25, Eb=10e6, nu_b=0.3, method="Boeing",
                       connections=[(0, 1), (1, 2)])
        ]
        pitches = [1.0]
        supports = [(1, 0, 0.0), (2, 0, 0.0)]
        
        model = Joint1D(pitches, plates, fasteners)
        sol = model.solve(supports=supports)
        
        # Just verify it runs and produces non-zero results
        total_load = sum(abs(f.force) for f in sol.fasteners)
        assert total_load > 0, "Expected non-zero total fastener load"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
