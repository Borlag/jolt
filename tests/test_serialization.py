import pytest
import json
import uuid
from jolt.model import JointSolution, FastenerResult, NodeResult, BarResult, BearingBypassResult, ReactionResult, FatigueResult
from jolt.config import JointConfiguration, Plate, FastenerRow

def test_joint_solution_serialization():
    # Create a dummy solution
    sol = JointSolution(
        displacements=[0.1, 0.2],
        stiffness_matrix=[[1.0, 0.0], [0.0, 1.0]],
        force_vector=[10.0, 20.0],
        fasteners=[
            FastenerResult(row=0, plate_i=0, plate_j=1, compliance=1e-5, stiffness=1e5, force=100.0, dof_i=0, dof_j=1)
        ],
        bearing_bypass=[
            BearingBypassResult(row=0, plate_name="P1", bearing=10.0, bypass=90.0)
        ],
        nodes=[
            NodeResult(plate_id=0, plate_name="P1", local_node=0, x=0.0, displacement=0.1, net_bypass=90.0, thickness=0.1, bypass_area=0.1)
        ],
        bars=[
            BarResult(plate_id=0, plate_name="P1", segment=0, axial_force=90.0, stiffness=1e6, modulus=10e6)
        ],
        reactions=[
            ReactionResult(plate_id=0, plate_name="P1", local_node=0, global_node=0, reaction=10.0)
        ],
        dof_map={(0, 0): 0, (0, 1): 1},
        fatigue_results=[
            FatigueResult(node_id=0, row=0, plate_name="P1", ktg=2.5, ktn=3.0, ktb=1.5, theta=0.5, ssf=1.2, bearing_load=10.0, bypass_load=90.0, sigma_ref=20.0, term_bearing=5.0, term_bypass=15.0, peak_stress=24.0, fsi=1.2)
        ],
        critical_points=[{"plate_name": "P1", "node_id": 0}],
        critical_node_id=0
    )
    
    # Serialize
    data = sol.to_dict()
    
    # Check summary fields
    assert "summary" in data
    assert data["summary"]["max_fsi_global"] == 1.2
    assert data["summary"]["max_fastener_load"] == 100.0
    assert data["summary"]["max_bypass"] == 90.0
    
    # Deserialize
    rehydrated = JointSolution.from_dict(data)
    
    # Check equality
    assert rehydrated.displacements == sol.displacements
    assert len(rehydrated.fasteners) == 1
    assert rehydrated.fasteners[0].force == 100.0
    assert rehydrated.fatigue_results[0].fsi == 1.2
    # Check DOF map reconstruction
    assert rehydrated.dof_map[(0, 0)] == 0

def test_joint_configuration_persistence():
    config = JointConfiguration(
        pitches=[1.0],
        plates=[Plate(name="P1", E=10e6, t=0.1, first_row=0, last_row=1, A_strip=[0.1])],
        fasteners=[FastenerRow(row=0, D=0.25, Eb=10e6, nu_b=0.3)],
        supports=[(0, 0, 0.0)],
        label="Test Config",
        units="Imperial"
    )
    
    # Check defaults
    assert config.model_id is not None
    assert config.schema_version == 1
    
    # Serialize
    data = config.to_dict()
    assert "model_id" in data
    assert "schema_version" in data
    
    # Deserialize
    rehydrated = JointConfiguration.from_dict(data)
    assert rehydrated.model_id == config.model_id
    assert rehydrated.label == "Test Config"

def test_legacy_json_compatibility():
    # Simulate legacy JSON without model_id, schema_version, or results
    legacy_data = {
        "pitches": [1.0],
        "plates": [{"name": "P1", "E": 10e6, "t": 0.1, "first_row": 0, "last_row": 1, "A_strip": [0.1]}],
        "fasteners": [{"row": 0, "D": 0.25, "Eb": 10e6, "nu_b": 0.3}],
        "supports": [[0, 0, 0.0]],
        "label": "Legacy Config",
        "units": "Imperial"
    }
    
    config = JointConfiguration.from_dict(legacy_data)
    
    # Should generate new ID and default version
    assert config.model_id is not None
    assert len(config.model_id) > 0
    assert config.schema_version == 1
    assert config.results is None
