
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jolt.model import Plate, FastenerRow, JointSolution, NodeResult, BearingBypassResult
from jolt.config import plate_to_dict, plate_from_dict
from jolt.fatigue import calculate_ssf

def test_persistence():
    print("Testing Persistence...")
    p = Plate(name="TestPlate", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1], fatigue_strength=50000.0)
    data = plate_to_dict(p)
    print(f"Serialized: {data}")
    assert data["fatigue_strength"] == 50000.0
    
    p2 = plate_from_dict(data)
    print(f"Deserialized: {p2.fatigue_strength}")
    assert p2.fatigue_strength == 50000.0
    print("Persistence OK")

def test_fsi_ranking():
    print("\nTesting FSI Ranking...")
    
    # Setup Plates
    p1 = Plate(name="Plate1", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1], fatigue_strength=60000.0) # f_max = 60k
    p2 = Plate(name="Plate2", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1], fatigue_strength=None)    # No f_max
    
    # Setup Solution
    sol = JointSolution(
        displacements=[], stiffness_matrix=[], force_vector=[], fasteners=[], bearing_bypass=[], nodes=[], bars=[], reactions=[], dof_map={},
        plates=[p1, p2]
    )
    
    # Mock Nodes
    # Node 1: Plate 1, Peak Stress 30k -> FSI = 0.5
    # Node 2: Plate 1, Peak Stress 45k -> FSI = 0.75 (Critical for Plate 1)
    # Node 3: Plate 2, Peak Stress 40k -> FSI = 40k (No allowable)
    
    sol.nodes = [
        NodeResult(plate_id=0, plate_name="Plate1", local_node=0, x=0, displacement=0, net_bypass=1000, thickness=0.1, bypass_area=0.1, row=1, legacy_id=1001),
        NodeResult(plate_id=0, plate_name="Plate1", local_node=1, x=1, displacement=0, net_bypass=1500, thickness=0.1, bypass_area=0.1, row=2, legacy_id=1002),
        NodeResult(plate_id=1, plate_name="Plate2", local_node=0, x=0, displacement=0, net_bypass=1000, thickness=0.1, bypass_area=0.1, row=1, legacy_id=2001),
    ]
    
    # Mock Fasteners
    f1 = FastenerRow(row=1, D=0.2, Eb=1e7, nu_b=0.3)
    f2 = FastenerRow(row=2, D=0.2, Eb=1e7, nu_b=0.3)
    fasteners = [f1, f2]
    
    # Mock Bearing/Bypass
    sol.bearing_bypass = [
        BearingBypassResult(row=1, plate_name="Plate1", bearing=500, bypass=1000),
        BearingBypassResult(row=2, plate_name="Plate1", bearing=750, bypass=1500),
        BearingBypassResult(row=1, plate_name="Plate2", bearing=500, bypass=1000),
    ]
    
    # Run Calculation
    # Note: We need to mock calculate_ssf or ensure it returns predictable values.
    # Since calculate_ssf is deterministic, we can just run it.
    # However, to control Peak Stress exactly, we might need to rely on the inputs.
    # Let's just check the relative ranking logic.
    
    sol.compute_fatigue_factors(fasteners)
    
    print("Critical Points:")
    for cp in sol.critical_points:
        print(cp)
        
    # Verification
    # We expect Plate 2 (Node 2001) to have huge FSI (raw stress) if mixed, 
    # but usually FSI is small. 
    # Wait, if one has f_max and other doesn't, the one without f_max will have FSI = Stress (e.g. 40000).
    # The one with f_max will have FSI ~ 0.5-1.0.
    # So Plate 2 should be Rank 1.
    
    assert len(sol.critical_points) == 2
    assert sol.critical_points[0]["plate_name"] == "Plate2"
    assert sol.critical_points[0]["rank"] == 1
    assert sol.critical_points[0]["fsi"] > 1000 # Should be raw stress
    
    assert sol.critical_points[1]["plate_name"] == "Plate1"
    assert sol.critical_points[1]["rank"] == 2
    assert sol.critical_points[1]["fsi"] < 10.0 # Should be ratio
    
    print("Ranking Logic OK")
    
    print("Checking FSI for all nodes...")
    assert len(sol.fatigue_results) == 3
    for res in sol.fatigue_results:
        print(f"Node {res.node_id}: FSI={res.fsi}, f_max={res.f_max}")
        assert res.fsi > 0
        if res.node_id == 1001: # Plate 1, Node 1
             assert res.f_max == 60000.0
        if res.node_id == 2001: # Plate 2
             assert res.f_max is None
             
    print("All Nodes FSI OK")

if __name__ == "__main__":
    test_persistence()
    test_fsi_ranking()
