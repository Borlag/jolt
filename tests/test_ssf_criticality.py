
import unittest
from unittest.mock import patch, MagicMock
from jolt.model import JointSolution, NodeResult, BearingBypassResult, FastenerRow, Plate

class TestSSFCriticality(unittest.TestCase):
    @patch('jolt.model.fatigue.calculate_ssf')
    def test_critical_node_selection(self, mock_calc_ssf):
        """
        Verify that the critical node is selected based on Peak Stress (SSF * sigma_ref),
        not just raw SSF.
        """
        # Setup JointSolution
        sol = JointSolution(
            displacements=[], stiffness_matrix=[], force_vector=[],
            fasteners=[], bearing_bypass=[], nodes=[], bars=[], reactions=[], dof_map={}
        )
        
        # Define two nodes
        # Node 1: High SSF (10.0), Low Sigma Ref (1000.0) -> Peak = 10,000
        # Node 2: Low SSF (2.0), High Sigma Ref (20,000.0) -> Peak = 40,000
        # Expected Critical: Node 2
        
        node1 = NodeResult(
            plate_id=0, plate_name="Plate1", local_node=0, x=0.0, displacement=0.0,
            net_bypass=100.0, thickness=0.1, bypass_area=1.0, row=1, legacy_id=101
        )
        node2 = NodeResult(
            plate_id=0, plate_name="Plate1", local_node=1, x=1.0, displacement=0.0,
            net_bypass=20000.0, thickness=0.1, bypass_area=1.0, row=2, legacy_id=102
        )
        sol.nodes = [node1, node2]
        
        # Define Bearing/Bypass results (needed for lookup)
        bb1 = BearingBypassResult(row=1, plate_name="Plate1", bearing=0.0, bypass=100.0)
        bb2 = BearingBypassResult(row=2, plate_name="Plate1", bearing=0.0, bypass=20000.0)
        sol.bearing_bypass = [bb1, bb2]
        
        # Define Fasteners (needed for lookup)
        f1 = FastenerRow(row=1, D=0.2, Eb=1e7, nu_b=0.3)
        f2 = FastenerRow(row=2, D=0.2, Eb=1e7, nu_b=0.3)
        fasteners = [f1, f2]
        
        # Mock calculate_ssf return values
        # The model calls calculate_ssf for each node.
        # We need to return dictionaries with 'ssf' and 'sigma_ref'.
        
        def side_effect(*args, **kwargs):
            # Identify which node is being processed based on load_bypass or other args
            # Node 1 has load_bypass=100.0
            # Node 2 has load_bypass=20000.0
            load_bypass = kwargs.get('load_bypass')
            
            if load_bypass == 100.0:
                return {
                    "ssf": 10.0, "sigma_ref": 1000.0, 
                    "ktg": 1.0, "ktn": 1.0, "ktb": 1.0, "theta": 1.0, 
                    "term_bearing": 0.0, "term_bypass": 0.0
                }
            elif load_bypass == 20000.0:
                return {
                    "ssf": 2.0, "sigma_ref": 20000.0, 
                    "ktg": 1.0, "ktn": 1.0, "ktb": 1.0, "theta": 1.0, 
                    "term_bearing": 0.0, "term_bypass": 0.0
                }
            return {}

        mock_calc_ssf.side_effect = side_effect
        
        # Run computation
        sol.compute_fatigue_factors(fasteners)
        
        # Check results
        print(f"Critical Node ID: {sol.critical_node_id}")
        
        # In the BUGGY version, it picks max SSF -> Node 101 (SSF 10.0)
        # In the FIXED version, it should pick max Peak Stress -> Node 102 (Peak 40,000)
        
        # We assert what we expect for the FIXED version.
        # If this fails, it confirms the bug (or that I haven't fixed it yet).
        self.assertEqual(sol.critical_node_id, 102, f"Should pick Node 102 (Peak 40k), but picked {sol.critical_node_id}")

if __name__ == '__main__':
    unittest.main()
