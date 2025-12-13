import unittest
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jolt.model import Joint1D, Plate, FastenerRow

class TestBoeingFix(unittest.TestCase):
    def test_boeing_2_layer(self):
        """
        Test a simple 2-layer Boeing joint.
        Verifies that the solver runs and produces reasonable results (non-zero load).
        """
        plates = [
            Plate(name="P1", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_right=1000.0),
            Plate(name="P2", E=10e6, t=0.1, first_row=1, last_row=2, A_strip=[0.1], Fx_left=0.0)
        ]
        fasteners = [
            FastenerRow(row=1, D=0.25, Eb=10e6, nu_b=0.3, method="Boeing")
        ]
        pitches = [1.0]
        
        joint = Joint1D(pitches, plates, fasteners)
        sol = joint.solve(supports=[(1, 0, 0.0)]) # Support P2 at left
        
        # Check fastener load
        # Should be close to 1000.0
        f_load = abs(sol.fasteners[0].force)
        print(f"\n2-Layer Fastener Load: {f_load}")
        self.assertGreater(f_load, 900.0)
        self.assertLess(f_load, 1100.0)

    def test_boeing_3_layer_case5(self):
        # Load Case_5_3_elements_row_a.json
        # Assuming it is in the root directory
        case_path = os.path.join(os.path.dirname(__file__), '..', "Case_5_3_elements_row_a.json")
        if not os.path.exists(case_path):
            self.skipTest(f"Case file not found: {case_path}")
            
        with open(case_path, "r") as f:
            data = json.load(f)
            
        # Reconstruct objects
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
        
        joint = Joint1D(pitches, plates, fasteners)
        
        supports = [tuple(s) for s in data["supports"]]
        
        sol = joint.solve(supports=supports)
        
        actual = []
        results = sorted(sol.fasteners, key=lambda x: (x.row, x.plate_i, x.plate_j))
        
        for res in results:
            actual.append(abs(res.force))
            
        # Expected values for JOSEF/JOLT Ladder Topology (Jarfall 1972)
        # boeing_beam is now the default for all Boeing methods.
        expected = [335.0, 532.2, 341.4, 467.8, 323.6]
        
        print(f"\nActual Shear Forces: {actual}")
        print(f"Expected Shear Forces: {expected}")
        
        self.assertEqual(len(actual), len(expected), "Mismatch in number of fastener interfaces")
        
        for i, (a, e) in enumerate(zip(actual, expected)):
            diff = abs(a - e)
            self.assertLess(diff, 2.0, f"Mismatch at index {i}: Actual {a}, Expected {e}, Diff {diff}")

if __name__ == '__main__':
    unittest.main()
