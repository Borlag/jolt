
import unittest
import math
from jolt.fasteners import huth_compliance
from jolt.model import Joint1D, Plate, FastenerRow

class TestHuthSymmetry(unittest.TestCase):
    def test_huth_manual_calc(self):
        """
        Verify Huth formula against manual calculation.
        Scenario: Two plates, t=2.0mm, D=5.0mm, Aluminum (E=70000 MPa).
        Fastener: Titanium (Ef=110000 MPa).
        """
        t1 = 2.0
        t2 = 2.0
        E = 70000.0
        D = 5.0
        Ef = 110000.0
        
        # Manual Calculation for Huth (Bolted Metal)
        # a = 2/3, b = 3.0
        # C = ((t1+t2)/(2d))^a * (b/n) * (1/t1E1 + 1/nt2E2 + 1/2t1Ef + 1/2nt2Ef)
        # n = 1 (single shear)
        
        a = 2.0/3.0
        b = 3.0
        n = 1.0
        
        term1 = ((t1 + t2) / (2.0 * D)) ** a
        term2 = b / n
        term3 = (1.0 / (t1 * E)) + (1.0 / (n * t2 * E)) + \
                (1.0 / (2.0 * t1 * Ef)) + (1.0 / (2.0 * n * t2 * Ef))
        
        expected_compliance = term1 * term2 * term3
        
        # Python Implementation Check
        calculated_compliance = huth_compliance(
            t1=t1, E1=E, t2=t2, E2=E, Ef=Ef, diameter=D, 
            shear="single", joint_type="bolted_metal"
        )
        
        print(f"Manual Huth: {expected_compliance:.6e}")
        print(f"Func Huth:   {calculated_compliance:.6e}")
        
        self.assertAlmostEqual(expected_compliance, calculated_compliance, places=9)

    def test_solver_integration(self):
        """
        Verify that the solver correctly applies the Huth compliance to the joint.
        The total flexibility of the connection should match the Huth formula.
        """
        t = 2.0
        E = 70000.0
        D = 5.0
        Ef = 110000.0
        
        # Create Model
        p1 = Plate(name="P1", E=E, t=t, first_row=1, last_row=2, A_strip=[10.0])
        p2 = Plate(name="P2", E=E, t=t, first_row=1, last_row=2, A_strip=[10.0])
        
        # Fastener with Huth method
        f1 = FastenerRow(row=1, D=D, Eb=Ef, nu_b=0.3, method="Huth (Bolted)")
        
        joint = Joint1D(pitches=[10.0], plates=[p1, p2], fasteners=[f1])
        
        # Solve with simple load to get displacement
        # Fix left side of P1, apply load to right side of P2? 
        # Or better: Fix P1, Pull P2.
        # The relative displacement at the fastener row should be F * C_total.
        
        # Support P1 at node 0 (left)
        # But we want to isolate the fastener spring.
        # Let's fix P1 at the fastener location (row 1) and Pull P2 at fastener location (row 1).
        # Wait, if we fix P1 at row 1, and Pull P2 at row 1, the displacement difference is exactly the spring stretch.
        
        # In Joint1D, row 1 corresponds to local_node 0 for these 1-segment plates?
        # Plate 1: first_row=1, last_row=1. segment_count = 0?
        # Wait, last_row - first_row = 0. So it's a point?
        # If segment_count is 0, it has 1 node (local 0).
        
        supports = [(0, 0, 0.0)] # Fix Plate 1 at row 1
        forces = [(1, 0, 1000.0)] # Pull Plate 2 at row 1 with 1000 N
        
        sol = joint.solve(supports, point_forces=forces)
        
        # Get displacements at row 1
        # Plate 1 is index 0, Plate 2 is index 1
        u1 = sol.displacements[joint._dof[(0, 0)]]
        u2 = sol.displacements[joint._dof[(1, 0)]]
        
        delta = u2 - u1
        load = 1000.0
        
        measured_compliance = delta / load
        
        # Calculate expected Huth compliance
        expected_c = huth_compliance(
            t1=t, E1=E, t2=t, E2=E, Ef=Ef, diameter=D, 
            shear="single", joint_type="bolted_metal"
        )
        
        print(f"Solver Delta: {delta:.6e}")
        print(f"Solver C:     {measured_compliance:.6e}")
        print(f"Expected C:   {expected_c:.6e}")
        
        # This will FAIL if the solver is still using the "split" logic (summing halves)
        # because (t+t)^a != t^a + t^a for a != 1.
        self.assertAlmostEqual(measured_compliance, expected_c, places=6)

if __name__ == '__main__':
    unittest.main()
