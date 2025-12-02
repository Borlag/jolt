import unittest
import math
from jolt.model import Joint1D, Plate, FastenerRow

class TestBoeing3Layer(unittest.TestCase):
    def test_boeing_3layer_topology_and_results(self):
        """
        Verify that a 3-layer Boeing69 joint uses Chain topology and produces correct results.
        """
        # 1. Setup 3-Layer Joint (Doubler, Skin, Strap)
        # Geometry and Properties
        E_al = 10.0e6
        D = 0.25
        Eb = 10.0e6
        
        # Plates: 3 layers
        # Plate 0: Doubler (Top)
        # Plate 1: Skin (Middle)
        # Plate 2: Strap (Bottom)
        
        # Pitches: 1.0 inch spacing
        pitches = [1.0, 1.0] 
        
        plates = [
            Plate(name="Doubler", E=E_al, t=0.1, first_row=1, last_row=2, A_strip=[1.0]),
            Plate(name="Skin",    E=E_al, t=0.1, first_row=1, last_row=2, A_strip=[1.0]),
            Plate(name="Strap",   E=E_al, t=0.1, first_row=1, last_row=2, A_strip=[1.0]),
        ]
        
        # Fasteners: 2 rows
        fasteners = [
            FastenerRow(row=1, D=D, Eb=Eb, nu_b=0.33, method="Boeing69"),
            FastenerRow(row=2, D=D, Eb=Eb, nu_b=0.33, method="Boeing69"),
        ]
        
        # Supports and Loads
        # Fix left end of Skin (Plate 1)
        supports = [
            (1, 0, 0.0) # Plate 1, Node 0 (Left)
        ]
        
        # Apply load at right end of Doubler and Strap? 
        # Let's apply load to Doubler and Strap to pull against Skin.
        # Or typical lap joint: Pull Doubler+Strap vs Skin.
        
        # Load: 1000 lbs on Doubler (Right) and 1000 lbs on Strap (Right)
        # Total 2000 lbs transferred to Skin.
        point_forces = [
            (0, 1, 1000.0), # Doubler, Right End
            (2, 1, 1000.0), # Strap, Right End
        ]
        
        # 2. Initialize and Solve
        joint = Joint1D(pitches, plates, fasteners)
        
        # Check Topology BEFORE solving (using debug_system or internal inspection)
        # Verify NO fastener DOFs are created
        ndof = joint._build_dofs()
        
        # Expected DOFs:
        # Plate 0: 2 nodes (Row 1, Row 2) -> 2 DOFs
        # Plate 1: 2 nodes (Row 1, Row 2) -> 2 DOFs
        # Plate 2: 2 nodes (Row 1, Row 2) -> 2 DOFs
        # Total = 6 DOFs. 
        # If Star topology was used, we would have +2 fastener DOFs = 8.
        
        self.assertEqual(ndof, 6, "Should have 6 DOFs (Chain Topology), not 8 (Star Topology)")
        
        # Check that "fastener" keys are NOT in _dof
        for f in fasteners:
            self.assertNotIn(("fastener", f.row), joint._dof)
            
        # 3. Solve
        solution = joint.solve(supports, point_forces)
        
        # 4. Verify Results
        
        # A. Stiffness Verification
        # Calculate expected pairwise compliance for Boeing69
        # t1=0.1, t2=0.1, D=0.25, E=10e6, Eb=10e6
        # Use the internal method to check consistency
        # But we should also verify against the formula manually to be sure.
        
        # Manual Boeing69 Calculation:
        # Shear: 4(t1+t2)/(9*G*A)
        # G = E/(2(1+nu)) = 10e6 / (2*1.33) = 3.759e6
        # A = pi*D^2/4 = 0.049087
        # Term_shear = 4*(0.2)/(9 * 3.759e6 * 0.049087) = 0.8 / 1.66e6 = 4.81e-7
        
        # Bending: (t1^3 + 5t1^2t2 + 5t1t2^2 + t2^3) / (40*Eb*I)
        # t1=t2=t -> (12 t^3) / (40*Eb*I)
        # I = pi*D^4/64 = 0.0001917
        # Term_bending = 12 * 0.001 / (40 * 10e6 * 0.0001917) = 0.012 / 76680 = 1.56e-7
        
        # Bearing: (1/t1)*(1/Eb+1/E1) + (1/t2)*(1/Eb+1/E2) / (1 if single shear else 2*planes)
        # Here we have 3 plates.
        # Pair 1: Doubler-Skin. Single shear plane? 
        # Wait, Boeing69 formula `shear_planes` argument.
        # In `_calculate_compliance_pairwise`, it calls `boeing69_compliance`.
        # It doesn't pass `shear_planes`. Default is 1.
        # So it treats each pair as a single shear joint.
        # Term_bearing = (1/0.1)*(2/10e6) + (1/0.1)*(2/10e6) = 20e-7 + 20e-7 = 40e-7.
        
        # Total Compliance approx = 0.48 + 0.15 + 4.0 = 4.63e-6 in/lb.
        # Stiffness = 1/C approx 215,000 lb/in.
        
        # Let's check the solution values.
        for f_res in solution.fasteners:
            # Check stiffness matches 1.0 / compliance
            self.assertAlmostEqual(f_res.stiffness, 1.0 / f_res.compliance, places=2)
            
            # Check compliance is reasonable (around 4-5e-6)
            self.assertTrue(3e-6 < f_res.compliance < 6e-6, f"Compliance {f_res.compliance} out of expected range")
            
        # B. Equilibrium Verification
        # Sum of reaction at support should equal applied load (2000 lbs).
        total_reaction = sum(r.reaction for r in solution.reactions)
        self.assertAlmostEqual(total_reaction, -2000.0, delta=0.1)
        
        # C. Bearing Force Balance
        # For each row, sum of bearing forces on all plates should be zero.
        # And sum of bearing forces should equal load transferred.
        
        # Check Row 1
        bf_doubler_1 = 0.0
        bf_skin_1 = 0.0
        bf_strap_1 = 0.0
        
        # We need to look at bearing_bypass results or reconstruct from solution.fasteners
        # solution.fasteners has bearing_force_upper/lower for each pair.
        # But bearing_bypass has total bearing per plate/row.
        
        bb_map = {(bb.plate_name, bb.row): bb for bb in solution.bearing_bypass}
        
        bf_doubler_1 = bb_map[("Doubler", 1)].bearing
        bf_skin_1 = bb_map[("Skin", 1)].bearing
        bf_strap_1 = bb_map[("Strap", 1)].bearing
        
        # Note: Bearing forces in JOLT are usually signed? 
        # In `solve`, we do:
        # bearing_forces[(idx, row)] += f
        # So they are signed.
        # But `BearingBypassResult` might store them? 
        # `BearingBypassResult` stores `bearing` which comes from `bearing_forces`.
        
        # Sum should be zero (internal equilibrium)
        self.assertAlmostEqual(bf_doubler_1 + bf_skin_1 + bf_strap_1, 0.0, delta=1e-5)
        
        # D. Bar Forces
        # Check axial forces in the segment (between row 1 and 2).
        # Load is applied at right (node 2).
        # At segment 0 (between row 1 and 2):
        # Doubler should carry some load.
        # Strap should carry some load.
        # Skin should carry the reaction.
        
        bars_seg0 = [b for b in solution.bars if b.segment == 0]
        self.assertEqual(len(bars_seg0), 3)
        
        # Verify symmetry (Doubler and Strap should be identical if geometry is identical)
        bar_doubler = next(b for b in bars_seg0 if b.plate_name == "Doubler")
        bar_strap = next(b for b in bars_seg0 if b.plate_name == "Strap")
        bar_skin = next(b for b in bars_seg0 if b.plate_name == "Skin")
        
        self.assertAlmostEqual(bar_doubler.axial_force, bar_strap.axial_force, delta=0.1)
        
        print("\nTest Results:")
        print(f"NDOF: {ndof}")
        print(f"Doubler Bar Force (Seg 0): {bar_doubler.axial_force}")
        print(f"Strap Bar Force (Seg 0): {bar_strap.axial_force}")
        print(f"Skin Bar Force (Seg 0): {bar_skin.axial_force}")
        print(f"Fastener 1 Stiffness: {solution.fasteners[0].stiffness}")
        
if __name__ == "__main__":
    unittest.main()
