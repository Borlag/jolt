"""Tests for 2D Beam FEM solver."""
import math
import pytest
from jolt.beam_fem import (
    fastener_beam_diameter,
    circular_beam_properties,
    axial_element_stiffness,
    beam_shear_element_stiffness,
    BeamFEMSolver,
    JointBeamFEM,
)


class TestBeamDiameterFormula:
    """Test the D6-29942 fastener beam diameter formula."""
    
    def test_boeing_example_diameter(self):
        """
        Verify the Boeing example from page 109.
        
        Given:
        - C_F = 5.968e-6 in/lb (from Swift equation)
        - E_fast = 10.5e6 psi
        - L_m = 1.0 in
        
        Expected: D_model = 0.406 in
        """
        C_F = 5.968e-6
        E_fast = 10.5e6
        L_m = 1.0
        
        D = fastener_beam_diameter(C_F, E_fast, L_m)
        
        assert abs(D - 0.406) < 0.01, f"Expected D=0.406, got D={D:.4f}"
    
    def test_compliance_roundtrip(self):
        """Verify that converting to diameter and back gives same compliance."""
        C_F = 5.0e-6
        E_fast = 10e6
        L_m = 1.0
        
        # Convert to diameter
        D = fastener_beam_diameter(C_F, E_fast, L_m)
        
        # Calculate compliance from beam shear formula
        I = math.pi * (D / 2.0) ** 4 / 4.0
        C_back = L_m ** 3 / (12.0 * E_fast * I)
        
        assert abs(C_back - C_F) / C_F < 0.001, f"Roundtrip error: {C_F} vs {C_back}"


class TestBeamElementStiffness:
    """Test beam element stiffness matrices."""
    
    def test_axial_stiffness_symmetry(self):
        """Axial stiffness matrix should be symmetric."""
        k = axial_element_stiffness(E=10e6, A=0.1, L=1.0)
        assert abs(k[0][1] - k[1][0]) < 1e-10
    
    def test_axial_stiffness_value(self):
        """Check axial stiffness value: k = EA/L."""
        E, A, L = 10e6, 0.1, 2.0
        k = axial_element_stiffness(E, A, L)
        expected = E * A / L
        
        assert abs(k[0][0] - expected) < 1e-6
        assert abs(k[1][1] - expected) < 1e-6
        assert abs(k[0][1] + expected) < 1e-6  # Off-diagonal is negative
    
    def test_shear_stiffness_value(self):
        """Check beam shear stiffness: k = 12EI/L^3."""
        E, I, L = 10e6, 0.001, 1.0
        k = beam_shear_element_stiffness(E, I, L)
        expected = 12.0 * E * I / (L ** 3)
        
        assert abs(k[0][0] - expected) < 1e-6


class TestSimpleTwoNodeProblem:
    """Test simple FEM problems."""
    
    def test_two_node_axial(self):
        """
        Simple two-node axial problem.
        
        Node 1 fixed, Node 2 has load P.
        Expected: u2 = P * L / (E * A)
        """
        from jolt.beam_fem import BeamElement
        
        solver = BeamFEMSolver()
        solver.add_node(1, 0.0, 0.0)
        solver.add_node(2, 1.0, 0.0)
        
        E, A = 10e6, 0.1
        I = 0.001  # Not used for axial
        
        elem = BeamElement(
            id=1, node_i=1, node_j=2, E=E, A=A, I=I, L=1.0, is_fastener=False
        )
        solver.elements.append(elem)
        
        P = 1000.0
        u, forces = solver.solve(supports=[(1, 0.0)], loads=[(2, P)])
        
        expected_u2 = P * 1.0 / (E * A)
        actual_u2 = u[solver._dof_map[2]]
        
        assert abs(actual_u2 - expected_u2) < 1e-9, f"Expected u={expected_u2}, got {actual_u2}"


class TestBoeingFEAExample:
    """
    Test against Boeing FEA example from pages 109-115.
    
    2-layer joint with 4 fasteners:
    - Doubler: 0.05 x 0.75 Al
    - Skin: 0.04 x 0.75 Al
    - Load: 750 lb on each end
    - Pitch: 1.0 in typical
    
    Expected fastener loads: F1=225, F2=121, F3=133, F4=272 lbs
    """
    
    def test_boeing_750lb_example(self):
        """Full Boeing 750 lb example."""
        # Material properties
        E_plate = 10.4e6  # psi
        E_fast = 10.5e6   # psi (BACR15BB6D)
        G_fast = 3.95e6   # psi
        
        # Geometry
        width = 0.75       # in
        t_doubler = 0.05   # in
        t_skin = 0.04      # in
        D_fast = 0.1875    # in (3/16")
        pitch = 1.0        # in
        
        # Swift compliance (from page 109)
        # C_F = [A + B * (D/td + D/ts)] / (E * D)
        # With A=5, B=0.80
        A_swift, B_swift = 5.0, 0.80
        C_F = (A_swift + B_swift * (D_fast / t_doubler + D_fast / t_skin)) / (E_plate * D_fast)
        
        # Build FEM model
        solver = BeamFEMSolver()
        
        # Nodes: Doubler (y=1) nodes 1-5, Skin (y=0) nodes 6-10
        # x positions: 0, 1, 2, 3, 4 inches
        for i in range(5):
            solver.add_node(i + 1, x=float(i), y=1.0)  # Doubler
            solver.add_node(i + 6, x=float(i), y=0.0)  # Skin
        
        # Plate elements
        from jolt.beam_fem import BeamElement
        A_doubler = width * t_doubler
        A_skin = width * t_skin
        I_doubler = width * t_doubler ** 3 / 12.0
        I_skin = width * t_skin ** 3 / 12.0
        
        elem_id = 1
        # Doubler elements (horizontal, nodes 1-2, 2-3, 3-4, 4-5)
        for i in range(4):
            solver.elements.append(BeamElement(
                id=elem_id, node_i=i+1, node_j=i+2,
                E=E_plate, A=A_doubler, I=I_doubler, L=pitch, is_fastener=False
            ))
            elem_id += 1
        
        # Skin elements (horizontal, nodes 6-7, 7-8, 8-9, 9-10)
        for i in range(4):
            solver.elements.append(BeamElement(
                id=elem_id, node_i=i+6, node_j=i+7,
                E=E_plate, A=A_skin, I=I_skin, L=pitch, is_fastener=False
            ))
            elem_id += 1
        
        # Fastener elements (vertical, nodes 2-7, 3-8, 4-9, 5-10)
        # Use compliance to get beam properties
        D_model = fastener_beam_diameter(C_F, E_fast, L_m=1.0)
        A_fast = math.pi * (D_model / 2.0) ** 2
        I_fast = math.pi * (D_model / 2.0) ** 4 / 4.0
        L_m = 1.0  # Model length
        
        fastener_elem_ids = []
        for i in range(4):
            eid = elem_id
            solver.elements.append(BeamElement(
                id=eid, node_i=i+2, node_j=i+7,
                E=E_fast, A=A_fast, I=I_fast, L=L_m, is_fastener=True
            ))
            fastener_elem_ids.append(eid)
            elem_id += 1
        
        # Boundary conditions: Node 1 fixed (doubler left end)
        # Loads: 750 lb pulling left on node 1 (reaction), 750 lb right on node 10
        # Actually: Load applied at ends, with left end fixed
        
        # Fix node 1 (doubler left)
        # Fix node 6 (skin left) - wait, looking at the diagram, only skin is loaded
        
        # Re-reading page 115: 750 lb pulls LEFT on doubler (node 1)
        # 750 lb pulls RIGHT on skin (node 10)
        # Node 1 is supported (fixed), skin receives the load
        
        # Solve
        u, forces = solver.solve(
            supports=[(1, 0.0)],  # Fix doubler left end
            loads=[(10, 750.0)]   # 750 lb on skin right end
        )
        
        # Get fastener forces
        f1 = abs(forces[fastener_elem_ids[0]])
        f2 = abs(forces[fastener_elem_ids[1]])
        f3 = abs(forces[fastener_elem_ids[2]])
        f4 = abs(forces[fastener_elem_ids[3]])
        
        print(f"\nBoeing 750 lb Example Results:")
        print(f"  F1 = {f1:.1f} lb (expected 225)")
        print(f"  F2 = {f2:.1f} lb (expected 121)")
        print(f"  F3 = {f3:.1f} lb (expected 133)")
        print(f"  F4 = {f4:.1f} lb (expected 272)")
        print(f"  Total = {f1+f2+f3+f4:.1f} lb (expected 750)")
        
        # Check sum equals applied load
        total = f1 + f2 + f3 + f4
        assert abs(total - 750.0) < 1.0, f"Total should be 750, got {total}"
        
        # Check individual forces (allow 10% tolerance)
        assert abs(f1 - 225) < 35, f"F1 expected ~225, got {f1}"
        assert abs(f2 - 121) < 20, f"F2 expected ~121, got {f2}"
        assert abs(f3 - 133) < 20, f"F3 expected ~133, got {f3}"
        assert abs(f4 - 272) < 40, f"F4 expected ~272, got {f4}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
