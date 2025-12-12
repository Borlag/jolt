"""
2D Beam FEM Solver for Joint Load Distribution.

This module implements a 2D Plane Frame FEM approach following Boeing D6-29942,
where fasteners are modeled as beam elements with shear deflection stiffness.

Key concepts:
- Plates: Horizontal beam elements with axial stiffness (EA/L)
- Fasteners: Vertical beam elements with shear deflection stiffness
- Fastener beam diameter calculated to match Swift/Boeing69 compliance
"""
import math
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class Node2D:
    """A node in the 2D FEM model."""
    id: int
    x: float
    y: float


@dataclass  
class BeamElement:
    """A 2D beam element connecting two nodes."""
    id: int
    node_i: int  # Start node ID
    node_j: int  # End node ID
    E: float     # Elastic modulus
    A: float     # Cross-sectional area
    I: float     # Moment of inertia (for bending - used for shear calc)
    L: float     # Length
    is_fastener: bool = False  # True for fastener elements
    
    @classmethod
    def from_nodes(cls, elem_id: int, n_i: Node2D, n_j: Node2D, E: float, A: float, I: float, is_fastener: bool = False):
        """Create element from node objects."""
        L = math.sqrt((n_j.x - n_i.x)**2 + (n_j.y - n_i.y)**2)
        return cls(id=elem_id, node_i=n_i.id, node_j=n_j.id, E=E, A=A, I=I, L=L, is_fastener=is_fastener)


def fastener_beam_diameter(C_F: float, E_fast: float, L_m: float = 1.0) -> float:
    """
    Calculate FEM beam diameter to match fastener compliance.
    
    From D6-29942:
    D_model = (L_m^3 * 64 / (12 * E_fast * C_F * pi))^0.25
    
    Args:
        C_F: Fastener compliance from Swift/Boeing69 (in/lb)
        E_fast: Fastener modulus of elasticity (psi)
        L_m: Model length for fastener beam (typically 1.0 for convenience)
    
    Returns:
        Beam diameter (in)
    """
    if C_F <= 0:
        raise ValueError("Compliance must be positive")
    
    numerator = L_m ** 3 * 64.0
    denominator = 12.0 * E_fast * C_F * math.pi
    
    return (numerator / denominator) ** 0.25


def circular_beam_properties(D: float) -> Tuple[float, float]:
    """
    Calculate area and moment of inertia for circular beam.
    
    Args:
        D: Diameter
    
    Returns:
        (Area, Moment of Inertia)
    """
    A = math.pi * (D / 2.0) ** 2
    I = math.pi * (D / 2.0) ** 4 / 4.0
    return A, I


def rectangular_beam_properties(width: float, height: float) -> Tuple[float, float]:
    """
    Calculate area and moment of inertia for rectangular beam.
    
    Args:
        width: Beam width (perpendicular to load)
        height: Beam height (in load direction, i.e., plate thickness)
    
    Returns:
        (Area, Moment of Inertia)
    """
    A = width * height
    I = width * height ** 3 / 12.0
    return A, I


def axial_element_stiffness(E: float, A: float, L: float) -> List[List[float]]:
    """
    2x2 local stiffness matrix for axial-only element (truss element).
    
    DOFs: [u_i, u_j] (axial displacements)
    
    k = EA/L * [[1, -1], [-1, 1]]
    """
    k = E * A / L
    return [
        [k, -k],
        [-k, k]
    ]


def beam_shear_element_stiffness(E: float, I: float, L: float) -> List[List[float]]:
    """
    2x2 local stiffness matrix for beam in pure shear (lateral deflection only).
    
    Based on fixed-fixed beam shear deflection:
    delta = P * L^3 / (12 * E * I)
    
    So k_shear = 12 * E * I / L^3
    
    DOFs: [v_i, v_j] (lateral displacements)
    """
    k = 12.0 * E * I / (L ** 3)
    return [
        [k, -k],
        [-k, k]
    ]


class BeamFEMSolver:
    """
    2D Beam FEM solver for joint analysis.
    
    Models plates as axial elements and fasteners as shear beam elements.
    """
    
    def __init__(self):
        self.nodes: Dict[int, Node2D] = {}
        self.elements: List[BeamElement] = []
        self.n_dof = 0
        
        # Node ID to DOF mapping
        # For plates: x-displacement (axial)
        # For fasteners: y-displacement (shear)
        self._dof_map: Dict[int, int] = {}
        
    def add_node(self, node_id: int, x: float, y: float) -> Node2D:
        """Add a node to the model."""
        node = Node2D(id=node_id, x=x, y=y)
        self.nodes[node_id] = node
        return node
    
    def add_plate_element(self, elem_id: int, node_i_id: int, node_j_id: int, 
                          E: float, width: float, thickness: float) -> BeamElement:
        """
        Add a plate (horizontal axial) element.
        
        Args:
            elem_id: Element ID
            node_i_id, node_j_id: Node IDs at element ends
            E: Plate modulus
            width: Joint width (pitch perpendicular to drawing)
            thickness: Plate thickness
        """
        n_i = self.nodes[node_i_id]
        n_j = self.nodes[node_j_id]
        A, I = rectangular_beam_properties(width, thickness)
        elem = BeamElement.from_nodes(elem_id, n_i, n_j, E, A, I, is_fastener=False)
        self.elements.append(elem)
        return elem
    
    def add_fastener_element(self, elem_id: int, node_i_id: int, node_j_id: int,
                             E: float, compliance: float, L_m: float = 1.0) -> BeamElement:
        """
        Add a fastener (vertical shear beam) element.
        
        Args:
            elem_id: Element ID
            node_i_id, node_j_id: Node IDs (on different plates)
            E: Fastener modulus
            compliance: Fastener compliance from Swift/Boeing69
            L_m: Model length for fastener beam
        """
        n_i = self.nodes[node_i_id]
        n_j = self.nodes[node_j_id]
        
        # Calculate beam diameter from compliance
        D = fastener_beam_diameter(compliance, E, L_m)
        A, I = circular_beam_properties(D)
        
        elem = BeamElement(
            id=elem_id,
            node_i=node_i_id,
            node_j=node_j_id,
            E=E,
            A=A,
            I=I,
            L=L_m,
            is_fastener=True
        )
        self.elements.append(elem)
        return elem
    
    def _build_dof_map(self) -> int:
        """
        Build DOF mapping for all nodes.
        
        Each node gets one DOF (x-displacement for plates, shared for connections).
        Returns total number of DOFs.
        """
        self._dof_map.clear()
        dof = 0
        for node_id in sorted(self.nodes.keys()):
            self._dof_map[node_id] = dof
            dof += 1
        return dof
    
    def _assemble_global_matrix(self, n_dof: int) -> Tuple[List[List[float]], List[float]]:
        """
        Assemble global stiffness matrix and force vector.
        
        Returns (K, F) where K is n_dof x n_dof and F is length n_dof.
        """
        K = [[0.0] * n_dof for _ in range(n_dof)]
        F = [0.0] * n_dof
        
        for elem in self.elements:
            dof_i = self._dof_map[elem.node_i]
            dof_j = self._dof_map[elem.node_j]
            
            if elem.is_fastener:
                # Fastener: shear stiffness
                k_local = beam_shear_element_stiffness(elem.E, elem.I, elem.L)
            else:
                # Plate: axial stiffness
                k_local = axial_element_stiffness(elem.E, elem.A, elem.L)
            
            # Add to global matrix
            dofs = [dof_i, dof_j]
            for i_local, i_global in enumerate(dofs):
                for j_local, j_global in enumerate(dofs):
                    K[i_global][j_global] += k_local[i_local][j_local]
        
        return K, F
    
    def _apply_bc(self, K: List[List[float]], F: List[float], 
                  supports: List[Tuple[int, float]]) -> None:
        """
        Apply boundary conditions (fixed nodes).
        
        Args:
            K: Global stiffness matrix (modified in place)
            F: Force vector (modified in place)
            supports: List of (node_id, prescribed_displacement)
        """
        n_dof = len(F)
        for node_id, disp in supports:
            dof = self._dof_map.get(node_id)
            if dof is None:
                continue
            
            # Penalty method for fixed DOF
            large = 1e30
            K[dof][dof] = large
            F[dof] = large * disp
    
    def _apply_loads(self, F: List[float], loads: List[Tuple[int, float]]) -> None:
        """
        Apply nodal loads.
        
        Args:
            F: Force vector (modified in place)
            loads: List of (node_id, force)
        """
        for node_id, force in loads:
            dof = self._dof_map.get(node_id)
            if dof is not None:
                F[dof] += force
    
    def _solve_system(self, K: List[List[float]], F: List[float]) -> List[float]:
        """Solve Ku = F using Gaussian elimination."""
        n = len(F)
        # Create augmented matrix
        M = [row[:] + [F[i]] for i, row in enumerate(K)]
        
        # Forward elimination
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(M[row][col]) > abs(M[max_row][col]):
                    max_row = row
            M[col], M[max_row] = M[max_row], M[col]
            
            # Eliminate
            for row in range(col + 1, n):
                if abs(M[col][col]) > 1e-15:
                    factor = M[row][col] / M[col][col]
                    for j in range(col, n + 1):
                        M[row][j] -= factor * M[col][j]
        
        # Back substitution
        u = [0.0] * n
        for i in range(n - 1, -1, -1):
            u[i] = M[i][n]
            for j in range(i + 1, n):
                u[i] -= M[i][j] * u[j]
            if abs(M[i][i]) > 1e-15:
                u[i] /= M[i][i]
        
        return u
    
    def _recover_element_forces(self, u: List[float]) -> Dict[int, float]:
        """
        Recover element forces from displacement solution.
        
        Returns dict of element_id -> force.
        """
        forces = {}
        for elem in self.elements:
            dof_i = self._dof_map[elem.node_i]
            dof_j = self._dof_map[elem.node_j]
            
            u_i = u[dof_i]
            u_j = u[dof_j]
            
            if elem.is_fastener:
                # Shear force = k * delta
                k = 12.0 * elem.E * elem.I / (elem.L ** 3)
                force = k * (u_j - u_i)
            else:
                # Axial force = EA/L * delta
                k = elem.E * elem.A / elem.L
                force = k * (u_j - u_i)
            
            forces[elem.id] = force
        
        return forces
    
    def solve(self, supports: List[Tuple[int, float]], 
              loads: List[Tuple[int, float]]) -> Tuple[List[float], Dict[int, float]]:
        """
        Solve the FEM system.
        
        Args:
            supports: List of (node_id, prescribed_displacement)
            loads: List of (node_id, force)
        
        Returns:
            (displacements, element_forces)
        """
        n_dof = self._build_dof_map()
        K, F = self._assemble_global_matrix(n_dof)
        self._apply_loads(F, loads)
        self._apply_bc(K, F, supports)
        u = self._solve_system(K, F)
        forces = self._recover_element_forces(u)
        return u, forces
    
    def get_node_displacement(self, node_id: int, u: List[float]) -> float:
        """Get displacement for a specific node."""
        dof = self._dof_map.get(node_id)
        if dof is not None:
            return u[dof]
        return 0.0


class JointBeamFEM:
    """
    High-level interface for modeling joints using 2D Beam FEM.
    
    Converts plate/fastener descriptions to FEM nodes and elements.
    """
    
    def __init__(self, pitches: List[float], joint_width: float = 1.0):
        """
        Initialize joint model.
        
        Args:
            pitches: Fastener pitches (distances between rows)
            joint_width: Width of joint strip (for plate area calculation)
        """
        self.pitches = pitches
        self.joint_width = joint_width
        self.solver = BeamFEMSolver()
        
        # Tracking
        self._plate_nodes: Dict[int, List[int]] = {}  # plate_idx -> [node_ids]
        self._fastener_elements: Dict[int, List[int]] = {}  # row -> [elem_ids]
        self._next_node_id = 1
        self._next_elem_id = 1
        
    def add_plate(self, plate_idx: int, E: float, thickness: float,
                  first_row: int, last_row: int) -> List[int]:
        """
        Add a plate to the model.
        
        Creates nodes at each row and elements between them.
        
        Returns list of node IDs for this plate.
        """
        # Calculate x positions from pitches
        x_positions = [0.0]
        for i, p in enumerate(self.pitches):
            x_positions.append(x_positions[-1] + p)
        
        # Y position based on plate index (stacked vertically)
        y = float(plate_idx)
        
        node_ids = []
        prev_node_id = None
        
        for row in range(first_row, last_row + 1):
            if row - 1 < len(x_positions):
                x = x_positions[row - 1]
            else:
                x = x_positions[-1]
            
            node_id = self._next_node_id
            self._next_node_id += 1
            self.solver.add_node(node_id, x, y)
            node_ids.append(node_id)
            
            # Add plate element to previous node
            if prev_node_id is not None:
                pitch_idx = row - first_row - 1
                if 0 <= pitch_idx < len(self.pitches):
                    elem_id = self._next_elem_id
                    self._next_elem_id += 1
                    self.solver.add_plate_element(
                        elem_id, prev_node_id, node_id,
                        E, self.joint_width, thickness
                    )
            
            prev_node_id = node_id
        
        self._plate_nodes[plate_idx] = node_ids
        return node_ids
    
    def add_fastener(self, row: int, plate_pairs: List[Tuple[int, int]],
                     E_fast: float, compliances: List[float], L_m: float = 1.0) -> List[int]:
        """
        Add fastener connections at a row.
        
        Args:
            row: Fastener row number
            plate_pairs: List of (plate_i, plate_j) pairs to connect
            E_fast: Fastener modulus
            compliances: Compliance for each pair
            L_m: Model length for fastener beam
        
        Returns list of element IDs for fasteners at this row.
        """
        elem_ids = []
        
        for (plate_i, plate_j), compliance in zip(plate_pairs, compliances):
            nodes_i = self._plate_nodes.get(plate_i, [])
            nodes_j = self._plate_nodes.get(plate_j, [])
            
            # Find node at this row for each plate
            # Row index in node list depends on plate's first_row
            # For simplicity, assume first node is at plate's first_row
            # This will need refinement for actual use
            
            if len(nodes_i) == 0 or len(nodes_j) == 0:
                continue
            
            # Find the node at this row
            node_i_id = None
            node_j_id = None
            
            for nid in nodes_i:
                node = self.solver.nodes[nid]
                node_row = self._get_row_from_x(node.x)
                if node_row == row:
                    node_i_id = nid
                    break
            
            for nid in nodes_j:
                node = self.solver.nodes[nid]
                node_row = self._get_row_from_x(node.x)
                if node_row == row:
                    node_j_id = nid
                    break
            
            if node_i_id is not None and node_j_id is not None:
                elem_id = self._next_elem_id
                self._next_elem_id += 1
                self.solver.add_fastener_element(
                    elem_id, node_i_id, node_j_id, E_fast, compliance, L_m
                )
                elem_ids.append(elem_id)
        
        self._fastener_elements[row] = elem_ids
        return elem_ids
    
    def _get_row_from_x(self, x: float) -> int:
        """Convert x position back to row number."""
        cum_x = 0.0
        for i, p in enumerate(self.pitches):
            if abs(x - cum_x) < 1e-9:
                return i + 1
            cum_x += p
        if abs(x - cum_x) < 1e-9:
            return len(self.pitches) + 1
        return -1
    
    def solve(self, supports: List[Tuple[int, float]], 
              loads: List[Tuple[int, float]]) -> Tuple[List[float], Dict[int, float]]:
        """Solve the joint model."""
        return self.solver.solve(supports, loads)
    
    def get_fastener_forces(self, element_forces: Dict[int, float]) -> Dict[int, List[float]]:
        """
        Get fastener forces organized by row.
        
        Returns dict of row -> [force1, force2, ...]
        """
        result = {}
        for row, elem_ids in self._fastener_elements.items():
            forces = [element_forces.get(eid, 0.0) for eid in elem_ids]
            result[row] = forces
        return result
