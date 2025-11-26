"""Core 1-D joint model (bars + fastener springs)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Dict, List, Optional, Sequence, Tuple, Set, Any

from .fasteners import (
    boeing69_compliance, 
    huth_compliance, 
    grumman_compliance, 
    swift_douglas_compliance, 
    tate_rosenfeld_compliance,
    morris_compliance
)
from .linalg import extract_submatrix, extract_vector, solve_dense


@dataclass
class Plate:
    """A single laminate/plate participating in the joint."""

    name: str
    E: float
    t: float
    first_row: int
    last_row: int
    A_strip: List[float]
    widths: Optional[List[float]] = None
    thicknesses: Optional[List[float]] = None
    Fx_left: float = 0.0
    Fx_right: float = 0.0

    def segment_count(self) -> int:
        return max(0, self.last_row - self.first_row)


@dataclass
class FastenerRow:
    row: int
    D: float
    Eb: float
    nu_b: float
    method: str = "Boeing69"
    k_manual: Optional[float] = None
    connections: Optional[List[Tuple[int, int]]] = None


@dataclass
class FastenerResult:
    row: int
    plate_i: int
    plate_j: int
    compliance: float
    stiffness: float
    force: float
    dof_i: int
    dof_j: int
    dof_j: int
    bearing_force_upper: float = 0.0
    bearing_force_lower: float = 0.0
    modulus: float = 0.0
    diameter: float = 0.0
    quantity: float = 1.0
    t1: float = 0.0
    t2: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class NodeResult:
    plate_id: int
    plate_name: str
    local_node: int
    x: float
    displacement: float
    net_bypass: float
    thickness: float
    bypass_area: Optional[float]
    order: int = 0
    multiple_thickness: bool = False
    row: int = 0
    legacy_id: int = 0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BarResult:
    plate_id: int
    plate_name: str
    segment: int
    axial_force: float
    stiffness: float
    modulus: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BearingBypassResult:
    row: int
    plate_name: str
    bearing: float
    bypass: float
    flow_left: float = 0.0
    flow_right: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ReactionResult:
    plate_id: int
    plate_name: str
    local_node: int
    global_node: int
    reaction: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class JointSolution:
    displacements: List[float]
    stiffness_matrix: List[List[float]]
    force_vector: List[float]
    fasteners: List[FastenerResult]
    bearing_bypass: List[BearingBypassResult]
    nodes: List[NodeResult]
    bars: List[BarResult]
    reactions: List[ReactionResult]
    dof_map: Dict[Tuple[int, int], int]
    plates: List[Plate] = field(default_factory=list)
    applied_forces: List[Dict[str, Any]] = field(default_factory=list)

    def fasteners_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "ID": f"{1000 * (fastener.plate_i + 1) + fastener.row}-{1000 * (fastener.plate_j + 1) + fastener.row}",
                "Load": abs(fastener.force),
                "Brg Force Upper": fastener.bearing_force_upper,
                "Brg Force Lower": fastener.bearing_force_lower,
                "Stiffness": fastener.stiffness,
                "Modulus": fastener.modulus,
                "Diameter": fastener.diameter,
                "Quantity": fastener.quantity,
                "Thickness Node 1": fastener.t1,
                "Thickness Node 2": fastener.t2,
                "Row": f"{fastener.row} ({fastener.plate_i}-{fastener.plate_j})", # Keep for internal use if needed
            }
            for index, fastener in enumerate(self.fasteners)
        ]

    def nodes_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "Node ID": node.legacy_id,
                "X Location": node.x,
                "Displacement": node.displacement,
                "Net Bypass Load": node.net_bypass,
                "Thickness": node.thickness,
                "Bypass Area": node.bypass_area,
                "Order": node.order,
                "Multiple Thickness": node.multiple_thickness,
                "plate_id": node.plate_id,
                "Plate": node.plate_name,
                "local_node": node.local_node,
            }
            for node in self.nodes
        ]

    def bars_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "ID": f"{1000 * (bar.plate_id + 1) + (bar.segment + 1)}-{1000 * (bar.plate_id + 1) + (bar.segment + 2)}",
                "Axial Force": bar.axial_force,
                "Stiffness": bar.stiffness,
                "Modulus": bar.modulus,
                "plate_id": bar.plate_id,
                "Plate": bar.plate_name,
                "seg": bar.segment,
            }
            for bar in self.bars
        ]

    def bearing_bypass_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "Row": item.row,
                "Plate": item.plate_name,
                "Bearing [lb]": item.bearing,
                "Bypass [lb]": item.bypass,
            }
            for item in self.bearing_bypass
        ]

    def reactions_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "Node ID": 1000 * (reaction.plate_id + 1) + (reaction.local_node + 1),
                "Force": reaction.reaction,
                "plate_id": reaction.plate_id,
                "Plate": reaction.plate_name,
                "local_node": reaction.local_node,
                "Global node": reaction.global_node,
            }
            for reaction in self.reactions
        ]

    def classic_results_as_dicts(self) -> List[Dict[str, float]]:
        results = []
        
        # Calculate Reference Load P
        # P is typically the total load transferred through the joint.
        # We calculate it as the maximum of (Sum of Rightward Loads) and (Sum of Leftward Loads)
        # to account for equilibrium (e.g. 1000 lb tension = 1000 lb right + 1000 lb left reaction).
        # We include both Plate End Loads and Point Forces.
        
        sum_right = 0.0
        sum_left = 0.0
        
        for p in self.plates:
            # Fx_right and Fx_left are defined as positive -> right
            if p.Fx_right > 0: sum_right += p.Fx_right
            else: sum_left += abs(p.Fx_right)
                
            if p.Fx_left > 0: sum_right += p.Fx_left
            else: sum_left += abs(p.Fx_left)
            
        for f in self.applied_forces:
            val = f.get("Value", 0.0)
            if val > 0: sum_right += val
            else: sum_left += abs(val)
            
        # If the system is in equilibrium with reactions, the reactions are not in applied_forces.
        # However, usually P is the "Applied Load".
        # If we have 1000 lb right and a support, sum_right=1000, sum_left=0. P=1000.
        # If we have 1000 lb right and 1000 lb left (explicit), sum_right=1000, sum_left=1000. P=1000.
        P = max(sum_right, sum_left)
        
        if P < 1e-9:
            P = 1.0 # Avoid division by zero
            
        fastener_rows = {f.row for f in self.fasteners}
        diam_by_row = {}
        for f in self.fasteners:
            if f.diameter > 0:
                diam_by_row[f.row] = f.diameter
        
        # Reconstruct node bearing map from fasteners
        node_bearing_map = {}
        for f in self.fasteners:
            node_bearing_map[(f.plate_i, f.row)] = f.bearing_force_upper
            node_bearing_map[(f.plate_j, f.row)] = f.bearing_force_lower

        plate_name_to_idx = {p.name: i for i, p in enumerate(self.plates)}
        
        # Create a map of BearingBypass results for quick lookup
        bb_map = {(bb.plate_name, bb.row): bb for bb in self.bearing_bypass}
        
        # Iterate through all nodes to ensure we capture every point of interest
        for node in self.nodes:
            # We only care about nodes that are part of the fastening/load transfer region?
            # The legacy table seems to show all nodes in the fastened region.
            # If a node is not in a fastener row, does it appear? 
            # In the screenshot, rows 2,3,4,5,6,7,8 are shown. These are fastener rows.
            # If there are nodes without fasteners, they might not be "Classic Results" relevant 
            # unless we want to show bypass stresses everywhere.
            # However, the user request specifically mentioned "Instead of row, there should be nodes".
            # And the screenshot shows "3002, 3003..." which correspond to fastener locations.
            # Let's stick to nodes that are at fastener rows for now, or maybe all nodes?
            # The legacy code filtered by `if bb.row not in fastener_rows`.
            # Let's keep that filter for "Classic Results" as it implies fastener interaction.
            
            if node.row not in fastener_rows:
                continue

            bb = bb_map.get((node.plate_name, node.row))
            if not bb:
                continue
                
            diameter = diam_by_row.get(node.row, 0.0)
            thickness = node.thickness
            area = node.bypass_area if node.bypass_area else 0.0
            
            plate_idx = plate_name_to_idx.get(node.plate_name)
            transfer = node_bearing_map.get((plate_idx, node.row), 0.0)
            
            # Legacy definition: Incoming = Transfer + Bypass (downstream)
            # Use absolute values for legacy reporting
            bypass = abs(bb.flow_right)
            incoming = transfer + bypass
            
            detail_stress = max(incoming, bypass) / area if area > 0 else 0.0
            bearing_stress = transfer / (diameter * thickness) if diameter * thickness > 0 else 0.0
            
            results.append({
                "Element": node.plate_name,
                "Node": node.legacy_id,
                "Row": node.row, # Keep for debug/reference if needed
                "Thickness": thickness,
                "Area": area,
                "Incoming Load": incoming,
                "Bypass Load": bypass,
                "Load Transfer": transfer,
                "L.Trans / P": transfer / P if P != 0 else 0.0,
                "Detail Stress": detail_stress,
                "Bearing Stress": bearing_stress,
                "Fbr / FDetail": bearing_stress / detail_stress if detail_stress > 1e-6 else 0.0,
            })
            
        # Sort by Element (Plate Name) then Node ID
        results.sort(key=lambda x: (x["Element"], x["Node"]))
            
        return results


class Joint1D:
    """Solve a one-dimensional load transfer problem."""

    def __init__(self, pitches: Sequence[float], plates: Sequence[Plate], fasteners: Sequence[FastenerRow]):
        self.pitches = list(pitches)
        self.plates = list(plates)
        self.fasteners = list(fasteners)
        self._dof: Dict[Tuple[int, int], int] = {}
        self._x: Dict[Tuple[int, int], float] = {}

    def _calculate_compliance_pairwise(
        self, 
        plate_i: Plate, 
        plate_j: Plate, 
        fastener: FastenerRow,
        t_i: float,
        t_j: float
    ) -> float:
        """
        Calculates total compliance between two plates based on the selected method.
        Correctly handles non-linear thickness dependencies (Huth, Grumman).
        """
        method = fastener.method.lower()
        
        # Manual Override
        if method == "manual" and fastener.k_manual:
            return 1.0 / fastener.k_manual

        # --- Empirical Methods (Total Connection Compliance) ---
        
        if "huth" in method:
            j_type = "bolted_metal"
            if "rivet" in method: j_type = "riveted_metal"
            elif "graphite" in method or "composite" in method: j_type = "bolted_graphite"
            
            return huth_compliance(
                t1=t_i, E1=plate_i.E,
                t2=t_j, E2=plate_j.E,
                Ef=fastener.Eb, diameter=fastener.D,
                shear="single",
                joint_type=j_type
            )

        elif "grumman" in method:
            return grumman_compliance(t_i, plate_i.E, t_j, plate_j.E, fastener.Eb, fastener.D)

        elif "swift" in method or "douglas" in method or "vought" in method:
            return swift_douglas_compliance(t_i, plate_i.E, t_j, plate_j.E, fastener.Eb, fastener.D)

        elif "tate" in method or "rosenfeld" in method:
            return tate_rosenfeld_compliance(t_i, plate_i.E, t_j, plate_j.E, fastener.Eb, fastener.D)
            
        elif "delft" in method or "morris" in method:
            return morris_compliance(t_i, plate_i.E, t_j, plate_j.E, fastener.Eb, fastener.D)

        # --- Component Methods (Boeing) ---
        else: 
            return boeing69_compliance(
                ti=t_i, Ei=plate_i.E,
                tj=t_j, Ej=plate_j.E,
                Eb=fastener.Eb, nu_b=fastener.nu_b,
                diameter=fastener.D
            )


    def _build_dofs(self) -> int:
        self._dof.clear()
        self._x.clear()
        ndof = 0
        for plate_index, plate in enumerate(self.plates):
            start_index = max(plate.first_row - 1, 0)
            x0 = sum(self.pitches[:start_index])
            xs = [x0]
            end_index = min(max(plate.last_row - 1, start_index), len(self.pitches))
            for segment_index in range(start_index, end_index):
                xs.append(xs[-1] + self.pitches[segment_index])
            for local_node, position in enumerate(xs):
                self._dof[(plate_index, local_node)] = ndof
                self._x[(plate_index, local_node)] = position
                ndof += 1
        
        for fastener in self.fasteners:
            self._dof[("fastener", fastener.row)] = ndof
            ndof += 1
            
        return ndof

    def _plates_at_row(self, row_index: int) -> List[int]:
        present = [
            idx
            for idx, plate in enumerate(self.plates)
            if plate.first_row <= row_index <= plate.last_row
        ]
        present.sort()
        return present

    def _calculate_single_stiffness(self, plate: Plate, fastener: FastenerRow, t_local: Optional[float] = None) -> Tuple[float, float]:
        t = t_local if t_local is not None else plate.t
        
        method = fastener.method.lower()
        
        if method == "manual":
            stiffness = fastener.k_manual if fastener.k_manual is not None else 1.0e6
            compliance = 1.0 / stiffness
            return compliance, stiffness
            
        elif "huth" in method:
            # Huth method
            # Approximation for star model: Calculate compliance of a symmetric joint (ti=tj=t)
            # and assign half to this plate.
            joint_type = "metal"
            if "graphite" in method or "composite" in method:
                joint_type = "composite"
            
            # Use 'single' shear for the symmetric pair calculation as a baseline unit
            c_sym = huth_compliance(
                ti=t, Ei=plate.E,
                tj=t, Ej=plate.E,
                Ef=fastener.Eb,
                diameter=fastener.D,
                shear="single",
                joint_type=joint_type
            )
            compliance = c_sym / 2.0
            
        elif "grumman" in method:
            # Grumman method
            # Approximation: Symmetric joint / 2
            c_sym = grumman_compliance(
                ti=t, Ei=plate.E,
                tj=t, Ej=plate.E,
                Ef=fastener.Eb,
                diameter=fastener.D
            )
            compliance = c_sym / 2.0
            
        else:
            # Boeing69 (Default)
            # Existing logic for partial compliance
            A_b = math.pi * (fastener.D / 2.0) ** 2
            I_b = math.pi * (fastener.D / 2.0) ** 4 / 4.0
            G_b = fastener.Eb / (2.0 * (1.0 + fastener.nu_b))

            t_shear = min(t, (fastener.D + t) / 2)
            C_shear = (4.0 * t_shear) / (9.0 * G_b * A_b)
            
            t_bending = min(t, fastener.D)
            C_bending = (6.0 * t_bending**3) / (40.0 * fastener.Eb * I_b)
            
            C_bearing = (1.0 / t) * (1.0 / fastener.Eb + 1.0 / plate.E)
                
            compliance = C_shear + C_bending + C_bearing
            
        stiffness = 1.0 / compliance if compliance > 0.0 else 1e12
        return compliance, stiffness

    def solve(
        self,
        supports: Sequence[Tuple[int, int, float]],
        point_forces: Optional[Sequence[Tuple[int, int, float]]] = None,
    ) -> JointSolution:
        point_forces = point_forces or []
        ndof = self._build_dofs()
        
        stiffness_matrix = [[0.0 for _ in range(ndof)] for _ in range(ndof)]
        force_vector = [0.0 for _ in range(ndof)]

        def _validate_support(plate_index: int, local_node: int) -> None:
            if plate_index < 0 or plate_index >= len(self.plates):
                raise ValueError(f"Support references invalid plate index {plate_index}")
            segments = self.plates[plate_index].segment_count()
            if local_node < 0 or local_node > segments:
                raise ValueError(
                    f"Support for plate '{self.plates[plate_index].name}' expects a local node between 0 and {segments};"
                    f" received {local_node}"
                )

        def _validate_force(plate_index: int, local_node: int) -> None:
            if plate_index < 0 or plate_index >= len(self.plates):
                raise ValueError(f"Force references invalid plate index {plate_index}")
            segments = self.plates[plate_index].segment_count()
            if local_node < 0 or local_node > segments:
                raise ValueError(
                    f"Force for plate '{self.plates[plate_index].name}' expects a local node between 0 and {segments};"
                    f" received {local_node}"
                )

        validated_supports = []
        for plate_index, local_node, value in supports:
            _validate_support(int(plate_index), int(local_node))
            validated_supports.append((int(plate_index), int(local_node), float(value)))

        validated_forces = []
        for plate_index, local_node, value in point_forces:
            _validate_force(int(plate_index), int(local_node))
            validated_forces.append((int(plate_index), int(local_node), float(value)))

        for plate_index, plate in enumerate(self.plates):
            segments = plate.segment_count()
            if len(plate.A_strip) != segments:
                raise ValueError(f"Plate '{plate.name}' expects {segments} bypass areas; received {len(plate.A_strip)}")
            for segment in range(segments):
                length = self.pitches[plate.first_row - 1 + segment]
                area = plate.A_strip[segment]
                k_bar = plate.E * area / length
                left = self._dof[(plate_index, segment)]
                right = self._dof[(plate_index, segment + 1)]
                stiffness_matrix[left][left] += k_bar
                stiffness_matrix[left][right] -= k_bar
                stiffness_matrix[right][left] -= k_bar
                stiffness_matrix[right][right] += k_bar
            force_vector[self._dof[(plate_index, 0)]] += plate.Fx_left
            force_vector[self._dof[(plate_index, segments)]] += plate.Fx_right

        springs: List[Tuple[int, int, float, int, int, float]] = [] 
        
        for fastener in self.fasteners:
            row_index = fastener.row
            present_indices = self._plates_at_row(row_index)
            if not present_indices:
                continue

            dof_fastener = self._dof[("fastener", row_index)]
            
            plates_at_row = []
            for plate_idx, plate in enumerate(self.plates):
                if plate.first_row <= row_index <= plate.last_row:
                    plates_at_row.append((plate_idx, plate))
            
            plates_at_row.sort(key=lambda x: x[0])
            
            # Check for Empirical Method
            method_key = fastener.method.lower()
            is_empirical = any(k in method_key for k in ["huth", "grumman", "swift", "douglas", "vought", "tate", "rosenfeld", "morris"])

            if is_empirical:
                # Pairwise Logic
                for i in range(len(plates_at_row) - 1):
                    idx_1, plate_1 = plates_at_row[i]
                    idx_2, plate_2 = plates_at_row[i+1]

                    # Resolve Thicknesses
                    t1 = plate_1.t
                    if plate_1.thicknesses:
                        ln1 = row_index - plate_1.first_row
                        s1 = ln1 if ln1 < len(plate_1.thicknesses) else ln1 - 1
                        if 0 <= s1 < len(plate_1.thicknesses): t1 = plate_1.thicknesses[s1]
                    
                    t2 = plate_2.t
                    if plate_2.thicknesses:
                        ln2 = row_index - plate_2.first_row
                        s2 = ln2 if ln2 < len(plate_2.thicknesses) else ln2 - 1
                        if 0 <= s2 < len(plate_2.thicknesses): t2 = plate_2.thicknesses[s2]

                    compliance = self._calculate_compliance_pairwise(plate_1, plate_2, fastener, t1, t2)
                    if compliance <= 1e-12: compliance = 1e-12
                    
                    # Star Model: Node stiffness = 2 * K_total
                    stiffness_total = 1.0 / compliance
                    node_stiffness = stiffness_total * 2.0

                    # Add to Plate 1
                    local_node_1 = row_index - plate_1.first_row
                    dof_1 = self._dof.get((idx_1, local_node_1))
                    if dof_1 is not None:
                        springs.append((dof_1, dof_fastener, node_stiffness, row_index, idx_1, compliance / 2.0))
                        
                        stiffness_matrix[dof_1][dof_1] += node_stiffness
                        stiffness_matrix[dof_1][dof_fastener] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_1] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_fastener] += node_stiffness

                    # Add to Plate 2
                    local_node_2 = row_index - plate_2.first_row
                    dof_2 = self._dof.get((idx_2, local_node_2))
                    if dof_2 is not None:
                        springs.append((dof_2, dof_fastener, node_stiffness, row_index, idx_2, compliance / 2.0))
                        
                        stiffness_matrix[dof_2][dof_2] += node_stiffness
                        stiffness_matrix[dof_2][dof_fastener] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_2] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_fastener] += node_stiffness
            else:
                plate_compliances = {}
                for plate_idx, plate in plates_at_row:
                    local_node = row_index - plate.first_row
                    t_local = plate.t
                    if plate.thicknesses:
                        seg_idx = local_node if local_node < len(plate.thicknesses) else local_node - 1
                        if 0 <= seg_idx < len(plate.thicknesses):
                            t_local = plate.thicknesses[seg_idx]
                    
                    compliance, stiffness = self._calculate_single_stiffness(plate, fastener, t_local)
                    plate_compliances[plate_idx] = compliance

                if len(plates_at_row) > 1 and "boeing" in fastener.method.lower():
                    I_b = math.pi * (fastener.D / 2.0) ** 4 / 4.0
                    factor = 5.0 / (40.0 * fastener.Eb * I_b)
                    
                    for i in range(len(plates_at_row) - 1):
                        p_idx_1, plate_1 = plates_at_row[i]
                        p_idx_2, plate_2 = plates_at_row[i+1]
                        
                        t1_local = plate_1.t
                        if plate_1.thicknesses:
                            ln1 = row_index - plate_1.first_row
                            s1 = ln1 if ln1 < len(plate_1.thicknesses) else ln1 - 1
                            if 0 <= s1 < len(plate_1.thicknesses): t1_local = plate_1.thicknesses[s1]
                            
                        t2_local = plate_2.t
                        if plate_2.thicknesses:
                            ln2 = row_index - plate_2.first_row
                            s2 = ln2 if ln2 < len(plate_2.thicknesses) else ln2 - 1
                            if 0 <= s2 < len(plate_2.thicknesses): t2_local = plate_2.thicknesses[s2]

                        t1_bend = min(t1_local, fastener.D)
                        t2_bend = min(t2_local, fastener.D)
                        
                        excess_compliance = factor * (t1_bend - t2_bend)**2 * (t1_bend + t2_bend)
                        
                        is_first_pair = (i == 0)
                        is_last_pair = (i == len(plates_at_row) - 2)
                        
                        if is_first_pair:
                            plate_compliances[p_idx_1] -= excess_compliance
                        elif is_last_pair:
                            plate_compliances[p_idx_2] -= excess_compliance
                        else:
                            plate_compliances[p_idx_1] -= excess_compliance / 2.0
                            plate_compliances[p_idx_2] -= excess_compliance / 2.0

                for plate_idx, plate in plates_at_row:
                    compliance = plate_compliances[plate_idx]
                    if compliance <= 1e-12:
                        compliance = 1e-12
                    
                    stiffness = 1.0 / compliance
                    
                    local_node = row_index - plate.first_row
                    dof_plate = self._dof.get((plate_idx, local_node))
                    
                    if dof_plate is not None:
                        springs.append((dof_plate, dof_fastener, stiffness, row_index, plate_idx, compliance))
                        
                        stiffness_matrix[dof_plate][dof_plate] += stiffness
                        stiffness_matrix[dof_plate][dof_fastener] -= stiffness
                        stiffness_matrix[dof_fastener][dof_plate] -= stiffness
                        stiffness_matrix[dof_fastener][dof_fastener] += stiffness

        for plate_index, local_node, magnitude in validated_forces:
            force_vector[self._dof[(plate_index, local_node)]] += magnitude

        fixed_dofs = [
            self._dof[(plate_index, local_node)] for plate_index, local_node, _ in validated_supports
        ]
        prescribed_values = [value for _, _, value in validated_supports]
        if not fixed_dofs:
            raise ValueError("At least one support must be defined to avoid rigid body motion")

        free_dofs = [idx for idx in range(ndof) if idx not in fixed_dofs]
        stiffness_ff = extract_submatrix(stiffness_matrix, free_dofs, free_dofs)
        stiffness_fp = extract_submatrix(stiffness_matrix, free_dofs, fixed_dofs)
        force_f = extract_vector(force_vector, free_dofs)
        rhs = [force_f[i] - sum(stiffness_fp[i][j] * prescribed_values[j] for j in range(len(fixed_dofs))) for i in range(len(free_dofs))]
        displacement_free = solve_dense(stiffness_ff, rhs)

        displacements = [0.0 for _ in range(ndof)]
        for idx, value in zip(free_dofs, displacement_free):
            displacements[idx] = value
        for idx, value in zip(fixed_dofs, prescribed_values):
            displacements[idx] = value

        springs_by_row: Dict[int, List[Tuple[int, int, float, int, float]]] = {}
        for dof_plate, dof_fast, stiffness, row_index, plate_idx, compliance in springs:
            if row_index not in springs_by_row:
                springs_by_row[row_index] = []
            springs_by_row[row_index].append((dof_plate, dof_fast, stiffness, plate_idx, compliance))

        fastener_results: List[FastenerResult] = []
        
        # Track total bearing load per node
        node_bearing_loads: Dict[Tuple[int, int], float] = {}
        
        # Temporary storage for calculating forces before creating FastenerResult objects
        temp_fastener_data = []

        for row_index, row_springs in springs_by_row.items():
            row_springs.sort(key=lambda x: x[3])
            
            star_forces = []
            for dof_plate, dof_fast, k, p_idx, c in row_springs:
                f = k * (displacements[dof_plate] - displacements[dof_fast])
                star_forces.append(f)
                
                # Bearing Force Logic: Cumulative sum of absolute shear forces acting on the node.
                # This aggregates load from ALL interfaces connected to this plate at this row.
                key = (p_idx, row_index)
                node_bearing_loads[key] = node_bearing_loads.get(key, 0.0) + abs(f)
                
            for i in range(len(row_springs) - 1):
                dof_plate_i, dof_fast_i, k_i, p_idx_i, c_i = row_springs[i]
                dof_plate_j, dof_fast_j, k_j, p_idx_j, c_j = row_springs[i+1]
                
                c_eff = c_i + c_j
                k_eff = 1.0 / c_eff if c_eff > 0 else 1e12
                f_eff = sum(star_forces[:i+1])
                
                t_i = self.plates[p_idx_i].t
                if self.plates[p_idx_i].thicknesses:
                    ln = row_index - self.plates[p_idx_i].first_row
                    s = ln if ln < len(self.plates[p_idx_i].thicknesses) else ln - 1
                    if 0 <= s < len(self.plates[p_idx_i].thicknesses): t_i = self.plates[p_idx_i].thicknesses[s]
                
                t_j = self.plates[p_idx_j].t
                if self.plates[p_idx_j].thicknesses:
                    ln = row_index - self.plates[p_idx_j].first_row
                    s = ln if ln < len(self.plates[p_idx_j].thicknesses) else ln - 1
                    if 0 <= s < len(self.plates[p_idx_j].thicknesses): t_j = self.plates[p_idx_j].thicknesses[s]

                
                temp_fastener_data.append({
                    "row": row_index,
                    "pi": p_idx_i,
                    "pj": p_idx_j,
                    "c": c_eff,
                    "k": k_eff,
                    "f": f_eff,
                    "di": dof_plate_i,
                    "dj": dof_plate_j,
                    "ti": t_i,
                    "tj": t_j
                })

        # Create FastenerResult objects with total bearing loads
        for item in temp_fastener_data:
            # Look up fastener properties
            # We need to find the FastenerRow object for this row
            fastener_obj = next((f for f in self.fasteners if f.row == item["row"]), None)
            
            brg_upper = node_bearing_loads.get((item["pi"], item["row"]), 0.0)
            brg_lower = node_bearing_loads.get((item["pj"], item["row"]), 0.0)
            
            fastener_results.append(
                FastenerResult(
                    row=item["row"],
                    plate_i=item["pi"],
                    plate_j=item["pj"],
                    compliance=item["c"],
                    stiffness=item["k"],
                    force=item["f"],
                    dof_i=item["di"],
                    dof_j=item["dj"],
                    bearing_force_upper=brg_upper,
                    bearing_force_lower=brg_lower,
                    modulus=fastener_obj.Eb if fastener_obj else 0.0,
                    diameter=fastener_obj.D if fastener_obj else 0.0,
                    quantity=1.0,
                    t1=item["ti"],
                    t2=item["tj"],
                )
            )

        fastener_results.sort(key=lambda item: (item.row, item.plate_i))

        bearing_results: List[BearingBypassResult] = []
        for row_index in range(1, len(self.pitches) + 1):
            for plate_index, plate in enumerate(self.plates):
                if not (plate.first_row <= row_index <= plate.last_row):
                    continue
                segments = plate.segment_count()
                flow_left = 0.0
                left_segment = row_index - 1 - plate.first_row
                if left_segment >= 0:
                    length = self.pitches[plate.first_row - 1 + left_segment]
                    area = plate.A_strip[left_segment]
                    stiffness_bar = plate.E * area / length
                    dof_left = self._dof[(plate_index, left_segment)]
                    dof_right = self._dof[(plate_index, left_segment + 1)]
                    flow_left = stiffness_bar * (displacements[dof_right] - displacements[dof_left])
                flow_right = 0.0
                right_segment = row_index - plate.first_row
                if right_segment < segments:
                    length = self.pitches[plate.first_row - 1 + right_segment]
                    area = plate.A_strip[right_segment]
                    stiffness_bar = plate.E * area / length
                    dof_left = self._dof[(plate_index, right_segment)]
                    dof_right = self._dof[(plate_index, right_segment + 1)]
                    flow_right = stiffness_bar * (displacements[dof_right] - displacements[dof_left])
                bearing = flow_right - flow_left
                bypass = flow_left if abs(flow_left) <= abs(flow_right) else flow_right
                bearing_results.append(BearingBypassResult(row=row_index, plate_name=plate.name, bearing=bearing, bypass=bypass, flow_left=flow_left, flow_right=flow_right))

        node_results: List[NodeResult] = []
        for plate_index, plate in enumerate(self.plates):
            segments = plate.segment_count()
            for local_node in range(segments + 1):
                dof_index = self._dof[(plate_index, local_node)]
                position = self._x[(plate_index, local_node)]
                
                # Calculate Net Bypass (Boeing Definition)
                force_left = 0.0
                if local_node > 0:
                    length = self.pitches[plate.first_row - 1 + (local_node - 1)]
                    area = plate.A_strip[local_node - 1]
                    stiffness_bar = plate.E * area / length
                    dof_left = self._dof[(plate_index, local_node - 1)]
                    force_left = stiffness_bar * (displacements[dof_index] - displacements[dof_left])
                
                force_right = 0.0
                if local_node < segments:
                    length = self.pitches[plate.first_row - 1 + local_node]
                    area = plate.A_strip[local_node]
                    stiffness_bar = plate.E * area / length
                    dof_right = self._dof[(plate_index, local_node + 1)]
                    force_right = stiffness_bar * (displacements[dof_right] - displacements[dof_index])
                
                net_bypass = min(abs(force_left), abs(force_right))
                
                bypass_area = plate.A_strip[local_node] if local_node < segments else plate.A_strip[-1]
                t_node = plate.t
                multiple_t = False
                if plate.thicknesses:
                    t_left = plate.thicknesses[local_node - 1] if local_node > 0 else None
                    t_right = plate.thicknesses[local_node] if local_node < segments else None
                    
                    if t_left is not None and t_right is not None and abs(t_left - t_right) > 1e-9:
                        multiple_t = True
                    
                    if t_right is not None:
                        t_node = t_right
                    elif t_left is not None:
                        t_node = t_left
                        
                node_results.append(
                    NodeResult(
                        plate_id=plate_index,
                        plate_name=plate.name,
                        local_node=local_node,
                        x=position,
                        displacement=displacements[dof_index],
                        net_bypass=net_bypass,
                        thickness=t_node,
                        bypass_area=bypass_area,
                        order=0,
                        multiple_thickness=multiple_t,
                        row=plate.first_row + local_node,
                        legacy_id=1000 * (plate_index + 1) + (plate.first_row + local_node)
                    )
                )

        bar_results: List[BarResult] = []
        for plate_index, plate in enumerate(self.plates):
            segments = plate.segment_count()
            for segment in range(segments):
                length = self.pitches[plate.first_row - 1 + segment]
                area = plate.A_strip[segment]
                stiffness_bar = plate.E * area / length
                dof_left = self._dof[(plate_index, segment)]
                dof_right = self._dof[(plate_index, segment + 1)]
                axial_force = stiffness_bar * (displacements[dof_right] - displacements[dof_left])
                bar_results.append(
                    BarResult(
                        plate_id=plate_index,
                        plate_name=plate.name,
                        segment=segment,
                        axial_force=axial_force,
                        stiffness=stiffness_bar,
                        modulus=plate.E,
                    )
                )

        reaction_results: List[ReactionResult] = []
        for plate_index, local_node, _ in supports:
            dof = self._dof[(plate_index, local_node)]
            reaction = sum(stiffness_matrix[dof][j] * displacements[j] for j in range(ndof)) - force_vector[dof]
            plate = self.plates[plate_index]
            global_node = plate.first_row + local_node
            reaction_results.append(
                ReactionResult(
                    plate_id=plate_index,
                    plate_name=plate.name,
                    local_node=local_node,
                    global_node=global_node,
                    reaction=reaction,
                )
            )

        return JointSolution(
            displacements=displacements,
            stiffness_matrix=stiffness_matrix,
            force_vector=force_vector,
            fasteners=fastener_results,
            bearing_bypass=bearing_results,
            nodes=node_results,
            bars=bar_results,
            reactions=reaction_results,
            dof_map=dict(self._dof),
            plates=list(self.plates),
            applied_forces=[
                {
                    "Node ID": 1000 * (pi + 1) + (ln + 1),
                    "Value": val,
                    "Reference Node": f"{1000 * (pi + 1) + (ln + 1)}",
                    "Plate": self.plates[pi].name
                }
                for pi, ln, val in validated_forces
            ]
        )



__all__ = [
    "Plate",
    "FastenerRow",
    "FastenerResult",
    "NodeResult",
    "BarResult",
    "BearingBypassResult",
    "ReactionResult",
    "JointSolution",
    "Joint1D",
]
