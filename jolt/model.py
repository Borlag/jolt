"""Core 1-D joint model (bars + fastener springs)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Dict, List, Optional, Sequence, Tuple, Set

from .fasteners import boeing69_compliance, grumman_compliance, huth_compliance
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
    applied_forces: List[Dict[str, Any]] = field(default_factory=list)

    def fasteners_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "Row": f"{fastener.row} ({fastener.plate_i}-{fastener.plate_j})", # Keeping for internal ID if needed, or just use Row index
                "Load": fastener.force,
                "Brg Force Upper": fastener.bearing_force_upper,
                "Brg Force Lower": fastener.bearing_force_lower,
                "Stiffness": fastener.stiffness,
                "Modulus": fastener.modulus,
                "Diameter": fastener.diameter,
                "Quantity": fastener.quantity,
                "Thickness Node 1": fastener.t1,
                "Thickness Node 2": fastener.t2,
                # Hidden/Classic fields
                "Fastener": f"F{fastener.row} ({fastener.plate_i}-{fastener.plate_j})",
                "CF [in/lb]": fastener.compliance,
                "k [lb/in]": fastener.stiffness,
                "F [lb]": fastener.force,
            }
            for index, fastener in enumerate(self.fasteners)
        ]

    def nodes_as_dicts(self) -> List[Dict[str, float]]:
        # Create a unique ID for each node: 1000 * (plate_id + 1) + local_node + 1 (assuming 1-based indexing for display)
        # Actually, let's follow the screenshot: 2002, 2003...
        # Plate 1 (Skin) -> 3000 series? Plate 0 (Tripler) -> 2000?
        # Let's use 1000 * (plate_id + 1) + (local_node + 1)
        return [
            {
                "Node ID": 1000 * (node.plate_id + 1) + (node.local_node + 1),
                "X Location": node.x,
                "Displacement": node.displacement,
                "Net Bypass Load": node.net_bypass,
                "Thickness": node.thickness,
                "Bypass Area": node.bypass_area,
                "Order": node.order,
                "Multiple Thickness": node.multiple_thickness,
                # Legacy
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
                "Force": bar.axial_force,
                "Stiffness": bar.stiffness,
                "Modulus": bar.modulus,
                # Legacy
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
                # Legacy
                "plate_id": reaction.plate_id,
                "Plate": reaction.plate_name,
                "local_node": reaction.local_node,
                "Global node": reaction.global_node,
            }
            for reaction in self.reactions
        ]

    def classic_results_as_dicts(self) -> List[Dict[str, float]]:
        results = []
        # Get unique rows from fastener results
        fastener_rows = {f.row for f in self.fasteners}
        
        # Map row -> diameter (from FastenerResult)
        diam_by_row = {}
        for f in self.fasteners:
            if f.diameter > 0:
                diam_by_row[f.row] = f.diameter
        
        # Map (plate_name, row) -> NodeResult
        node_map = {(n.plate_name, n.row): n for n in self.nodes}
        
        for bb in self.bearing_bypass:
            if bb.row not in fastener_rows:
                continue
            
            node_res = node_map.get((bb.plate_name, bb.row))
            if not node_res:
                continue
                
            diameter = diam_by_row.get(bb.row, 0.0)
            thickness = node_res.thickness
            area = node_res.bypass_area if node_res.bypass_area else 0.0
            
            incoming = bb.flow_left
            bypass = bb.flow_right
            transfer = abs(bb.bearing)
            
            detail_stress = max(abs(incoming), abs(bypass)) / area if area > 0 else 0.0
            bearing_stress = transfer / (diameter * thickness) if diameter * thickness > 0 else 0.0
            
            results.append({
                "Thickness": thickness,
                "Area": area,
                "No of Fasteners": 1.0,
                "Fastener Diameter": diameter,
                "Incoming Load": incoming,
                "Bypass Load": bypass,
                "Load Transfer": transfer,
                "L.Trans / P": transfer / abs(incoming) if abs(incoming) > 1e-6 else 0.0,
                "Detail Stress": detail_stress,
                "Bearing Stress": bearing_stress,
                "Fbr / FDetail": bearing_stress / detail_stress if detail_stress > 1e-6 else 0.0,
                "Row": bb.row,
                "Plate": bb.plate_name,
            })
            
        return results


class Joint1D:
    """Solve a one-dimensional load transfer problem."""

    def __init__(self, pitches: Sequence[float], plates: Sequence[Plate], fasteners: Sequence[FastenerRow]):
        self.pitches = list(pitches)
        self.plates = list(plates)
        self.fasteners = list(fasteners)
        self._dof: Dict[Tuple[int, int], int] = {}
        self._x: Dict[Tuple[int, int], float] = {}

    # --- degree-of-freedom helpers -------------------------------------------------
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
        
        # Add DOFs for fastener nodes (one per row)
        for fastener in self.fasteners:
            self._dof[("fastener", fastener.row)] = ndof
            ndof += 1
            
        return ndof

    # --- compliance helpers --------------------------------------------------------
    def _plates_at_row(self, row_index: int) -> List[int]:
        present = [
            idx
            for idx, plate in enumerate(self.plates)
            if plate.first_row <= row_index <= plate.last_row
        ]
        present.sort()
        return present

    def _compliance_for_pair(
        self,
        fastener: FastenerRow,
        top: Plate,
        bottom: Plate,
        ti: float,
        tj: float,
        shear_planes: int,
        *,
        shear_ti: Optional[float] = None,
        shear_tj: Optional[float] = None,
        bending_ti: Optional[float] = None,
        bending_tj: Optional[float] = None,
        bearing_ti: Optional[float] = None,
        bearing_tj: Optional[float] = None,
    ) -> Tuple[float, float]:
        if fastener.method == "Manual":
            if fastener.k_manual is None or fastener.k_manual <= 0.0:
                raise ValueError("Manual fastener rows require a positive stiffness value")
            stiffness = float(fastener.k_manual)
            return 1.0 / stiffness, stiffness
        if fastener.method == "Boeing69":
            compliance = boeing69_compliance(
                ti,
                top.E,
                tj,
                bottom.E,
                fastener.Eb,
                fastener.nu_b,
                fastener.D,
                shear_planes=shear_planes,
                shear_ti=shear_ti,
                shear_tj=shear_tj,
                bending_ti=bending_ti,
                bending_tj=bending_tj,
                bearing_ti=bearing_ti,
                bearing_tj=bearing_tj,
            )
        elif fastener.method == "Huth_metal":
            shear_mode = "double" if shear_planes > 1 else "single"
            compliance = huth_compliance(
                ti,
                top.E,
                tj,
                bottom.E,
                fastener.Eb,
                fastener.D,
                shear_mode,
                "bolted_metal",
            )
        elif fastener.method == "Huth_graphite":
            shear_mode = "double" if shear_planes > 1 else "single"
            compliance = huth_compliance(
                ti,
                top.E,
                tj,
                bottom.E,
                fastener.Eb,
                fastener.D,
                shear_mode,
                "bolted_graphite",
            )
        elif fastener.method == "Grumman":
            compliance = grumman_compliance(ti, top.E, tj, bottom.E, fastener.Eb, fastener.D)
        else:
            compliance = boeing69_compliance(
                ti,
                top.E,
                tj,
                bottom.E,
                fastener.Eb,
                fastener.nu_b,
                fastener.D,
                shear_planes=shear_planes,
                shear_ti=shear_ti,
                shear_tj=shear_tj,
                bending_ti=bending_ti,
                bending_tj=bending_tj,
                bearing_ti=bearing_ti,
                bearing_tj=bearing_tj,
            )
        stiffness = 1.0 / compliance if compliance > 0.0 else 1e12
        return compliance, stiffness

    def _calculate_single_stiffness(self, plate: Plate, fastener: FastenerRow, t_local: Optional[float] = None) -> Tuple[float, float]:
        """Calculate stiffness of a single plate connecting to the fastener node."""
        t = t_local if t_local is not None else plate.t
        # Calculate fastener properties
        A_b = math.pi * (fastener.D / 2.0) ** 2
        I_b = math.pi * (fastener.D / 2.0) ** 4 / 4.0
        G_b = fastener.Eb / (2.0 * (1.0 + fastener.nu_b))

        # 1. Shear: Cap thickness at Diameter (Boeing D6-29942 Thick Plate Rule)
        # Physical Justification: For t > D, the effective shear path is limited by the bolt diameter.
        t_shear = min(t, (fastener.D + t) / 2)
        C_shear = (4.0 * t_shear) / (9.0 * G_b * A_b)
        
        # 2. Bending: Cap thickness at Diameter
        t_bending = min(t, fastener.D)
        C_bending = (6.0 * t_bending**3) / (40.0 * fastener.Eb * I_b)
        
        # 3. Bearing: Single plate bearing uses full t (Standard)
        C_bearing = (1.0 / t) * (1.0 / fastener.Eb + 1.0 / plate.E)
            
        compliance = C_shear + C_bending + C_bearing
        stiffness = 1.0 / compliance if compliance > 0.0 else 1e12
        return compliance, stiffness

    def _resolve_fastener_pairs(
        self,
        fastener: FastenerRow,
        row_index: int,
        present: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, int]]:
        present = list(present) if present is not None else self._plates_at_row(row_index)
        if len(present) < 2:
            return []

        if fastener.connections is None:
            return list(zip(present[:-1], present[1:]))

        order = {plate_idx: position for position, plate_idx in enumerate(present)}
        resolved: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()

        for pair in fastener.connections:
            if len(pair) != 2:
                raise ValueError("Fastener connections must be defined as pairs of plate indices")
            raw_top, raw_bottom = int(pair[0]), int(pair[1])
            if raw_top not in order or raw_bottom not in order:
                raise ValueError(
                    f"Fastener at row {row_index} cannot connect plates {raw_top} and {raw_bottom}; at least one plate is not present"
                )
            if raw_top == raw_bottom:
                raise ValueError("Fastener connections require two different plates")
            top_idx, bottom_idx = (raw_top, raw_bottom)
            if order[top_idx] > order[bottom_idx]:
                top_idx, bottom_idx = bottom_idx, top_idx
            if abs(order[top_idx] - order[bottom_idx]) != 1:
                raise ValueError("Fastener connections must reference adjacent plates at the selected row")
            key = (top_idx, bottom_idx)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(key)

        return resolved

    # --- solution ------------------------------------------------------------------
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



        # 1. Assemble Plate Stiffness (Bars)
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

        # 2. Assemble Fastener Stiffness (Star Topology)
        # Store springs for result generation: (dof_plate, dof_fast, k, row, plate_idx, compliance)
        springs: List[Tuple[int, int, float, int, int, float]] = [] 
        
        for fastener in self.fasteners:
            row_index = fastener.row
            present_indices = self._plates_at_row(row_index)
            if not present_indices:
                continue

            dof_fastener = self._dof[("fastener", row_index)]
            
            # 2. Identify plates at this row and sort them
            plates_at_row = []
            for plate_idx, plate in enumerate(self.plates):
                if plate.first_row <= row_index <= plate.last_row:
                    plates_at_row.append((plate_idx, plate))
            
            # Sort by plate index to establish neighbor relationships
            plates_at_row.sort(key=lambda x: x[0])
            
            # 3. Calculate base compliances and apply corrections
            # Base compliance (Star Topology)
            plate_compliances = {}
            for plate_idx, plate in plates_at_row:
                # Determine local thickness
                local_node = row_index - plate.first_row
                # If local_node is at the end of a segment, which thickness to use?
                # Fastener is at a node.
                # If thickness varies, we should probably use the average or min, or the one specified for the "land".
                # Let's use the thickness of the segment to the right (local_node) if available, else left (local_node-1).
                # Note: local_node goes from 0 to segments.
                # If local_node < segments, use thicknesses[local_node].
                # Else use thicknesses[local_node-1].
                t_local = plate.t
                if plate.thicknesses:
                    seg_idx = local_node if local_node < len(plate.thicknesses) else local_node - 1
                    if 0 <= seg_idx < len(plate.thicknesses):
                        t_local = plate.thicknesses[seg_idx]
                
                compliance, stiffness = self._calculate_single_stiffness(plate, fastener, t_local)
                plate_compliances[plate_idx] = compliance

            # Apply Coupling Correction
            # The Star model (6t^3) is too compliant for unequal thicknesses compared to Boeing (coupled).
            # We subtract the excess compliance from the interfaces.
            # Excess = 5 * (ti - tj)^2 * (ti + tj) / (40 * Eb * Ib)
            if len(plates_at_row) > 1:
                I_b = math.pi * (fastener.D / 2.0) ** 4 / 4.0
                factor = 5.0 / (40.0 * fastener.Eb * I_b)
                
                for i in range(len(plates_at_row) - 1):
                    p_idx_1, plate_1 = plates_at_row[i]
                    p_idx_2, plate_2 = plates_at_row[i+1]
                    
                    # Use capped thickness for bending correction to match base springs
                    # We need t_local for these plates too
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
                    
                    # Excess = 5 * (ti - tj)^2 * (ti + tj) / (40 E I)
                    excess_compliance = factor * (t1_bend - t2_bend)**2 * (t1_bend + t2_bend)
                    
                    # DISTRIBUTE CORRECTION
                    # To avoid coupling errors in 3-plate stacks (where the middle plate 
                    # affects both interfaces), we push the correction to the "outer" 
                    # plates of the stack whenever possible.
                    
                    is_first_pair = (i == 0)
                    is_last_pair = (i == len(plates_at_row) - 2)
                    
                    if is_first_pair:
                        # First pair: Subtract full excess from Top Plate
                        plate_compliances[p_idx_1] -= excess_compliance
                    elif is_last_pair:
                        # Last pair: Subtract full excess from Bottom Plate
                        plate_compliances[p_idx_2] -= excess_compliance
                    else:
                        # Middle pairs (rare in standard joints): Split correction
                        plate_compliances[p_idx_1] -= excess_compliance / 2.0
                        plate_compliances[p_idx_2] -= excess_compliance / 2.0

            # 4. Assemble Stiffness Matrix
            for plate_idx, plate in plates_at_row:
                # Get corrected compliance
                compliance = plate_compliances[plate_idx]
                # Ensure compliance doesn't become non-physical (negative or zero)
                if compliance <= 1e-12:
                    compliance = 1e-12
                
                stiffness = 1.0 / compliance
                
                # Get DOFs
                local_node = row_index - plate.first_row
                dof_plate = self._dof.get((plate_idx, local_node))
                
                if dof_plate is not None:
                    springs.append((dof_plate, dof_fastener, stiffness, row_index, plate_idx, compliance))
                    
                    # Add to global stiffness matrix
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

        # Group springs by row for Virtual Pair processing
        springs_by_row: Dict[int, List[Tuple[int, int, float, int, float]]] = {}
        for dof_plate, dof_fast, stiffness, row_index, plate_idx, compliance in springs:
            if row_index not in springs_by_row:
                springs_by_row[row_index] = []
            springs_by_row[row_index].append((dof_plate, dof_fast, stiffness, plate_idx, compliance))

        fastener_results: List[FastenerResult] = []
        
        for row_index, row_springs in springs_by_row.items():
            # Sort by plate index (assuming plate index corresponds to stack order)
            # This is critical for defining "intervals" correctly.
            row_springs.sort(key=lambda x: x[3]) # Sort by plate_idx
            
            # Calculate individual forces first
            star_forces = []
            for dof_plate, dof_fast, k, p_idx, c in row_springs:
                f = k * (displacements[dof_plate] - displacements[dof_fast])
                star_forces.append(f)
                
            # Generate Virtual Pair results (Intervals)
            # For N plates, we have N-1 intervals.
            # Interval i connects Plate i and Plate i+1.
            for i in range(len(row_springs) - 1):
                # Data for Plate i (Top)
                dof_plate_i, dof_fast_i, k_i, p_idx_i, c_i = row_springs[i]
                # Data for Plate i+1 (Bottom)
                dof_plate_j, dof_fast_j, k_j, p_idx_j, c_j = row_springs[i+1]
                
                # Effective Stiffness of the pair: Series combination of the two star links
                # K_eff = 1 / (C_i + C_j)
                c_eff = c_i + c_j
                k_eff = 1.0 / c_eff if c_eff > 0 else 1e12
                
                # Effective Force: Cumulative load from top
                # The shear force transferring across the interface between Plate i and Plate i+1
                # is the sum of all loads entering the fastener from plates above the interface.
                f_eff = sum(star_forces[:i+1])
                
                # Get thicknesses for result
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

                fastener_results.append(
                    FastenerResult(
                        row=row_index,
                        plate_i=p_idx_i,
                        plate_j=p_idx_j,
                        compliance=c_eff,
                        stiffness=k_eff,
                        force=f_eff,
                        dof_i=dof_plate_i, # Keep DOFs for reference, though force is composite
                        dof_j=dof_plate_j,
                        bearing_force_upper=star_forces[i],
                        bearing_force_lower=star_forces[i+1],
                        modulus=self.fasteners[self._dof[("fastener", row_index)] - ndof + len(self.fasteners)].Eb if "fastener" in str(self._dof) else 0.0, # Tricky to get fastener obj here efficiently without lookup
                        # Actually we have 'fastener' object in the outer loop? No, we iterated springs_by_row.
                        # We need to look up fastener by row_index.
                        diameter=0.0, # Placeholder, will update below
                        quantity=1.0,
                        t1=t_i,
                        t2=t_j,
                    )
                )
        
        # Update fastener details (diameter, modulus)
        fastener_map = {f.row: f for f in self.fasteners}
        for res in fastener_results:
            f_obj = fastener_map.get(res.row)
            if f_obj:
                res.diameter = f_obj.D
                res.modulus = f_obj.Eb

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
                bypass_flow = 0.0
                if local_node > 0:
                    length = self.pitches[plate.first_row - 1 + (local_node - 1)]
                    area = plate.A_strip[local_node - 1]
                    stiffness_bar = plate.E * area / length
                    dof_left = self._dof[(plate_index, local_node - 1)]
                    bypass_flow += stiffness_bar * (displacements[dof_index] - displacements[dof_left])
                if local_node < segments:
                    length = self.pitches[plate.first_row - 1 + local_node]
                    area = plate.A_strip[local_node]
                    stiffness_bar = plate.E * area / length
                    dof_right = self._dof[(plate_index, local_node + 1)]
                    bypass_flow += stiffness_bar * (displacements[dof_right] - displacements[dof_index])
                bypass_flow *= 0.5
                bypass_area = plate.A_strip[local_node] if local_node < segments else plate.A_strip[-1]
                t_node = plate.t
                multiple_t = False
                if plate.thicknesses:
                    # Check if thickness changes at this node (if it's an internal node)
                    t_left = plate.thicknesses[local_node - 1] if local_node > 0 else None
                    t_right = plate.thicknesses[local_node] if local_node < segments else None
                    
                    if t_left is not None and t_right is not None and abs(t_left - t_right) > 1e-9:
                        multiple_t = True
                    
                    # For display, prefer right, then left
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
                        net_bypass=bypass_flow,
                        thickness=t_node,
                        bypass_area=bypass_area,
                        order=0,
                        multiple_thickness=multiple_t,
                        row=plate.first_row + local_node,
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
            applied_forces=[
                {
                    "Node ID": 1000 * (pi + 1) + (ln + 1),
                    "Value": val,
                    "Reference Node": f"{1000 * (pi + 1) + (ln + 1)}", # Matches Boeing "Reference Node"
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
