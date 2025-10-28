"""Core 1-D joint model (bars + fastener springs)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
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

    def fasteners_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "Row": fastener.row,
                "Fastener": f"F{index + 1}",
                "CF [in/lb]": fastener.compliance,
                "k [lb/in]": fastener.stiffness,
                "F [lb]": fastener.force,
                "iDOF": fastener.dof_i,
                "jDOF": fastener.dof_j,
            }
            for index, fastener in enumerate(self.fasteners)
        ]

    def nodes_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "plate_id": node.plate_id,
                "Plate": node.plate_name,
                "local_node": node.local_node,
                "X [in]": node.x,
                "u [in]": node.displacement,
                "Net Bypass [lb]": node.net_bypass,
                "t [in]": node.thickness,
                "Bypass Area [in^2]": node.bypass_area,
            }
            for node in self.nodes
        ]

    def bars_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "plate_id": bar.plate_id,
                "Plate": bar.plate_name,
                "seg": bar.segment,
                "Force [lb]": bar.axial_force,
                "k_bar [lb/in]": bar.stiffness,
                "E [psi]": bar.modulus,
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
                "plate_id": reaction.plate_id,
                "Plate": reaction.plate_name,
                "local_node": reaction.local_node,
                "Global node": reaction.global_node,
                "Reaction [lb]": reaction.reaction,
            }
            for reaction in self.reactions
        ]


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

        springs: List[Tuple[int, int, float, int, int, int, float]] = []
        for fastener in self.fasteners:
            row_index = fastener.row
            present = self._plates_at_row(row_index)
            pairs = self._resolve_fastener_pairs(fastener, row_index, present)
            if not pairs:
                continue

            position_map = {plate_idx: idx for idx, plate_idx in enumerate(present)}
            for upper_idx, lower_idx in pairs:
                upper_plate = self.plates[upper_idx]
                lower_plate = self.plates[lower_idx]
                local_upper = row_index - upper_plate.first_row
                local_lower = row_index - lower_plate.first_row
                dof_upper = self._dof[(upper_idx, local_upper)]
                dof_lower = self._dof[(lower_idx, local_lower)]
                upper_position = position_map[upper_idx]
                lower_position = position_map[lower_idx]
                ti = max(upper_plate.t, 1e-12)
                tj = max(lower_plate.t, 1e-12)
                compliance_total, stiffness_total = self._compliance_for_pair(
                    fastener,
                    upper_plate,
                    lower_plate,
                    ti,
                    tj,
                    1,
                )
                compliance = compliance_total
                stiffness = stiffness_total
                stiffness_matrix[dof_upper][dof_upper] += stiffness
                stiffness_matrix[dof_upper][dof_lower] -= stiffness
                stiffness_matrix[dof_lower][dof_upper] -= stiffness
                stiffness_matrix[dof_lower][dof_lower] += stiffness
                springs.append((dof_upper, dof_lower, stiffness, row_index, upper_idx, lower_idx, compliance))

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

        fastener_results: List[FastenerResult] = []
        for dof_i, dof_j, stiffness, row_index, plate_i, plate_j, compliance in springs:
            force = stiffness * (displacements[dof_i] - displacements[dof_j])
            fastener_results.append(
                FastenerResult(
                    row=row_index,
                    plate_i=plate_i,
                    plate_j=plate_j,
                    compliance=compliance,
                    stiffness=stiffness,
                    force=force,
                    dof_i=dof_i,
                    dof_j=dof_j,
                )
            )
        fastener_results.sort(key=lambda item: (item.row, min(item.plate_i, item.plate_j), max(item.plate_i, item.plate_j)))

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
                bearing_results.append(BearingBypassResult(row=row_index, plate_name=plate.name, bearing=bearing, bypass=bypass))

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
                node_results.append(
                    NodeResult(
                        plate_id=plate_index,
                        plate_name=plate.name,
                        local_node=local_node,
                        x=position,
                        displacement=displacements[dof_index],
                        net_bypass=bypass_flow,
                        thickness=plate.t,
                        bypass_area=bypass_area,
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
