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
import jolt.fatigue as fatigue


@dataclass
class FatigueResult:
    node_id: int
    row: int
    plate_name: str
    ktg: float
    ktn: float
    ktb: float
    theta: float
    ssf: float
    bearing_load: float
    bypass_load: float
    sigma_ref: float
    term_bearing: float
    term_bypass: float
    peak_stress: float = 0.0
    fsi: float = 0.0
    f_max: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)



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
    material_name: Optional[str] = None
    Fx_left: float = 0.0
    Fx_right: float = 0.0
    fatigue_strength: Optional[float] = None


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
    
    # Fatigue / Hole Configuration
    hole_centered: bool = True
    hole_offset: float = 0.0
    hole_condition_factor: float = 1.0 # alpha
    hole_filling_factor: float = 1.0 # beta
    
    # Countersink Configuration
    is_countersunk: bool = False
    cs_depth: float = 0.0
    cs_angle: float = 100.0
    cs_affects_bypass: bool = False
    cs_layers: List[str] = field(default_factory=list) # List of layer names that are countersunk


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
    fatigue_results: List[FatigueResult] = field(default_factory=list)
    critical_points: List[Dict[str, Any]] = field(default_factory=list)
    critical_node_id: Optional[int] = None

    def compute_fatigue_factors(self, fasteners: List[FastenerRow]):
        """
        Compute SSF for all nodes at fastener locations.
        Populates self.fatigue_results and self.critical_points.
        """
        self.fatigue_results = []
        self.critical_points = []
        
        fastener_map = {f.row: f for f in fasteners}
        bb_map = {(bb.plate_name, bb.row): bb for bb in self.bearing_bypass}
        plate_map = {p.name: p for p in self.plates}
        
        # Temporary storage for ranking: plate_name -> list of results
        plate_results: Dict[str, List[Dict[str, Any]]] = {}

        for node in self.nodes:
            if node.row not in fastener_map:
                continue
            
            f_def = fastener_map[node.row]
            
            # Get Geometry
            D = f_def.D
            t = node.thickness
            area = node.bypass_area if node.bypass_area else 0.0
            
            if t > 1e-9 and area > 1e-9:
                W = area / t
            else:
                W = 0.0

            # Get Loads
            bb = bb_map.get((node.plate_name, node.row))
            if not bb:
                continue
                
            P_bearing = abs(bb.bearing)
            P_bypass = abs(node.net_bypass)
            
            # Countersink Logic
            is_cs = f_def.is_countersunk
            cs_depth_ratio = 0.0
            if is_cs:
                if node.plate_name in f_def.cs_layers:
                    if f_def.cs_depth > 0:
                        cs_depth_ratio = f_def.cs_depth / t
                else:
                    is_cs = False
            
            # Calculate SSF
            res = fatigue.calculate_ssf(
                load_transfer=P_bearing,
                load_bypass=P_bypass,
                D=D,
                W=W,
                t=t,
                offset=f_def.hole_offset,
                is_countersunk=is_cs,
                cs_depth_ratio=cs_depth_ratio,
                alpha=f_def.hole_condition_factor,
                beta=f_def.hole_filling_factor,
                shear_type="single", 
                cs_affects_bypass=f_def.cs_affects_bypass
            )
            
            peak_stress = res["ssf"] * res["sigma_ref"]
            
            # --- FSI Logic ---
            plate = plate_map.get(node.plate_name)
            f_max = plate.fatigue_strength if plate and plate.fatigue_strength else None
            
            if f_max and f_max > 0:
                fsi = peak_stress / f_max
            else:
                fsi = peak_stress # Fallback if no strength defined

            f_res = FatigueResult(
                node_id=node.legacy_id,
                row=node.row,
                plate_name=node.plate_name,
                ktg=res["ktg"],
                ktn=res["ktn"],
                ktb=res["ktb"],
                theta=res["theta"],
                ssf=res["ssf"],
                bearing_load=P_bearing,
                bypass_load=P_bypass,
                sigma_ref=res["sigma_ref"],
                term_bearing=res["term_bearing"],
                term_bypass=res["term_bypass"],
                peak_stress=peak_stress,
                fsi=fsi,
                f_max=f_max
            )
            
            self.fatigue_results.append(f_res)
            
            entry = {
                "node_id": node.legacy_id,
                "plate_name": node.plate_name,
                "row": node.row,
                "peak_stress": peak_stress,
                "ssf": res["ssf"],
                "f_max": f_max,
                "fsi": fsi,
                "x": node.x, 
                "y": 0.0 
            }
            
            if node.plate_name not in plate_results:
                plate_results[node.plate_name] = []
            plate_results[node.plate_name].append(entry)

        # Ranking Logic: Find max FSI per plate, then rank plates
        ranked_candidates = []
        
        for plate_name, results in plate_results.items():
            # Find the worst node for this plate
            worst_node = max(results, key=lambda x: x["fsi"])
            ranked_candidates.append(worst_node)
            
        # Sort all plate maxima by FSI descending
        ranked_candidates.sort(key=lambda x: x["fsi"], reverse=True)
        
        # Assign Ranks and store
        for i, cand in enumerate(ranked_candidates):
            cand["rank"] = i + 1
            self.critical_points.append(cand)
            
        # Set legacy critical_node_id to the top rank
        if self.critical_points:
            self.critical_node_id = self.critical_points[0]["node_id"]
        else:
            self.critical_node_id = None

    def fatigue_results_as_dicts(self) -> List[Dict[str, float]]:
        return [res.as_dict() for res in self.fatigue_results]


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
        
        sum_right = 0.0
        sum_left = 0.0
        
        for p in self.plates:
            if p.Fx_right > 0: sum_right += p.Fx_right
            else: sum_left += abs(p.Fx_right)
                
            if p.Fx_left > 0: sum_right += p.Fx_left
            else: sum_left += abs(p.Fx_left)
            
        for f in self.applied_forces:
            val = f.get("Value", 0.0)
            if val > 0: sum_right += val
            else: sum_left += abs(val)
            
        P = max(sum_right, sum_left)
        
        if P < 1e-9:
            P = 1.0 
            
        fastener_rows = {f.row for f in self.fasteners}
        diam_by_row = {}
        for f in self.fasteners:
            if f.diameter > 0:
                diam_by_row[f.row] = f.diameter
        
        node_bearing_map = {}
        for f in self.fasteners:
            node_bearing_map[(f.plate_i, f.row)] = f.bearing_force_upper
            node_bearing_map[(f.plate_j, f.row)] = f.bearing_force_lower

        plate_name_to_idx = {p.name: i for i, p in enumerate(self.plates)}
        
        bb_map = {(bb.plate_name, bb.row): bb for bb in self.bearing_bypass}
        
        for node in self.nodes:
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
            
            bypass = abs(bb.flow_right)
            incoming = transfer + bypass
            
            detail_stress = max(incoming, bypass) / area if area > 0 else 0.0
            bearing_stress = transfer / (diameter * thickness) if diameter * thickness > 0 else 0.0
            
            results.append({
                "Element": node.plate_name,
                "Node": node.legacy_id,
                "Row": node.row, 
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
            joint_type = "metal"
            if "graphite" in method or "composite" in method:
                joint_type = "composite"
            
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
            c_sym = grumman_compliance(
                ti=t, Ei=plate.E,
                tj=t, Ej=plate.E,
                Ef=fastener.Eb,
                diameter=fastener.D
            )
            compliance = c_sym / 2.0
            
        else:
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
            
            method_key = fastener.method.lower()
            is_empirical = any(k in method_key for k in ["huth", "grumman", "swift", "douglas", "vought", "tate", "rosenfeld", "morris"])

            if is_empirical:
                for i in range(len(plates_at_row) - 1):
                    idx_1, plate_1 = plates_at_row[i]
                    idx_2, plate_2 = plates_at_row[i+1]

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
                    
                    stiffness_total = 1.0 / compliance
                    node_stiffness = stiffness_total * 2.0

                    local_node_1 = row_index - plate_1.first_row
                    dof_1 = self._dof.get((idx_1, local_node_1))
                    if dof_1 is not None:
                        springs.append((dof_1, dof_fastener, node_stiffness, row_index, idx_1, compliance / 2.0))
                        
                        stiffness_matrix[dof_1][dof_1] += node_stiffness
                        stiffness_matrix[dof_1][dof_fastener] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_1] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_fastener] += node_stiffness

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
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                force_vector[dof] += magnitude

        for plate_index, local_node, displacement in validated_supports:
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                stiffness_matrix[dof] = [0.0] * ndof
                stiffness_matrix[dof][dof] = 1.0
                force_vector[dof] = displacement

        displacements = solve_dense(stiffness_matrix, force_vector)

        fastener_results = []
        for dof_plate, dof_fastener, stiffness, row, plate_idx, compliance in springs:
            u_plate = displacements[dof_plate]
            u_fastener = displacements[dof_fastener]
            force = stiffness * (u_fastener - u_plate)
            
            # Find the other plate connected to this fastener for result reporting
            # This is tricky because the spring list is flat.
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
            
            method_key = fastener.method.lower()
            is_empirical = any(k in method_key for k in ["huth", "grumman", "swift", "douglas", "vought", "tate", "rosenfeld", "morris"])

            if is_empirical:
                for i in range(len(plates_at_row) - 1):
                    idx_1, plate_1 = plates_at_row[i]
                    idx_2, plate_2 = plates_at_row[i+1]

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
                    
                    stiffness_total = 1.0 / compliance
                    node_stiffness = stiffness_total * 2.0

                    local_node_1 = row_index - plate_1.first_row
                    dof_1 = self._dof.get((idx_1, local_node_1))
                    if dof_1 is not None:
                        springs.append((dof_1, dof_fastener, node_stiffness, row_index, idx_1, compliance / 2.0))
                        
                        stiffness_matrix[dof_1][dof_1] += node_stiffness
                        stiffness_matrix[dof_1][dof_fastener] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_1] -= node_stiffness
                        stiffness_matrix[dof_fastener][dof_fastener] += node_stiffness

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
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                force_vector[dof] += magnitude

        # We can aggregate forces by fastener row and plate index.
        # Create a copy of the stiffness matrix for reaction calculation
        # (The solver modifies the matrix for boundary conditions)
        stiffness_matrix_orig = [row[:] for row in stiffness_matrix]
        force_vector_orig = force_vector[:]

        for plate_index, local_node, displacement in validated_supports:
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                stiffness_matrix[dof] = [0.0] * ndof
                stiffness_matrix[dof][dof] = 1.0
                force_vector[dof] = displacement

        displacements = solve_dense(stiffness_matrix, force_vector)

        fastener_results = []
        for dof_plate, dof_fastener, stiffness, row, plate_idx, compliance in springs:
            u_plate = displacements[dof_plate]
            u_fastener = displacements[dof_fastener]
            force = stiffness * (u_fastener - u_plate)
            
            # Find the other plate connected to this fastener for result reporting
            # This is tricky because the spring list is flat.
            # We can aggregate forces by fastener row and plate index.
            pass

        # Re-process fasteners to create clean results
        # We need to know the force transferred at each interface.
        # The solver gives us forces between plate and fastener node.
        # F_plate_to_fastener = k * (u_fastener - u_plate)
        
        # Aggregate bearing forces per plate/row
        bearing_forces: Dict[Tuple[int, int], float] = {} # (plate_idx, row) -> force
        
        for dof_plate, dof_fastener, stiffness, row, plate_idx, compliance in springs:
            u_plate = displacements[dof_plate]
            u_fastener = displacements[dof_fastener]
            force = stiffness * (u_fastener - u_plate) # Force exerted BY fastener ON plate? 
            # If u_fastener > u_plate, spring stretches. Force on plate is positive (right).
            # Force on fastener is negative (left).
            # Let's check sign convention.
            # If plate moves right (u_plate > 0) and fastener is fixed (0), force on plate is -k*u. Correct.
            
            current = bearing_forces.get((plate_idx, row), 0.0)
            bearing_forces[(plate_idx, row)] = current + force

        # Create FastenerResult objects
        # We need to pair them up.
        # For each fastener row, we have multiple plates.
        # We need to report forces between pairs?
        # The legacy output reports "Fastener 1 (0-1)", "Fastener 1 (1-2)" etc.
        # So we need to iterate through plates at each row.
        
        fastener_results_list = []
        
        for fastener in self.fasteners:
            row_index = fastener.row
            plates_at_row = self._plates_at_row(row_index)
            
            # Sort by plate index
            plates_at_row.sort()
            
            for i in range(len(plates_at_row) - 1):
                p_i = plates_at_row[i]
                p_j = plates_at_row[i+1]
                
                # Force transferred between p_i and p_j?
                # In the star model, the fastener node balances the forces.
                # Force from p_i = F_i. Force from p_j = F_j.
                # If only 2 plates, F_i + F_j = 0 (equilibrium of fastener node).
                # So transfer is F_i.
                # If 3 plates, F_i + F_j + F_k = 0.
                # Transfer across interface i-j?
                # Shear force in the fastener shank between i and j.
                # Shear(i, j) = Sum(Forces on plates <= i)
                
                shear_force = 0.0
                for k in range(i + 1):
                    pk = plates_at_row[k]
                    shear_force += bearing_forces.get((pk, row_index), 0.0)
                
                # Compliance?
                # We need to retrieve the compliance we calculated.
                # It's not stored easily. Re-calculate or store in a map.
                # For reporting, we can re-calculate.
                
                plate_obj_i = self.plates[p_i]
                plate_obj_j = self.plates[p_j]
                
                t1 = plate_obj_i.t
                if plate_obj_i.thicknesses:
                    ln1 = row_index - plate_obj_i.first_row
                    s1 = ln1 if ln1 < len(plate_obj_i.thicknesses) else ln1 - 1
                    if 0 <= s1 < len(plate_obj_i.thicknesses): t1 = plate_obj_i.thicknesses[s1]
                
                t2 = plate_obj_j.t
                if plate_obj_j.thicknesses:
                    ln2 = row_index - plate_obj_j.first_row
                    s2 = ln2 if ln2 < len(plate_obj_j.thicknesses) else ln2 - 1
                    if 0 <= s2 < len(plate_obj_j.thicknesses): t2 = plate_obj_j.thicknesses[s2]

                # Use pairwise compliance calculation
                # Note: This might differ slightly from the star model effective compliance if not careful,
                # but it's the physical compliance of the connection.
                comp = self._calculate_compliance_pairwise(plate_obj_i, plate_obj_j, fastener, t1, t2)
                stiff = 1.0 / comp if comp > 0 else 1e12
                
                # Bearing forces on the holes
                # For the interface i-j, the bearing force on plate i is relevant?
                # Or is it the total bearing force on the hole?
                # Legacy output shows "Brg Force Upper" and "Brg Force Lower".
                # Upper = Force on Plate i. Lower = Force on Plate j.
                # But if there are 3 plates, for interface 1-2 (middle), 
                # Upper is Plate 1? Or Plate 1+2?
                # Let's assume Upper = Force on Plate i, Lower = Force on Plate j (local to this interface?)
                # Actually, usually we report the bearing force on the plate at that hole.
                # So for interface i-j, we show Bearing Force on i and Bearing Force on j.
                
                f_res = FastenerResult(
                    row=row_index,
                    plate_i=p_i,
                    plate_j=p_j,
                    compliance=comp,
                    stiffness=stiff,
                    force=shear_force, # Shear force in the bolt
                    dof_i=self._dof.get((p_i, row_index - self.plates[p_i].first_row), -1),
                    dof_j=self._dof.get((p_j, row_index - self.plates[p_j].first_row), -1),
                    bearing_force_upper=bearing_forces.get((p_i, row_index), 0.0),
                    bearing_force_lower=bearing_forces.get((p_j, row_index), 0.0),
                    modulus=fastener.Eb,
                    diameter=fastener.D,
                    quantity=1.0,
                    t1=t1,
                    t2=t2
                )
                fastener_results_list.append(f_res)

        # Nodes Results
        node_results = []
        for plate_idx, plate in enumerate(self.plates):
            segments = plate.segment_count()
            # Nodes are 0 to segments
            for local_node in range(segments + 1):
                dof = self._dof.get((plate_idx, local_node))
                disp = displacements[dof] if dof is not None else 0.0
                
                # Determine legacy ID (row based)
                # Global node index?
                # Legacy ID usually: 1000*(plate+1) + row?
                # Or 1000*(plate+1) + local_node + 1?
                # The prompt implies row-based IDs like 3002, 3003.
                # Let's use 1000*(plate+1) + (first_row + local_node)
                row_abs = plate.first_row + local_node
                legacy_id = 1000 * (plate_idx + 1) + row_abs
                
                # Thickness and Area
                # Area is defined for segments (between nodes).
                # Node thickness?
                t_node = plate.t
                if plate.thicknesses:
                    # Use thickness of segment to the right, or left if last
                    s_idx = local_node if local_node < len(plate.thicknesses) else local_node - 1
                    if 0 <= s_idx < len(plate.thicknesses):
                        t_node = plate.thicknesses[s_idx]
                
                # Bypass Area
                # Average of adjacent segments? Or segment to the right?
                # Usually bypass load is calculated at the node (net section).
                # So we need the area at the node (net area).
                # But input is A_strip (gross or net?). Usually Gross.
                # Let's use the segment area to the right (downstream) as "Bypass Area" reference?
                # Or just the area at that location.
                area_node = 0.0
                if 0 <= local_node < len(plate.A_strip):
                    area_node = plate.A_strip[local_node]
                elif local_node > 0 and local_node - 1 < len(plate.A_strip):
                    area_node = plate.A_strip[local_node - 1]
                    
                # Net Bypass Load
                # Load in the plate at this node.
                # F_axial = k * (u_right - u_left).
                # At node i, load on the right side (segment i) vs left side (segment i-1).
                # Net Bypass usually means the load passing through the plate at the hole line.
                # = Average of load in segment left and segment right?
                # Or just the load in the segment?
                # If there is a fastener, Load_Left - Load_Right = Bearing_Force.
                # Bypass = Load_Right (if flow is left to right).
                # Let's calculate loads in bars first.
                
                node_results.append(NodeResult(
                    plate_id=plate_idx,
                    plate_name=plate.name,
                    local_node=local_node,
                    x=self._x.get((plate_idx, local_node), 0.0),
                    displacement=disp,
                    net_bypass=0.0, # To be filled
                    thickness=t_node,
                    bypass_area=area_node,
                    row=row_abs,
                    legacy_id=legacy_id
                ))

        # Bars Results
        bar_results = []
        for plate_idx, plate in enumerate(self.plates):
            segments = plate.segment_count()
            for seg in range(segments):
                dof_l = self._dof.get((plate_idx, seg))
                dof_r = self._dof.get((plate_idx, seg + 1))
                
                u_l = displacements[dof_l] if dof_l is not None else 0.0
                u_r = displacements[dof_r] if dof_r is not None else 0.0
                
                length = self.pitches[plate.first_row - 1 + seg]
                area = plate.A_strip[seg]
                k_bar = plate.E * area / length
                
                axial_force = k_bar * (u_r - u_l)
                
                bar_results.append(BarResult(
                    plate_id=plate_idx,
                    plate_name=plate.name,
                    segment=seg,
                    axial_force=axial_force,
                    stiffness=k_bar,
                    modulus=plate.E
                ))

        # Fill Net Bypass in Nodes
        # Map (plate, seg) -> force
        bar_force_map = {(b.plate_id, b.segment): b.axial_force for b in bar_results}
        
        for node in node_results:
            # Load Right (downstream)
            f_right = bar_force_map.get((node.plate_id, node.local_node), 0.0)
            # Load Left (upstream)
            f_left = bar_force_map.get((node.plate_id, node.local_node - 1), 0.0)
            
            # Bypass is the load remaining in the plate after the fastener.
            # If flow is left->right, Bypass = f_right.
            # If flow is right->left, Bypass = f_left?
            # Let's define Bypass as the load on the side away from the reference end?
            # Standard convention: Bypass is the load in the segment 'downstream' of the fastener.
            # But 'downstream' depends on load direction.
            # Let's use the maximum absolute load? No, that includes transfer.
            # Let's use the average? No.
            # Usually Bypass = Load leaving the station.
            # Let's use f_right for now (assuming left-to-right flow).
            # But we should probably report both or the one consistent with flow.
            # For 1D model, we often just list the bar forces.
            # Let's store f_right as "Net Bypass" for now, as it matches "Load in segment i".
            node.net_bypass = f_right

        # Bearing/Bypass Table
        bb_results = []
        for node in node_results:
            # Check if there is a fastener at this row
            # We need to find if this node corresponds to a fastener location.
            # We have node.row.
            brg = bearing_forces.get((node.plate_id, node.row), 0.0)
            
            # Bypass: As discussed, usually the load in the adjacent element.
            # Let's use f_right (load in segment starting at this node).
            bypass = node.net_bypass
            
            f_right = bar_force_map.get((node.plate_id, node.local_node), 0.0)
            f_left = bar_force_map.get((node.plate_id, node.local_node - 1), 0.0)
            
            bb_results.append(BearingBypassResult(
                row=node.row,
                plate_name=node.plate_name,
                bearing=brg,
                bypass=bypass,
                flow_left=f_left,
                flow_right=f_right
            ))

        # Reactions (using original stiffness matrix)
        reaction_results = []
        for plate_index, local_node, _ in validated_supports:
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                # Reaction = K_orig * u - F_applied
                internal_force = 0.0
                for j in range(ndof):
                    internal_force += stiffness_matrix_orig[dof][j] * displacements[j]
                
                # Calculate applied force at this node
                f_app = 0.0
                # Plate end loads
                plate = self.plates[plate_index]
                if local_node == 0: f_app += plate.Fx_left
                if local_node == plate.segment_count(): f_app += plate.Fx_right
                
                # Point forces
                for pi, ln, val in validated_forces:
                    if pi == plate_index and ln == local_node:
                        f_app += val
                
                reaction = internal_force - f_app
                
                reaction_results.append(ReactionResult(
                    plate_id=plate_index,
                    plate_name=self.plates[plate_index].name,
                    local_node=local_node,
                    global_node=self.plates[plate_index].first_row + local_node,
                    reaction=reaction
                ))

        solution = JointSolution(
            displacements=displacements,
            stiffness_matrix=stiffness_matrix,
            force_vector=force_vector,
            fasteners=fastener_results_list,
            bearing_bypass=bb_results,
            nodes=node_results,
            bars=bar_results,
            reactions=reaction_results,
            dof_map=self._dof,
            plates=self.plates,
            applied_forces=[{"plate": p, "node": n, "Value": v} for p, n, v in validated_forces]
        )
        
        # Compute Fatigue Factors
        solution.compute_fatigue_factors(self.fasteners)
        
        return solution

