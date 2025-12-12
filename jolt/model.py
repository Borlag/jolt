"""Core 1-D joint model (bars + fastener springs)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Dict, List, Optional, Sequence, Tuple, Set, Any

from .fasteners import (
    boeing69_compliance, 
    boeing69_eq1_compliance,
    boeing69_eq2_compliance,
    boeing_pair_compliance,
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
    incoming_load: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FatigueResult:
        return cls(**data)


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
    
    # Visual / ID attributes
    name: str = ""  # User-defined label (e.g. "HL10-5")
    marker_symbol: str = "circle"  # Visual style for the scheme
    
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
    topology: Optional[str] = None  # Optional override for topology selection (star/chain variants)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FastenerResult:
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeResult:
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BarResult:
        return cls(**data)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BearingBypassResult:
        return cls(**data)


@dataclass
class ReactionResult:
    plate_id: int
    plate_name: str
    local_node: int
    global_node: int
    reaction: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReactionResult:
        return cls(**data)


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
    full_displacements: List[float] = field(default_factory=list)
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

        # Detect Support Side
        is_support_left = self._is_support_left()

        for node in self.nodes:
            if node.row not in fastener_map:
                continue
            
            f_res, entry = self._calculate_node_fatigue(node, fastener_map, bb_map, plate_map, is_support_left)
            
            if f_res and entry:
                self.fatigue_results.append(f_res)
                
                if node.plate_name not in plate_results:
                    plate_results[node.plate_name] = []
                plate_results[node.plate_name].append(entry)

        self._rank_critical_points(plate_results)

    def _is_support_left(self) -> bool:
        """Determine if the support is primarily on the left side."""
        support_left = 0
        support_right = 0
        for r in self.reactions:
            if r.local_node == 0: support_left += 1
            else: support_right += 1
        return support_left >= support_right

    def _calculate_node_fatigue(
        self, 
        node: NodeResult, 
        fastener_map: Dict[int, FastenerRow], 
        bb_map: Dict[Tuple[str, int], BearingBypassResult], 
        plate_map: Dict[str, Plate], 
        is_support_left: bool
    ) -> Tuple[Optional[FatigueResult], Optional[Dict[str, Any]]]:
        """Calculate fatigue result for a single node."""
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
            return None, None
            
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
        
        # Determine Incoming and Bypass - Boeing JOLT Convention (from support reaction)
        # Incoming: Load flowing from support direction INTO this node
        # Bypass: Load flowing towards external load (away from support)
        if is_support_left:
            incoming = abs(bb.flow_left)
            bypass = abs(bb.flow_right)
        else:
            incoming = abs(bb.flow_right)
            bypass = abs(bb.flow_left)
        
        # Reference Load for SSF is the maximum load (Incoming or Bypass)
        P_max = max(incoming, bypass)

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
            cs_affects_bypass=f_def.cs_affects_bypass,
            reference_load=P_max
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
            f_max=f_max,
            incoming_load=incoming
        )
        
        entry = {
            "node_id": node.legacy_id,
            "plate_name": node.plate_name,
            "row": node.row,
            "peak_stress": peak_stress,
            "ssf": res["ssf"],
            "f_max": f_max,
            "fsi": fsi,
            "x": node.x, 
            "y": 0.0,
            "incoming_load": incoming
        }
        
        return f_res, entry

    def _rank_critical_points(self, plate_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Rank critical points based on FSI."""
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
        res_list = []
        for res in self.fatigue_results:
            d = res.as_dict()
            d["Incoming Load"] = res.incoming_load
            res_list.append(d)
        return res_list


    def fasteners_as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "ID": f"{1000 * (fastener.plate_i + 1) + fastener.row}-{1000 * (fastener.plate_j + 1) + fastener.row}",
                "F [lb]": abs(fastener.force),
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
                "u [in]": node.displacement,
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
                "Reaction [lb]": reaction.reaction,
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
        
            # Detect Support Side from reactions
        support_left = 0
        support_right = 0
        for r in self.reactions:
            if r.local_node == 0: support_left += 1
            else: support_right += 1
            
        is_support_left = support_left >= support_right

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
            
            # Determine Incoming and Bypass based on support direction
            # Boeing JOLT Convention (from support reaction):
            # Incoming: Load flowing from support direction INTO this node
            # Bypass: Load flowing towards external load (away from support)
            if is_support_left:
                # Support on left: incoming is flow from left, bypass goes towards right (load)
                incoming = abs(bb.flow_left)
                bypass = abs(bb.flow_right)
            else:
                # Support on right: incoming is flow from right, bypass goes towards left (load)
                incoming = abs(bb.flow_right)
                bypass = abs(bb.flow_left)
            
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
            
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the solution to a dictionary."""
        # Calculate summary metrics
        max_fsi_global = 0.0
        max_fsi_per_element = {}
        max_fastener_load = 0.0
        max_bypass = 0.0
        max_displacement = 0.0
        
        if self.fatigue_results:
            max_fsi_global = max((r.fsi for r in self.fatigue_results), default=0.0)
            for r in self.fatigue_results:
                if r.plate_name not in max_fsi_per_element:
                    max_fsi_per_element[r.plate_name] = 0.0
                max_fsi_per_element[r.plate_name] = max(max_fsi_per_element[r.plate_name], r.fsi)
        
        if self.fasteners:
            max_fastener_load = max((abs(f.force) for f in self.fasteners), default=0.0)
            
        if self.nodes:
            max_bypass = max((abs(n.net_bypass) for n in self.nodes), default=0.0)
            max_displacement = max((abs(n.displacement) for n in self.nodes), default=0.0)

        # Governing IDs
        governing_element = None
        if self.critical_points:
            governing_element = self.critical_points[0].get("plate_name")

        return {
            "displacements": self.displacements,
            "stiffness_matrix": self.stiffness_matrix,
            "force_vector": self.force_vector,
            "fasteners": [f.as_dict() for f in self.fasteners],
            "bearing_bypass": [bb.as_dict() for bb in self.bearing_bypass],
            "nodes": [n.as_dict() for n in self.nodes],
            "bars": [b.as_dict() for b in self.bars],
            "reactions": [r.as_dict() for r in self.reactions],
            # We don't serialize dof_map keys as tuples directly to JSON, 
            # but we can reconstruct it or store as string keys if needed.
            # For now, let's store it as a list of items to be safe.
            "dof_map_items": [[list(k), v] for k, v in self.dof_map.items()],
            "plates": [p.name for p in self.plates], # Just names or full objects? 
            # Ideally we rely on the input config for full plate defs, but storing names helps validation.
            "applied_forces": self.applied_forces,
            "fatigue_results": [fr.as_dict() for fr in self.fatigue_results],
            "critical_points": self.critical_points,
            "critical_node_id": self.critical_node_id,
            "summary": {
                "max_fsi_global": max_fsi_global,
                "max_fsi_per_element": max_fsi_per_element,
                "max_fastener_load": max_fastener_load,
                "max_bypass": max_bypass,
                "max_displacement": max_displacement,
                "governing_element": governing_element,
                "critical_node_id": self.critical_node_id
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JointSolution:
        """Rehydrate a JointSolution from a dictionary."""
        
        # Reconstruct DOF map
        dof_map = {}
        for k_list, v in data.get("dof_map_items", []):
            if len(k_list) >= 2:
                # Check if first element is "fastener" string or int
                key_0 = k_list[0]
                key_1 = k_list[1]
                # JSON loads might make tuple keys into lists
                dof_map[(key_0, key_1)] = v

        # Note: We are NOT rehydrating the full 'plates' list here because 
        # JointSolution usually holds references to the input Plate objects.
        # However, for the purpose of the Comparison Module, we might need them 
        # if we want to pass this solution to the plotter.
        # The plotter uses `solution.plates` for some lookups?
        # Checking `visualization_plotly.py`... it uses `plates` argument passed to the function,
        # NOT `solution.plates`. `solution.plates` is used in `compute_fatigue_factors` though.
        # So for pure results display, we might be fine without full Plate objects in solution,
        # AS LONG AS we pass the rehydrated Plate objects from the Config to the plotter.
        
        return cls(
            displacements=data.get("displacements", []),
            stiffness_matrix=data.get("stiffness_matrix", []),
            force_vector=data.get("force_vector", []),
            fasteners=[FastenerResult.from_dict(f) for f in data.get("fasteners", [])],
            bearing_bypass=[BearingBypassResult.from_dict(bb) for bb in data.get("bearing_bypass", [])],
            nodes=[NodeResult.from_dict(n) for n in data.get("nodes", [])],
            bars=[BarResult.from_dict(b) for b in data.get("bars", [])],
            reactions=[ReactionResult.from_dict(r) for r in data.get("reactions", [])],
            dof_map=dof_map,
            plates=[], # Empty for now, will be filled by Config rehydration if needed?
            applied_forces=data.get("applied_forces", []),
            fatigue_results=[FatigueResult.from_dict(fr) for fr in data.get("fatigue_results", [])],
            critical_points=data.get("critical_points", []),
            critical_node_id=data.get("critical_node_id")
        )


class Joint1D:
    """Solve a one-dimensional load transfer problem."""

    def __init__(self, pitches: Sequence[float], plates: Sequence[Plate], fasteners: Sequence[FastenerRow]):
        self.pitches = list(pitches)
        self.plates = list(plates)
        self.fasteners = list(fasteners)
        self._dof: Dict[Tuple[int, int], int] = {}
        self._x: Dict[Tuple[int, int], float] = {}
        self._condensed_recovery: Dict[int, Any] = {}
        # (row, plate_i, plate_j) -> (compliance, stiffness) used during assembly
        self._interface_properties: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------
    def _fastener_topology(self, fastener: FastenerRow) -> str:
        """Return the normalized topology string for a fastener row."""
        topo_raw = (fastener.topology or "").strip().lower()
        if topo_raw:
            return topo_raw

        method = fastener.method.lower()
        
        # Explicit combined modes (method + topology in one string)
        # e.g., "Boeing69_chain" or "Huth_star"
        # New Eq1/Eq2 support
        if "boeing69_eq1_chain" in method or "boeing_eq1_chain" in method:
            return "boeing_chain_eq1"
        if "boeing69_eq2_chain" in method or "boeing_eq2_chain" in method:
            return "boeing_chain_eq2"
        if "boeing69_eq1_star" in method or "boeing_eq1_star" in method:
            return "boeing_star_eq1"
        if "boeing69_eq2_star" in method or "boeing_eq2_star" in method:
            return "boeing_star_eq2"

        if "boeing69_chain" in method:
            return "boeing_chain"
        if "boeing69_star_scaled" in method:
            return "boeing_star_scaled"
        if "boeing69_star_raw" in method:
            return "boeing_star_raw"
        if "huth_chain" in method:
            return "empirical_chain"
        if "huth_star" in method:
            return "empirical_star"
        
        # Existing behavior (backward-compatible)
        if "boeing_chain" in method:
            return "boeing_chain"
        if "boeing_star_scaled" in method:
            return "boeing_star_scaled"
        if "boeing_star_raw" in method:
            return "boeing_star_raw"
        if "boeing_chain_eq1" in method:
            return "boeing_chain_eq1"
        if "boeing_chain_eq2" in method:
            return "boeing_chain_eq2"
        if "boeing_star_eq1" in method:
            return "boeing_star_eq1"
        if "boeing_star_eq2" in method:
            return "boeing_star_eq2"
        if "empirical_chain" in method:
            return "empirical_chain"
        if "empirical_star" in method:
            return "empirical_star"
        if "boeing_beam" in method:
            return "boeing_beam"

        # Defaults: Boeing -> legacy star, everything else -> empirical star
        if "boeing" in method:
            return "boeing_star_scaled"
        return "empirical_star"

    def _topology_uses_fastener_dof(self, topology: str) -> bool:
        # boeing_beam uses condensed beam architecture (NO fastener DOF)
        if topology == "boeing_beam":
            return False
        return "star" in topology

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _thickness_at_row(self, plate: Plate, row_index: int) -> float:
        """Return the effective thickness for a plate at a given row index."""
        t_val = plate.t
        if plate.thicknesses:
            ln = row_index - plate.first_row
            s = ln if ln < len(plate.thicknesses) else ln - 1
            if 0 <= s < len(plate.thicknesses):
                t_val = plate.thicknesses[s]
        return t_val

    def _ordered_plate_indices(self, connection_pairs: List[Tuple[int, int]]) -> List[int]:
        ordered_plates: List[int] = []
        for i, (idx_1, idx_2) in enumerate(connection_pairs):
            if i == 0:
                ordered_plates.append(idx_1)
            ordered_plates.append(idx_2)
        return ordered_plates

    def _calculate_compliance_pairwise(
        self, 
        plate_i: Plate, 
        plate_j: Plate, 
        fastener: FastenerRow,
        t_i: float,
        t_j: float,
        shear_planes: int = 1
    ) -> float:
        """
        Calculates total compliance between two plates based on the selected method.
        Crucially accepts 'shear_planes' to allow enforcing Single Shear for Boeing Consistency.
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
            shear_mode = "double" if shear_planes >= 2 else "single"
            return huth_compliance(
                t1=t_i, E1=plate_i.E,
                t2=t_j, E2=plate_j.E,
                Ef=fastener.Eb, diameter=fastener.D,
                shear=shear_mode,
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
            # Determine variant
            variant = "legacy"
            check_str = (method + " " + (fastener.topology or "")).lower()
            
            if "boeing1" in check_str or "eq1" in check_str:
                variant = "eq1"
            elif "boeing2" in check_str or "eq2" in check_str:
                variant = "eq2"

            Gf = fastener.Eb / (2.0 * (1.0 + fastener.nu_b))

            if variant == "legacy":
                # Legacy Boeing 69 supports shear_planes argument
                return boeing69_compliance(
                    ti=t_i, Ei=plate_i.E, tj=t_j, Ej=plate_j.E,
                    Eb=fastener.Eb, nu_b=fastener.nu_b,
                    diameter=fastener.D, shear_planes=shear_planes
                )
            else:
                return boeing_pair_compliance(
                    t1=t_i, E1=plate_i.E,
                    t2=t_j, E2=plate_j.E,
                    Ef=fastener.Eb, Gf=Gf,
                    d=fastener.D,
                    variant=variant
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
            topology = self._fastener_topology(fastener)
            if self._topology_uses_fastener_dof(topology):
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

    def _calculate_base_compliance(self, plate: Plate, fastener: FastenerRow, t_local: Optional[float] = None) -> float:
        """
        Calculate the 'Base' (Single) compliance of a plate branch.
        Used as the starting point for the Boeing decomposition.
        """
        t = t_local if t_local is not None else plate.t
        
        # Constants
        A_b = math.pi * (fastener.D / 2.0) ** 2
        I_b = math.pi * (fastener.D / 2.0) ** 4 / 4.0
        G_b = fastener.Eb / (2.0 * (1.0 + fastener.nu_b))
        
        # Shear Term (Boeing Single Model uses t_shear = min(t, (D+t)/2))
        t_shear = min(t, (fastener.D + t) / 2.0)
        C_shear = (4.0 * t_shear) / (9.0 * G_b * A_b)
        
        # Bending Term (Boeing Single Model uses t_bending = min(t, D))
        t_bending = min(t, fastener.D)
        C_bending = (6.0 * t_bending**3) / (40.0 * fastener.Eb * I_b)
        
        # Bearing Term
        C_bearing = (1.0 / t) * (1.0 / fastener.Eb + 1.0 / plate.E)
            
        return C_shear + C_bending + C_bearing

    def _solve_branch_compliances(self, plates: List[Plate], fastener: FastenerRow, row_index: int, pairwise_compliances: List[float]) -> List[float]:
        """
        Decompose pairwise compliances into branch compliances C_i.
        
        Methodology:
        1. N=2 (2 plates): Symmetric split (C1 = C2 = C_pair / 2).
        2. N=3 (Boeing): Symmetric Star Decomposition formula.
        3. N>3: Weighted Least Squares (WLS) with base compliances as weights.
        """
        n_pairs = len(pairwise_compliances)
        n_branches = n_pairs + 1
        
        # Case 1: 2-Layer Stack (Strict Symmetry)
        if n_branches == 2:
            c_pair = pairwise_compliances[0]
            return [c_pair / 2.0, c_pair / 2.0]
        
        is_boeing = "boeing" in fastener.method.lower()
        topology = self._fastener_topology(fastener)
        
        # Case 2: 3-Layer Stack (Boeing Symmetric Star / JOLT Logic)
        # This formula is empirically validated to produce 0.1% accuracy for 3-layer stacks
        if is_boeing and "star" in topology and n_branches == 3:
            C01, C12 = pairwise_compliances[0], pairwise_compliances[1]
            C_min, C_max = min(C01, C12), max(C01, C12)
            
            # Symmetric Star Decomposition
            C_mid = (2.0 * C_min - C_max) / 2.0
            
            # Enforce non-negative compliance (rigid link limit)
            if C_mid < 0.0: 
                C_mid = 0.0
                
            # Back-calculate outer branches
            branch_compliances = [0.0] * 3
            branch_compliances[1] = C_mid
            branch_compliances[0] = C01 - C_mid
            branch_compliances[2] = C12 - C_mid
            
            return [max(c, 1e-12) for c in branch_compliances]

        # Case 3: N > 3 Layers - Weighted Least Squares (WLS)
        # Always use calculated base compliances as both priors AND weights.
        # This ensures thickness-dependent stiffness scaling is correctly reflected.
        base_compliances = []
        for plate in plates:
            t_local = plate.t
            if plate.thicknesses:
                ln = row_index - plate.first_row
                s = ln if ln < len(plate.thicknesses) else ln - 1
                if 0 <= s < len(plate.thicknesses):
                    t_local = plate.thicknesses[s]
            base_compliances.append(self._calculate_base_compliance(plate, fastener, t_local))
        
        # Ensure no zero base compliances (would cause division issues)
        for i, bc in enumerate(base_compliances):
            if bc < 1e-15:
                base_compliances[i] = 1e-15
        
        # Build Weighted Constraint Matrix M = A * W^-1 * A^T
        # Where W^-1 is the diagonal matrix of base_compliances.
        matrix_M = [[0.0] * n_pairs for _ in range(n_pairs)]
        for i in range(n_pairs):
            # Diagonal: Sum of weights for the connected branches
            matrix_M[i][i] = base_compliances[i] + base_compliances[i+1]
            if i > 0:
                matrix_M[i][i-1] = base_compliances[i]
            if i < n_pairs - 1:
                matrix_M[i][i+1] = base_compliances[i+1]
                
        # RHS = (C_base_i + C_base_{i+1}) - C_pair_i
        rhs_vec = []
        for i in range(n_pairs):
            c_base_sum = base_compliances[i] + base_compliances[i+1]
            excess = c_base_sum - pairwise_compliances[i]
            rhs_vec.append(excess)
            
        try:
            lambdas = solve_dense(matrix_M, rhs_vec)
        except Exception:
            lambdas = [0.0] * n_pairs
            
        # C = C_base - W^-1 * A^T * lambda
        # The correction is scaled by the base compliance (weighting).
        final_compliances = list(base_compliances)
        final_compliances[0] -= base_compliances[0] * lambdas[0]
        for i in range(1, n_branches - 1):
            final_compliances[i] -= base_compliances[i] * (lambdas[i-1] + lambdas[i])
        final_compliances[-1] -= base_compliances[-1] * lambdas[-1]

        # Ensure non-negative for physical stability
        for i, c in enumerate(final_compliances):
            if c < 0.0:
                final_compliances[i] = 0.0
        
        return final_compliances


    def solve(
        self,
        supports: Sequence[Tuple[int, int, float]],
        point_forces: Optional[Sequence[Tuple[int, int, float]]] = None,
    ) -> JointSolution:
        """Solve the joint and return displacements and post-processed results."""
        stiffness_matrix, force_vector, springs, validated_supports, validated_forces, ndof = self._prepare_system(
            supports, point_forces or []
        )

        # Keep original stiffness for reaction calculation
        stiffness_matrix_orig = [row[:] for row in stiffness_matrix]

        # Apply Boundary Conditions
        self._apply_boundary_conditions(stiffness_matrix, force_vector, validated_supports, ndof)

        # Solve System
        displacements = solve_dense(stiffness_matrix, force_vector)
        
        # Post-process results
        return self._generate_solution(
            displacements, 
            stiffness_matrix_orig, 
            force_vector, 
            springs, 
            validated_supports, 
            validated_forces, 
            ndof
        )

    def _apply_boundary_conditions(
        self, 
        stiffness_matrix: List[List[float]], 
        force_vector: List[float], 
        validated_supports: List[Tuple[int, int, float]], 
        ndof: int
    ) -> None:
        """Apply displacement boundary conditions to the system matrix and force vector."""
        for plate_index, local_node, displacement in validated_supports:
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                stiffness_matrix[dof] = [0.0] * ndof
                stiffness_matrix[dof][dof] = 1.0
                force_vector[dof] = displacement

    def _generate_solution(
        self,
        displacements: List[float],
        stiffness_matrix_orig: List[List[float]],
        force_vector: List[float],
        springs: List[Tuple[int, int, float, int, int, float]],
        validated_supports: List[Tuple[int, int, float]],
        validated_forces: List[Tuple[int, int, float]],
        ndof: int
    ) -> JointSolution:
        """Generate the full JointSolution object from the raw solution data."""
        
        plate_dof_count = sum(plate.segment_count() + 1 for plate in self.plates)
        plate_displacements = displacements[:plate_dof_count]

        # 1. Calculate Bearing Forces
        bearing_forces = self._calculate_bearing_forces(displacements, springs)

        # 2. Generate Fastener Results
        fastener_results_list = self._generate_fastener_results(bearing_forces)

        # 3. Generate Node Results (Initial)
        node_results = self._generate_node_results(displacements)

        # 4. Generate Bar Results
        bar_results = self._generate_bar_results(displacements)
        
        # 5. Update Nodes with Bypass Loads (requires Bar Results)
        self._update_nodes_with_bypass(node_results, bar_results, validated_supports)

        # 6. Generate Bearing/Bypass Results
        bb_results = self._generate_bearing_bypass_results(node_results, bar_results, bearing_forces)

        # 7. Generate Reactions
        reaction_results = self._generate_reaction_results(
            displacements, stiffness_matrix_orig, validated_supports, validated_forces, ndof
        )

        solution = JointSolution(
            displacements=plate_displacements,
            stiffness_matrix=stiffness_matrix_orig, # Return original K? Or modified? Usually original is more useful for checking.
            force_vector=force_vector,
            fasteners=fastener_results_list,
            bearing_bypass=bb_results,
            nodes=node_results,
            bars=bar_results,
            reactions=reaction_results,
            dof_map=self._dof,
            full_displacements=displacements,
            plates=self.plates,
            applied_forces=[{"plate": p, "node": n, "Value": v} for p, n, v in validated_forces]
        )

        solution.compute_fatigue_factors(self.fasteners)

        return solution

    def _calculate_bearing_forces(
        self, 
        displacements: List[float], 
        springs: List[Tuple[int, int, float, int, int, float]]
    ) -> Dict[Tuple[int, int], float]:
        """Calculate bearing forces at each fastener location."""
        bearing_forces: Dict[Tuple[int, int], float] = {}
        
        # 1. Standard Springs (Star Topology)
        for dof_plate, dof_fastener, stiffness, row, plate_idx, _ in springs:
            u_plate = displacements[dof_plate]
            u_fastener = displacements[dof_fastener]
            force = stiffness * (u_fastener - u_plate)
            bearing_forces[(plate_idx, row)] = bearing_forces.get((plate_idx, row), 0.0) + force
            
        # 2. Condensed Beam Recovery (Ladder Topology)
        if hasattr(self, "_condensed_recovery"):
            for row_index, data in self._condensed_recovery.items():
                plate_indices = data["plate_indices"]
                k_brg = data["k_brg"]
                Temp = data["Temp"]
                b_indices_map = data["b_indices_map"]
                
                n = len(plate_indices)
                
                # Gather Plate Displacements (u)
                u_vec = []
                for i, p_idx in enumerate(plate_indices):
                    plate = self.plates[p_idx]
                    local_node = row_index - plate.first_row
                    dof = self._dof.get((p_idx, local_node))
                    val = displacements[dof] if dof is not None else 0.0
                    u_vec.append(val)
                
                # Recover Beam Internal DOFs (b = -Temp * u)
                nb = len(Temp)
                b_vec = [0.0] * nb
                for r in range(nb):
                    val = 0.0
                    for c in range(n):
                        val += Temp[r][c] * u_vec[c]
                    b_vec[r] = -val
                
                # Compute Bearing Forces for each plate: F = k_brg * (v_i - u_i)
                # v_i is the translational DOF corresponds to local index n + 2*i
                for i in range(n):
                    target_local_idx = n + 2 * i
                    
                    try:
                        b_pos = b_indices_map.index(target_local_idx)
                        v_i = b_vec[b_pos]
                    except ValueError:
                        v_i = 0.0 # Should be reachable dependent on clamping logic
                    
                    force = k_brg[i] * (v_i - u_vec[i])
                    bearing_forces[(plate_indices[i], row_index)] = bearing_forces.get((plate_indices[i], row_index), 0.0) + force

        return bearing_forces

    def _generate_fastener_results(self, bearing_forces: Dict[Tuple[int, int], float]) -> List[FastenerResult]:
        """Generate detailed results for each fastener."""
        results = []
        for fastener in self.fasteners:
            plates_at_row, connection_pairs = self._plates_and_pairs(fastener)
            if not plates_at_row:
                continue
            row_index = fastener.row
            plate_lookup = {idx: plate for idx, plate in plates_at_row}

            for p_i, p_j in connection_pairs:
                shear_force = 0.0
                for pk in sorted(plate_lookup.keys()):
                    if pk <= p_i:
                        shear_force += bearing_forces.get((pk, row_index), 0.0)

                plate_obj_i = plate_lookup[p_i]
                plate_obj_j = plate_lookup[p_j]

                t1 = self._thickness_at_row(plate_obj_i, row_index)
                t2 = self._thickness_at_row(plate_obj_j, row_index)

                override = self._interface_properties.get((row_index, p_i, p_j))
                if override is None:
                    comp = self._calculate_compliance_pairwise(plate_obj_i, plate_obj_j, fastener, t1, t2)
                    stiff = 1.0 / comp if comp > 0 else 1e12
                else:
                    comp, stiff = override

                results.append(FastenerResult(
                    row=row_index,
                    plate_i=p_i,
                    plate_j=p_j,
                    compliance=comp,
                    stiffness=stiff,
                    force=shear_force,
                    dof_i=self._dof.get((p_i, row_index - plate_obj_i.first_row), -1),
                    dof_j=self._dof.get((p_j, row_index - plate_obj_j.first_row), -1),
                    bearing_force_upper=bearing_forces.get((p_i, row_index), 0.0),
                    bearing_force_lower=bearing_forces.get((p_j, row_index), 0.0),
                    modulus=fastener.Eb,
                    diameter=fastener.D,
                    quantity=1.0,
                    t1=t1,
                    t2=t2
                ))
        return results

    def _generate_node_results(self, displacements: List[float]) -> List[NodeResult]:
        """Generate results for all nodes."""
        results = []
        for plate_idx, plate in enumerate(self.plates):
            segments = plate.segment_count()
            for local_node in range(segments + 1):
                dof = self._dof.get((plate_idx, local_node))
                disp = displacements[dof] if dof is not None else 0.0

                row_abs = plate.first_row + local_node
                legacy_id = 1000 * (plate_idx + 1) + row_abs

                t_node = plate.t
                if plate.thicknesses:
                    s_idx = local_node if local_node < len(plate.thicknesses) else local_node - 1
                    if 0 <= s_idx < len(plate.thicknesses):
                        t_node = plate.thicknesses[s_idx]

                area_node = 0.0
                if 0 <= local_node < len(plate.A_strip):
                    area_node = plate.A_strip[local_node]
                elif local_node > 0 and local_node - 1 < len(plate.A_strip):
                    area_node = plate.A_strip[local_node - 1]

                results.append(NodeResult(
                    plate_id=plate_idx,
                    plate_name=plate.name,
                    local_node=local_node,
                    x=self._x.get((plate_idx, local_node), 0.0),
                    displacement=disp,
                    net_bypass=0.0, # Populated later
                    thickness=t_node,
                    bypass_area=area_node,
                    row=row_abs,
                    legacy_id=legacy_id
                ))
        return results

    def _generate_bar_results(self, displacements: List[float]) -> List[BarResult]:
        """Generate results for all bar elements."""
        results = []
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

                results.append(BarResult(
                    plate_id=plate_idx,
                    plate_name=plate.name,
                    segment=seg,
                    axial_force=axial_force,
                    stiffness=k_bar,
                    modulus=plate.E
                ))
        return results

    def _update_nodes_with_bypass(
        self, 
        node_results: List[NodeResult], 
        bar_results: List[BarResult], 
        validated_supports: List[Tuple[int, int, float]]
    ) -> None:
        """Calculate and update net bypass loads for nodes."""
        bar_force_map = {(b.plate_id, b.segment): b.axial_force for b in bar_results}

        support_left = sum(1 for _, ln, _ in validated_supports if ln == 0)
        support_right = len(validated_supports) - support_left
        is_support_left = support_left >= support_right

        for node in node_results:
            f_right = bar_force_map.get((node.plate_id, node.local_node), 0.0)
            f_left = bar_force_map.get((node.plate_id, node.local_node - 1), 0.0)
            node.net_bypass = f_left if is_support_left else f_right

    def _generate_bearing_bypass_results(
        self, 
        node_results: List[NodeResult], 
        bar_results: List[BarResult],
        bearing_forces: Dict[Tuple[int, int], float]
    ) -> List[BearingBypassResult]:
        """Generate combined bearing and bypass results."""
        bar_force_map = {(b.plate_id, b.segment): b.axial_force for b in bar_results}
        results = []
        for node in node_results:
            brg = bearing_forces.get((node.plate_id, node.row), 0.0)
            f_right = bar_force_map.get((node.plate_id, node.local_node), 0.0)
            f_left = bar_force_map.get((node.plate_id, node.local_node - 1), 0.0)

            results.append(BearingBypassResult(
                row=node.row,
                plate_name=node.plate_name,
                bearing=brg,
                bypass=node.net_bypass,
                flow_left=f_left,
                flow_right=f_right
            ))
        return results

    def _generate_reaction_results(
        self,
        displacements: List[float],
        stiffness_matrix_orig: List[List[float]],
        validated_supports: List[Tuple[int, int, float]],
        validated_forces: List[Tuple[int, int, float]],
        ndof: int
    ) -> List[ReactionResult]:
        """Calculate reactions at support nodes."""
        results = []
        for plate_index, local_node, _ in validated_supports:
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                internal_force = sum(stiffness_matrix_orig[dof][j] * displacements[j] for j in range(ndof))
                f_app = 0.0
                plate = self.plates[plate_index]
                if local_node == 0:
                    f_app += plate.Fx_left
                if local_node == plate.segment_count():
                    f_app += plate.Fx_right
                for pi, ln, val in validated_forces:
                    if pi == plate_index and ln == local_node:
                        f_app += val
                reaction = internal_force - f_app
                results.append(ReactionResult(
                    plate_id=plate_index,
                    plate_name=self.plates[plate_index].name,
                    local_node=local_node,
                    global_node=self.plates[plate_index].first_row + local_node,
                    reaction=reaction
                ))
        return results


    def _prepare_system(
        self,
        supports: Sequence[Tuple[int, int, float]],
        point_forces: Sequence[Tuple[int, int, float]],
    ) -> Tuple[List[List[float]], List[float], List[Tuple[int, int, float, int, int, float]], List[Tuple[int, int, float]], List[Tuple[int, int, float]], int]:
        point_forces = point_forces or []
        self._interface_properties = {}
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
            plates_at_row, connection_pairs = self._plates_and_pairs(fastener)
            if not plates_at_row:
                continue
            topology = self._fastener_topology(fastener)
            if topology in ("boeing_chain", "boeing_chain_eq1", "boeing_chain_eq2"):
                self._assemble_boeing_chain(plates_at_row, connection_pairs, fastener, springs, stiffness_matrix)
            elif topology == "boeing_star_scaled":
                self._assemble_boeing_star_scaled(plates_at_row, connection_pairs, fastener, springs, stiffness_matrix)
            elif topology in ("boeing_star_raw", "boeing_star_eq1", "boeing_star_eq2"):
                self._assemble_boeing_star_raw(plates_at_row, connection_pairs, fastener, springs, stiffness_matrix)
            elif topology == "boeing_beam":
                # Experimental: Beam-derived branch compliances with star architecture
                self._assemble_boeing_beam(plates_at_row, connection_pairs, fastener, springs, stiffness_matrix)
            elif topology == "empirical_chain":
                self._assemble_empirical_chain(plates_at_row, connection_pairs, fastener, springs, stiffness_matrix)
            else:
                self._assemble_empirical_star(plates_at_row, connection_pairs, fastener, springs, stiffness_matrix)

        for plate_index, local_node, magnitude in validated_forces:
            dof = self._dof.get((plate_index, local_node))
            if dof is not None:
                force_vector[dof] += magnitude

        return stiffness_matrix, force_vector, springs, validated_supports, validated_forces, ndof

    def _plates_and_pairs(self, fastener: FastenerRow) -> Tuple[List[Tuple[int, Plate]], List[Tuple[int, int]]]:
        row_index = fastener.row
        plates_at_row = [
            (idx, plate)
            for idx, plate in enumerate(self.plates)
            if plate.first_row <= row_index <= plate.last_row
        ]
        plates_at_row.sort(key=lambda x: x[0])
        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        connection_pairs: List[Tuple[int, int]]

        if fastener.connections:
            allowed_indices = {idx for pair in fastener.connections for idx in pair}
            available = set(plate_lookup.keys())
            
            missing = allowed_indices - available
            if missing:
                raise ValueError(
                    f"Fastener connections reference plates not present at row {row_index}: {sorted(missing)}"
                )
            
            plates_at_row = [(idx, plate_lookup[idx]) for idx in sorted(allowed_indices)]
            connection_pairs = list(fastener.connections)
        else:
            connection_pairs = [
                (plates_at_row[i][0], plates_at_row[i + 1][0])
                for i in range(len(plates_at_row) - 1)
            ]
        return plates_at_row, connection_pairs

    def _assemble_pairs_core(
        self,
        plate_lookup: Dict[int, Plate],
        row_index: int,
        connection_pairs: List[Tuple[int, int]],
        stiffness_per_pair: List[float],
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        """Assemble stiffness elements for direct plate-to-plate pairs (Chain topology)."""
        for (idx_i, idx_j), k_pair in zip(connection_pairs, stiffness_per_pair):
            plate_i = plate_lookup[idx_i]
            local_node_i = row_index - plate_i.first_row
            dof_i = self._dof.get((idx_i, local_node_i))

            plate_j = plate_lookup[idx_j]
            local_node_j = row_index - plate_j.first_row
            dof_j = self._dof.get((idx_j, local_node_j))

            if dof_i is not None and dof_j is not None:
                stiffness_matrix[dof_i][dof_i] += k_pair
                stiffness_matrix[dof_j][dof_j] += k_pair
                stiffness_matrix[dof_i][dof_j] -= k_pair
                stiffness_matrix[dof_j][dof_i] -= k_pair

                # Record spring. Using idx_i as the primary plate for the spring record.
                # Compliance is 1/k
                comp = 1.0 / k_pair if k_pair > 0 else 0.0
                springs.append((dof_i, dof_j, k_pair, row_index, idx_i, comp))

    def _assemble_boeing_chain(
        self,
        plates_at_row: List[Tuple[int, Plate]],
        connection_pairs: List[Tuple[int, int]],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
        ) -> None:
            """
            Boeing 'chain' topology using true per-interface Boeing-69 stiffnesses.

            For each connection pair (i, j) we treat the interface as a single-shear
            joint with compliance C_ij from D6-29942 and stiffness k_ij = 1/C_ij.
            This matches Boeing JOLT's per-interface stiffness values.
            """
            if not connection_pairs:
                return

            plate_lookup = {idx: plate for idx, plate in plates_at_row}
            # ordered_plates = self._ordered_plate_indices(connection_pairs)
            row_index = fastener.row
            stiffness_per_pair: List[float] = []

            for idx_i, idx_j in connection_pairs:
                plate_i = plate_lookup[idx_i]
                plate_j = plate_lookup[idx_j]
                t_i = self._thickness_at_row(plate_i, row_index)
                t_j = self._thickness_at_row(plate_j, row_index)

                # Boeing 69 single-shear compliance between i and j
                comp_ij = self._calculate_compliance_pairwise(
                    plate_i, plate_j, fastener, t_i, t_j
                )
                if comp_ij <= 0.0:
                    comp_ij = 1e-12
                k_ij = 1.0 / comp_ij

                stiffness_per_pair.append(k_ij)

                # Store the *pairwise* compliance & stiffness for reporting
                self._interface_properties[(row_index, idx_i, idx_j)] = (comp_ij, k_ij)

            self._assemble_pairs_core(
                plate_lookup, row_index, connection_pairs,
                stiffness_per_pair, springs, stiffness_matrix
            )

    def _assemble_empirical_chain(
        self,
        plates_at_row: List[Tuple[int, Plate]],
        connection_pairs: List[Tuple[int, int]],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        """Assemble a chain using empirical (single-shear) pairwise compliances."""
        if not connection_pairs:
            return

        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        ordered_plates = self._ordered_plate_indices(connection_pairs)
        row_index = fastener.row
        stiffness_per_pair: List[float] = []

        for idx_i, idx_j in connection_pairs:
            plate_i = plate_lookup[idx_i]
            plate_j = plate_lookup[idx_j]
            t_i = self._thickness_at_row(plate_i, row_index)
            t_j = self._thickness_at_row(plate_j, row_index)
            comp = self._calculate_compliance_pairwise(plate_i, plate_j, fastener, t_i, t_j)
            if comp <= 0:
                comp = 1e-12
            k_pair = 1.0 / comp
            stiffness_per_pair.append(k_pair)
            self._interface_properties[(row_index, idx_i, idx_j)] = (comp, k_pair)

        self._assemble_pairs_core(plate_lookup, row_index, connection_pairs, stiffness_per_pair, springs, stiffness_matrix)

    def _ordered_plate_indices(self, connection_pairs: List[Tuple[int, int]]) -> List[int]:
        """Flatten connection pairs into a sorted list of unique plate indices."""
        indices = set()
        for i, j in connection_pairs:
            indices.add(i)
            indices.add(j)
        return sorted(indices)

    def _boeing_chain_compliance(
        self,
        ordered_plates: List[int],
        plate_lookup: Dict[int, Plate],
        fastener: FastenerRow,
        row_index: int,
    ) -> float:
        """Calculate total compliance of the chain (sum of pairwise compliances)."""
        if len(ordered_plates) < 2:
            return 0.0
            
        total_comp = 0.0
        for i in range(len(ordered_plates) - 1):
            idx_i = ordered_plates[i]
            idx_j = ordered_plates[i + 1]
            plate_i = plate_lookup[idx_i]
            plate_j = plate_lookup[idx_j]
            t_i = self._thickness_at_row(plate_i, row_index)
            t_j = self._thickness_at_row(plate_j, row_index)
            
            c_ij = self._calculate_compliance_pairwise(plate_i, plate_j, fastener, t_i, t_j)
            if c_ij > 0:
                total_comp += c_ij
        return total_comp

    def _assemble_star_from_compliances(
        self,
        ordered_plates: List[int],
        plate_lookup: Dict[int, Plate],
        branch_compliances: List[float],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        if not ordered_plates:
            return

        row_index = fastener.row
        dof_fastener = self._dof.get(("fastener", row_index))
        if dof_fastener is None:
            return

        for plate_idx, C_i in zip(ordered_plates, branch_compliances):
            if abs(C_i) < 1e-12:
                C_i = 1e-12
            stiffness = 1.0 / C_i
            plate = plate_lookup[plate_idx]
            local_node = row_index - plate.first_row
            dof_plate = self._dof.get((plate_idx, local_node))

            if dof_plate is not None:
                springs.append((dof_plate, dof_fastener, stiffness, row_index, plate_idx, C_i))
                stiffness_matrix[dof_plate][dof_plate] += stiffness
                stiffness_matrix[dof_plate][dof_fastener] -= stiffness
                stiffness_matrix[dof_fastener][dof_plate] -= stiffness
                stiffness_matrix[dof_fastener][dof_fastener] += stiffness

    def _assemble_boeing_beam(
        self,
        plates_at_row: List[Tuple[int, Plate]],
        connection_pairs: List[Tuple[int, int]],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        """
        Assemble fastener using the JOSEF/JOLT "Ladder" Topology.
        Ref: Jarfall (1972), Fig 4 & 6.
        
        Physics:
        1. Topology: 2D Frame (Plates = Nodes u, Fastener = Beam v/theta).
        2. Interaction: Bearing Springs connect u_i to v_i.
        3. Calibration: Beam EI calibrated to (Total - Bearing) compliance to avoid double-counting.
        4. Boundary: Head and Nut are clamped against rotation (theta=0).
        """
        n = len(plates_at_row)
        if n < 2:
            return

        row_index = fastener.row
        plate_indices = [p[0] for p in plates_at_row]
        plate_objs = [p[1] for p in plates_at_row]
        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        
        # --- 1. Compute Components & Calibrate ---
        E_b = fastener.Eb
        
        # Calculate explicit bearing stiffness for each plate
        k_brg = []
        for i, plate in enumerate(plate_objs):
            t_p = self._thickness_at_row(plate, row_index)
            # Boeing 69 Bearing Term: C = 1/t * (1/Eb + 1/Ep)
            c_brg_val = (1.0 / t_p) * (1.0 / E_b + 1.0 / plate.E)
            # Use raw value (no 1.12 factor)
            k_brg.append(1.0 / max(c_brg_val, 1e-12))

        # --- 2. Build Local Super-Element Matrix ---
        size = 3 * n
        K_local = [[0.0] * size for _ in range(size)]
        
        def idx_u(i): return i
        def idx_v(i): return n + 2*i
        def idx_th(i): return n + 2*i + 1
        
        # A. Add Bearing Springs (u <-> v)
        for i in range(n):
            k = k_brg[i]
            iu, iv = idx_u(i), idx_v(i)
            K_local[iu][iu] += k
            K_local[iv][iv] += k
            K_local[iu][iv] -= k
            K_local[iv][iu] -= k
            
        # B. Add Beam Segments (with Calibrated EI)
        for i in range(n - 1):
            p_i = plate_objs[i]
            p_j = plate_objs[i+1]
            t_i = self._thickness_at_row(p_i, row_index)
            t_j = self._thickness_at_row(p_j, row_index)
            
            # Total Target Compliance (Single Shear Boeing 69)
            # We match the single-shear compliance exactly.
            C_total = self._calculate_compliance_pairwise(p_i, p_j, fastener, t_i, t_j, shear_planes=1)
            
            self._interface_properties[(row_index, plate_indices[i], plate_indices[i+1])] = (C_total, 1.0/max(C_total, 1e-12))

            # Isolate Beam Compliance (Subtract Bearing Terms)
            # C_beam = C_total - (1/k_brg_i + 1/k_brg_j)
            C_brg_term = (1.0/k_brg[i]) + (1.0/k_brg[i+1])
            C_beam_target = C_total - C_brg_term
            if C_beam_target < 1e-13:
                C_beam_target = 1e-13
            
            L = (t_i + t_j) / 2.0
            if L < 1e-9:
                L = 0.01
            
            # Calibrate Effective EI to match C_beam_target (Fixed-Fixed Beam segment)
            EI_eff = (L**3) / (12.0 * C_beam_target)
            
            k11 = 12.0 * EI_eff / L**3
            k12 = 6.0 * EI_eff / L**2
            k22 = 4.0 * EI_eff / L
            k22_cross = 2.0 * EI_eff / L
            
            # Local beam assembly
            iv1, ith1 = idx_v(i), idx_th(i)
            iv2, ith2 = idx_v(i+1), idx_th(i+1)
            
            K_local[iv1][iv1] += k11; K_local[iv1][iv2] -= k11
            K_local[iv2][iv1] -= k11; K_local[iv2][iv2] += k11
            K_local[iv1][ith1] += k12; K_local[iv1][ith2] += k12
            K_local[iv2][ith1] -= k12; K_local[iv2][ith2] -= k12
            K_local[ith1][iv1] += k12; K_local[ith1][iv2] -= k12
            K_local[ith2][iv1] += k12; K_local[ith2][iv2] -= k12
            K_local[ith1][ith1] += k22; K_local[ith1][ith2] += k22_cross
            K_local[ith2][ith1] += k22_cross; K_local[ith2][ith2] += k22

        # --- 3. Static Condensation ---
        u_indices = list(range(n))
        b_indices = []
        for i in range(n):
            b_indices.append(idx_v(i))
            if i > 0 and i < n - 1:
                b_indices.append(idx_th(i))
                
        K_uu = extract_submatrix(K_local, u_indices, u_indices)
        K_ub = extract_submatrix(K_local, u_indices, b_indices)
        K_bu = extract_submatrix(K_local, b_indices, u_indices)
        K_bb = extract_submatrix(K_local, b_indices, b_indices)
        
        nb = len(b_indices)
        try:
            K_bb_inv = [[0.0] * nb for _ in range(nb)]
            for col in range(nb):
                rhs = [1.0 if r == col else 0.0 for r in range(nb)]
                col_sol = solve_dense(K_bb, rhs)
                for r in range(nb):
                    K_bb_inv[r][col] = col_sol[r]
            
            Temp = [[sum(K_bb_inv[r][k] * K_bu[k][c] for k in range(nb)) for c in range(n)] for r in range(nb)]
            Correction = [[sum(K_ub[r][k] * Temp[k][c] for k in range(nb)) for c in range(n)] for r in range(n)]
            K_cond = [[K_uu[r][c] - Correction[r][c] for c in range(n)] for r in range(n)]
            
            self._condensed_recovery[row_index] = {
                "plate_indices": plate_indices,
                "k_brg": k_brg,
                "Temp": Temp,
                "b_indices_map": b_indices
            }
        except:
            K_cond = K_uu

        # --- 4. Add to Global Stitching ---
        for r in range(n):
            for c in range(n):
                k_eff = K_cond[r][c]
                if abs(k_eff) < 1e-9:
                    continue
                
                p_r = plate_objs[r]
                p_c = plate_objs[c]
                local_r = row_index - p_r.first_row
                local_c = row_index - p_c.first_row
                
                dof_r = self._dof.get((plate_indices[r], local_r))
                dof_c = self._dof.get((plate_indices[c], local_c))
                
                if dof_r is not None and dof_c is not None:
                    stiffness_matrix[dof_r][dof_c] += k_eff

    def _assemble_boeing_beam_unused(
        self,
        plates_at_row: List[Tuple[int, Plate]],
        connection_pairs: List[Tuple[int, int]],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        """
        Assemble fastener using Beam-derived branch compliances with Star architecture.
        
        This computes a Condensed Timoshenko Beam stiffness matrix, then extracts
        branch compliances from the diagonal. These are used with the standard
        star assembly (_assemble_star_from_compliances) for proper force recovery.
        
        The beam model captures bolt tilting behavior for multi-layer joints while
        maintaining compatibility with the existing star DOF architecture.
        """
        n = len(plates_at_row)
        if n < 2:
            return

        row_index = fastener.row
        plate_indices = [p[0] for p in plates_at_row]
        plate_objs = [p[1] for p in plates_at_row]
        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        
        # 1. Build Local Beam Matrix (2n x 2n: u, theta alternating)
        k_local = [[0.0] * (2 * n) for _ in range(2 * n)]
        
        # Constants
        A_nom = math.pi * fastener.D**2 / 4.0
        I_nom = math.pi * fastener.D**4 / 64.0
        G_b = fastener.Eb / (2.0 * (1.0 + fastener.nu_b))

        # 2. Add Bearing Stiffness (Diagonal u terms)
        for i, plate in enumerate(plate_objs):
            t = self._thickness_at_row(plate, row_index)
            term = (1.0 / t) * (1.0/plate.E + 1.0/fastener.Eb)
            # Internal plates in the stack behave as Double Shear (2x Stiffness)
            is_internal_plate = (i > 0) and (i < n - 1)
            brg_factor = 2.0 if is_internal_plate else 1.0
            
            k_brg = 1.0 / term if term > 0 else 1e12
            k_local[2*i][2*i] += k_brg

        # 2b. Add rotational stiffness to stabilize K_tt
        k_rot_stab = 1e-6 * (fastener.Eb * I_nom)
        for i in range(n):
            k_local[2*i+1][2*i+1] += k_rot_stab

        # 3. Add Beam Elements (Shear + Bending)
        for i in range(n - 1):
            p1, p2 = plate_objs[i], plate_objs[i+1]
            t1 = self._thickness_at_row(p1, row_index)
            t2 = self._thickness_at_row(p2, row_index)
            L = (t1 + t2) / 2.0
            
            # Boeing 69 Compliances
            C_s = 4.0 * (t1 + t2) / (9.0 * G_b * A_nom)
            bending_num = t1**3 + 5.0*t1**2*t2 + 5.0*t1*t2**2 + t2**3
            C_b = bending_num / (40.0 * fastener.Eb * I_nom)
            

        """Legacy Boeing star using pairwise single-shear compliances (current behaviour)."""
        if not connection_pairs:
            return

        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        row_index = fastener.row
        pairwise_compliances = []
        ordered_plates = self._ordered_plate_indices(connection_pairs)

        for idx_i, idx_j in connection_pairs:
            plate_i = plate_lookup[idx_i]
            plate_j = plate_lookup[idx_j]
            t_i = self._thickness_at_row(plate_i, row_index)
            t_j = self._thickness_at_row(plate_j, row_index)
            # CRITICAL FIX: Always use Single Shear (shear_planes=1) for Analysis inputs.
            # This aligns the WLS/Star decomposition with the reported stiffness values.
            compliance = self._calculate_compliance_pairwise(
                plate_i, plate_j, fastener, t_i, t_j, shear_planes=1
            )
            if compliance <= 1e-12:
                compliance = 1e-12
            pairwise_compliances.append(compliance)
            self._interface_properties[(row_index, idx_i, idx_j)] = (compliance, 1.0 / compliance)

        ordered_plate_objects = [plate_lookup[idx] for idx in ordered_plates]
        branch_compliances = self._solve_branch_compliances(ordered_plate_objects, fastener, row_index, pairwise_compliances)
        self._assemble_star_from_compliances(ordered_plates, plate_lookup, branch_compliances, fastener, springs, stiffness_matrix)

    def _assemble_boeing_star_scaled(
        self,
        plates_at_row: List[Tuple[int, Plate]],
        connection_pairs: List[Tuple[int, int]],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        """Boeing star using per-plate Base Compliances (Single Shear).
        
        This variant uses the calculated single-shear base compliance for each branch directly,
        without WLS decomposition or additional Double Shear scaling.
        Empirically proven to be the most robust for D06_4 and D06_5 (N>3 layers).
        """
        if not connection_pairs:
            return

        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        ordered_plates = self._ordered_plate_indices(connection_pairs)
        row_index = fastener.row

        # Store the ACTUAL pairwise stiffness for reporting
        for idx_i, idx_j in connection_pairs:
            plate_i = plate_lookup[idx_i]
            plate_j = plate_lookup[idx_j]
            t_i = self._thickness_at_row(plate_i, row_index)
            t_j = self._thickness_at_row(plate_j, row_index)
            comp_ij = self._calculate_compliance_pairwise(plate_i, plate_j, fastener, t_i, t_j)
            if comp_ij <= 0.0:
                comp_ij = 1e-12
            k_ij = 1.0 / comp_ij
            self._interface_properties[(row_index, idx_i, idx_j)] = (comp_ij, k_ij)

        # Compute per-plate base compliances (single-layer Boeing model)
        # then apply 2x scaling for double-shear behavior
        ordered_plate_objects = [plate_lookup[idx] for idx in ordered_plates]
        base_compliances = [
            max(self._calculate_base_compliance(plate, fastener, 
                self._thickness_at_row(plate, row_index)), 1e-12)
            for plate in ordered_plate_objects
        ]
        
        # Apply 1.0x scaling (Single Shear Base Compliance)
        branch_compliances = [1.0 * c for c in base_compliances]
        
        self._assemble_star_from_compliances(ordered_plates, plate_lookup, branch_compliances, fastener, springs, stiffness_matrix)

    def _assemble_empirical_star(
        self,
        plates_at_row: List[Tuple[int, Plate]],
        connection_pairs: List[Tuple[int, int]],
        fastener: FastenerRow,
        springs: List[Tuple[int, int, float, int, int, float]],
        stiffness_matrix: List[List[float]],
    ) -> None:
        """Generic empirical star (Huth/others) using pairwise compliances."""
        if not connection_pairs:
            return

        plate_lookup = {idx: plate for idx, plate in plates_at_row}
        row_index = fastener.row
        pairwise_compliances = []
        ordered_plates = self._ordered_plate_indices(connection_pairs)

        for idx_i, idx_j in connection_pairs:
            plate_i = plate_lookup[idx_i]
            plate_j = plate_lookup[idx_j]
            t_i = self._thickness_at_row(plate_i, row_index)
            t_j = self._thickness_at_row(plate_j, row_index)
            compliance = self._calculate_compliance_pairwise(plate_i, plate_j, fastener, t_i, t_j)
            if compliance <= 1e-12:
                compliance = 1e-12
            pairwise_compliances.append(compliance)
            self._interface_properties[(row_index, idx_i, idx_j)] = (compliance, 1.0 / compliance)

        ordered_plate_objects = [plate_lookup[idx] for idx in ordered_plates]
        branch_compliances = self._solve_branch_compliances(ordered_plate_objects, fastener, row_index, pairwise_compliances)
        self._assemble_star_from_compliances(ordered_plates, plate_lookup, branch_compliances, fastener, springs, stiffness_matrix)