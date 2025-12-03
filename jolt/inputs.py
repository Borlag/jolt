"""Input processing logic for JOLT 1D Joint application."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from jolt import FastenerRow, Plate


def process_refined_rows(
    pitches: List[float],
    plates: List[Plate],
    fasteners: List[FastenerRow],
    extra_node_locs: List[float],
) -> Tuple[List[float], List[Plate], List[FastenerRow]]:
    """
    Refines the grid by adding extra nodes at specified locations.
    
    Args:
        pitches: Original list of pitches.
        plates: Original list of plates (referencing original row indices).
        fasteners: Original list of fasteners (referencing original row indices).
        extra_node_locs: List of absolute X coordinates for new nodes.
        
    Returns:
        Tuple of (new_pitches, new_plates, new_fasteners).
    """
    if not extra_node_locs:
        return pitches, plates, fasteners

    # 1. Calculate original node positions
    original_nodes = [0.0]
    for p in pitches:
        original_nodes.append(original_nodes[-1] + p)
    
    # 2. Merge and sort unique node locations
    # Use a small tolerance for equality checks to avoid floating point issues
    tolerance = 1e-9
    
    all_locs = sorted(original_nodes + extra_node_locs)
    unique_locs = []
    if all_locs:
        unique_locs.append(all_locs[0])
        for loc in all_locs[1:]:
            if loc > unique_locs[-1] + tolerance:
                unique_locs.append(loc)
    
    # 3. Calculate new pitches
    new_pitches = []
    for i in range(len(unique_locs) - 1):
        new_pitches.append(unique_locs[i+1] - unique_locs[i])
        
    # 4. Create mapping from X coordinate to New Row Index
    # We map the *original* node positions to their new indices
    x_to_new_row: Dict[float, int] = {}
    for idx, x in enumerate(unique_locs):
        x_to_new_row[x] = idx
        
    def get_new_row(x_target: float) -> int:
        # Find closest match
        best_idx = -1
        min_dist = float('inf')
        for x, idx in x_to_new_row.items():
            dist = abs(x - x_target)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        if min_dist > tolerance:
            raise ValueError(f"Could not map location {x_target} to new grid.")
        return best_idx

    # 5. Update Plates
    new_plates = []
    for plate in plates:
        # Get original X coordinates
        # Note: plate.first_row is 1-based in input, so subtract 1 for list access
        if plate.first_row - 1 >= len(original_nodes) or plate.last_row - 1 >= len(original_nodes):
             # Should not happen if input is valid
             continue
             
        x_start = original_nodes[plate.first_row - 1]
        x_end = original_nodes[plate.last_row - 1]
        
        # get_new_row returns 0-based index, convert to 1-based
        new_first = get_new_row(x_start) + 1
        new_last = get_new_row(x_end) + 1
        
        new_plates.append(replace(plate, first_row=new_first, last_row=new_last))
        
    # 6. Update Fasteners
    new_fasteners = []
    for f in fasteners:
        row_idx = int(f.row) - 1 # 1-based to 0-based
        if row_idx >= len(original_nodes):
            continue
        x_pos = original_nodes[row_idx]
        new_row = get_new_row(x_pos) + 1 # 0-based to 1-based
        new_fasteners.append(replace(f, row=new_row))
        
    return new_pitches, new_plates, new_fasteners


def process_node_based(
    nodes: Dict[int, float],
    elements: List[Dict[str, Any]],
    fasteners: List[Dict[str, Any]],
) -> Tuple[List[float], List[Plate], List[FastenerRow]]:
    """
    Converts node/element definitions into pitches/plates/fasteners.
    
    Args:
        nodes: Dict mapping Node ID -> X coordinate.
        elements: List of dicts defining elements. Expected keys:
                  'layer_name', 'start_node', 'end_node', 'E', 't', 'width', 'nu'.
        fasteners: List of dicts defining fasteners. Expected keys:
                   'node_id', 'diameter', 'E', 'v', 'method', 'connected_layers' (list of layer names).
                   
    Returns:
        Tuple of (pitches, plates, fasteners).
    """
    if not nodes:
        return [], [], []

    # 1. Create Grid from unique sorted X coordinates
    unique_x = sorted(list(set(nodes.values())))
    x_to_row = {x: i + 1 for i, x in enumerate(unique_x)} # 1-based indexing
    
    pitches = []
    for i in range(len(unique_x) - 1):
        pitches.append(unique_x[i+1] - unique_x[i])
        
    # 2. Process Elements -> Plates
    # Group elements by layer_name to merge contiguous ones
    layer_groups: Dict[str, List[Dict[str, Any]]] = {}
    for el in elements:
        name = el.get('layer_name', 'Unnamed')
        if name not in layer_groups:
            layer_groups[name] = []
        layer_groups[name].append(el)
        
    plates = []
    
    for layer_name, group in layer_groups.items():
        # Sort elements by start position
        group.sort(key=lambda e: nodes.get(e['start_node'], 0.0))
        
        current_plate_start_row: Optional[int] = None
        current_plate_end_row: Optional[int] = None
        current_props: Dict[str, Any] = {}
        current_areas: List[float] = []
        
        for el in group:
            n_start = el['start_node']
            n_end = el['end_node']
            if n_start not in nodes or n_end not in nodes:
                continue
                
            x_start = nodes[n_start]
            x_end = nodes[n_end]
            
            # Ensure start < end
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            
            r_start = x_to_row[x_start]
            r_end = x_to_row[x_end]
            
            # Calculate area for this element
            element_area = el.get('width', 1.0) * el.get('t', 0.1)
            
            # Check continuity with previous element in this group
            is_continuous = (
                current_plate_end_row is not None 
                and r_start == current_plate_end_row
                and el.get('E') == current_props.get('E')
                and el.get('t') == current_props.get('t')
                # We allow width to vary, so we don't check width for continuity of the *Plate object*
            )
            
            if is_continuous:
                # Extend current plate
                current_plate_end_row = r_end
                current_areas.append(element_area)
            else:
                # Commit previous plate if exists
                if current_plate_start_row is not None and current_plate_end_row is not None:
                    plates.append(Plate(
                        name=layer_name,
                        E=current_props.get('E', 1e7),
                        t=current_props.get('t', 0.1),
                        A_strip=current_areas,
                        first_row=current_plate_start_row,
                        last_row=current_plate_end_row
                    ))
                
                # Start new plate
                current_plate_start_row = r_start
                current_plate_end_row = r_end
                current_props = el
                current_areas = [element_area]
                
        # Commit last plate
        if current_plate_start_row is not None and current_plate_end_row is not None:
            plates.append(Plate(
                name=layer_name,
                E=current_props.get('E', 1e7),
                t=current_props.get('t', 0.1),
                A_strip=current_areas,
                first_row=current_plate_start_row,
                last_row=current_plate_end_row
            ))
    # 3. Process Fasteners
    # We need to map "connected_layers" (names) to plate indices
    # Since there can be multiple plates with the same name (if discontinuous),
    # we need to find which specific plate exists at the fastener's location.
    
    final_fasteners = []
    
    for f_idx, f in enumerate(fasteners):
        n_id = f.get('node_id')
        if n_id not in nodes:
            continue
        x_pos = nodes[n_id]
        row_idx = x_to_row[x_pos]
        
        # Find which plates exist at this row and match the requested layer names
        connected_names = f.get('connected_layers', [])
        connected_plate_indices = []
        
        for p_idx, plate in enumerate(plates):
            if plate.name in connected_names:
                # Check if plate exists at this row
                if plate.first_row <= row_idx <= plate.last_row:
                    connected_plate_indices.append(p_idx)
        
        # Identify pairs of plates to connect
        connections = []
        if len(connected_plate_indices) >= 2:
            # Create a chain of connections: (0,1), (1,2), etc.
            # This models a single fastener passing through multiple plates.
            for i in range(len(connected_plate_indices) - 1):
                connections.append((connected_plate_indices[i], connected_plate_indices[i+1]))
            
            topology = f.get('topology') or None
            final_fasteners.append(FastenerRow(
                row=row_idx,
                D=f.get('diameter', 0.19),
                Eb=f.get('E', 1e7),
                nu_b=f.get('v', 0.3),
                method=f.get('method', 'boeing69'),
                connections=connections,
                topology=topology
            ))
            
    return pitches, plates, final_fasteners
