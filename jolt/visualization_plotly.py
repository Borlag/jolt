"""Plotly visualization module for JOLT 1D Joint application."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore
except Exception:
    go = None
    pio = None

from jolt import (
    FastenerResult,
    FastenerRow,
    JointSolution,
    Plate,
)



def render_joint_diagram_plotly(
    pitches: Sequence[float],
    plates: Sequence[Plate],
    fasteners: Sequence[FastenerRow],
    supports: Sequence[Tuple[int, int, float]],
    solution: JointSolution,
    units: Dict[str, str],
    mode: str = "scheme",
    font_size: int = 10,
) -> Optional[go.Figure]:
    from jolt.ui.utils import resolve_fastener_connections
    if go is None:
        return None

    if not plates:
        return None

    max_pitch = max(pitches) if pitches else 1.0
    vertical_spacing = 1.6
    
    # Calculate global node positions
    global_nodes: List[float] = [0.0]
    for pitch in pitches:
        global_nodes.append(global_nodes[-1] + pitch)

    y_levels: Dict[int, float] = {
        idx: (len(plates) - idx - 1) * vertical_spacing for idx in range(len(plates))
    }

    node_coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
    segment_midpoints: Dict[Tuple[int, int], Tuple[float, float]] = {}
    reaction_map: Dict[Tuple[int, int], float] = {}
    for reaction in solution.reactions:
        reaction_map[(reaction.plate_id, reaction.local_node)] = reaction.reaction

    fig = go.Figure()

    # Opacity for base geometry in fatigue mode
    base_opacity = 0.3 if mode == "fatigue" else 1.0

    # --- Background Grid (Vertical Lines) ---
    for xi in global_nodes:
        fig.add_shape(
            type="line",
            x0=xi, y0=min(y_levels.values()) - 1.0,
            x1=xi, y1=max(y_levels.values()) + 1.0,
            line=dict(color="LightGray", width=1, dash="dash"),
            layer="below"
        )

    # --- Plates (Horizontal Lines) ---
    for plate_idx, plate in enumerate(plates):
        segments = plate.segment_count()
        if segments <= 0:
            continue

        start_index = max(plate.first_row - 1, 0)
        end_index = min(max(plate.last_row - 1, start_index), len(pitches))
        x0 = sum(pitches[:start_index])
        coords: List[float] = [x0]
        for seg_idx in range(start_index, end_index):
            if seg_idx >= len(pitches):
                break
            coords.append(coords[-1] + pitches[seg_idx])
        if len(coords) < 2:
            continue

        y = y_levels[plate_idx]
        
        # Plate Line
        # We need to handle gaps (A_strip <= 1e-9).
        # We'll build lists of x, y coordinates, inserting None to break the line at gaps.
        x_plot = []
        y_plot = []
        
        # Iterate through segments to build the plot path
        # coords has len(segments) + 1 points
        for seg_idx in range(len(coords) - 1):
            strip_idx = start_index - (plate.first_row - 1) + seg_idx
            if 0 <= strip_idx < len(plate.A_strip):
                is_gap = plate.A_strip[strip_idx] <= 1e-9
            else:
                is_gap = False 
            
            if is_gap:
                x_plot.append(None)
                y_plot.append(None)
            else:
                if not x_plot or x_plot[-1] is None:
                    x_plot.append(coords[seg_idx])
                    y_plot.append(y)
                
                x_plot.append(coords[seg_idx+1])
                y_plot.append(y)

        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            line=dict(width=4),
            opacity=base_opacity,
            name=f"{plate.name} ({plate.material_name})" if plate.material_name else plate.name,
            legendgroup=plate.name,
            connectgaps=False 
        ))

        # Nodes
        for local_node, x_node in enumerate(coords):
            node_coords[(plate_idx, local_node)] = (x_node, y)
            fig.add_trace(go.Scatter(
                x=[x_node],
                y=[y],
                mode="markers",
                marker=dict(size=10, color="white", line=dict(width=2, color="black")),
                opacity=base_opacity,
                showlegend=False,
                hoverinfo="text",
                text=f"{plate.name} n{plate.first_row + local_node}"
            ))
            # Node Label
            fig.add_annotation(
                x=x_node,
                y=y + 0.2 * vertical_spacing,
                text=f"n{plate.first_row + local_node}",
                showarrow=False,
                font=dict(size=font_size),
                yshift=5
            )

        # Segment Midpoints
        for segment in range(len(coords) - 1):
            x_mid = 0.5 * (coords[segment] + coords[segment + 1])
            segment_midpoints[(plate_idx, segment)] = (x_mid, y)

        # End Loads
        if abs(plate.Fx_left) > 0.0:
            direction = 1 if plate.Fx_left >= 0 else -1
            start_x = coords[0]
            fig.add_annotation(
                x=start_x,
                y=y,
                ax=start_x - direction * 0.6 * max_pitch,
                ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
            fig.add_annotation(
                x=start_x - direction * 0.65 * max_pitch,
                y=y + 0.3 * vertical_spacing,
                text=f"{plate.Fx_left:+.0f} {units['force']}",
                showarrow=False,
                font=dict(color="red", size=font_size + 2, weight="bold")
            )

        if abs(plate.Fx_right) > 0.0:
            direction = 1 if plate.Fx_right >= 0 else -1
            end_x = coords[-1]
            fig.add_annotation(
                x=end_x,
                y=y,
                ax=end_x - direction * 0.6 * max_pitch,
                ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
            fig.add_annotation(
                x=end_x - direction * 0.65 * max_pitch,
                y=y + 0.3 * vertical_spacing,
                text=f"{plate.Fx_right:+.0f} {units['force']}",
                showarrow=False,
                font=dict(color="red", size=font_size + 2, weight="bold")
            )

    # --- Fasteners ---
    fastener_plot_data: Dict[int, List[Tuple[int, FastenerResult, List[Tuple[float, float]]]]] = defaultdict(list)
    
    # Track seen labels for legend control
    seen_labels: Set[str] = set()

    for fast_idx, fastener in enumerate(fasteners):
        row_index = int(fastener.row)
        attachments: Dict[int, Tuple[float, float]] = {}
        for plate_idx, plate in enumerate(plates):
            if plate.first_row <= row_index <= plate.last_row:
                local = row_index - plate.first_row
                coord = node_coords.get((plate_idx, local))
                if coord is not None:
                    attachments[plate_idx] = coord
        
        pairs = resolve_fastener_connections(fastener, plates)
        used_coords: List[Tuple[float, float]] = []
        
        # Determine Label
        label = fastener.name if fastener.name else f"F{fastener.row}"
        
        # Determine Marker Symbol (Plotly format)
        symbol = fastener.marker_symbol if hasattr(fastener, "marker_symbol") else "circle"
        
        # Legend Logic
        # Group by Label. Only show legend for the first time we see this label.
        show_leg = label not in seen_labels
        if show_leg:
            seen_labels.add(label)

        # Add Trace for the Fastener (Vertical Line)
        for i, (top_idx, bottom_idx) in enumerate(pairs):
            coord_top = attachments.get(top_idx)
            coord_bottom = attachments.get(bottom_idx)
            if coord_top is None or coord_bottom is None:
                continue
            
            x_val = coord_top[0]
            y_vals = [coord_top[1], coord_bottom[1]]
            
            # Only the first segment of a fastener stack gets the legend entry (if it's the first time seeing this label)
            # Actually, if we use legendgroup=label, clicking one toggles all with that label.
            # So we set legendgroup=label for ALL segments of ALL fasteners with this name.
            # And showlegend=True only for the VERY FIRST segment of the VERY FIRST fastener with this name.
            
            is_first_segment = (i == 0)
            
            fig.add_trace(go.Scatter(
                x=[x_val, x_val],
                y=y_vals,
                mode="lines+markers",
                # Dashed purple line
                line=dict(color="purple", width=2, dash="dash"),
                # Custom Marker
                marker=dict(
                    symbol=symbol, 
                    size=10, 
                    color="purple",
                    line=dict(width=1, color="black") 
                ),
                opacity=base_opacity,
                name=label,          
                showlegend=(show_leg and is_first_segment), 
                legendgroup=label, # Group by Name
                hoverinfo="text",
                text=f"{label} (Row {row_index})"
            ))
            used_coords.extend([coord_top, coord_bottom])

        # Annotation Label on the Diagram (Text next to the fastener)
        if used_coords:
            x_label = used_coords[0][0]
            y_label = max(pt[1] for pt in used_coords)
            
            fig.add_annotation(
                x=x_label,
                y=y_label + 0.3 * vertical_spacing,
                text=label,
                showarrow=False,
                font=dict(color="purple", size=font_size),
                bgcolor="rgba(255, 255, 255, 0.8)" # Add background for readability
            )

    # --- Supports ---
    for s_idx, (plate_idx, local_node, value) in enumerate(supports):
        coord = node_coords.get((plate_idx, local_node))
        if coord is None:
            continue
        x_node, y_node = coord
        global_node = plates[plate_idx].first_row + local_node
        reaction = reaction_map.get((plate_idx, local_node))
        
        fig.add_trace(go.Scatter(
            x=[x_node],
            y=[y_node - 0.1 * vertical_spacing], # Moved closer to node
            mode="markers",
            marker=dict(symbol="triangle-up", size=15, color="green"), # Changed to triangle-up
            showlegend=False,
            hoverinfo="text",
            text=f"Support S{s_idx}"
        ))
        
        label_text = (
            f"S{s_idx}: {plates[plate_idx].name} n{global_node}<br>"
            f"u={value:+.3f} {units['length']}"
            + (f"<br>R={reaction:+.1f} {units['force']}" if reaction is not None else "")
        )
        fig.add_annotation(
            x=x_node,
            y=y_node - 0.25 * vertical_spacing, # Moved label up as well
            text=label_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="green",
            font=dict(color="green", size=font_size)
        )

    # --- Mode: Displacements ---
    if mode == "displacements":
        node_displacements: Dict[Tuple[int, int], float] = {}
        max_disp = 0.0
        for (plate_idx, local_node), coord in node_coords.items():
            dof = solution.dof_map.get((plate_idx, local_node))
            if dof is None:
                continue
            disp = solution.displacements[dof]
            node_displacements[(plate_idx, local_node)] = disp
            max_disp = max(max_disp, abs(disp))

        scale = 0.0
        if max_disp > 1e-9:
            scale = 0.55 * max_pitch / max_disp

        for (plate_idx, local_node), (x_node, y_node) in node_coords.items():
            disp = node_displacements.get((plate_idx, local_node))
            if disp is None:
                continue
            target_x = x_node + disp * scale
            
            fig.add_annotation(
                x=target_x,
                y=y_node + 0.1 * vertical_spacing,
                ax=x_node,
                ay=y_node + 0.1 * vertical_spacing,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="orange"
            )
            fig.add_annotation(
                x=x_node,
                y=y_node + 0.25 * vertical_spacing,
                text=f"u = {disp:+.6f} {units['length']}",
                showarrow=False,
                font=dict(color="orange", size=font_size, weight="bold")
            )

    # --- Mode: Loads ---
    if mode == "loads":
        for bar in solution.bars:
            coord = segment_midpoints.get((bar.plate_id, bar.segment))
            if coord is None:
                continue
            x_mid, y_mid = coord
            fig.add_annotation(
                x=x_mid,
                y=y_mid + 0.2 * vertical_spacing,
                text=f"{bar.axial_force:+.1f} {units['force']}",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="blue",
                font=dict(color="blue", size=font_size, weight="bold")
            )

        # Collect fastener data for load labels
        for fast_idx, fastener in enumerate(solution.fasteners):
            row_index = int(fastener.row)
            coords: List[Tuple[float, float]] = []
            for plate_idx in (fastener.plate_i, fastener.plate_j):
                if not (0 <= plate_idx < len(plates)):
                    continue
                plate = plates[plate_idx]
                if plate.first_row <= row_index <= plate.last_row:
                    local = row_index - plate.first_row
                    coord = node_coords.get((plate_idx, local))
                    if coord is not None:
                        coords.append(coord)
            if len(coords) < 2:
                continue
            coords.sort(key=lambda pt: pt[1], reverse=True)
            fastener_plot_data[row_index].append((fast_idx, fastener, coords))

        for row_index, items in fastener_plot_data.items():
            items.sort(
                key=lambda entry: (
                    min(entry[1].plate_i, entry[1].plate_j),
                    max(entry[1].plate_i, entry[1].plate_j),
                )
            )
            count = len(items)
            for position, (fast_idx, fastener, coords) in enumerate(items):
                y_center = sum(pt[1] for pt in coords) / len(coords)
                offset = 0.0
                if count > 1:
                    offset = (position - 0.5 * (count - 1)) * (0.5 * vertical_spacing)
                x_pos = coords[0][0] + 0.15 * max_pitch
                
                fig.add_annotation(
                    x=x_pos,
                    y=y_center + offset,
                    text=f"F{fastener.row}<sup>{fastener.plate_i}-{fastener.plate_j}</sup> = {fastener.force:+.1f} {units['force']}",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.92)",
                    bordercolor="purple",
                    font=dict(color="purple", size=font_size, weight="bold"),
                    xanchor="left"
                )

    # --- Mode: Fatigue ---
    if mode == "fatigue":
        # Iterate through critical points (already sorted by Rank)
        critical_points = getattr(solution, "critical_points", [])
        for i, cp in enumerate(critical_points):
            node_id = cp["node_id"]
            rank = cp.get("rank", i + 1)
            fsi = cp.get("fsi", 0.0)
            peak = cp.get("peak_stress", 0.0)
            
            # Find coordinates
            # We need to find the node object to get plate_id and local_node
            node_obj = next((n for n in solution.nodes if n.legacy_id == node_id), None)
            if not node_obj:
                continue
                
            coord = node_coords.get((node_obj.plate_id, node_obj.local_node))
            if not coord:
                continue
                
            x_c, y_c = coord
            
            # Determine Color and Size based on Rank
            if rank == 1:
                color = "red"
                size = 25
                symbol = "circle-open"
                line_width = 3
            else:
                color = "orange"
                size = 20
                symbol = "circle-open"
                line_width = 2
                
            # Marker
            fig.add_trace(go.Scatter(
                x=[x_c],
                y=[y_c],
                mode="markers",
                marker=dict(size=size, color=f"rgba(255, 0, 0, 0.1)", line=dict(width=line_width, color=color), symbol=symbol),
                name=f"Rank {rank}",
                showlegend=(i < 5), # Only show first few in legend to avoid clutter
                hoverinfo="text",
                hovertext=f"Rank: {rank}<br>Node: {node_id}<br>FSI: {fsi:.2f}<br>Peak: {peak:.0f}"
            ))
            
            # Annotation
            # Stagger annotations to avoid overlap if possible, or just place them
            # Alternating top/bottom might help?
            ay_offset = -40 if i % 2 == 0 else 40
            
            label_text = f"CRIT {rank}<br>FSI: {fsi:.2f}"
            
            fig.add_annotation(
                x=x_c,
                y=y_c,
                text=label_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=color,
                ax=0,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=color,
                font=dict(color=color, size=font_size, weight="bold")
            )

    # --- Layout Configuration ---
    
    # Primary X-Axis (Bottom) - Global Nodes (n1, n2...)
    fig.update_xaxes(
        tickvals=global_nodes,
        ticktext=[f"n{i + 1}" for i in range(len(global_nodes))],
        title_text=f"x [{units['length']}]",
        showgrid=False,
        zeroline=False
    )
    
    # Secondary X-Axis (Top) - Rows aligned with Nodes
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickvals=global_nodes,
            ticktext=[f"Row {i + 1}" for i in range(len(global_nodes))],
            showgrid=False,
            zeroline=False
        )
    )

    fig.update_yaxes(
        tickvals=[y_levels[idx] for idx in range(len(plates))],
        ticktext=[plate.name for plate in plates],
        showgrid=False,
        zeroline=False,
        range=[min(y_levels.values()) - 1.0, max(y_levels.values()) + 1.0]
    )

    titles = {
        "scheme": "Scheme overview",
        "displacements": "Nodal displacements",
        "loads": "Load distribution",
        "fatigue": "Fatigue Criticality (FSI)",
    }
    
    fig.update_layout(
        title=titles.get(mode, titles["scheme"]),
        plot_bgcolor="white",
        showlegend=True,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        dragmode="pan" 
    )

    # --- Critical Point Highlight (Legacy / Non-Fatigue Mode) ---
    # User requested to remove this from Scheme and Displacements
    if solution.critical_node_id and mode not in ["fatigue", "scheme", "displacements"]:
        crit_node = next((n for n in solution.nodes if n.legacy_id == solution.critical_node_id), None)
        if crit_node:
            coord = node_coords.get((crit_node.plate_id, crit_node.local_node))
            if coord:
                x_c, y_c = coord
                
                # Find SSF and Peak Stress value for annotation
                ssf_val = 0.0
                peak_val = 0.0
                if solution.fatigue_results:
                     f_res = next((r for r in solution.fatigue_results if r.node_id == solution.critical_node_id), None)
                     if f_res:
                         ssf_val = f_res.ssf
                         peak_val = getattr(f_res, "peak_stress", 0.0)

                # Add Red Halo / Marker with Legend
                fig.add_trace(go.Scatter(
                    x=[x_c],
                    y=[y_c],
                    mode="markers",
                    marker=dict(size=25, color="rgba(255, 0, 0, 0.3)", line=dict(width=3, color="red"), symbol="circle-open"),
                    name="Critical Node", # Legend entry
                    hoverinfo="text",
                    hovertext=f"Critical Node: {solution.critical_node_id}<br>Peak Stress: {peak_val:.0f} {units['stress']}<br>SSF: {ssf_val:.2f}"
                ))
                
                fig.add_annotation(
                    x=x_c,
                    y=y_c,
                    text=f"CRIT (Peak={peak_val:.0f})",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0,
                    ay=-50,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="red",
                    font=dict(color="red", size=font_size + 1, weight="bold")
                )

    return fig
