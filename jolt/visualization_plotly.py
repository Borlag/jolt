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
from jolt.ui.utils import resolve_fastener_connections


def render_joint_diagram_plotly(
    pitches: Sequence[float],
    plates: Sequence[Plate],
    fasteners: Sequence[FastenerRow],
    supports: Sequence[Tuple[int, int, float]],
    solution: JointSolution,
    mode: str = "scheme",
    font_size: int = 10,
) -> Optional[go.Figure]:
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
            # Map local segment index to plate's A_strip
            # coords starts at start_index (relative to pitches), which corresponds to plate.first_row - 1
            # So seg_idx 0 in coords corresponds to plate.A_strip[start_index - (plate.first_row - 1)]
            # Wait, start_index = max(plate.first_row - 1, 0).
            # If plate.first_row = 1, start_index = 0.
            # plate.A_strip[0] corresponds to segment between row 1 and 2.
            # coords[0] is x at row 1. coords[1] is x at row 2.
            # So segment seg_idx corresponds to plate.A_strip[start_index - (plate.first_row - 1) + seg_idx]
            
            strip_idx = start_index - (plate.first_row - 1) + seg_idx
            if 0 <= strip_idx < len(plate.A_strip):
                is_gap = plate.A_strip[strip_idx] <= 1e-9
            else:
                is_gap = False # Should not happen if indices are correct
            
            if is_gap:
                # If it's a gap, we don't draw this segment.
                # If we were drawing, we need to break the line.
                x_plot.append(None)
                y_plot.append(None)
            else:
                # If we are starting a new segment and the previous was a gap (or start),
                # we need to ensure the start point is added.
                # However, Plotly connects points.
                # To draw segment from coords[seg_idx] to coords[seg_idx+1]:
                # We add coords[seg_idx], then coords[seg_idx+1].
                # But if we just append points, we get a continuous line.
                # If we have [x0, x1, None, x2, x3], we get line x0-x1 and x2-x3.
                
                # Check if we need to add the start point of this segment
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
            name=plate.name,
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
                ax=start_x + direction * 0.6 * max_pitch,
                ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
            fig.add_annotation(
                x=start_x + direction * 0.65 * max_pitch,
                y=y + 0.3 * vertical_spacing,
                text=f"{plate.Fx_left:+.0f} lb",
                showarrow=False,
                font=dict(color="red", size=font_size + 2, weight="bold")
            )

        if abs(plate.Fx_right) > 0.0:
            direction = 1 if plate.Fx_right >= 0 else -1
            end_x = coords[-1]
            fig.add_annotation(
                x=end_x,
                y=y,
                ax=end_x + direction * 0.6 * max_pitch,
                ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
            fig.add_annotation(
                x=end_x + direction * 0.65 * max_pitch,
                y=y + 0.3 * vertical_spacing,
                text=f"{plate.Fx_right:+.0f} lb",
                showarrow=False,
                font=dict(color="red", size=font_size + 2, weight="bold")
            )

    # --- Fasteners ---
    fastener_plot_data: Dict[int, List[Tuple[int, FastenerResult, List[Tuple[float, float]]]]] = defaultdict(list)
    
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
        
        for top_idx, bottom_idx in pairs:
            coord_top = attachments.get(top_idx)
            coord_bottom = attachments.get(bottom_idx)
            if coord_top is None or coord_bottom is None:
                continue
            
            x_val = coord_top[0]
            y_vals = [coord_top[1], coord_bottom[1]]
            
            fig.add_trace(go.Scatter(
                x=[x_val, x_val],
                y=y_vals,
                mode="lines+markers",
                line=dict(color="purple", width=2, dash="dash"),
                marker=dict(size=6, color="purple"),
                showlegend=False,
                hoverinfo="skip"
            ))
            used_coords.extend([coord_top, coord_bottom])

        if used_coords:
            x_label = used_coords[0][0]
            y_label = max(pt[1] for pt in used_coords)
            fig.add_annotation(
                x=x_label,
                y=y_label + 0.3 * vertical_spacing,
                text=f"F{fastener.row}",
                showarrow=False,
                font=dict(color="purple", size=font_size)
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
            y=[y_node - 0.25 * vertical_spacing],
            mode="markers",
            marker=dict(symbol="triangle-down", size=15, color="green"),
            showlegend=False,
            hoverinfo="text",
            text=f"Support S{s_idx}"
        ))
        
        label_text = (
            f"S{s_idx}: {plates[plate_idx].name} n{global_node}<br>"
            f"u={value:+.3f} in"
            + (f"<br>R={reaction:+.1f} lb" if reaction is not None else "")
        )
        fig.add_annotation(
            x=x_node,
            y=y_node - 0.5 * vertical_spacing,
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
        if max_disp > 0:
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
                text=f"u = {disp * 1000:.3f} mil",
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
                text=f"{bar.axial_force:+.1f} lb",
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
                    text=f"F{fastener.row}<sup>{fastener.plate_i}-{fastener.plate_j}</sup> = {fastener.force:+.1f} lb",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.92)",
                    bordercolor="purple",
                    font=dict(color="purple", size=font_size, weight="bold"),
                    xanchor="left"
                )

    # --- Layout Configuration ---
    
    # Primary X-Axis (Bottom) - Global Nodes (n1, n2...)
    fig.update_xaxes(
        tickvals=global_nodes,
        ticktext=[f"n{i + 1}" for i in range(len(global_nodes))],
        title_text="x [in]",
        showgrid=False,
        zeroline=False
    )
    
    # Secondary X-Axis (Top) - Rows aligned with Nodes
    # User requested: "the row should go along the nodes"
    # So we align Row 1 with n1, Row 2 with n2, etc.
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
    }
    
    fig.update_layout(
        title=titles.get(mode, titles["scheme"]),
        plot_bgcolor="white",
        showlegend=True,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        dragmode="pan" # Allow panning by default
    )

    return fig
