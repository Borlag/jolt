"""Visualization module for JOLT 1D Joint application."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
except Exception:
    plt = None
    Line2D = None

from jolt import (
    FastenerResult,
    FastenerRow,
    JointSolution,
    Plate,
)


def _resolve_fastener_connections(
    fastener: FastenerRow, plates: Sequence[Plate]
) -> List[Tuple[int, int]]:
    # This helper is needed here for visualization as well.
    # It is also in ui/utils.py, but to avoid circular imports or complex deps,
    # we might duplicate or import it. For now, let's import it if we can,
    # or just duplicate the logic since it's small and pure.
    # Actually, let's try to keep it DRY. I'll assume it's in jolt.ui.utils
    # but wait, visualization shouldn't depend on UI.
    # It should probably be in the core model or a shared utils.
    # For now, I will include the logic here to keep this module self-contained
    # regarding the plotting logic, or better yet, I'll move these helpers to 
    # a common place later. For this step, I'll duplicate the small helper 
    # to ensure the plot function works independently.
    
    def _plates_present_at_row(plates_seq: Sequence[Plate], row_idx: int) -> List[int]:
        present = [
            idx
            for idx, plate in enumerate(plates_seq)
            if plate.first_row <= row_idx <= plate.last_row
        ]
        present.sort()
        return present

    row_index = int(fastener.row)
    present = _plates_present_at_row(plates, row_index)
    if len(present) < 2:
        return []
    if fastener.connections is None:
        return list(zip(present[:-1], present[1:]))

    order = {plate_idx: position for position, plate_idx in enumerate(present)}
    resolved: List[Tuple[int, int]] = []
    seen = set()

    for pair in fastener.connections:
        if len(pair) != 2:
            continue
        raw_top, raw_bottom = int(pair[0]), int(pair[1])
        if raw_top not in order or raw_bottom not in order:
            continue
        top_idx, bottom_idx = (raw_top, raw_bottom)
        if order[top_idx] > order[bottom_idx]:
            top_idx, bottom_idx = bottom_idx, top_idx
        if abs(order[top_idx] - order[bottom_idx]) != 1:
            continue
        key = (top_idx, bottom_idx)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)

    return resolved


def render_joint_diagram(
    pitches: Sequence[float],
    plates: Sequence[Plate],
    fasteners: Sequence[FastenerRow],
    supports: Sequence[Tuple[int, int, float]],
    solution: JointSolution,
    mode: str = "scheme",
):
    if plt is None:
        raise RuntimeError("Matplotlib is required to render the joint diagram")

    max_pitch = max(pitches) if pitches else 1.0
    vertical_spacing = 1.6
    fig_height = 2.8 + vertical_spacing * max(len(plates) - 1, 0)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    global_nodes: List[float] = [0.0]
    for pitch in pitches:
        global_nodes.append(global_nodes[-1] + pitch)

    if global_nodes:
        margin = 0.12 * max_pitch
        ax.set_xlim(global_nodes[0] - margin, global_nodes[-1] + margin)
        if len(global_nodes) >= 2:
            for idx in range(len(global_nodes) - 1):
                if idx % 2 == 0:
                    ax.axvspan(
                        global_nodes[idx],
                        global_nodes[idx + 1],
                        color="0.96",
                        zorder=0,
                    )
        ax.set_xticks(global_nodes)
        ax.set_xticklabels([f"n{i + 1}" for i in range(len(global_nodes))])

    y_levels: Dict[int, float] = {
        idx: (len(plates) - idx - 1) * vertical_spacing for idx in range(len(plates))
    }

    node_coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
    segment_midpoints: Dict[Tuple[int, int], Tuple[float, float]] = {}
    reaction_map: Dict[Tuple[int, int], float] = {}
    for reaction in solution.reactions:
        reaction_map[(reaction.plate_id, reaction.local_node)] = reaction.reaction

    for xi in global_nodes:
        ax.axvline(
            xi,
            color="0.85",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )

    top_axis = ax.secondary_xaxis("top") if len(global_nodes) >= 2 else None
    if top_axis is not None:
        centers = [
            0.5 * (global_nodes[i] + global_nodes[i + 1])
            for i in range(len(global_nodes) - 1)
        ]
        top_axis.set_xticks(centers)
        top_axis.set_xticklabels([f"Row {i + 1}" for i in range(len(centers))])
        top_axis.tick_params(axis="x", labelsize=9)
        top_axis.set_xlabel("Fastener rows")

    legend_handles: List = []
    legend_labels: List[str] = []

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
        ys = [y] * len(coords)
        (line,) = ax.plot(coords, ys, lw=2.2, label=plate.name)
        legend_handles.append(line)
        legend_labels.append(plate.name)

        for local_node, x_node in enumerate(coords):
            node_coords[(plate_idx, local_node)] = (x_node, y)
            global_row = plate.first_row + local_node
            ax.scatter(x_node, y, s=46, c="white", edgecolors="black", zorder=5)
            ax.text(
                x_node,
                y + 0.22 * vertical_spacing,
                f"n{plate.first_row + local_node}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for segment in range(len(coords) - 1):
            x_mid = 0.5 * (coords[segment] + coords[segment + 1])
            segment_midpoints[(plate_idx, segment)] = (x_mid, y)

        if abs(plate.Fx_left) > 0.0:
            direction = 1 if plate.Fx_left >= 0 else -1
            start_x = coords[0]
            ax.annotate(
                "",
                xy=(start_x + direction * 0.6 * max_pitch, y),
                xytext=(start_x, y),
                arrowprops=dict(arrowstyle="<|-", color="tab:red", lw=2.2),
            )
            ax.scatter(start_x, y, marker="s", s=70, c="tab:red", zorder=6)
            ax.text(
                start_x + direction * 0.65 * max_pitch,
                y + 0.55 * vertical_spacing,
                f"{plate.Fx_left:+.0f} lb",
                ha="left" if direction >= 0 else "right",
                va="bottom",
                fontsize=9,
                color="tab:red",
                fontweight="bold",
            )

        if abs(plate.Fx_right) > 0.0:
            direction = 1 if plate.Fx_right >= 0 else -1
            end_x = coords[-1]
            ax.annotate(
                "",
                xy=(end_x + direction * 0.6 * max_pitch, y),
                xytext=(end_x, y),
                arrowprops=dict(arrowstyle="<|-", color="tab:red", lw=2.2),
            )
            ax.scatter(end_x, y, marker="s", s=70, c="tab:red", zorder=6)
            ax.text(
                end_x + direction * 0.65 * max_pitch,
                y + 0.55 * vertical_spacing,
                f"{plate.Fx_right:+.0f} lb",
                ha="left" if direction >= 0 else "right",
                va="bottom",
                fontsize=9,
                color="tab:red",
                fontweight="bold",
            )

    for fast_idx, fastener in enumerate(fasteners):
        row_index = int(fastener.row)
        attachments: Dict[int, Tuple[float, float]] = {}
        for plate_idx, plate in enumerate(plates):
            if plate.first_row <= row_index <= plate.last_row:
                local = row_index - plate.first_row
                coord = node_coords.get((plate_idx, local))
                if coord is not None:
                    attachments[plate_idx] = coord
        pairs = _resolve_fastener_connections(fastener, plates)
        used_coords: List[Tuple[float, float]] = []
        for top_idx, bottom_idx in pairs:
            coord_top = attachments.get(top_idx)
            coord_bottom = attachments.get(bottom_idx)
            if coord_top is None or coord_bottom is None:
                continue
            x_val = coord_top[0]
            y_vals = [coord_top[1], coord_bottom[1]]
            ax.plot(
                [x_val, x_val],
                y_vals,
                ls="--",
                color="tab:purple",
                lw=1.6,
                marker="o",
                markersize=4,
                zorder=4,
            )
            used_coords.extend([coord_top, coord_bottom])
        if used_coords:
            x_label = used_coords[0][0]
            y_label = max(pt[1] for pt in used_coords)
            ax.text(
                x_label,
                y_label + 0.45 * vertical_spacing,
                f"F{fast_idx + 1} @ n{row_index}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="tab:purple",
            )

    for s_idx, (plate_idx, local_node, value) in enumerate(supports):
        coord = node_coords.get((plate_idx, local_node))
        if coord is None:
            continue
        x_node, y_node = coord
        global_node = plates[plate_idx].first_row + local_node
        reaction = reaction_map.get((plate_idx, local_node))
        ax.plot(
            [x_node, x_node],
            [y_node, y_node - 0.35 * vertical_spacing],
            color="tab:green",
            lw=2.0,
            zorder=4,
        )
        ax.scatter(
            x_node,
            y_node - 0.35 * vertical_spacing,
            marker="v",
            s=100,
            c="tab:green",
            zorder=5,
        )
        ax.text(
            x_node,
            y_node - 0.65 * vertical_spacing,
            (
                f"S{s_idx}: {plates[plate_idx].name} n{global_node}\n"
                f"u={value:+.3f} in"
                + (f"\nR={reaction:+.1f} lb" if reaction is not None else "")
            ),
            ha="center",
            va="top",
            fontsize=8,
            color="tab:green",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="tab:green",
                linewidth=0.5,
                boxstyle="round,pad=0.2",
            ),
        )

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
            ax.annotate(
                "",
                xy=(target_x, y_node + 0.1 * vertical_spacing),
                xytext=(x_node, y_node + 0.1 * vertical_spacing),
                arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.6),
            )
            ax.text(
                x_node,
                y_node + 0.35 * vertical_spacing,
                f"u = {disp * 1000:.3f} mil",
                ha="center",
                va="bottom",
                fontsize=8,
                color="tab:orange",
                fontweight="bold",
            )

    if mode == "loads":
        for bar in solution.bars:
            coord = segment_midpoints.get((bar.plate_id, bar.segment))
            if coord is None:
                continue
            x_mid, y_mid = coord
            ax.text(
                x_mid,
                y_mid + 0.26 * vertical_spacing,
                f"{bar.axial_force:+.1f} lb",
                ha="center",
                va="bottom",
                fontsize=8,
                color="tab:blue",
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="tab:blue",
                    linewidth=0.5,
                    boxstyle="round,pad=0.2",
                ),
            )

        fastener_plot_data: Dict[int, List[Tuple[int, FastenerResult, List[Tuple[float, float]]]]] = defaultdict(list)
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
            x_val = coords[0][0]
            y_vals = [pt[1] for pt in coords]
            ax.plot(
                [x_val, x_val],
                y_vals,
                ls="--",
                color="tab:purple",
                lw=1.6,
                marker="o",
                markersize=4,
                zorder=4,
            )
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
                    offset = (position - 0.5 * (count - 1)) * (0.6 * vertical_spacing)
                x_pos = coords[0][0] + 0.2 * max_pitch
                ax.text(
                    x_pos,
                    y_center + offset,
                    f"F{fast_idx + 1} = {fastener.force:+.1f} lb",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="tab:purple",
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.92,
                        edgecolor="tab:purple",
                        linewidth=0.5,
                        boxstyle="round,pad=0.2",
                    ),
                )

    ax.set_xlabel("x [in]")
    ax.set_yticks([y_levels[idx] for idx in range(len(plates))])
    ax.set_yticklabels([plate.name for plate in plates])
    ax.set_ylim(
        min(y_levels.values(), default=0.0) - 1.1 * vertical_spacing,
        max(y_levels.values(), default=0.0) + 0.9 * vertical_spacing,
    )
    ax.grid(False)

    if Line2D is not None and legend_handles:
        extra_handles = [
            Line2D([0], [0], ls="--", color="tab:purple", lw=1.6, label="Fastener"),
            Line2D([0], [0], marker="v", color="tab:green", linestyle="", markersize=10, label="Support"),
            Line2D([0], [0], marker="s", color="tab:red", linestyle="", markersize=8, label="Load"),
        ]
        legend_handles = legend_handles + extra_handles
        legend_labels = legend_labels + [h.get_label() for h in extra_handles]

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right")

    titles = {
        "scheme": "Scheme overview (nodes n#, supports S#, fasteners F#)",
        "displacements": "Nodal displacements",
        "loads": "Load distribution (bars & fasteners)",
    }
    ax.set_title(titles.get(mode, titles["scheme"]))
    fig.tight_layout()
    return fig
