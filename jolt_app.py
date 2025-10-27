"""Streamlit application for the JOLT 1D joint model."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Union

try:  # Optional imports used for the UI only
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
except Exception:  # pragma: no cover - plotting is optional for tests
    plt = None  # type: ignore
    Line2D = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import streamlit as st  # type: ignore
except Exception as exc:  # pragma: no cover - tests do not require Streamlit
    raise RuntimeError(
        "The Streamlit interface requires the 'streamlit' package. Install it to launch the app."
    ) from exc

from jolt import (
    FastenerResult,
    FastenerRow,
    Joint1D,
    JointSolution,
    JointConfiguration,
    Plate,
    figure76_example,
)


st.set_page_config(page_title="JOLT 1D Joint", layout="wide")


def _plates_present_at_row(plates: Sequence[Plate], row_index: int) -> List[int]:
    present = [
        idx
        for idx, plate in enumerate(plates)
        if plate.first_row <= row_index <= plate.last_row
    ]
    present.sort()
    return present


def _available_fastener_pairs(fastener: FastenerRow, plates: Sequence[Plate]) -> List[Tuple[int, int]]:
    row_index = int(fastener.row)
    present = _plates_present_at_row(plates, row_index)
    return list(zip(present[:-1], present[1:]))


def _resolve_fastener_connections(
    fastener: FastenerRow, plates: Sequence[Plate]
) -> List[Tuple[int, int]]:
    row_index = int(fastener.row)
    present = _plates_present_at_row(plates, row_index)
    if len(present) < 2:
        return []
    if fastener.connections is None:
        return list(zip(present[:-1], present[1:]))

    order = {plate_idx: position for position, plate_idx in enumerate(present)}
    resolved: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

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
def _serialize_configuration(
    pitches: Sequence[float],
    plates: Sequence[Plate],
    fasteners: Sequence[FastenerRow],
    supports: Sequence[Tuple[int, int, float]],
    label: str,
    unloading: str,
    point_forces: Optional[Sequence[Tuple[int, int, float]]] = None,
) -> Dict[str, Any]:
    configuration = JointConfiguration(
        pitches=list(pitches),
        plates=list(plates),
        fasteners=list(fasteners),
        supports=[(int(item[0]), int(item[1]), float(item[2])) for item in supports],
        point_forces=[
            (int(item[0]), int(item[1]), float(item[2]))
            for item in (point_forces or [])
        ],
        label=label,
        unloading=unloading,
    )
    return configuration.to_dict()


def _apply_configuration(config: Union[Dict[str, Any], JointConfiguration]) -> JointConfiguration:
    configuration = (
        config if isinstance(config, JointConfiguration) else JointConfiguration.from_dict(config)
    )
    st.session_state.pitches = configuration.pitches
    st.session_state.plates = configuration.plates
    st.session_state.fasteners = configuration.fasteners
    st.session_state.supports = configuration.supports
    st.session_state.point_forces = configuration.point_forces
    st.session_state.config_label = configuration.label or "Case"
    st.session_state.config_unloading = configuration.unloading
    return configuration
    st.session_state.config_label = str(config.get("label", "Case"))
    st.session_state.config_unloading = str(config.get("unloading", ""))


def _node_table(solution: JointSolution) -> List[dict]:
    return solution.nodes_as_dicts()


def _bar_table(solution: JointSolution) -> List[dict]:
    return solution.bars_as_dicts()


def _bearing_table(solution: JointSolution) -> List[dict]:
    return solution.bearing_bypass_as_dicts()


def _reaction_table(solution: JointSolution) -> List[dict]:
    return solution.reactions_as_dicts()


def _render_joint_diagram(
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
                    offset = (position - 0.5 * (count - 1)) * (0.35 * vertical_spacing)
                x_pos = coords[0][0] + 0.12 * max_pitch
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


# ---------- 2.1 Example configuration ----------

def load_example_figure76():
    return figure76_example()


# ---------- 2.2 Session state ----------

if "pitches" not in st.session_state:
    (
        st.session_state.pitches,
        st.session_state.plates,
        st.session_state.fasteners,
        st.session_state.supports,
    ) = load_example_figure76()

if "saved_models" not in st.session_state:
    st.session_state.saved_models: List[Dict[str, Any]] = []
if "config_label" not in st.session_state:
    st.session_state.config_label = "Case 1"
if "config_unloading" not in st.session_state:
    st.session_state.config_unloading = ""
if "point_forces" not in st.session_state:
    st.session_state.point_forces: List[Tuple[int, int, float]] = []


# ---------- 2.3 Input panels ----------

with st.sidebar:
    st.header("Geometry")
    n_rows = st.number_input("Number of rows", 1, 50, len(st.session_state.pitches))
    if n_rows != len(st.session_state.pitches):
        if n_rows > len(st.session_state.pitches):
            last = st.session_state.pitches[-1]
            st.session_state.pitches.extend([last] * (n_rows - len(st.session_state.pitches)))
        else:
            st.session_state.pitches = st.session_state.pitches[:n_rows]

        # Keep plate definitions consistent with the new number of rows so that
        # widgets never receive values outside their allowed ranges. This also
        # avoids Streamlit complaining about default values being greater than
        # the selected maximum when the user reduces the number of rows.
        for plate in st.session_state.plates:
            max_first_allowed = max(1, n_rows - 1) if n_rows > 1 else 1
            plate.first_row = max(1, min(int(plate.first_row), max_first_allowed))
            if n_rows > 1:
                min_last_allowed = plate.first_row + 1
                plate.last_row = max(min_last_allowed, min(int(plate.last_row), n_rows))
            else:
                plate.last_row = plate.first_row
            segments = plate.segment_count()
            if segments <= 0:
                plate.A_strip = []
            elif len(plate.A_strip) != segments:
                default_area = plate.A_strip[0] if plate.A_strip else 0.05
                plate.A_strip = [default_area] * segments

        for fastener in st.session_state.fasteners:
            if n_rows <= 0:
                fastener.row = 1
            else:
                fastener.row = max(1, min(int(fastener.row), n_rows))
    cols = st.columns(2)
    with cols[0]:
        same_pitch = st.checkbox("All pitches equal", value=True)
    if same_pitch:
        value = st.number_input("Pitch value [in]", 0.01, 100.0, st.session_state.pitches[0], step=0.001, format="%.3f")
        st.session_state.pitches = [float(value)] * n_rows
    else:
        st.write("Pitches [in]")
        st.session_state.pitches = [
            st.number_input(f"p[{i+1}]", 0.001, 100.0, st.session_state.pitches[i], key=f"pitch_{i}", step=0.001, format="%.3f")
            for i in range(n_rows)
        ]

    st.divider()
    st.subheader("Plates (Layers)")
    for idx, plate in enumerate(st.session_state.plates):
        with st.expander(f"Layer {idx}: {plate.name}", expanded=False):
            c1, c2, c3 = st.columns(3)
            plate.name = c1.text_input("Name", plate.name, key=f"pl_name_{idx}")
            plate.E = c2.number_input("E [psi]", 1e5, 5e8, plate.E, key=f"pl_E_{idx}", step=1e5, format="%.0f")
            plate.t = c3.number_input("t [in]", 0.001, 2.0, plate.t, key=f"pl_t_{idx}", step=0.001, format="%.3f")
            d1, d2, _ = st.columns(3)
            # The values were clamped right after the number of rows changed so
            # that the default values here are always valid. We still apply the
            # min/max constraints to guarantee the relationship first_row ≤ last_row.
            max_first_allowed = max(1, n_rows - 1) if n_rows > 1 else 1
            first_row_value = max(1, min(int(plate.first_row), max_first_allowed))
            plate.first_row = int(
                d1.number_input("First row", 1, max_first_allowed, first_row_value, key=f"pl_fr_{idx}")
            )
            if n_rows > 1:
                min_last_allowed = plate.first_row + 1
                max_last_allowed = max(min_last_allowed, n_rows)
                last_row_value = max(min_last_allowed, min(int(plate.last_row), max_last_allowed))
            else:
                min_last_allowed = plate.first_row
                max_last_allowed = plate.first_row
                last_row_value = plate.first_row
            plate.last_row = int(
                d2.number_input(
                    "Last row",
                    min_last_allowed,
                    max_last_allowed,
                    last_row_value,
                    key=f"pl_lr_{idx}",
                )
            )
            segments = plate.segment_count()
            st.write(f"Segments = {segments}")
            if segments <= 0:
                st.info("Select at least two rows to define plate segments for this layer.")
                plate.A_strip = []
            else:
                if len(plate.A_strip) != segments:
                    default_area = plate.A_strip[0] if plate.A_strip else 0.05
                    plate.A_strip = [default_area] * segments
                same_area = st.checkbox("Same bypass area for all segments", value=True, key=f"sameA_{idx}")
                if same_area:
                    default_area = plate.A_strip[0] if plate.A_strip else 0.05
                    area_val = st.number_input(
                        "Bypass area per segment [in²]",
                        1e-5,
                        10.0,
                        default_area,
                        key=f"pl_A_all_{idx}",
                        step=0.001,
                        format="%.3f",
                    )
                    plate.A_strip = [float(area_val)] * segments
                else:
                    for seg in range(segments):
                        plate.A_strip[seg] = st.number_input(
                            f"A[{seg+1}] [in²]",
                            1e-5,
                            10.0,
                            plate.A_strip[seg],
                            key=f"pl_A_{idx}_{seg}",
                            step=0.001,
                            format="%.3f",
                        )
            e1, e2 = st.columns(2)
            plate.Fx_left = e1.number_input(
                "End load LEFT [+→] [lb]", -1e6, 1e6, plate.Fx_left, key=f"pl_Fl_{idx}", step=1.0, format="%.1f"
            )
            plate.Fx_right = e2.number_input(
                "End load RIGHT [+→] [lb]", -1e6, 1e6, plate.Fx_right, key=f"pl_Fr_{idx}", step=1.0, format="%.1f"
            )
    cadd, cex = st.columns([1, 1])
    if cadd.button("➕ Add layer"):
        default_last = n_rows if n_rows > 1 else 1
        default_segments = max(default_last - 1, 0)
        st.session_state.plates.append(
            Plate(
                name=f"Layer{len(st.session_state.plates)}",
                E=1.0e7,
                t=0.05,
                first_row=1,
                last_row=default_last,
                A_strip=[0.05] * default_segments,
            )
        )
    if cex.button("🗑 Remove last layer") and len(st.session_state.plates) > 1:
        st.session_state.plates.pop()

    st.divider()
    st.subheader("Fasteners")
    fcols = st.columns([1, 3])
    with fcols[0]:
        if st.button("➕ Add fastener"):
            if st.session_state.fasteners:
                template = st.session_state.fasteners[-1]
            else:
                template = FastenerRow(row=1, D=0.188, Eb=1.0e7, nu_b=0.3)
            default_row = min(len(st.session_state.pitches), template.row if template.row > 0 else 1)
            template_connections = (
                None
                if template.connections is None
                else [tuple(pair) for pair in template.connections]
            )
            st.session_state.fasteners.append(
                replace(
                    template,
                    row=max(1, default_row) if n_rows > 0 else 1,
                    connections=template_connections,
                )
            )
    with fcols[1]:
        st.write("Configure any number of fasteners and assign them to available nodes.")

    remove_fasteners: List[int] = []
    methods = ["Boeing69", "Huth_metal", "Huth_graphite", "Grumman", "Manual"]
    for idx, fastener in enumerate(st.session_state.fasteners):
        with st.expander(f"Fastener {idx + 1} — row {fastener.row}", expanded=(len(st.session_state.fasteners) <= 6)):
            c0, c1, c2, c3, c4 = st.columns([1, 1, 1, 1, 0.5])
            clamped_row = min(max(int(fastener.row), 1), max(n_rows, 1))
            fastener.row = int(
                c0.number_input(
                    "Node index", 1, max(n_rows, 1), clamped_row, key=f"fr_row_{idx}", step=1
                )
            )
            fastener.D = c1.number_input("Diameter d [in]", 0.01, 2.0, fastener.D, key=f"fr_d_{idx}", step=0.001, format="%.3f")
            fastener.Eb = c2.number_input("Bolt E [psi]", 1e5, 5e8, fastener.Eb, key=f"fr_Eb_{idx}", step=1e5, format="%.0f")
            fastener.nu_b = c3.number_input("Bolt ν", 0.0, 0.49, fastener.nu_b, key=f"fr_nu_{idx}", step=0.01, format="%.2f")
            fastener.method = c4.selectbox("Method", methods, index=methods.index(fastener.method), key=f"fr_m_{idx}")
            if fastener.method == "Manual":
                fastener.k_manual = st.number_input(
                    "Manual k [lb/in]", 1.0, 1e12, fastener.k_manual or 1.0e6, key=f"fr_km_{idx}", step=1e5, format="%.0f"
                )
            available_pairs = _available_fastener_pairs(fastener, st.session_state.plates)
            if available_pairs:
                st.markdown("**Interfaces connected by this fastener**")
                if fastener.connections is None:
                    selected_pairs: Set[Tuple[int, int]] = set(available_pairs)
                else:
                    selected_pairs = {tuple(pair) for pair in fastener.connections}
                updated_pairs: List[Tuple[int, int]] = []
                for top_idx, bottom_idx in available_pairs:
                    label = f"{st.session_state.plates[top_idx].name} ↔ {st.session_state.plates[bottom_idx].name}"
                    checked = (top_idx, bottom_idx) in selected_pairs
                    checked = st.checkbox(
                        label,
                        value=checked,
                        key=f"fr_conn_{idx}_{top_idx}_{bottom_idx}",
                    )
                    if checked:
                        updated_pairs.append((top_idx, bottom_idx))
                fastener.connections = updated_pairs
                if not updated_pairs:
                    st.warning("This fastener will not connect any interfaces at the selected row.")
            else:
                fastener.connections = []
                st.info("Only one layer is present at this row; no spring will be added.")
            if st.button("✖ Remove", key=f"fr_rm_{idx}"):
                remove_fasteners.append(idx)
    if remove_fasteners:
        for idx in sorted(remove_fasteners, reverse=True):
            if 0 <= idx < len(st.session_state.fasteners):
                st.session_state.fasteners.pop(idx)

    st.divider()
    st.subheader("Supports (Dirichlet u=0)")
    if st.button("➕ Add support"):
        st.session_state.supports.append((0, 0, 0.0))
    remove_ids: List[int] = []
    for idx, (plate_idx, local_node, value) in enumerate(st.session_state.supports):
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            max_plate_idx = max(0, len(st.session_state.plates) - 1)
            clamped_plate_idx = min(max(int(plate_idx), 0), max_plate_idx)
            plate_idx = c1.number_input(
                f"Support {idx} — Plate index (0..)", 0, max_plate_idx, clamped_plate_idx, key=f"sp_pi_{idx}"
            )
            selected_plate = st.session_state.plates[int(plate_idx)]
            segments = selected_plate.segment_count()
            max_local = segments
            clamped_local = min(max(int(local_node), 0), max_local)
            local_node = c2.number_input("Local node (0..nSeg)", 0, max_local, clamped_local, key=f"sp_ln_{idx}")
            value = c3.number_input("u [in]", -1.0, 1.0, float(value), key=f"sp_val_{idx}", step=0.001, format="%.3f")
            st.session_state.supports[idx] = (int(plate_idx), int(local_node), float(value))
            if c4.button("✖", key=f"sp_rm_{idx}"):
                remove_ids.append(idx)
    if remove_ids:
        for idx in sorted(remove_ids, reverse=True):
            st.session_state.supports.pop(idx)

    st.divider()
    st.subheader("Saved configurations")
    uploaded_file = st.file_uploader("Load configuration JSON", type="json", key="cfg_upload")
    if uploaded_file is not None:
        try:
            configuration = JointConfiguration.from_json(uploaded_file)
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Failed to load configuration: {exc}")
        else:
            display_name = getattr(uploaded_file, "name", "uploaded")
            _apply_configuration(configuration)
            st.session_state["_last_loaded_config"] = configuration.label or display_name
            st.success(
                f"Loaded configuration '{configuration.label or display_name}'."
            )
            st.experimental_rerun()
    saved_configs = st.session_state.saved_models
    if saved_configs:
        indices = list(range(len(saved_configs)))
        selected_idx = st.selectbox(
            "Available cases",
            indices,
            format_func=lambda idx: saved_configs[idx]["label"],
            key="saved_config_select",
        )
        chosen = saved_configs[selected_idx]
        load_col, delete_col, export_col = st.columns([1, 1, 1])
        if load_col.button("Load", key="load_saved_config"):
            configuration = _apply_configuration(chosen)
            st.session_state["_last_loaded_config"] = configuration.label or chosen["label"]
            st.experimental_rerun()
        if delete_col.button("Delete", key="delete_saved_config"):
            st.session_state.saved_models.pop(selected_idx)
            st.experimental_rerun()
        export_data = json.dumps(chosen, indent=2).encode("utf-8")
        export_name = chosen["label"].strip().replace(" ", "_") or "configuration"
        export_col.download_button(
            "⬇️ Export",
            data=export_data,
            file_name=f"{export_name}.json",
            mime="application/json",
            key="export_saved_config",
        )
        if "_last_loaded_config" in st.session_state:
            st.caption(f"Loaded: {st.session_state['_last_loaded_config']}")
    else:
        st.info("No saved configurations yet. Save one after solving a case.")

    st.divider()
    if st.button("Load ▶ JOLT Figure 76"):
        (
            st.session_state.pitches,
            st.session_state.plates,
            st.session_state.fasteners,
            st.session_state.supports,
        ) = load_example_figure76()
        st.session_state.point_forces = []
        st.session_state.config_label = "JOLT Figure 76"
        st.session_state.config_unloading = ""
        st.session_state["_last_loaded_config"] = "JOLT Figure 76"
        st.experimental_rerun()


# ---------- 2.4 Solution ----------

st.title("JOLT 1D Joint — Bars + Springs")

pitches = st.session_state.pitches
plates = st.session_state.plates
fasteners = st.session_state.fasteners
supports = st.session_state.supports
point_forces = st.session_state.point_forces

if st.button("Solve", type="primary"):
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports, point_forces=point_forces or None)

    if plt is not None:
        tabs = st.tabs(["Scheme", "Displacements", "Loads"])
        modes = ["scheme", "displacements", "loads"]
        for tab, mode in zip(tabs, modes):
            with tab:
                fig = _render_joint_diagram(
                    pitches=pitches,
                    plates=plates,
                    fasteners=fasteners,
                    supports=supports,
                    solution=solution,
                    mode=mode,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:  # pragma: no cover - executed only when matplotlib is unavailable
        st.info("Matplotlib is not installed. Install it to see the schematic plot.")

    st.subheader("Fasteners")
    fastener_dicts = solution.fasteners_as_dicts()
    if pd is not None:
        df_fast = pd.DataFrame(fastener_dicts)
        st.dataframe(
            df_fast.style.format({"CF [in/lb]": "{:.3e}", "k [lb/in]": "{:.3e}", "F [lb]": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:  # pragma: no cover
        st.table(fastener_dicts)

    reaction_dicts = _reaction_table(solution)
    df_react = None
    if reaction_dicts:
        st.subheader("Support reactions")
        if pd is not None:
            df_react = pd.DataFrame(reaction_dicts)
            st.dataframe(
                df_react.style.format({"Reaction [lb]": "{:+.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:  # pragma: no cover
            st.table(reaction_dicts)

    st.subheader("Nodes")
    node_dicts = _node_table(solution)
    if pd is not None:
        df_nodes = pd.DataFrame(node_dicts)
        st.dataframe(
            df_nodes.style.format(
                {
                    "X [in]": "{:.3f}",
                    "u [in]": "{:.6e}",
                    "Net Bypass [lb]": "{:.2f}",
                    "t [in]": "{:.3f}",
                    "Bypass Area [in^2]": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:  # pragma: no cover
        st.table(node_dicts)

    st.subheader("Bars (plate segments)")
    bar_dicts = _bar_table(solution)
    if pd is not None:
        df_bars = pd.DataFrame(bar_dicts)
        st.dataframe(
            df_bars.style.format({"Force [lb]": "{:.2f}", "k_bar [lb/in]": "{:.3e}", "E [psi]": "{:.3e}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:  # pragma: no cover
        st.table(bar_dicts)

    st.subheader("Bearing / Bypass by row & plate")
    bearing_dicts = _bearing_table(solution)
    if pd is not None:
        df_bb = pd.DataFrame(bearing_dicts)
        st.dataframe(
            df_bb.style.format({"Bearing [lb]": "{:.2f}", "Bypass [lb]": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:  # pragma: no cover
        st.table(bearing_dicts)

    if pd is not None:
        st.download_button("Export fasteners CSV", data=df_fast.to_csv(index=False).encode("utf-8"), file_name="fasteners.csv", mime="text/csv")
        st.download_button("Export nodes CSV", data=df_nodes.to_csv(index=False).encode("utf-8"), file_name="nodes.csv", mime="text/csv")
        st.download_button("Export bars CSV", data=df_bars.to_csv(index=False).encode("utf-8"), file_name="bars.csv", mime="text/csv")
        st.download_button(
            "Export bearing_bypass CSV",
            data=df_bb.to_csv(index=False).encode("utf-8"),
            file_name="bearing_bypass.csv",
            mime="text/csv",
        )
        if df_react is not None:
            st.download_button(
                "Export reactions CSV",
                data=df_react.to_csv(index=False).encode("utf-8"),
                file_name="reactions.csv",
                mime="text/csv",
            )

    st.divider()
    with st.expander("Save / export configuration", expanded=False):
        default_label = st.session_state.config_label or f"Case {len(st.session_state.saved_models) + 1}"
        label = st.text_input("Configuration label", default_label, key="save_cfg_label")
        unloading_note = st.text_input(
            "Unloading / load case description",
            st.session_state.config_unloading,
            key="save_cfg_unloading",
        )
        config_dict = _serialize_configuration(
            pitches=pitches,
            plates=plates,
            fasteners=fasteners,
            supports=supports,
            label=label,
            unloading=unloading_note,
            point_forces=st.session_state.get("point_forces", []),
        )
        save_col, download_col = st.columns([1, 1])
        if save_col.button("💾 Save to session", key="save_cfg_button"):
            st.session_state.config_label = label
            st.session_state.config_unloading = unloading_note
            existing = next((idx for idx, cfg in enumerate(st.session_state.saved_models) if cfg["label"] == label), None)
            if existing is not None:
                st.session_state.saved_models[existing] = config_dict
                st.success(f"Updated saved configuration '{label}'.")
            else:
                st.session_state.saved_models.append(config_dict)
                st.success(f"Saved configuration '{label}'.")
        download_payload = json.dumps(config_dict, indent=2).encode("utf-8")
        file_stub = label.strip().replace(" ", "_") or "configuration"
        download_col.download_button(
            "⬇️ Download JSON",
            data=download_payload,
            file_name=f"{file_stub}.json",
            mime="application/json",
            key="download_cfg_json",
        )
else:
    st.info("Соберите схему в левой панели и нажмите **Solve**. Для воспроизведения скрина используйте **Load ▶ JOLT Figure 76**.")
