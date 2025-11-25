"""UI components for the JOLT application."""
from dataclasses import replace
from typing import List, Set, Tuple, Dict, Any
import streamlit as st
import pandas as pd
import json
import hashlib

from jolt import Plate, FastenerRow, JointSolution, JointConfiguration
from .utils import available_fastener_pairs
from .state import apply_configuration, clear_configuration_widget_state, serialize_configuration

_UPLOAD_DIGEST_KEY = "_cfg_upload_digest"

def render_sidebar() -> Tuple[List[float], List[Plate], List[FastenerRow], List[Tuple[int, int, float]], Dict[int, float]]:
    with st.sidebar:
        st.header("Configuration")
        
        # Mode Selection
        # Mode Selection
        # Check if we need to force a specific mode (e.g. after loading)
        forced_mode = st.session_state.pop("_force_input_mode", None)
        if forced_mode:
            st.session_state["input_mode_selector"] = forced_mode
            
        input_mode = st.selectbox(
            "Input Mode",
            ["Standard", "Refined Row", "Node-based"],
            index=0,
            key="input_mode_selector"
        )
        st.session_state.input_mode = input_mode

        # --- Standard / Refined Mode ---
        if input_mode in ["Standard", "Refined Row"]:
            _render_geometry_section()
            pitches = st.session_state.pitches
            
            extra_nodes = []
            if input_mode == "Refined Row":
                with st.expander("Extra Nodes (Refined)", expanded=True):
                    st.markdown("Add intermediate nodes at specific X coordinates.")
                    extra_nodes_str = st.text_area("X Coordinates (comma-separated)", value="", help="e.g., 1.5, 3.2")
                    if extra_nodes_str:
                        try:
                            extra_nodes = [float(x.strip()) for x in extra_nodes_str.split(",") if x.strip()]
                        except ValueError:
                            st.error("Invalid coordinates")
                st.session_state.extra_nodes = extra_nodes

            _render_plates_section()
            plates = st.session_state.plates
            
            _render_fasteners_section()
            fasteners = st.session_state.fasteners
            
            if input_mode == "Refined Row" and extra_nodes:
                from jolt.inputs import process_refined_rows
                pitches, plates, fasteners = process_refined_rows(pitches, plates, fasteners, extra_nodes)

        # --- Node-based Mode ---
        else:
            pitches, plates, fasteners = _render_node_based_inputs()

        # Supports
        # We pass the *current* pitches and plates to supports section so it can validate indices
        supports = _render_supports_section(pitches, plates)
        
        # Point forces
        point_forces = {}
        # TODO: Add point force input UI
        
        _render_saved_configs_section()

    return pitches, plates, fasteners, supports, point_forces


def _render_node_based_inputs() -> Tuple[List[float], List[Plate], List[FastenerRow]]:
    st.subheader("Nodes")
    # Initialize or migrate to DataFrame
    if "node_table" not in st.session_state:
        st.session_state.node_table = pd.DataFrame([{"id": 0, "x": 0.0}, {"id": 1, "x": 10.0}])
    elif isinstance(st.session_state.node_table, list):
        st.session_state.node_table = pd.DataFrame(st.session_state.node_table)
    
    # Ensure columns exist if empty
    if st.session_state.node_table.empty and "id" not in st.session_state.node_table.columns:
        st.session_state.node_table = pd.DataFrame(columns=["id", "x"])
    
    node_df = st.data_editor(
        st.session_state.node_table,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.NumberColumn("Node ID", step=1, required=True),
            "x": st.column_config.NumberColumn("X [in]", required=True, format="%.3f"),
        },
        key="node_editor",
        hide_index=True,
    )
    st.session_state.node_table = node_df
    
    # Parse using to_dict("records") to handle DataFrame
    nodes_dict = {row["id"]: row["x"] for row in node_df.to_dict("records")}

    st.subheader("Elements")
    if "element_table" not in st.session_state:
        st.session_state.element_table = pd.DataFrame([
            {"layer": "Skin", "start": 0, "end": 1, "E": 1e7, "t": 0.1, "w": 1.0}
        ])
    elif isinstance(st.session_state.element_table, list):
        st.session_state.element_table = pd.DataFrame(st.session_state.element_table)
        
    if st.session_state.element_table.empty and "layer" not in st.session_state.element_table.columns:
        st.session_state.element_table = pd.DataFrame(columns=["layer", "start", "end", "E", "t", "w"])
        
    elem_df = st.data_editor(
        st.session_state.element_table,
        num_rows="dynamic",
        column_config={
            "layer": st.column_config.TextColumn("Layer Name", required=True),
            "start": st.column_config.NumberColumn("Start Node", step=1, required=True),
            "end": st.column_config.NumberColumn("End Node", step=1, required=True),
            "E": st.column_config.NumberColumn("Modulus [psi]", default=1e7),
            "t": st.column_config.NumberColumn("Thickness [in]", default=0.1),
            "w": st.column_config.NumberColumn("Width [in]", default=1.0),
        },
        key="element_editor",
        hide_index=True,
    )
    st.session_state.element_table = elem_df
    
    elements = []
    missing_nodes = set()
    for row in elem_df.to_dict("records"):
        start_n = row["start"]
        end_n = row["end"]
        if start_n not in nodes_dict:
            missing_nodes.add(start_n)
        if end_n not in nodes_dict:
            missing_nodes.add(end_n)
            
        elements.append({
            "layer_name": row["layer"],
            "start_node": start_n,
            "end_node": end_n,
            "E": row["E"],
            "t": row["t"],
            "width": row["w"]
        })

    if missing_nodes:
        st.error(f"The following nodes are used in Elements but not defined in Nodes: {sorted(list(missing_nodes))}")

    st.subheader("Fasteners")
    if "fastener_table_nb" not in st.session_state:
        st.session_state.fastener_table_nb = pd.DataFrame(columns=["node", "d", "layers", "E", "v", "method"])
    elif isinstance(st.session_state.fastener_table_nb, list):
        st.session_state.fastener_table_nb = pd.DataFrame(st.session_state.fastener_table_nb)
        
    # Ensure columns exist
    required_cols = ["node", "d", "layers", "E", "v", "method"]
    for col in required_cols:
        if col not in st.session_state.fastener_table_nb.columns:
            if col == "E":
                st.session_state.fastener_table_nb[col] = 1e7
            elif col == "v":
                st.session_state.fastener_table_nb[col] = 0.3
            elif col == "method":
                st.session_state.fastener_table_nb[col] = "boeing69"
            else:
                # Should not happen for node/d/layers if table was created before, but safe default
                st.session_state.fastener_table_nb[col] = None
        
    fast_df = st.data_editor(
        st.session_state.fastener_table_nb,
        num_rows="dynamic",
        column_config={
            "node": st.column_config.NumberColumn("Node ID", step=1, required=True),
            "d": st.column_config.NumberColumn("Diameter [in]", default=0.19),
            "layers": st.column_config.TextColumn("Layers (comma-sep)", help="e.g. Skin, Doubler"),
            "E": st.column_config.NumberColumn("Modulus [psi]", default=1e7),
            "v": st.column_config.NumberColumn("Poisson Ratio", default=0.3),
            "method": st.column_config.SelectboxColumn(
                "Method",
                options=["boeing69", "huth", "grumman"],
                default="boeing69",
                required=True
            )
        },
        key="fastener_editor_nb",
        hide_index=True,
    )
    st.session_state.fastener_table_nb = fast_df
    
    fasteners_in = []
    fastener_warnings = []
    
    # Collect available layers per node for validation
    node_layers: Dict[int, Set[str]] = {}
    for el in elements:
        start, end = el["start_node"], el["end_node"]
        layer = el["layer_name"]
        if start not in node_layers: node_layers[start] = set()
        if end not in node_layers: node_layers[end] = set()
        node_layers[start].add(layer)
        node_layers[end].add(layer)

    for row_idx, row in enumerate(fast_df.to_dict("records")):
        node_id = row["node"]
        layers_str = str(row.get("layers", ""))
        # Support both comma and semicolon
        layers_str = layers_str.replace(";", ",")
        layers = [x.strip() for x in layers_str.split(",") if x.strip()]
        
        if node_id not in nodes_dict:
            fastener_warnings.append(f"Row {row_idx+1}: Node {node_id} not defined.")
            continue
            
        if len(layers) < 2:
            fastener_warnings.append(f"Row {row_idx+1}: Fastener at Node {node_id} must connect at least 2 layers (found {len(layers)}: {layers}).")
            continue
            
        # Check if layers exist at this node
        available = node_layers.get(node_id, set())
        missing_layers = [l for l in layers if l not in available]
        if missing_layers:
            fastener_warnings.append(f"Row {row_idx+1}: Layers {missing_layers} not found at Node {node_id}. Available: {sorted(list(available))}")
            
        fasteners_in.append({
            "node_id": node_id,
            "diameter": row["d"],
            "connected_layers": layers,
            "E": row.get("E", 1e7),
            "v": row.get("v", 0.3),
            "method": row.get("method", "boeing69")
        })

    if fastener_warnings:
        for warn in fastener_warnings:
            st.warning(warn)

    from jolt.inputs import process_node_based
    return process_node_based(nodes_dict, elements, fasteners_in)


def _render_geometry_section():
    default_rows = len(st.session_state.pitches)
    n_rows = int(
        st.number_input(
            "Number of rows",
            1,
            50,
            st.session_state.get("n_rows", default_rows),
            key=f"n_rows_v{st.session_state.get('_widget_version', 0)}",
        )
    )
    st.session_state["n_rows"] = n_rows
    if n_rows != len(st.session_state.pitches):
        if n_rows > len(st.session_state.pitches):
            last = st.session_state.pitches[-1]
            st.session_state.pitches.extend([last] * (n_rows - len(st.session_state.pitches)))
        else:
            st.session_state.pitches = st.session_state.pitches[:n_rows]

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
            st.number_input(f"p[{i+1}]", 0.001, 100.0, st.session_state.pitches[i], key=f"pitch_{i}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            for i in range(n_rows)
        ]

def _render_plates_section():
    n_rows = len(st.session_state.pitches)
    for idx, plate in enumerate(st.session_state.plates):
        # Ensure backward compatibility for session state objects
        if not hasattr(plate, "widths"):
            plate.widths = None
        if not hasattr(plate, "thicknesses"):
            plate.thicknesses = None
            
        with st.expander(f"Layer {idx}: {plate.name}", expanded=False):
            c1, c2, c3 = st.columns(3)
            plate.name = c1.text_input("Name", plate.name, key=f"pl_name_{idx}_v{st.session_state.get('_widget_version', 0)}")
            plate.E = c2.number_input("E [psi]", 1e5, 5e8, plate.E, key=f"pl_E_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1e5, format="%.0f")
            plate.t = c3.number_input("t [in]", 0.001, 2.0, plate.t, key=f"pl_t_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            d1, d2, _ = st.columns(3)
            max_first_allowed = max(1, n_rows - 1) if n_rows > 1 else 1
            first_row_value = max(1, min(int(plate.first_row), max_first_allowed))
            plate.first_row = int(
                d1.number_input("First row", 1, max_first_allowed, first_row_value, key=f"pl_fr_{idx}_v{st.session_state.get('_widget_version', 0)}")
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
                    key=f"pl_lr_{idx}_v{st.session_state.get('_widget_version', 0)}",
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
                # Definition Mode Selection
                def_mode = st.radio(
                    "Definition Mode", 
                    ["Area", "Width/Thickness"], 
                    index=1 if (plate.widths and plate.thicknesses) else 0,
                    key=f"def_mode_{idx}_v{st.session_state.get('_widget_version', 0)}", 
                    horizontal=True
                )
                
                if def_mode == "Width/Thickness":
                    # Initialize if needed
                    if not plate.widths or len(plate.widths) != segments:
                        plate.widths = [1.0] * segments
                    if not plate.thicknesses or len(plate.thicknesses) != segments:
                        plate.thicknesses = [plate.t] * segments
                        
                    same_dims = st.checkbox("Constant Width & Thickness", value=True, key=f"sameWT_{idx}_v{st.session_state.get('_widget_version', 0)}")
                    
                    if same_dims:
                        c_w, c_t = st.columns(2)
                        w_val = c_w.number_input("Width [in]", 0.001, 100.0, plate.widths[0], key=f"pl_w_all_{idx}", step=0.1, format="%.3f")
                        t_val = c_t.number_input("Thickness [in]", 0.001, 2.0, plate.thicknesses[0], key=f"pl_t_all_{idx}", step=0.001, format="%.3f")
                        plate.widths = [w_val] * segments
                        plate.thicknesses = [t_val] * segments
                        # Update Area immediately for visual feedback or consistency
                        plate.A_strip = [w * t for w, t in zip(plate.widths, plate.thicknesses)]
                    else:
                        st.write("Segment Dimensions:")
                        for seg in range(segments):
                            c_w, c_t = st.columns(2)
                            w_val = c_w.number_input(f"w[{seg+1}]", 0.001, 100.0, plate.widths[seg], key=f"pl_w_{idx}_{seg}", step=0.1, format="%.3f")
                            t_val = c_t.number_input(f"t[{seg+1}]", 0.001, 2.0, plate.thicknesses[seg], key=f"pl_t_{idx}_{seg}", step=0.001, format="%.3f")
                            plate.widths[seg] = w_val
                            plate.thicknesses[seg] = t_val
                            plate.A_strip[seg] = w_val * t_val
                else:
                    # Clear W/T to indicate Area mode preference? Or just ignore them.
                    # Better to keep them but maybe reset them if Area is changed?
                    # For now, just show Area inputs.
                    same_area = st.checkbox("Same bypass area for all segments", value=True, key=f"sameA_{idx}_v{st.session_state.get('_widget_version', 0)}")
                    if same_area:
                        default_area = plate.A_strip[0] if plate.A_strip else 0.05
                        area_val = st.number_input(
                            "Bypass area per segment [in²]",
                            1e-5,
                            10.0,
                            default_area,
                            key=f"pl_A_all_{idx}_v{st.session_state.get('_widget_version', 0)}",
                            step=0.001,
                            format="%.3f",
                        )
                        plate.A_strip = [float(area_val)] * segments
                    else:
                        for seg in range(segments):
                            c_gap, c_val = st.columns([0.3, 0.7])
                            is_gap = plate.A_strip[seg] <= 1e-9
                            with c_gap:
                                # Use a unique key for the checkbox
                                make_gap = st.checkbox("Gap", value=is_gap, key=f"gap_{idx}_{seg}_v{st.session_state.get('_widget_version', 0)}")
                            with c_val:
                                if make_gap:
                                    plate.A_strip[seg] = 0.0
                                    st.text_input(f"A[{seg+1}]", value="0.0", disabled=True, key=f"pl_A_dis_{idx}_{seg}_v{st.session_state.get('_widget_version', 0)}")
                                else:
                                    val_to_show = plate.A_strip[seg]
                                    if val_to_show <= 1e-9:
                                        val_to_show = 0.05
                                    plate.A_strip[seg] = st.number_input(
                                        f"A[{seg+1}] [in²]",
                                        1e-5,
                                        10.0,
                                        val_to_show,
                                        key=f"pl_A_{idx}_{seg}_v{st.session_state.get('_widget_version', 0)}",
                                        step=0.001,
                                        format="%.3f",
                                    )
            e1, e2 = st.columns(2)
            plate.Fx_left = e1.number_input(
                "End load LEFT [+→] [lb]", -1e6, 1e6, plate.Fx_left, key=f"pl_Fl_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1.0, format="%.1f"
            )
            plate.Fx_right = e2.number_input(
                "End load RIGHT [+→] [lb]", -1e6, 1e6, plate.Fx_right, key=f"pl_Fr_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1.0, format="%.1f"
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

def _render_fasteners_section():
    n_rows = len(st.session_state.pitches)
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
                    "Node index", 1, max(n_rows, 1), clamped_row, key=f"fr_row_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1
                )
            )
            fastener.D = c1.number_input("Diameter d [in]", 0.01, 2.0, fastener.D, key=f"fr_d_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            fastener.Eb = c2.number_input("Bolt E [psi]", 1e5, 5e8, fastener.Eb, key=f"fr_Eb_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1e5, format="%.0f")
            fastener.nu_b = c3.number_input("Bolt ν", 0.0, 0.49, fastener.nu_b, key=f"fr_nu_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.01, format="%.2f")
            try:
                method_index = methods.index(fastener.method)
            except ValueError:
                method_index = 0
                fastener.method = methods[method_index]
            fastener.method = c4.selectbox("Method", methods, index=method_index, key=f"fr_m_{idx}_v{st.session_state.get('_widget_version', 0)}")
            if fastener.method == "Manual":
                fastener.k_manual = st.number_input(
                    "Manual k [lb/in]", 1.0, 1e12, fastener.k_manual or 1.0e6, key=f"fr_km_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1e5, format="%.0f"
                )
            available_pairs = available_fastener_pairs(fastener, st.session_state.plates)
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
                        key=f"fr_conn_{idx}_{top_idx}_{bottom_idx}_v{st.session_state.get('_widget_version', 0)}",
                    )
                    if checked:
                        updated_pairs.append((top_idx, bottom_idx))
                fastener.connections = updated_pairs
                if not updated_pairs:
                    st.warning("This fastener will not connect any interfaces at the selected row.")
            else:
                fastener.connections = []
                st.info("Only one layer is present at this row; no spring will be added.")
            if st.button("✖ Remove", key=f"fr_rm_{idx}_v{st.session_state.get('_widget_version', 0)}"):
                remove_fasteners.append(idx)
    if remove_fasteners:
        for idx in sorted(remove_fasteners, reverse=True):
            if 0 <= idx < len(st.session_state.fasteners):
                st.session_state.fasteners.pop(idx)

def _render_supports_section(pitches: List[float], plates: List[Plate]) -> List[Tuple[int, int, float]]:
    if st.button("➕ Add support"):
        st.session_state.supports.append((0, 0, 0.0))
    remove_ids: List[int] = []
    for idx, (plate_idx, local_node, value) in enumerate(st.session_state.supports):
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            max_plate_idx = max(0, len(plates) - 1)
            clamped_plate_idx = min(max(int(plate_idx), 0), max_plate_idx)
            plate_idx = c1.number_input(
                f"Support {idx} — Plate index (0..)", 0, max_plate_idx, clamped_plate_idx, key=f"sp_pi_{idx}_v{st.session_state.get('_widget_version', 0)}"
            )
            # Use the passed 'plates' list, not session state (unless they are the same)
            if int(plate_idx) < len(plates):
                selected_plate = plates[int(plate_idx)]
                segments = selected_plate.segment_count()
            else:
                segments = 0
                
            max_local = segments
            clamped_local = min(max(int(local_node), 0), max_local)
            local_node = c2.number_input("Local node (0..nSeg)", 0, max_local, clamped_local, key=f"sp_ln_{idx}_v{st.session_state.get('_widget_version', 0)}")
            value = c3.number_input("u [in]", -1.0, 1.0, float(value), key=f"sp_val_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            st.session_state.supports[idx] = (int(plate_idx), int(local_node), float(value))
            if c4.button("✖", key=f"sp_rm_{idx}_v{st.session_state.get('_widget_version', 0)}"):
                remove_ids.append(idx)
    if remove_ids:
        for idx in sorted(remove_ids, reverse=True):
            st.session_state.supports.pop(idx)
    return st.session_state.supports

def _render_saved_configs_section():
    load_feedback = st.session_state.pop("_load_feedback", None)
    if load_feedback:
        st.success(load_feedback)
    if st.session_state.pop("_reset_cfg_upload", False):
        st.session_state.pop("cfg_upload", None)
    uploaded_file = st.file_uploader("Load configuration JSON", type="json", key="cfg_upload")
    if uploaded_file is None:
        st.session_state.pop(_UPLOAD_DIGEST_KEY, None)
    else:
        file_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get(_UPLOAD_DIGEST_KEY) != digest:
            try:
                payload = json.loads(file_bytes.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Top-level JSON entry must be an object")
                configuration = JointConfiguration.from_dict(payload)
            except Exception as exc:
                st.error(f"Failed to load configuration: {exc}")
            else:
                display_name = getattr(uploaded_file, "name", "uploaded")
                apply_configuration(configuration)
                st.session_state["_last_loaded_config"] = configuration.label or display_name
                st.session_state["_load_feedback"] = (
                    f"Loaded configuration '{configuration.label or display_name}'."
                )
                st.session_state["_reset_cfg_upload"] = True
                st.session_state[_UPLOAD_DIGEST_KEY] = digest
                st.rerun()
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
            configuration = apply_configuration(chosen)
            st.session_state["_last_loaded_config"] = configuration.label or chosen["label"]
            st.session_state["_load_feedback"] = (
                f"Loaded configuration '{configuration.label or chosen['label']}'."
            )
            st.rerun()
        if delete_col.button("Delete", key="delete_saved_config"):
            st.session_state.saved_models.pop(selected_idx)
            st.rerun()
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

def _load_example_figure76():
    from jolt import figure76_example
    clear_configuration_widget_state()
    (
        st.session_state.pitches,
        st.session_state.plates,
        st.session_state.fasteners,
        st.session_state.supports,
    ) = figure76_example()
    st.session_state.point_forces = []
    st.session_state.config_label = "JOLT Figure 76"
    st.session_state.config_unloading = ""
    st.session_state["_last_loaded_config"] = "JOLT Figure 76"
    st.session_state["n_rows"] = len(st.session_state.pitches)
    st.rerun()

def render_solution_tables(solution: JointSolution):
    # 1. Nodes Table
    st.subheader("Nodes")
    node_dicts = solution.nodes_as_dicts()
    if pd is not None:
        df_nodes = pd.DataFrame(node_dicts)
        # Reorder columns to match Boeing: X Location, Displacement, Net Bypass Load, Thickness, Bypass Area, Order, Multiple Thickness
        cols = ["Node ID", "X Location", "Displacement", "Net Bypass Load", "Thickness", "Bypass Area", "Order", "Multiple Thickness"]
        # Filter existing columns
        cols = [c for c in cols if c in df_nodes.columns]
        
        st.dataframe(
            df_nodes[cols].style.format(
                {
                    "X Location": "{:.3f}",
                    "Displacement": "{:.6e}",
                    "Net Bypass Load": "{:.1f}",
                    "Thickness": "{:.3f}",
                    "Bypass Area": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.table(node_dicts)

    # 2. Plates Table (Bars)
    st.subheader("Plates")
    bar_dicts = solution.bars_as_dicts()
    if pd is not None:
        df_bars = pd.DataFrame(bar_dicts)
        cols = ["ID", "Force", "Stiffness", "Modulus"]
        cols = [c for c in cols if c in df_bars.columns]
        st.dataframe(
            df_bars[cols].style.format({"Force": "{:.1f}", "Stiffness": "{:.2e}", "Modulus": "{:.3e}"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.table(bar_dicts)

    # 3. Fasteners Table
    st.subheader("Fasteners")
    fastener_dicts = solution.fasteners_as_dicts()
    if pd is not None:
        df_fast = pd.DataFrame(fastener_dicts)
        cols = ["Row", "Load", "Brg Force Upper", "Brg Force Lower", "Stiffness", "Modulus", "Diameter", "Quantity", "Thickness Node 1", "Thickness Node 2"]
        cols = [c for c in cols if c in df_fast.columns]
        st.dataframe(
            df_fast[cols].style.format({
                "Load": "{:.1f}", 
                "Brg Force Upper": "{:.1f}",
                "Brg Force Lower": "{:.1f}",
                "Stiffness": "{:.2e}", 
                "Modulus": "{:.3e}",
                "Diameter": "{:.3f}",
                "Quantity": "{:.1f}",
                "Thickness Node 1": "{:.3f}",
                "Thickness Node 2": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )
    else:
        st.table(fastener_dicts)

    # 4. Reactions Table
    if solution.reactions:
        st.subheader("Reactions")
        reaction_dicts = solution.reactions_as_dicts()
        if pd is not None:
            df_react = pd.DataFrame(reaction_dicts)
            cols = ["Node ID", "Force"]
            cols = [c for c in cols if c in df_react.columns]
            st.dataframe(
                df_react[cols].style.format({"Force": "{:.1f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(reaction_dicts)
            
    # 5. Classic Results
    st.subheader("Classic Results")
    classic_dicts = solution.classic_results_as_dicts()
    if classic_dicts:
        if pd is not None:
            df_classic = pd.DataFrame(classic_dicts)
            cols = ["Row", "Thickness", "Area", "No of Fasteners", "Fastener Diameter", "Incoming Load", "Bypass Load", "Load Transfer", "L.Trans / P", "Detail Stress", "Bearing Stress", "Fbr / FDetail"]
            cols = [c for c in cols if c in df_classic.columns]
            st.dataframe(
                df_classic[cols].style.format({
                    "Thickness": "{:.3f}",
                    "Area": "{:.3f}",
                    "No of Fasteners": "{:.1f}",
                    "Fastener Diameter": "{:.3f}",
                    "Incoming Load": "{:.1f}",
                    "Bypass Load": "{:.1f}",
                    "Load Transfer": "{:.1f}",
                    "L.Trans / P": "{:.3f}",
                    "Detail Stress": "{:.0f}",
                    "Bearing Stress": "{:.0f}",
                    "Fbr / FDetail": "{:.3f}",
                }),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(classic_dicts)

    # 6. Loads
    if solution.applied_forces:
        st.subheader("Loads")
        if pd is not None:
            df_loads = pd.DataFrame(solution.applied_forces)
            cols = ["Value", "Reference Node"]
            st.dataframe(
                df_loads[cols].style.format({"Value": "{:.1f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(solution.applied_forces)

    # 7. Min/Max Results
    st.subheader("Min/Max Results")
    max_fast_force = max([abs(f.force) for f in solution.fasteners]) if solution.fasteners else 0.0
    max_plate_load = max([b.axial_force for b in solution.bars]) if solution.bars else 0.0
    min_plate_load = min([b.axial_force for b in solution.bars]) if solution.bars else 0.0
    
    min_max_data = [
        {"Result": "Abs Max Fastener Force", "Load": max_fast_force, "Quantity": "1.0", "ID": ""}, # ID could be found
        {"Result": "Max Plate Load", "Load": max_plate_load, "Quantity": "", "ID": ""},
        {"Result": "Min Plate Load", "Load": min_plate_load, "Quantity": "", "ID": ""},
    ]
    
    if pd is not None:
        st.dataframe(
            pd.DataFrame(min_max_data).style.format({"Load": "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.table(min_max_data)

    # Exports
    if pd is not None:
        st.download_button("Export fasteners CSV", data=df_fast.to_csv(index=False).encode("utf-8"), file_name="fasteners.csv", mime="text/csv")
        st.download_button("Export nodes CSV", data=df_nodes.to_csv(index=False).encode("utf-8"), file_name="nodes.csv", mime="text/csv")
        st.download_button("Export bars CSV", data=df_bars.to_csv(index=False).encode("utf-8"), file_name="bars.csv", mime="text/csv")
        if classic_dicts:
             st.download_button("Export classic results CSV", data=df_classic.to_csv(index=False).encode("utf-8"), file_name="classic_results.csv", mime="text/csv")


def render_save_section(pitches, plates, fasteners, supports, point_forces):
    st.divider()
    with st.expander("Save / export configuration", expanded=False):
        default_label = st.session_state.config_label or f"Case {len(st.session_state.saved_models) + 1}"
        label = st.text_input("Configuration label", default_label, key="save_cfg_label")
        unloading_note = st.text_input(
            "Unloading / load case description",
            st.session_state.config_unloading,
            key="save_cfg_unloading",
        )
        config_dict = serialize_configuration(
            pitches=pitches,
            plates=plates,
            fasteners=fasteners,
            supports=supports,
            label=label,
            unloading=unloading_note,
            point_forces=point_forces,
        )
        feedback = st.session_state.pop("_save_feedback", None)
        if feedback:
            st.success(feedback)
        save_col, download_col = st.columns([1, 1])
        if save_col.button("💾 Save to session", key="save_cfg_button"):
            st.session_state.config_label = label
            st.session_state.config_unloading = unloading_note
            existing = next((idx for idx, cfg in enumerate(st.session_state.saved_models) if cfg["label"] == label), None)
            if existing is not None:
                st.session_state.saved_models[existing] = config_dict
                st.session_state["_save_feedback"] = f"Updated saved configuration '{label}'."
            else:
                st.session_state.saved_models.append(config_dict)
                st.session_state["_save_feedback"] = f"Saved configuration '{label}'."
            st.rerun()
        download_payload = json.dumps(config_dict, indent=2).encode("utf-8")
        file_stub = label.strip().replace(" ", "_") or "configuration"
        download_col.download_button(
            "⬇️ Download JSON",
            data=download_payload,
            file_name=f"{file_stub}.json",
            mime="application/json",
            key="download_cfg_json",
        )
