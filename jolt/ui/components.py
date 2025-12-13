"""UI components for the JOLT application."""
from dataclasses import replace
from typing import List, Set, Tuple, Dict, Any
import streamlit as st
import pandas as pd
import json
import hashlib
import math

from jolt import Plate, FastenerRow, JointSolution, JointConfiguration
from jolt.units import UnitSystem, UnitConverter
from .utils import available_fastener_pairs
from .state import apply_configuration, clear_configuration_widget_state, serialize_configuration, convert_session_state, safe_load_model

_UPLOAD_DIGEST_KEY = "_cfg_upload_digest"
_TOPOLOGY_OPTIONS = [
    ("Auto (based on method)", ""),
    ("Boeing chain (JOLT)", "boeing_chain"),
    ("Boeing star (scaled double-shear)", "boeing_star_scaled"),
    ("Boeing star (legacy single-shear)", "boeing_star_raw"),
    ("Empirical chain (Huth / other)", "empirical_chain"),
    ("Empirical star (default for non-Boeing)", "empirical_star"),
    ("Boeing Beam (Jarfall Ladder)", "boeing_beam"),
]

def render_sidebar() -> Tuple[List[float], List[Plate], List[FastenerRow], List[Tuple[int, int, float]], Dict[int, float], Dict[str, str]]:
    with st.sidebar:
        st.header("Configuration")
        
        # Unit System Selection
        current_units = st.session_state.get("unit_system", UnitSystem.IMPERIAL)
        
        # Use a callback to handle conversion immediately upon change
        def on_unit_change():
            new_unit = st.session_state._unit_selector
            convert_session_state(new_unit)

        unit_choice = st.radio(
            "Unit System",
            [UnitSystem.IMPERIAL, UnitSystem.SI],
            index=0 if current_units == UnitSystem.IMPERIAL else 1,
            key="_unit_selector",
            horizontal=True,
            on_change=on_unit_change,
            format_func=lambda x: "Imperial" if x == UnitSystem.IMPERIAL else "SI"
        )
        
        # Get labels for current system
        units = UnitConverter.get_labels(current_units)

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
            _render_geometry_section(units)
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

            _render_plates_section(units)
            plates = st.session_state.plates
            
            _render_fasteners_section(units)
            fasteners = st.session_state.fasteners
            
            if input_mode == "Refined Row" and extra_nodes:
                from jolt.inputs import process_refined_rows
                pitches, plates, fasteners = process_refined_rows(pitches, plates, fasteners, extra_nodes)

        # --- Node-based Mode ---
        else:
            pitches, plates, fasteners = _render_node_based_inputs(units)

        # Supports
        # We pass the *current* pitches and plates to supports section so it can validate indices
        supports = _render_supports_section(pitches, plates, units)
        
        # Point forces
        point_forces = {}
        # TODO: Add point force input UI
        
        _render_saved_configs_section()

    return pitches, plates, fasteners, supports, point_forces, units


def _render_node_based_inputs(units: Dict[str, str]) -> Tuple[List[float], List[Plate], List[FastenerRow]]:
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
            "x": st.column_config.NumberColumn(f"X [{units['length']}]", required=True, format="%.3f"),
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
            "E": st.column_config.NumberColumn(f"Modulus [{units['stress']}]", default=1e7),
            "t": st.column_config.NumberColumn(f"Thickness [{units['length']}]", default=0.1),
            "w": st.column_config.NumberColumn(f"Width [{units['length']}]", default=1.0),
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
        st.session_state.fastener_table_nb = pd.DataFrame(columns=["node", "d", "layers", "E", "v", "method", "topology"])
    elif isinstance(st.session_state.fastener_table_nb, list):
        st.session_state.fastener_table_nb = pd.DataFrame(st.session_state.fastener_table_nb)

    # Ensure columns exist
    required_cols = ["node", "d", "layers", "E", "v", "method", "topology"]
    for col in required_cols:
        if col not in st.session_state.fastener_table_nb.columns:
            if col == "E":
                st.session_state.fastener_table_nb[col] = 1e7
            elif col == "v":
                st.session_state.fastener_table_nb[col] = 0.3
            elif col == "method":
                st.session_state.fastener_table_nb[col] = "boeing"
            elif col == "topology":
                st.session_state.fastener_table_nb[col] = ""
            else:
                # Should not happen for node/d/layers if table was created before, but safe default
                st.session_state.fastener_table_nb[col] = None

    fast_df = st.data_editor(
        st.session_state.fastener_table_nb,
        num_rows="dynamic",
        column_config={
            "node": st.column_config.NumberColumn("Node ID", step=1, required=True),
            "d": st.column_config.NumberColumn(f"Diameter [{units['length']}]", default=0.19),
            "layers": st.column_config.TextColumn("Layers (comma-sep)", help="e.g. Skin, Doubler"),
            "E": st.column_config.NumberColumn(f"Modulus [{units['stress']}]", default=1e7),
            "v": st.column_config.NumberColumn("Poisson Ratio", default=0.3),
            "method": st.column_config.SelectboxColumn(
                "Method",
                options=["boeing", "huth", "grumman"],
                default="boeing",
                required=True
            ),
            "topology": st.column_config.SelectboxColumn(
                "Topology",
                options=[opt[1] for opt in _TOPOLOGY_OPTIONS],
                default="",
                help="Leave blank to let the method choose; otherwise pick a chain/star variant."
            ),
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
            "method": row.get("method", "boeing69"),
            "topology": row.get("topology", ""),
        })

    if fastener_warnings:
        for warn in fastener_warnings:
            st.warning(warn)

    from jolt.inputs import process_node_based
    return process_node_based(nodes_dict, elements, fasteners_in)


def _render_geometry_section(units: Dict[str, str]):
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
        value = st.number_input(f"Pitch value [{units['length']}]", 0.001, 1000.0, st.session_state.pitches[0], step=0.001, format="%.3f")
        st.session_state.pitches = [float(value)] * n_rows
    else:
        st.write(f"Pitches [{units['length']}]")
        st.session_state.pitches = [
            st.number_input(f"p[{i+1}]", 0.001, 100.0, st.session_state.pitches[i], key=f"pitch_{i}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            for i in range(n_rows)
        ]

def _render_plates_section(units: Dict[str, str]):
    n_rows = len(st.session_state.pitches)
    for idx, plate in enumerate(st.session_state.plates):
        # Ensure backward compatibility for session state objects
        if not hasattr(plate, "widths"):
            plate.widths = None
        if not hasattr(plate, "thicknesses"):
            plate.thicknesses = None
        if not hasattr(plate, "material_name"):
            plate.material_name = None
        if not hasattr(plate, "fatigue_strength"):
            plate.fatigue_strength = None
            
        with st.expander(f"Layer {idx}: {plate.name}", expanded=False):
            c1, c2, c3 = st.columns(3)
            plate.name = c1.text_input("Name", plate.name, key=f"pl_name_{idx}_v{st.session_state.get('_widget_version', 0)}")
            # Dynamic ranges based on unit system
            is_si = units["stress"] == "MPa"
            e_min = 1.0 if is_si else 1e4
            e_max = 1e7 if is_si else 1e9
            e_step = 1000.0 if is_si else 1e5
            
            plate.E = c2.number_input(f"E [{units['stress']}]", e_min, e_max, plate.E, key=f"pl_E_{idx}_v{st.session_state.get('_widget_version', 0)}", step=e_step, format="%.0f")
            plate.t = c3.number_input(f"t [{units['length']}]", 0.001, 1000.0, plate.t, key=f"pl_t_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            
            # Row 2: Material Name and Fatigue Strength
            m1, m2 = st.columns([2, 1])
            plate.material_name = m1.text_input("Material Name", plate.material_name or "", key=f"pl_mat_{idx}_v{st.session_state.get('_widget_version', 0)}")
            
            fs_val = plate.fatigue_strength if plate.fatigue_strength is not None else 0.0
            new_fs = m2.number_input(
                f"Fatigue Strength (f_max) [{units['stress']}]", 
                0.0, 
                e_max, 
                fs_val, 
                key=f"pl_fs_{idx}_v{st.session_state.get('_widget_version', 0)}",
                step=e_step/10.0,
                format="%.1f",
                help="Material strength from S-N curve for target life (e.g., 10^5 cycles)."
            )
            plate.fatigue_strength = new_fs if new_fs > 0 else None

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
                        
                    # Auto-detect if dims are constant
                    is_const_w = all(abs(w - plate.widths[0]) < 1e-9 for w in plate.widths) if plate.widths else True
                    is_const_t = all(abs(t - plate.thicknesses[0]) < 1e-9 for t in plate.thicknesses) if plate.thicknesses else True
                    default_same = is_const_w and is_const_t
                        
                    same_dims = st.checkbox("Constant Width & Thickness", value=default_same, key=f"sameWT_{idx}_v{st.session_state.get('_widget_version', 0)}")
                    
                    if same_dims:
                        c_w, c_t = st.columns(2)
                        w_val = c_w.number_input(f"Width [{units['length']}]", 0.001, 5000.0, plate.widths[0], key=f"pl_w_all_{idx}", step=0.1, format="%.3f")
                        t_val = c_t.number_input(f"Thickness [{units['length']}]", 0.001, 1000.0, plate.thicknesses[0], key=f"pl_t_all_{idx}", step=0.001, format="%.3f")
                        plate.widths = [w_val] * segments
                        plate.thicknesses = [t_val] * segments
                        # Update Area immediately for visual feedback or consistency
                        plate.A_strip = [w * t for w, t in zip(plate.widths, plate.thicknesses)]
                    else:
                        st.write("Segment Dimensions:")
                        for seg in range(segments):
                            c_w, c_t = st.columns(2)
                            w_val = c_w.number_input(f"w[{seg+1}]", 0.001, 5000.0, plate.widths[seg], key=f"pl_w_{idx}_{seg}", step=0.1, format="%.3f")
                            t_val = c_t.number_input(f"t[{seg+1}]", 0.001, 1000.0, plate.thicknesses[seg], key=f"pl_t_{idx}_{seg}", step=0.001, format="%.3f")
                            plate.widths[seg] = w_val
                            plate.thicknesses[seg] = t_val
                            plate.A_strip[seg] = w_val * t_val
                else:
                    # Auto-detect if areas are constant
                    is_const_a = all(abs(a - plate.A_strip[0]) < 1e-9 for a in plate.A_strip) if plate.A_strip else True
                    
                    same_area = st.checkbox("Same bypass area for all segments", value=is_const_a, key=f"sameA_{idx}_v{st.session_state.get('_widget_version', 0)}")
                    if same_area:
                        default_area = plate.A_strip[0] if plate.A_strip else 0.05
                        area_val = st.number_input(
                            f"Bypass area per segment [{units['area']}]",
                            1e-5,
                            10000.0,
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
                                        f"A[{seg+1}] [{units['area']}]",
                                        1e-5,
                                        10000.0,
                                        val_to_show,
                                        key=f"pl_A_{idx}_{seg}_v{st.session_state.get('_widget_version', 0)}",
                                        step=0.001,
                                        format="%.3f",
                                    )
            e1, e2 = st.columns(2)
            plate.Fx_left = e1.number_input(
                f"End load LEFT [+→] [{units['force']}]", -1e9, 1e9, plate.Fx_left, key=f"pl_Fl_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1.0, format="%.1f"
            )
            plate.Fx_right = e2.number_input(
                f"End load RIGHT [+→] [{units['force']}]", -1e9, 1e9, plate.Fx_right, key=f"pl_Fr_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1.0, format="%.1f"
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

def _render_fasteners_section(units: Dict[str, str]):
    n_rows = len(st.session_state.pitches)
    layer_names = [p.name for p in st.session_state.plates]
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
    
    # --- Bulk Add Feature ---
    with st.expander("Bulk Add Fasteners", expanded=False):
        st.markdown("Add or update fasteners for a range of rows.")
        bc1, bc2, bc3 = st.columns(3)
        b_start = bc1.number_input("Start Row", 1, max(n_rows, 1), 1, key=f"bulk_start_v{st.session_state.get('_widget_version', 0)}")
        b_end = bc2.number_input("End Row", 1, max(n_rows, 1), max(n_rows, 1), key=f"bulk_end_v{st.session_state.get('_widget_version', 0)}")
        b_method = bc3.selectbox("Method", [
            "Boeing", 
            "Huth (Bolted Metal)", 
            "Huth (Riveted Metal)", 
            "Huth (Bolted Graphite)", 
            "Grumman", 
            "Swift (Douglas)", 
            "Tate-Rosenfeld", 
            "Morris", 
            "Manual"
        ], key=f"bulk_method_v{st.session_state.get('_widget_version', 0)}")
        
        bc4, bc5, bc6 = st.columns(3)
        # Dynamic ranges
        is_si = units["stress"] == "MPa"
        e_min = 1.0 if is_si else 1e4
        e_max = 1e7 if is_si else 1e9
        e_step = 1000.0 if is_si else 1e5

        b_d = bc4.number_input(f"Diameter [{units['length']}]", 0.001, 1000.0, 0.188, step=0.001, format="%.3f", key=f"bulk_d_v{st.session_state.get('_widget_version', 0)}")
        b_E = bc5.number_input(f"Bolt E [{units['stress']}]", e_min, e_max, 1e7, step=e_step, format="%.0f", key=f"bulk_E_v{st.session_state.get('_widget_version', 0)}")
        b_nu = bc6.number_input("Bolt ν", 0.0, 0.49, 0.3, step=0.01, format="%.2f", key=f"bulk_nu_v{st.session_state.get('_widget_version', 0)}")
        
        # Bulk Countersink
        st.markdown("**Bulk Countersink Settings**")
        bcs1, bcs2, bcs3 = st.columns(3)
        b_is_cs = bcs1.checkbox("Countersunk", value=False, key=f"bulk_iscs_v{st.session_state.get('_widget_version', 0)}")
        
        b_cs_depth = 0.0
        b_cs_angle = 100.0
        b_cs_affects_bypass = False
        b_cs_layers = []
        
        if b_is_cs:
            b_cs_depth = bcs2.number_input(f"CS Depth [{units['length']}]", 0.0, 100.0, 0.0, step=0.001, format="%.3f", key=f"bulk_csd_v{st.session_state.get('_widget_version', 0)}")
            b_cs_angle = bcs3.number_input("CS Angle [deg]", 0.0, 180.0, 100.0, step=1.0, key=f"bulk_csa_v{st.session_state.get('_widget_version', 0)}")
            
            bcs4, bcs5 = st.columns([1, 2])
            b_cs_affects_bypass = bcs4.checkbox("Affects Bypass Area", value=False, help="If checked, reduces gross area for bypass stress calculation.", key=f"bulk_csab_v{st.session_state.get('_widget_version', 0)}")
            
            # Collect all layer names for dropdowns
            b_cs_layers = bcs5.multiselect("CS Layers", options=layer_names, key=f"bulk_csl_v{st.session_state.get('_widget_version', 0)}")

        # Bulk Fatigue Factors
        st.markdown("**Bulk Fatigue Factors**")
        bff1, bff2 = st.columns(2)
        
        # Define options mapping
        alpha_options = {
            "1.0 - Standard hole drilled": 1.0,
            "1.1 - Drilled hole with deburring": 1.1,
            "1.2 - Reamed hole": 1.2,
            "1.3 - Cold worked hole": 1.3,
            "0.9 - Rough drilled hole": 0.9,
        }
        beta_options = {
            "1.0 - Standard fit": 1.0,
            "0.8 - Interference fit": 0.8,
            "1.2 - Clearance fit": 1.2,
            "0.5 - Taper-Lok": 0.5,
        }
        
        b_alpha_key = bff1.selectbox("Hole Condition (α)", list(alpha_options.keys()), index=0, key=f"bulk_alpha_v{st.session_state.get('_widget_version', 0)}")
        b_beta_key = bff2.selectbox("Hole Filling (β)", list(beta_options.keys()), index=0, key=f"bulk_beta_v{st.session_state.get('_widget_version', 0)}")
        
        b_alpha = alpha_options[b_alpha_key]
        b_beta = beta_options[b_beta_key]

        if st.button("Apply to Range"):
            # Create a map of existing fasteners by row for quick lookup/update
            existing_map = {f.row: i for i, f in enumerate(st.session_state.fasteners)}
            
            for r in range(int(b_start), int(b_end) + 1):
                if r in existing_map:
                    # Update existing
                    idx = existing_map[r]
                    f = st.session_state.fasteners[idx]
                    
                    # Update basic props
                    f = replace(f, D=b_d, Eb=b_E, nu_b=b_nu, method=b_method)
                    
                    # Update Fatigue props
                    f = replace(f, hole_condition_factor=b_alpha, hole_filling_factor=b_beta)
                    
                    # Update CS props
                    if b_is_cs:
                        f = replace(f, is_countersunk=True, cs_depth=b_cs_depth, cs_angle=b_cs_angle, cs_affects_bypass=b_cs_affects_bypass, cs_layers=list(b_cs_layers))
                    else:
                        f = replace(f, is_countersunk=False) # Reset if unchecked? Or keep existing? Usually bulk apply overwrites.
                    
                    st.session_state.fasteners[idx] = f
                else:
                    # Append new
                    new_f = FastenerRow(
                        row=r, 
                        D=b_d, 
                        Eb=b_E, 
                        nu_b=b_nu, 
                        method=b_method,
                        hole_condition_factor=b_alpha,
                        hole_filling_factor=b_beta
                    )
                    if b_is_cs:
                        new_f = replace(new_f, is_countersunk=True, cs_depth=b_cs_depth, cs_angle=b_cs_angle, cs_affects_bypass=b_cs_affects_bypass, cs_layers=list(b_cs_layers))
                    st.session_state.fasteners.append(new_f)
                    
            # Sort fasteners by row
            st.session_state.fasteners.sort(key=lambda x: x.row)
            st.success(f"Updated fasteners for rows {b_start}-{b_end}")
            st.rerun()

    methods = [
        "Boeing", 
        "Huth (Bolted Metal)", 
        "Huth (Riveted Metal)", 
        "Huth (Bolted Graphite)", 
        "Grumman", 
        "Swift (Douglas)", 
        "Tate-Rosenfeld", 
        "Morris", 
        "Manual"
    ]
    
    # SSF Factor Options
    hole_conditions = {
        "Standard hole drilled": 1.0,
        "Broached or reamed": 0.9,
        "Cold worked holes": 0.75 # Average of 0.7-0.8
    }
    
    hole_fillings = {
        "Open holes": 1.0,
        "Lock bolt (steel)": 0.75,
        "Rivets": 0.75,
        "Threaded bolts": 0.825, # Average of 0.75-0.9
        "Taper-Lok": 0.5,
        "Hi-Lok": 0.75
    }

    # --- Fastener Validation Logic ---
    # 1. Check for Same Name but Different Markers
    # 2. Check for Different Names but Same Markers (Warning only, maybe common)
    
    name_to_markers: Dict[str, Set[str]] = {}
    marker_to_names: Dict[str, Set[str]] = {}
    
    for f in st.session_state.fasteners:
        nm = f.name if f.name else f"F{f.row}" # Use fallback name if empty
        mk = f.marker_symbol if hasattr(f, "marker_symbol") and f.marker_symbol else "circle"
        
        if nm not in name_to_markers: name_to_markers[nm] = set()
        name_to_markers[nm].add(mk)
        
        if mk not in marker_to_names: marker_to_names[mk] = set()
        marker_to_names[mk].add(nm)

    # Validation: Combined Check
    conflicts_name = [nm for nm, mks in name_to_markers.items() if len(mks) > 1]
    conflicts_marker = [mk for mk, nms in marker_to_names.items() if len(nms) > 1]
    
    if conflicts_name or conflicts_marker:
        msg = []
        if conflicts_name:
            msg.append(f"Inconsistent markers for names: {conflicts_name}.")
        if conflicts_marker:
            msg.append(f"Shared markers for different names: {conflicts_marker}.")
            
        st.warning(" ".join(msg))
        
        if st.button("Fix: Auto-assign Distinct Markers"):
            # 1. Unify markers for same names (Priority)
            for nm in conflicts_name:
                counts = {}
                for f in st.session_state.fasteners:
                    fn = f.name if f.name else f"F{f.row}"
                    if fn == nm:
                        mk = f.marker_symbol if hasattr(f, "marker_symbol") else "circle"
                        counts[mk] = counts.get(mk, 0) + 1
                best_marker = max(counts, key=counts.get)
                
                for i, f in enumerate(st.session_state.fasteners):
                    fn = f.name if f.name else f"F{f.row}"
                    if fn == nm:
                        f.marker_symbol = best_marker
                        st.session_state.fasteners[i] = f
            
            # 2. Assign distinct markers for different names
            # Re-evaluate mapping after unification
            final_name_map = {}
            for f in st.session_state.fasteners:
                fn = f.name if f.name else f"F{f.row}"
                final_name_map[fn] = f.marker_symbol
            
            available_symbols = [
                "circle", "x", "diamond", "star", "square", "triangle-up", 
                "triangle-down", "cross", "hexagon", "pentagon"
            ]
            
            all_names = sorted(list(final_name_map.keys()))
            new_map = {}
            for i, nm in enumerate(all_names):
                new_map[nm] = available_symbols[i % len(available_symbols)]
                
            for i, f in enumerate(st.session_state.fasteners):
                fn = f.name if f.name else f"F{f.row}"
                if fn in new_map:
                    f.marker_symbol = new_map[fn]
                    st.session_state.fasteners[i] = f
            
            # CRITICAL: Increment widget version to force UI refresh
            st.session_state["_widget_version"] = st.session_state.get("_widget_version", 0) + 1
            st.success("Markers auto-assigned!")
            st.rerun()

    for idx, fastener in enumerate(st.session_state.fasteners):
        # Patch for new fields if hot-reloading
        if not hasattr(fastener, "name"):
            fastener.name = ""
        if not hasattr(fastener, "marker_symbol"):
            fastener.marker_symbol = "circle"

        with st.expander(f"Fastener {idx + 1} — row {fastener.row}", expanded=(len(st.session_state.fasteners) <= 6)):
            
            # --- NEW VISUAL CONFIGURATION SECTION ---
            c_name, c_sym = st.columns([3, 1])
            
            # 1. Fastener Name
            fastener.name = c_name.text_input(
                "Fastener Name / Label", 
                value=fastener.name, 
                key=f"fr_name_{idx}_v{st.session_state.get('_widget_version', 0)}",
                placeholder=f"F{fastener.row}"
            )
            
            # 2. Marker Symbol Selection
            # Mapping readable names to internal Plotly symbols
            symbol_options = {
                "Bold Dot (●)": "circle",
                "Cross (X)": "x", 
                "Diamond (◆)": "diamond",
                "Star (★)": "star", 
                "Square (■)": "square",
                "Triangle (▲)": "triangle-up"
            }
            
            # Reverse lookup for default index
            current_sym = fastener.marker_symbol or "circle"
            # Handle legacy/default fallback
            default_idx = 0
            vals = list(symbol_options.values())
            if current_sym in vals:
                default_idx = vals.index(current_sym)
            
            selected_label = c_sym.selectbox(
                "Marker", 
                list(symbol_options.keys()), 
                index=default_idx,
                key=f"fr_sym_{idx}_v{st.session_state.get('_widget_version', 0)}"
            )
            fastener.marker_symbol = symbol_options[selected_label]
            
            st.markdown("---")
            c0, c1, c2, c3, c4 = st.columns([0.7, 0.8, 1.0, 0.7, 1.8])
            clamped_row = min(max(int(fastener.row), 1), max(n_rows, 1))
            fastener.row = int(
                c0.number_input(
                    "Node index", 1, max(n_rows, 1), clamped_row, key=f"fr_row_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1
                )
            )
            # Dynamic ranges
            is_si = units["stress"] == "MPa"
            e_min = 1.0 if is_si else 1e4
            e_max = 1e7 if is_si else 1e9
            e_step = 1000.0 if is_si else 1e5

            fastener.D = c1.number_input(f"Diameter d [{units['length']}]", 0.01, 1000.0, fastener.D, key=f"fr_d_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
            fastener.Eb = c2.number_input(f"Bolt E [{units['stress']}]", e_min, e_max, fastener.Eb, key=f"fr_Eb_{idx}_v{st.session_state.get('_widget_version', 0)}", step=e_step, format="%.0f")
            fastener.nu_b = c3.number_input("Bolt ν", 0.0, 0.49, fastener.nu_b, key=f"fr_nu_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.01, format="%.2f")
            try:
                method_index = methods.index(fastener.method)
            except ValueError:
                method_index = 0
                fastener.method = methods[method_index]
            fastener.method = c4.selectbox("Method", methods, index=method_index, key=f"fr_m_{idx}_v{st.session_state.get('_widget_version', 0)}")
            if fastener.method == "Manual":
                fastener.k_manual = st.number_input(
                    f"Manual k [{units['stiffness']}]", 1.0, 1e12, fastener.k_manual or 1.0e6, key=f"fr_km_{idx}_v{st.session_state.get('_widget_version', 0)}", step=1e5, format="%.0f"
                )

            topo_values = [opt[1] for opt in _TOPOLOGY_OPTIONS]
            topo_labels = {opt[1]: opt[0] for opt in _TOPOLOGY_OPTIONS}
            current_topology = fastener.topology or ""
            topo_index = topo_values.index(current_topology) if current_topology in topo_values else 0
            selected_topology = st.selectbox(
                "Topology (load-sharing layout)",
                topo_values,
                index=topo_index,
                format_func=lambda v: topo_labels.get(v, topo_labels[""]),
                help="Chain (Boeing) matches JOLT; star options keep an internal fastener DOF.",
                key=f"fr_topology_{idx}_v{st.session_state.get('_widget_version', 0)}",
            )
            fastener.topology = selected_topology or None
            
            # --- Fatigue Configuration ---
            st.markdown("---")
            st.markdown("**Fatigue Configuration**")
            
            # Hole Configuration
            h1, h2, h3, h4 = st.columns(4)
            fastener.hole_centered = h1.checkbox("Centered Hole", value=fastener.hole_centered, key=f"fr_hc_{idx}")
            if not fastener.hole_centered:
                fastener.hole_offset = h2.number_input(f"Offset [{units['length']}]", 0.0, 1000.0, fastener.hole_offset, key=f"fr_ho_{idx}", step=0.001, format="%.3f")
            else:
                fastener.hole_offset = 0.0
                h2.text_input("Offset", value="0.0 (Centered)", disabled=True, key=f"fr_ho_dis_{idx}")
            
            # Hole Condition (Alpha)
            # Find closest key or default
            alpha_key = "Standard hole drilled"
            min_diff = 100.0
            for k, v in hole_conditions.items():
                if abs(fastener.hole_condition_factor - v) < min_diff:
                    min_diff = abs(fastener.hole_condition_factor - v)
                    alpha_key = k
            
            sel_alpha = h3.selectbox("Hole Condition (α)", list(hole_conditions.keys()), index=list(hole_conditions.keys()).index(alpha_key), key=f"fr_alpha_sel_{idx}")
            fastener.hole_condition_factor = hole_conditions[sel_alpha]
            
            # Hole Filling (Beta)
            beta_key = "Open holes"
            min_diff = 100.0
            for k, v in hole_fillings.items():
                if abs(fastener.hole_filling_factor - v) < min_diff:
                    min_diff = abs(fastener.hole_filling_factor - v)
                    beta_key = k
            
            sel_beta = h4.selectbox("Hole Filling (β)", list(hole_fillings.keys()), index=list(hole_fillings.keys()).index(beta_key), key=f"fr_beta_sel_{idx}")
            fastener.hole_filling_factor = hole_fillings[sel_beta]
            
            # Countersink Configuration
            cs1, cs2 = st.columns([1, 3])
            fastener.is_countersunk = cs1.checkbox("Countersunk", value=fastener.is_countersunk, key=f"fr_iscs_{idx}")
            
            if fastener.is_countersunk:
                # CS Definition Mode
                cs_mode = cs2.radio("CS Definition", ["Depth & Angle", "Depth & Diameter"], horizontal=True, key=f"fr_csmode_{idx}")
                
                csa1, csa2, csa3 = st.columns(3)
                
                # Depth is common
                fastener.cs_depth = csa1.number_input(f"CS Depth [{units['length']}]", 0.0, 100.0, fastener.cs_depth, key=f"fr_csdepth_{idx}", step=0.001, format="%.3f")
                
                if "Angle" in cs_mode:
                    fastener.cs_angle = csa2.number_input("CS Angle [deg]", 0.0, 180.0, fastener.cs_angle, key=f"fr_csangle_{idx}", step=1.0)
                else:
                    # Diameter
                    cs_diam = csa2.number_input(f"CS Diameter [{units['length']}]", fastener.D, 500.0, fastener.D + 2 * fastener.cs_depth, key=f"fr_csdiam_{idx}")
                    if fastener.cs_depth > 1e-6 and cs_diam > fastener.D:
                        theta_rad = 2 * math.atan((cs_diam - fastener.D) / (2 * fastener.cs_depth))
                        fastener.cs_angle = math.degrees(theta_rad)
                
                # Affected Layers
                fastener.cs_layers = csa3.multiselect(
                    "Countersunk Layers", 
                    options=layer_names,
                    default=fastener.cs_layers,
                    key=f"fr_cslayers_{idx}"
                )
                
                # Affects Bypass?
                fastener.cs_affects_bypass = st.checkbox(
                    "Reduce Bypass Area (Experimental)", 
                    value=fastener.cs_affects_bypass,
                    key=f"fr_csbypass_{idx}",
                    help="If checked, the gross area for stress calculation will be reduced by the countersink void."
                )

            st.markdown("---")
            
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

def _render_supports_section(pitches: List[float], plates: List[Plate], units: Dict[str, str]) -> List[Tuple[int, int, float]]:
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
            value = c3.number_input(f"u [{units['length']}]", -1.0, 1.0, float(value), key=f"sp_val_{idx}_v{st.session_state.get('_widget_version', 0)}", step=0.001, format="%.3f")
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
                
                loaded_configs = []
                if isinstance(payload, list):
                    # Bulk load
                    for item in payload:
                        if isinstance(item, dict):
                            loaded_configs.append(JointConfiguration.from_dict(item))
                elif isinstance(payload, dict):
                    # Single load
                    loaded_configs.append(JointConfiguration.from_dict(payload))
                else:
                    raise ValueError("JSON must be an object or a list of objects")
                
                if not loaded_configs:
                    raise ValueError("No valid configurations found in JSON")

            except Exception as exc:
                st.error(f"Failed to load configuration: {exc}")
            else:
                display_name = getattr(uploaded_file, "name", "uploaded")
                
                # Add or Update loaded configs in session state
                existing_map = {m.model_id: i for i, m in enumerate(st.session_state.saved_models)}
                added_count = 0
                updated_count = 0
                
                for config in loaded_configs:
                    if config.model_id in existing_map:
                        # Update existing model
                        idx = existing_map[config.model_id]
                        st.session_state.saved_models[idx] = config
                        updated_count += 1
                    else:
                        # Add new model
                        st.session_state.saved_models.append(config)
                        existing_map[config.model_id] = len(st.session_state.saved_models) - 1
                        added_count += 1
                
                # Apply the first one if it's a single load, or just notify for bulk
                if len(loaded_configs) == 1:
                    config = loaded_configs[0]
                    apply_configuration(config)
                    st.session_state["_last_loaded_config"] = config.label or display_name
                    st.session_state["_load_feedback"] = (
                        f"Loaded configuration '{config.label or display_name}'."
                    )
                else:
                    msg_parts = []
                    if added_count > 0:
                        msg_parts.append(f"Imported {added_count} new")
                    if updated_count > 0:
                        msg_parts.append(f"Updated {updated_count} existing")
                    
                    summary = ", ".join(msg_parts) if msg_parts else "No changes"
                    st.session_state["_load_feedback"] = (
                        f"{summary} configurations from '{display_name}'."
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
            format_func=lambda idx: saved_configs[idx].label,
            key="saved_config_select",
        )
        chosen = saved_configs[selected_idx]
        load_col, delete_col, export_col = st.columns([1, 1, 1])
        if load_col.button("Load", key="load_saved_config"):
            configuration = apply_configuration(chosen)
            st.session_state["_last_loaded_config"] = configuration.label or chosen.label
            st.session_state["_load_feedback"] = (
                f"Loaded configuration '{configuration.label or chosen.label}'."
            )
            st.rerun()
        if delete_col.button("Delete", key="delete_saved_config"):
            st.session_state.saved_models.pop(selected_idx)
            st.rerun()
        export_data = json.dumps(chosen.to_dict(), indent=2).encode("utf-8")
        export_name = chosen.label.strip().replace(" ", "_") or "configuration"
        export_col.download_button(
            "⬇️ Export",
            data=export_data,
            file_name=f"{export_name}.json",
            mime="application/json",
            key="export_saved_config",
            # on_click=None # No callback needed
        )
        if "_last_loaded_config" in st.session_state:
            st.caption(f"Loaded: {st.session_state['_last_loaded_config']}")
            
        # --- EXPORT ALL ---
        st.markdown("---")
        if st.button("Export All Cases"):
            all_cases_data = [m.to_dict() for m in saved_configs]
            all_cases_json = json.dumps(all_cases_data, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download All Cases (.json)",
                data=all_cases_json,
                file_name=f"jolt_all_cases_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="export_all_cases"
            )
            
        # --- RE-SOLVE ALL ---
        if st.button("Re-Solve All Cases", help="Re-calculate results for all saved cases using their stored inputs."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                updated_count = 0
                for i, config in enumerate(saved_configs):
                    status_text.text(f"Solving {config.label}...")
                    
                    # Re-build and solve
                    model = config.build_model()
                    solution = model.solve(
                        supports=config.supports,
                        point_forces=config.point_forces
                    )
                    
                    # Update results in place (safe because config is already deepcopied in session state)
                    config.results = solution.to_dict()
                    updated_count += 1
                    progress_bar.progress((i + 1) / len(saved_configs))
                
                st.success(f"Successfully re-solved {updated_count} cases.")
            except Exception as e:
                st.error(f"Failed to re-solve cases: {e}")
            finally:
                status_text.empty()
                progress_bar.empty()
    else:
        st.info("No saved configurations yet. Save one after solving a case.")

def _load_example_case_5_3():
    from jolt import case_5_3_elements_example
    clear_configuration_widget_state()
    (
        st.session_state.pitches,
        st.session_state.plates,
        st.session_state.fasteners,
        st.session_state.supports,
    ) = case_5_3_elements_example()
    st.session_state.point_forces = []
    st.session_state.config_label = "Case 5.3 Elements"
    st.session_state.config_unloading = ""
    st.session_state["_last_loaded_config"] = "Case 5.3 Elements"
    st.session_state["n_rows"] = len(st.session_state.pitches)
    st.rerun()

def render_solution_tables(solution: JointSolution, units: Dict[str, str]):
    # Global Controls
    col_exp, col_col, _ = st.columns([1, 1, 4])
    if col_exp.button("Expand All"):
        st.session_state["_results_expanded"] = True
    if col_col.button("Collapse All"):
        st.session_state["_results_expanded"] = False
        
    expanded_state = st.session_state.get("_results_expanded", False)

    # 1. Nodes Table
    with st.expander("Nodes", expanded=expanded_state):
        node_dicts = solution.nodes_as_dicts()
        if pd is not None:
            df_nodes = pd.DataFrame(node_dicts)
            # Rename columns to include units
            rename_map = {
                "X Location": f"X Location [{units['length']}]",
                "Displacement": f"Displacement [{units['length']}]",
                "Net Bypass Load": f"Net Bypass Load [{units['force']}]",
                "Thickness": f"Thickness [{units['length']}]",
                "Bypass Area": f"Bypass Area [{units['area']}]"
            }
            df_nodes = df_nodes.rename(columns=rename_map)

            cols = ["Node ID", f"X Location [{units['length']}]", f"Displacement [{units['length']}]", f"Net Bypass Load [{units['force']}]", f"Thickness [{units['length']}]", f"Bypass Area [{units['area']}]", "Order", "Multiple Thickness"]
            cols = [c for c in cols if c in df_nodes.columns]
            
            st.dataframe(
                df_nodes[cols].style.format(
                    {
                        f"X Location [{units['length']}]": "{:.3f}",
                        f"Displacement [{units['length']}]": "{:.6e}",
                        f"Net Bypass Load [{units['force']}]": "{:.1f}",
                        f"Thickness [{units['length']}]": "{:.3f}",
                        f"Bypass Area [{units['area']}]": "{:.3f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(node_dicts)

    # 2. Plates Table (Bars)
    with st.expander("Plates", expanded=expanded_state):
        bar_dicts = solution.bars_as_dicts()
        if pd is not None:
            df_bars = pd.DataFrame(bar_dicts)
            # Rename columns to include units
            rename_map = {
                "Axial Force": f"Axial Force [{units['force']}]",
                "Stiffness": f"Stiffness [{units['stiffness']}]",
                "Modulus": f"Modulus [{units['stress']}]"
            }
            df_bars = df_bars.rename(columns=rename_map)

            cols = ["ID", f"Axial Force [{units['force']}]", f"Stiffness [{units['stiffness']}]", f"Modulus [{units['stress']}]"]
            cols = [c for c in cols if c in df_bars.columns]
            st.dataframe(
                df_bars[cols].style.format({f"Axial Force [{units['force']}]": "{:.1f}", f"Stiffness [{units['stiffness']}]": "{:.2e}", f"Modulus [{units['stress']}]": "{:.3e}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(bar_dicts)

    # 3. Fasteners Table
    with st.expander("Fasteners", expanded=expanded_state):
        fastener_dicts = solution.fasteners_as_dicts()
        if pd is not None:
            df_fast = pd.DataFrame(fastener_dicts)
            # Ensure ID column exists or use Row as fallback
            if "ID" not in df_fast.columns and "Row" in df_fast.columns:
                 df_fast["ID"] = df_fast["Row"]
                 
            # Rename columns to include units
            rename_map = {
                "Load": f"Load [{units['force']}]",
                "Brg Force Upper": f"Brg Force Upper [{units['force']}]",
                "Brg Force Lower": f"Brg Force Lower [{units['force']}]",
                "Stiffness": f"Stiffness [{units['stiffness']}]",
                "Modulus": f"Modulus [{units['stress']}]",
                "Diameter": f"Diameter [{units['length']}]",
                "Thickness Node 1": f"Thickness Node 1 [{units['length']}]",
                "Thickness Node 2": f"Thickness Node 2 [{units['length']}]"
            }
            df_fast = df_fast.rename(columns=rename_map)

            cols = ["ID", f"Load [{units['force']}]", f"Brg Force Upper [{units['force']}]", f"Brg Force Lower [{units['force']}]", f"Stiffness [{units['stiffness']}]", f"Modulus [{units['stress']}]", f"Diameter [{units['length']}]", "Quantity", f"Thickness Node 1 [{units['length']}]", f"Thickness Node 2 [{units['length']}]"]
            cols = [c for c in cols if c in df_fast.columns]
            st.dataframe(
                df_fast[cols].style.format({
                    f"Load [{units['force']}]": "{:.1f}", 
                    f"Brg Force Upper [{units['force']}]": "{:.1f}",
                    f"Brg Force Lower [{units['force']}]": "{:.1f}",
                    f"Stiffness [{units['stiffness']}]": "{:.2e}", 
                    f"Modulus [{units['stress']}]": "{:.3e}",
                    f"Diameter [{units['length']}]": "{:.3f}",
                    "Quantity": "{:.1f}",
                    f"Thickness Node 1 [{units['length']}]": "{:.3f}",
                    f"Thickness Node 2 [{units['length']}]": "{:.3f}",
                }),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(fastener_dicts)

    # 4. Reactions Table
    if solution.reactions:
        with st.expander("Reactions", expanded=expanded_state):
            reaction_dicts = solution.reactions_as_dicts()
            if pd is not None:
                df_react = pd.DataFrame(reaction_dicts)
                # Rename columns to include units
                rename_map = {
                    "Force": f"Force [{units['force']}]"
                }
                df_react = df_react.rename(columns=rename_map)

                cols = ["Node ID", f"Force [{units['force']}]"]
                cols = [c for c in cols if c in df_react.columns]
                st.dataframe(
                    df_react[cols].style.format({f"Force [{units['force']}]": "{:.1f}"}),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.table(reaction_dicts)
                
    # 5. Classic Results Table
    with st.expander("Classic Results", expanded=expanded_state):
        classic_dicts = solution.classic_results_as_dicts()
        if pd is not None:
            df_classic = pd.DataFrame(classic_dicts)
            # Rename columns to include units
            rename_map = {
                "Thickness": f"Thickness [{units['length']}]",
                "Area": f"Area [{units['area']}]",
                "Incoming Load": f"Incoming Load [{units['force']}]",
                "Bypass Load": f"Bypass Load [{units['force']}]",
                "Load Transfer": f"Load Transfer [{units['force']}]",
                "Detail Stress": f"Detail Stress [{units['stress']}]",
                "Bearing Stress": f"Bearing Stress [{units['stress']}]"
            }
            df_classic = df_classic.rename(columns=rename_map)

            cols = ["Element", "Node", f"Thickness [{units['length']}]", f"Area [{units['area']}]", f"Incoming Load [{units['force']}]", f"Bypass Load [{units['force']}]", f"Load Transfer [{units['force']}]", "L.Trans / P", f"Detail Stress [{units['stress']}]", f"Bearing Stress [{units['stress']}]", "Fbr / FDetail"]
            cols = [c for c in cols if c in df_classic.columns]
            
            st.dataframe(
                df_classic[cols].style.format({
                    f"Thickness [{units['length']}]": "{:.3f}",
                    f"Area [{units['area']}]": "{:.3f}",
                    f"Incoming Load [{units['force']}]": "{:.1f}",
                    f"Bypass Load [{units['force']}]": "{:.1f}",
                    f"Load Transfer [{units['force']}]": "{:.1f}",
                    "L.Trans / P": "{:.3f}",
                    f"Detail Stress [{units['stress']}]": "{:.1f}",
                    f"Bearing Stress [{units['stress']}]": "{:.1f}",
                    "Fbr / FDetail": "{:.3f}",
                }),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(classic_dicts)
            
    # 6. Loads Table (Bearing/Bypass)
    with st.expander("Loads", expanded=expanded_state):
        bb_dicts = solution.bearing_bypass_as_dicts()
        if pd is not None:
            df_bb = pd.DataFrame(bb_dicts)
            # Rename columns to include units
            rename_map = {
                "Bearing [lb]": f"Bearing [{units['force']}]",
                "Bypass [lb]": f"Bypass [{units['force']}]"
            }
            df_bb = df_bb.rename(columns=rename_map)
            
            st.dataframe(
                df_bb.style.format({f"Bearing [{units['force']}]": "{:.1f}", f"Bypass [{units['force']}]": "{:.1f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.table(bb_dicts)
            
    # 7. Min/Max Results
    with st.expander("Min/Max Results", expanded=expanded_state):
        # Calculate min/max
        max_disp = max([abs(n.displacement) for n in solution.nodes]) if solution.nodes else 0.0
        max_load = max([abs(b.axial_force) for b in solution.bars]) if solution.bars else 0.0
        max_shear = max([abs(f.force) for f in solution.fasteners]) if solution.fasteners else 0.0
        
        st.write(f"**Max Displacement:** {max_disp:.6e} {units['length']}")
        st.write(f"**Max Axial Load:** {max_load:.1f} {units['force']}")
        st.write(f"**Max Fastener Shear:** {max_shear:.1f} {units['force']}")
        
    # 8. Force Balance Check
    with st.expander("Force Balance Check", expanded=expanded_state):
        # Sum of External Forces
        sum_ext_right = 0.0
        sum_ext_left = 0.0
        
        # Plate Ends
        for p in solution.plates:
            if p.Fx_right > 0: sum_ext_right += p.Fx_right
            else: sum_ext_left += abs(p.Fx_right)
            if p.Fx_left > 0: sum_ext_right += p.Fx_left
            else: sum_ext_left += abs(p.Fx_left)
            
        # Applied Point Forces
        for f in solution.applied_forces:
            val = f.get("Value", 0.0)
            if val > 0: sum_ext_right += val
            else: sum_ext_left += abs(val)
            
        # Reactions
        sum_react_right = 0.0
        sum_react_left = 0.0
        for r in solution.reactions:
            if r.reaction > 0: sum_react_right += r.reaction
            else: sum_react_left += abs(r.reaction)
            
        total_right = sum_ext_right + sum_react_right
        total_left = sum_ext_left + sum_react_left
        
        diff = total_right - total_left
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rightward Force", f"{total_right:.1f} {units['force']}")
        c2.metric("Total Leftward Force", f"{total_left:.1f} {units['force']}")
        c3.metric("Net Force (Imbalance)", f"{diff:.1e} {units['force']}", delta_color="inverse")
        
        if abs(diff) > 1e-3:
            st.error("Warning: Significant force imbalance detected!")
        else:
            st.success("System is in equilibrium.")

    # 9. Fatigue Analysis (SSF)
    with st.expander("Fatigue Analysis (SSF)", expanded=expanded_state):
        if solution.fatigue_results:
            fatigue_dicts = solution.fatigue_results_as_dicts()
            if pd is not None:
                df_fatigue = pd.DataFrame(fatigue_dicts)
                
                # Rename columns to include units
                rename_map = {
                    "bearing_load": f"bearing_load [{units['force']}]",
                    "bypass_load": f"bypass_load [{units['force']}]",
                    "sigma_ref": f"sigma_ref [{units['stress']}]",
                    "peak_stress": f"peak_stress [{units['stress']}]"
                }
                df_fatigue = df_fatigue.rename(columns=rename_map)

                # Merge Ranking Info
                # Create a map from node_id to ranking info (only for critical nodes)
                critical_points = getattr(solution, "critical_points", [])
                rank_map = {cp["node_id"]: cp for cp in critical_points}
                
                # Add columns
                # Rank comes from the critical points list
                df_fatigue["Rank"] = df_fatigue["node_id"].map(lambda x: rank_map.get(x, {}).get("rank", None))
                
                # FSI and f_max come directly from the FatigueResult object now (available for all nodes)
                # Note: df_fatigue already has 'fsi' and 'f_max' columns from the object dict
                
                # Rename FSI/f_max columns to include units if needed, or just ensure they are present
                # The as_dict() method should have included them.
                if "f_max" in df_fatigue.columns:
                     df_fatigue = df_fatigue.rename(columns={"f_max": f"f_max [{units['stress']}]"})
                
                # Highlight Critical Node
                def highlight_crit(row):
                    if row["node_id"] == solution.critical_node_id:
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)

                cols = ["Rank", "fsi", "node_id", "plate_name", "ktg", "ktn", "ktb", "theta", "ssf", f"bearing_load [{units['force']}]", f"bypass_load [{units['force']}]", f"sigma_ref [{units['stress']}]", f"peak_stress [{units['stress']}]", f"f_max [{units['stress']}]"]
                cols = [c for c in cols if c in df_fatigue.columns]
                
                # Sort by Rank if available, else by node_id
                if "Rank" in df_fatigue.columns:
                    df_fatigue = df_fatigue.sort_values(by=["Rank", "node_id"], na_position='last')
                    # Convert to string to avoid PyArrow mixed type issues (float vs empty string)
                    df_fatigue["Rank"] = df_fatigue["Rank"].apply(lambda x: str(int(x)) if pd.notnull(x) and x != "" else "")

                # Check for missing FSI columns (stale state)
                if "fsi" not in df_fatigue.columns:
                    st.warning("⚠️ New fatigue columns (FSI, f_max) are missing. Please click 'Solve' to update the results.")

                st.dataframe(
                    df_fatigue[cols].style.format({
                        "Rank": "{}", 
                        "fsi": "{:.2f}",
                        "ktg": "{:.2f}",
                        "ktn": "{:.2f}",
                        "ktb": "{:.2f}",
                        "theta": "{:.2f}",
                        "ssf": "{:.2f}",
                        f"bearing_load [{units['force']}]": "{:.1f}",
                        f"bypass_load [{units['force']}]": "{:.1f}",
                        f"sigma_ref [{units['stress']}]": "{:.1f}",
                        f"peak_stress [{units['stress']}]": "{:.1f}",
                        f"f_max [{units['stress']}]": "{:.1f}",
                    }, na_rep="-").apply(highlight_crit, axis=1),
                    width="stretch",
                    hide_index=True,
                )
                
                if solution.critical_node_id:
                    st.error(f"Critical Node: {solution.critical_node_id}")
            else:
                st.table(fatigue_dicts)
        else:
            st.info("No fatigue results available (check fastener rows).")

def render_save_section(
    pitches: List[float],
    plates: List[Plate],
    fasteners: List[FastenerRow],
    supports: List[Tuple[int, int, float]],
    point_forces: Dict[int, float],
):
    st.markdown("---")
    st.subheader("Save & Load Configuration")
    
    # --- SAVE SECTION ---
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    label = c1.text_input(
        "Label for this case",
        value=st.session_state.get("config_label", ""),
        key="config_label_input",
    )
    if c2.button("Save Case"):
        if not label:
            st.error("Please provide a label.")
        else:
            # Get current solution if available
            current_solution = st.session_state.get("solution")
            
            config = serialize_configuration(
                label=label,
                pitches=pitches,
                plates=plates,
                fasteners=fasteners,
                supports=supports,
                unloading=st.session_state.get("config_unloading", ""),
                point_forces=[],
                solution=current_solution,
                description=f"Saved on {pd.Timestamp.now()}" if pd is not None else "Saved configuration"
            )
            st.session_state.saved_models.append(config)
            st.success(f"Configuration '{label}' saved.")
            
            # Offer download immediately
            # config is now a JointConfiguration object, so we use to_json() or to_dict()
            json_str = config.to_json(indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"{label.strip().replace(' ', '_')}.json",
                mime="application/json",
                key=f"save_download_{pd.Timestamp.now().timestamp()}"
            )

    # --- LOAD SECTION REMOVED (Consolidated to Sidebar) ---

