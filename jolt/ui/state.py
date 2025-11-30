"""State management for the JOLT UI."""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import replace
import json
import streamlit as st
from jolt import JointConfiguration, Plate, FastenerRow, figure76_example, JointSolution
from jolt.units import UnitSystem, UnitConverter


def clear_configuration_widget_state() -> None:
    """Remove cached widget values that conflict with a newly loaded model."""
    prefixes = [
        "n_rows",
        "pitch_",
        "pl_name_",
        "pl_E_",
        "pl_t_",
        "pl_fr_",
        "pl_lr_",
        "pl_A_",
        "sameA_",
        "pl_Fl_",
        "pl_Fr_",
        "fr_row_",
        "fr_d_",
        "fr_Eb_",
        "fr_nu_",
        "fr_m_",
        "fr_km_",
        "fr_conn_",
        "sp_pi_",
        "sp_ln_",
        "sp_val_",
        "pl_A_all_",
        "pl_A_",
        "save_cfg_",
        # Node-based / Refined mode keys
        "node_editor",
        "element_editor",
        "fastener_editor_nb",
    ]

    # Exact keys to remove
    exact_keys = [
        "node_table",
        "element_table",
        "fastener_table_nb",
        "extra_nodes",
    ]

    removed_count = 0
    for key in list(st.session_state.keys()):
        if key in exact_keys or any(key.startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)
            removed_count += 1
            # print(f"Cleared widget state: {key}") # Debug logging
    
    print(f"Cleared {removed_count} widget state keys.")


from copy import deepcopy

def serialize_configuration(
    pitches: Sequence[float],
    plates: Sequence[Plate],
    fasteners: Sequence[FastenerRow],
    supports: Sequence[Tuple[int, int, float]],
    label: str,
    unloading: str,
    point_forces: Optional[Sequence[Tuple[int, int, float]]] = None,
    solution: Optional[JointSolution] = None,
    description: str = "",
) -> JointConfiguration:
    configuration = JointConfiguration(
        pitches=list(pitches),
        plates=deepcopy(list(plates)),
        fasteners=deepcopy(list(fasteners)),
        supports=[(int(item[0]), int(item[1]), float(item[2])) for item in supports],
        point_forces=[
            (int(item[0]), int(item[1]), float(item[2]))
            for item in (point_forces or [])
        ],
        label=label,
        unloading=unloading,
        description=description,
        units=st.session_state.get("unit_system", UnitSystem.IMPERIAL),
    )
    if solution:
        configuration.results = solution.to_dict()
        
    return configuration


def apply_configuration(config: Union[Dict[str, Any], JointConfiguration]) -> JointConfiguration:
    # Handle case where config is already a JointConfiguration object
    # We check for 'to_dict' method or if it's not a dict to be safe against class reloading issues
    if isinstance(config, JointConfiguration) or hasattr(config, "to_dict"):
        configuration = config
    else:
        configuration = JointConfiguration.from_dict(config)
    clear_configuration_widget_state()
    st.session_state.pitches = list(configuration.pitches)
    st.session_state.plates = deepcopy(list(configuration.plates))
    st.session_state.fasteners = deepcopy(list(configuration.fasteners))
    st.session_state.supports = list(configuration.supports)
    st.session_state.point_forces = list(configuration.point_forces)
    st.session_state.config_label = configuration.label or "Case"
    st.session_state.config_unloading = configuration.unloading
    st.session_state["n_rows"] = len(st.session_state.pitches)
    st.session_state["unit_system"] = configuration.units
    st.session_state.pop("solution", None)  # Clear existing solution on load
    
    # Switch to Standard mode to view the loaded configuration
    # We use a temporary flag because modifying the widget key after instantiation raises an error.
    # This flag will be consumed by render_sidebar before creating the widget.
    st.session_state["_force_input_mode"] = "Standard"
    
    # Update widget key suffix to force re-creation of all input widgets
    # This guarantees that new values are used and old state is ignored.
    current_ver = st.session_state.get("_widget_version", 0)
    st.session_state["_widget_version"] = current_ver + 1
    
    return configuration


def convert_session_state(target_system: str) -> None:
    """Convert all physical values in session state to the target unit system."""
    current_system = st.session_state.get("unit_system", UnitSystem.IMPERIAL)
    if current_system == target_system:
        return

    # Determine conversion direction (Imperial -> SI or SI -> Imperial)
    # If target is SI, we convert FROM Imperial (assuming current is Imperial)
    # If target is Imperial, we convert FROM SI (assuming current is SI)
    # The UnitConverter methods take 'to_system' as argument.
    
    to_sys = UnitSystem(target_system)
    
    # 1. Pitches (Length)
    st.session_state.pitches = [
        UnitConverter.convert_length(p, to_sys) for p in st.session_state.pitches
    ]
    
    # 2. Plates
    new_plates = []
    for p in st.session_state.plates:
        # E (Stress)
        new_E = UnitConverter.convert_stress(p.E, to_sys)
        # t (Length)
        new_t = UnitConverter.convert_length(p.t, to_sys)
        # A_strip (Area)
        new_A = [UnitConverter.convert_area(a, to_sys) for a in p.A_strip]
        # Widths/Thicknesses if present
        new_widths = None
        if p.widths:
            new_widths = [UnitConverter.convert_length(w, to_sys) for w in p.widths]
        new_thicknesses = None
        if p.thicknesses:
            new_thicknesses = [UnitConverter.convert_length(t, to_sys) for t in p.thicknesses]
            
        # Forces (Force)
        new_Fx_left = UnitConverter.convert_force(p.Fx_left, to_sys)
        new_Fx_right = UnitConverter.convert_force(p.Fx_right, to_sys)
        
        # fatigue_strength (Stress)
        new_fs = None
        current_fs = getattr(p, "fatigue_strength", None)
        if current_fs is not None:
            new_fs = UnitConverter.convert_stress(current_fs, to_sys)
        
        new_plates.append(replace(
            p, 
            E=new_E, 
            t=new_t, 
            A_strip=new_A, 
            widths=new_widths, 
            thicknesses=new_thicknesses,
            Fx_left=new_Fx_left,
            Fx_right=new_Fx_right,
            fatigue_strength=new_fs
        ))
    st.session_state.plates = new_plates
    
    # 3. Fasteners
    new_fasteners = []
    for f in st.session_state.fasteners:
        # D (Length)
        new_D = UnitConverter.convert_length(f.D, to_sys)
        # Eb (Stress)
        new_Eb = UnitConverter.convert_stress(f.Eb, to_sys)
        # k_manual (Stiffness)
        new_k = None
        if f.k_manual is not None:
            new_k = UnitConverter.convert_stiffness(f.k_manual, to_sys)
            
        # Countersink Depth (Length)
        new_cs_depth = UnitConverter.convert_length(f.cs_depth, to_sys)
        # Hole Offset (Length)
        new_hole_offset = UnitConverter.convert_length(f.hole_offset, to_sys)
        
        new_fasteners.append(replace(
            f, 
            D=new_D, 
            Eb=new_Eb, 
            k_manual=new_k,
            cs_depth=new_cs_depth,
            hole_offset=new_hole_offset
        ))
    st.session_state.fasteners = new_fasteners
    
    # 4. Supports (Displacement -> Length)
    new_supports = []
    for (pid, node, val) in st.session_state.supports:
        new_val = UnitConverter.convert_length(val, to_sys)
        new_supports.append((pid, node, new_val))
    st.session_state.supports = new_supports
    
    # 5. Point Forces (Force)
    new_forces = []
    for (pid, node, val) in st.session_state.point_forces:
        new_val = UnitConverter.convert_force(val, to_sys)
        new_forces.append((pid, node, new_val))
    st.session_state.point_forces = new_forces
    
    # 6. Extra Nodes (Length)
    if "extra_nodes" in st.session_state and st.session_state.extra_nodes:
        st.session_state.extra_nodes = [
            UnitConverter.convert_length(x, to_sys) for x in st.session_state.extra_nodes
        ]
        
    # 7. Node Table (if present)
    if "node_table" in st.session_state and not st.session_state.node_table.empty:
        df = st.session_state.node_table
        if "x" in df.columns:
            df["x"] = df["x"].apply(lambda x: UnitConverter.convert_length(x, to_sys))
            st.session_state.node_table = df
            
    # 8. Element Table (if present)
    if "element_table" in st.session_state and not st.session_state.element_table.empty:
        df = st.session_state.element_table
        if "E" in df.columns:
            df["E"] = df["E"].apply(lambda x: UnitConverter.convert_stress(x, to_sys))
        if "t" in df.columns:
            df["t"] = df["t"].apply(lambda x: UnitConverter.convert_length(x, to_sys))
        if "w" in df.columns:
            df["w"] = df["w"].apply(lambda x: UnitConverter.convert_length(x, to_sys))
        st.session_state.element_table = df
        
    # 9. Fastener Table NB (if present)
    if "fastener_table_nb" in st.session_state and not st.session_state.fastener_table_nb.empty:
        df = st.session_state.fastener_table_nb
        if "d" in df.columns:
            df["d"] = df["d"].apply(lambda x: UnitConverter.convert_length(x, to_sys))
        if "E" in df.columns:
            df["E"] = df["E"].apply(lambda x: UnitConverter.convert_stress(x, to_sys))
        st.session_state.fastener_table_nb = df

    # Update System
    st.session_state.unit_system = target_system
    
    # Clear widget state to force refresh of inputs
    clear_configuration_widget_state()
    
    # Increment widget version
    st.session_state["_widget_version"] = st.session_state.get("_widget_version", 0) + 1


def initialize_session_state():
    if "pitches" not in st.session_state:
        (
            st.session_state.pitches,
            st.session_state.plates,
            st.session_state.fasteners,
            st.session_state.supports,
        ) = figure76_example()
        st.session_state["n_rows"] = len(st.session_state.pitches)

    if "saved_models" not in st.session_state:
        st.session_state.saved_models = []
    if "config_label" not in st.session_state:
        st.session_state.config_label = "Case 1"
    if "config_unloading" not in st.session_state:
        st.session_state.config_unloading = ""
    if "point_forces" not in st.session_state:
        st.session_state.point_forces = []
    if "_widget_version" not in st.session_state:
        st.session_state["_widget_version"] = 0
    if "unit_system" not in st.session_state:
        st.session_state.unit_system = UnitSystem.IMPERIAL


def safe_load_model(source: Any) -> Tuple[Optional[JointConfiguration], str]:
    """
    Load a model safely from a JSON source (string or file-like), returning (config, error_message).
    If successful, error_message is empty.
    """
    try:
        config = JointConfiguration.from_json(source)
        # Basic validation
        if not config.pitches:
             return None, "Model has no pitches defined."
        return config, ""
    except json.JSONDecodeError:
        return None, "Invalid JSON format."
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"
