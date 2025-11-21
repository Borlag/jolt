"""State management for the JOLT UI."""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import streamlit as st
from jolt import JointConfiguration, Plate, FastenerRow, figure76_example


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


def serialize_configuration(
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


def apply_configuration(config: Union[Dict[str, Any], JointConfiguration]) -> JointConfiguration:
    configuration = (
        config if isinstance(config, JointConfiguration) else JointConfiguration.from_dict(config)
    )
    clear_configuration_widget_state()
    st.session_state.pitches = list(configuration.pitches)
    st.session_state.plates = list(configuration.plates)
    st.session_state.fasteners = list(configuration.fasteners)
    st.session_state.supports = list(configuration.supports)
    st.session_state.point_forces = list(configuration.point_forces)
    st.session_state.config_label = configuration.label or "Case"
    st.session_state.config_unloading = configuration.unloading
    st.session_state["n_rows"] = len(st.session_state.pitches)
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
