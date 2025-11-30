"""UI for comparing multiple JOLT models."""
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jolt import JointConfiguration, JointSolution, Plate, FastenerRow
from jolt.units import UnitSystem, UnitConverter
from jolt.visualization_plotly import render_joint_diagram_plotly

def render_comparison_tab(saved_models: List[JointConfiguration], current_units: str):
    """Render the Model Comparison tab."""
    
    st.header("Model Comparison")
    
    if not saved_models:
        st.info("No saved models available. Please solve and save a model in the 'Model Definition' tab, or import a JSON file.")
        return

    # --- 1. Model Selection ---
    # Filter out models without results for the selection list (or mark them)
    model_options = {}
    for i, m in enumerate(saved_models):
        has_results = m.results is not None
        label = f"{m.label} (ID: {m.model_id[:8]})"
        if not has_results:
            label += " [No Results]"
        model_options[label] = i

    # Multi-select
    selected_indices = st.multiselect(
        "Select Models to Compare",
        options=list(model_options.keys()),
        format_func=lambda x: x,
        key="comparison_model_selector"
    )
    
    # Filter actual selected models
    selected_models: List[JointConfiguration] = []
    for label in selected_indices:
        idx = model_options[label]
        model = saved_models[idx]
        if model.results:
            selected_models.append(model)
        else:
            st.warning(f"Model '{model.label}' has no results and cannot be compared.")

    if len(selected_models) < 2:
        st.info("Select at least 2 models with results to perform a comparison.")
        # If 1 model selected, we could show its summary, but comparison needs 2.
        if len(selected_models) == 1:
            st.subheader("Single Model Summary")
            _render_single_model_summary(selected_models[0], current_units)
        return

    # --- 2. Reference Model ---
    ref_options = [m.label for m in selected_models]
    ref_label = st.selectbox("Reference Model (Baseline)", ref_options, index=0, key="ref_model_selector")
    ref_model = next((m for m in selected_models if m.label == ref_label), selected_models[0])

    # --- 3. Unit Consistency Check ---
    # Check if all selected models have the same unit system
    unit_systems = {m.units for m in selected_models}
    if len(unit_systems) > 1:
        st.error(f"⚠️ Unit Mismatch! Selected models use different unit systems: {unit_systems}. Comparison may be invalid.")
        # We could try to convert, but for now just warn.
    
    # --- 4. Element Filter ---
    # Collect all unique plate names across selected models
    all_plate_names = set()
    for m in selected_models:
        for p in m.plates:
            all_plate_names.add(p.name)
            
    sorted_plates = sorted(list(all_plate_names))
    if not sorted_plates:
        st.error("No plates found in selected models.")
        return
        
    target_element = st.selectbox("Select Element/Layer to Compare", sorted_plates, key="comp_element_selector")

    # --- 5. Metrics & Settings ---
    c1, c2, c3 = st.columns(3)
    with c1:
        show_fsi = st.checkbox("FSI (Fatigue)", value=True)
        show_loads = st.checkbox("Fastener Loads", value=True)
    with c2:
        show_bypass = st.checkbox("Bypass Loads", value=False)
        show_disp = st.checkbox("Displacements", value=False)
    with c3:
        normalized = st.checkbox("Normalize to Reference (%)", value=True)
        tolerance = st.slider("Highlight Tolerance (%)", 0.0, 50.0, 5.0, step=0.5)

    st.markdown("---")

    # --- 6. Comparison Logic & Visualization ---
    
    # Rehydrate solutions for selected models ONLY
    # We store them in a dict for easy access: model_id -> solution
    solutions: Dict[str, JointSolution] = {}
    for m in selected_models:
        if m.results:
            solutions[m.model_id] = JointSolution.from_dict(m.results)

    # A. Summary Table
    st.subheader("Summary Comparison")
    summary_data = []
    
    ref_sol = solutions[ref_model.model_id]
    
    # Helper to get summary value safely
    def get_summary_val(sol: JointSolution, key: str, element: Optional[str] = None) -> float:
        # We need to re-calculate or extract from rehydrated solution if not in summary dict
        # But wait, JointSolution.from_dict populates the object fields.
        # We can compute on the fly from the object.
        if key == "max_fsi":
            if element:
                # Find max FSI for this element
                return max((r.fsi for r in sol.fatigue_results if r.plate_name == element), default=0.0)
            return max((r.fsi for r in sol.fatigue_results), default=0.0)
        elif key == "max_load":
            return max((abs(f.force) for f in sol.fasteners), default=0.0)
        elif key == "max_bypass":
            # Filter by element if needed? Usually global or per element.
            # Let's do global for summary table row
            return max((abs(n.net_bypass) for n in sol.nodes), default=0.0)
        return 0.0

    for m in selected_models:
        sol = solutions[m.model_id]
        row = {"Model": m.label}
        
        # FSI
        fsi = get_summary_val(sol, "max_fsi", target_element)
        row["Max FSI"] = fsi
        
        # Load
        load = get_summary_val(sol, "max_load")
        row["Max Fastener Load"] = load
        
        # Deltas
        if m.model_id != ref_model.model_id:
            ref_fsi = get_summary_val(ref_sol, "max_fsi", target_element)
            ref_load = get_summary_val(ref_sol, "max_load")
            
            if normalized:
                fsi_delta = ((fsi - ref_fsi) / ref_fsi * 100) if ref_fsi > 1e-9 else 0.0
                load_delta = ((load - ref_load) / ref_load * 100) if ref_load > 1e-9 else 0.0
                row["FSI Δ (%)"] = fsi_delta
                row["Load Δ (%)"] = load_delta
            else:
                row["FSI Δ"] = fsi - ref_fsi
                row["Load Δ"] = load - ref_load
        else:
            if normalized:
                row["FSI Δ (%)"] = 0.0
                row["Load Δ (%)"] = 0.0
            else:
                row["FSI Δ"] = 0.0
                row["Load Δ"] = 0.0
                
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    format_dict = {col: "{:.2f}" for col in df_summary.columns if col != "Model"}
    st.dataframe(df_summary.style.format(format_dict))

    # B. Charts
    
    # FSI Chart
    if show_fsi:
        st.subheader(f"FSI Distribution - {target_element}")
        fig_fsi = go.Figure()
        
        for m in selected_models:
            sol = solutions[m.model_id]
            # Filter results for target element
            # We need to sort by node order or x-coordinate
            # Let's use 'row' or 'node_id'
            
            # Get nodes for this plate
            relevant_results = [r for r in sol.fatigue_results if r.plate_name == target_element]
            # Sort by row (assuming row corresponds to position)
            relevant_results.sort(key=lambda r: r.row)
            
            x_vals = [r.row for r in relevant_results] # Use Row index for X-axis
            y_vals = [r.fsi for r in relevant_results]
            
            fig_fsi.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                name=m.label,
                opacity=0.7 if m.model_id != ref_model.model_id else 1.0
            ))
            
        fig_fsi.update_layout(
            xaxis_title="Node Index (Row)",
            yaxis_title="FSI",
            barmode='group',
            title=f"FSI by Node Index ({target_element})",
            annotations=[dict(
                x=0.5, y=-0.2, xref="paper", yref="paper",
                text="Note: Different models may have different mesh discretization; node indices do not necessarily align physically.",
                showarrow=False, font=dict(size=10, color="gray")
            )]
        )
        st.plotly_chart(fig_fsi, use_container_width=True)

    # Load Chart
    if show_loads:
        st.subheader("Fastener Load Distribution")
        fig_load = go.Figure()
        
        for m in selected_models:
            sol = solutions[m.model_id]
            # Sort fasteners by row
            fasteners = sorted(sol.fasteners, key=lambda f: f.row)
            
            x_vals = [f.row for f in fasteners]
            y_vals = [abs(f.force) for f in fasteners]
            
            fig_load.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                name=m.label,
                opacity=0.7 if m.model_id != ref_model.model_id else 1.0
            ))
            
        fig_load.update_layout(
            xaxis_title="Fastener Row",
            yaxis_title=f"Load [{current_units}]", # Assuming consistent units or just label
            barmode='group',
            title="Fastener Loads"
        )
        st.plotly_chart(fig_load, use_container_width=True)

    # C. Joint Diagrams (Grid Layout)
    st.subheader("Joint Diagrams")
    
    # Layout logic
    n_models = len(selected_models)
    cols = 2 if n_models <= 4 else 3
    
    # Create columns container
    grid_cols = st.columns(cols)
    
    for i, m in enumerate(selected_models):
        col_idx = i % cols
        with grid_cols[col_idx]:
            st.markdown(f"**{m.label}**")
            sol = solutions[m.model_id]
            
            # Render diagram
            # We need to pass units dict. We assume current_units (e.g. "Imperial")
            # But render_joint_diagram_plotly expects a dict of unit labels.
            # We can get it from UnitConverter
            unit_labels = UnitConverter.get_labels(current_units) # Or m.units if we want to respect model units
            
            fig = render_joint_diagram_plotly(
                pitches=m.pitches,
                plates=m.plates,
                fasteners=m.fasteners,
                supports=m.supports,
                solution=sol,
                units=unit_labels,
                mode="scheme", # Default to scheme
                font_size=10
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"viz_{m.model_id}")
            else:
                st.error("Failed to render diagram.")


def _render_single_model_summary(model: JointConfiguration, current_units: str):
    """Helper to show a simple summary for a single model."""
    if not model.results:
        return
    sol = JointSolution.from_dict(model.results)
    
    st.write(f"**Description:** {model.description}")
    st.write(f"**Units:** {model.units}")
    
    # Show summary metrics from the dict if available, or compute
    # The 'summary' key in results dict might be available
    summary = model.results.get("summary", {})
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Max FSI", f"{summary.get('max_fsi_global', 0):.2f}")
        c2.metric("Max Load", f"{summary.get('max_fastener_load', 0):.1f}")
        c3.metric("Max Bypass", f"{summary.get('max_bypass', 0):.1f}")
    else:
        st.write("No pre-computed summary available.")
