"""UI for comparing multiple JOLT models."""
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jolt import JointConfiguration, JointSolution, Plate, FastenerRow
from jolt.units import UnitSystem, UnitConverter
from jolt.visualization_plotly import render_joint_diagram_plotly

# Import feature flag
from . import ENABLE_JOLT_BENCHMARKS

# Conditionally import benchmark modules only when feature is enabled
if ENABLE_JOLT_BENCHMARKS:
    from jolt.benchmarks import (
        D06_ROW_A_LABEL,
        JOLT_FASTENER_REFS,
        compare_solution_to_jolt,
    )
    from jolt.analysis import (
        TOPOLOGY_VARIANTS_BOEING,
        solve_with_topologies,
    )

def render_comparison_tab(
    saved_models: List[JointConfiguration], 
    current_units: str,
    current_model: Optional[Dict[str, Any]] = None,
):
    """Render the Model Comparison tab.
    
    Args:
        saved_models: List of saved JointConfiguration objects
        current_units: Current unit system string
        current_model: Optional dict with current model definition from sidebar
            Keys: pitches, plates, fasteners, supports, point_forces
    """
    
    st.header("Model Comparison")
    
    # --- JOLT Benchmark Section (guarded by feature flag) ---
    if ENABLE_JOLT_BENCHMARKS:
        _render_jolt_benchmark_section(current_model, current_units)
    
    st.markdown("---")
    
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
        if len(selected_models) == 1:
            st.subheader("Single Model Summary")
            _render_single_model_summary(selected_models[0], current_units)
        return

    # --- 2. Reference Model ---
    ref_options = [m.label for m in selected_models]
    ref_label = st.selectbox("Reference Model (Baseline)", ref_options, index=0, key="ref_model_selector")
    ref_model = next((m for m in selected_models if m.label == ref_label), selected_models[0])

    # --- 3. Unit Consistency Check ---
    unit_systems = {m.units for m in selected_models}
    if len(unit_systems) > 1:
        st.error(f"⚠️ Unit Mismatch! Selected models use different unit systems: {unit_systems}. Comparison may be invalid.")
    
    # --- 4. Element Filter ---
    all_plate_names = set()
    for m in selected_models:
        for p in m.plates:
            all_plate_names.add(p.name)
            
    sorted_plates = sorted(list(all_plate_names))
    if not sorted_plates:
        st.error("No plates found in selected models.")
        return
        
    # Add "All Elements" option
    element_options = ["All Elements"] + sorted_plates
    target_element = st.selectbox("Select Element/Layer to Compare", element_options, key="comp_element_selector")

    # --- 5. Metrics & Settings ---
    c1, c2, c3 = st.columns(3)
    with c1:
        show_fsi = st.checkbox("FSI (Fatigue)", value=True)
        show_loads = st.checkbox("Fastener Loads", value=True)
        # Color Pickers
        col_ref = st.color_picker("Reference Color", "#1f77b4") # Default Blue
    with c2:
        show_bypass = st.checkbox("Bypass Loads", value=False)
        show_disp = st.checkbox("Displacements", value=False)
        col_comp = st.color_picker("Comparison Color", "#ff7f0e") # Default Orange
    with c3:
        normalized = st.checkbox("Normalize to Reference (%)", value=True)
        tolerance = st.slider("Highlight Tolerance (%)", 0.0, 50.0, 5.0, step=0.5)

    st.markdown("---")

    # --- 6. Comparison Logic & Visualization ---
    
    # Rehydrate solutions for selected models ONLY
    solutions: Dict[str, JointSolution] = {}
    for m in selected_models:
        if m.results:
            solutions[m.model_id] = JointSolution.from_dict(m.results)

    # A. Summary Table (Critical Node Focus)
    st.subheader("Summary Comparison (Critical Node)")
    summary_data = []
    
    ref_sol = solutions[ref_model.model_id]
    
    # Helper to find critical node (Max FSI)
    def get_critical_node_info(sol: JointSolution):
        if not sol.fatigue_results:
            return None
        # Find max FSI result
        crit_res = max(sol.fatigue_results, key=lambda r: r.fsi)
        # Find corresponding node object for bypass/displacement
        node_obj = next((n for n in sol.nodes if n.legacy_id == crit_res.node_id), None)
        
        return {
            "fsi": crit_res.fsi,
            "bearing": crit_res.bearing_load,
            "bypass": crit_res.bypass_load,
            "location": f"{crit_res.plate_name} - Row {crit_res.row}",
            "node_id": crit_res.node_id
        }

    ref_crit = get_critical_node_info(ref_sol)

    # Styling function for Highlight Tolerance
    def highlight_high_delta(val):
        """Highlight red if delta > tolerance."""
        if isinstance(val, str) and "(" in val and "%" in val:
            try:
                # Extract percentage: "123.45 (+10.5%)" -> 10.5
                pct_str = val.split("(")[1].split("%")[0].replace("+", "").replace("-", "")
                pct = float(pct_str)
                if pct > tolerance:
                    return "color: red; font-weight: bold"
            except:
                pass
        return ""

    for m in selected_models:
        sol = solutions[m.model_id]
        crit = get_critical_node_info(sol)
        
        row = {"Model": m.label}
        
        if crit:
            row["Critical Loc"] = crit["location"]
            row["Max FSI"] = crit["fsi"]
            row["Bearing @ Crit"] = crit["bearing"]
            row["Bypass @ Crit"] = crit["bypass"]
            
            if m.model_id != ref_model.model_id and ref_crit:
                if normalized:
                    fsi_delta = ((crit["fsi"] - ref_crit["fsi"]) / ref_crit["fsi"] * 100) if ref_crit["fsi"] > 1e-9 else 0.0
                    row["FSI Δ (%)"] = fsi_delta
                else:
                    row["FSI Δ"] = crit["fsi"] - ref_crit["fsi"]
        else:
            row["Critical Loc"] = "N/A"
            row["Max FSI"] = 0.0
            
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    
    # Apply styling
    # We want to highlight the Delta columns
    delta_cols = [c for c in df_summary.columns if "Δ" in c]
    
    def highlight_numeric_delta(val):
        if isinstance(val, (int, float)) and normalized:
             if abs(val) > tolerance:
                 return "color: red; font-weight: bold"
        return ""

    format_dict = {col: "{:.2f}" for col in df_summary.columns if col != "Model" and col != "Critical Loc"}
    st.dataframe(df_summary.style.format(format_dict).map(highlight_numeric_delta, subset=delta_cols))

    # B. Node Inspector
    st.subheader("Node Inspector")
    st.markdown("Select a specific node to inspect for each model.")
    
    # Dictionary to store the selected node object for each model
    # Key: model_id, Value: NodeResult object (or None)
    selected_nodes_map = {}
    
    # Create columns for selectors if there are few models, otherwise list them
    # Using columns for up to 3 models looks good
    insp_cols = st.columns(min(len(selected_models), 3))
    
    for i, m in enumerate(selected_models):
        col_idx = i % 3
        with insp_cols[col_idx]:
            sol = solutions[m.model_id]
            
            # Filter nodes based on target_element
            if target_element == "All Elements":
                nodes = sol.nodes
                nodes.sort(key=lambda n: (n.plate_name, n.row))
                options = [f"{n.plate_name} - Row {n.row}" for n in nodes]
                node_map = {f"{n.plate_name} - Row {n.row}": n for n in nodes}
            else:
                nodes = [n for n in sol.nodes if n.plate_name == target_element]
                nodes.sort(key=lambda n: n.row)
                options = [f"Row {n.row}" for n in nodes]
                node_map = {f"Row {n.row}": n for n in nodes}
            
            if options:
                # Try to preserve selection if possible, or default to first
                # We use a unique key for each selectbox
                sel_lbl = st.selectbox(f"{m.label}", options, key=f"insp_node_sel_{m.model_id}")
                selected_nodes_map[m.model_id] = node_map[sel_lbl]
            else:
                st.warning(f"No nodes in {m.label}")
                selected_nodes_map[m.model_id] = None

    # Build Inspector Table
    inspector_data = []
    
    # Get Reference values from the node SELECTED for the reference model
    ref_fsi_val = 0.0
    ref_bearing_val = 0.0
    ref_bypass_val = 0.0
    ref_fast_load = 0.0
    
    ref_node_obj = selected_nodes_map.get(ref_model.model_id)
    
    if ref_node_obj:
        ref_bypass_val = abs(ref_node_obj.net_bypass)
        ref_sol = solutions[ref_model.model_id]
        
        # Find associated fatigue/fastener results for this node
        # We need to look them up in the solution lists
        ref_fatigue = next((r for r in ref_sol.fatigue_results if r.plate_name == ref_node_obj.plate_name and r.row == ref_node_obj.row), None)
        if ref_fatigue:
            ref_fsi_val = ref_fatigue.fsi
            ref_bearing_val = ref_fatigue.bearing_load
        
        ref_fast = next((f for f in ref_sol.fasteners if f.row == ref_node_obj.row), None)
        if ref_fast: ref_fast_load = abs(ref_fast.force)

    # Helper to format value and delta
    def fmt_val_delta(val, ref_val, is_ref):
        if is_ref:
            return f"{val:.2f}"
        
        if normalized:
            delta = ((val - ref_val) / ref_val * 100) if abs(ref_val) > 1e-9 else 0.0
            return f"{val:.2f} ({delta:+.1f}%)"
        else:
            delta = val - ref_val
            return f"{val:.2f} ({delta:+.2f})"

    for m in selected_models:
        sol = solutions[m.model_id]
        is_ref = (m.model_id == ref_model.model_id)
        
        # Get the node explicitly selected for this model
        target_node = selected_nodes_map.get(m.model_id)
        
        row = {"Model": m.label}
        
        if target_node:
            # Add Node Location info to the row for clarity
            row["Node"] = f"{target_node.plate_name} @ {target_node.row}"

            # FSI
            fsi_val = 0.0
            fatigue = next((r for r in sol.fatigue_results if r.plate_name == target_node.plate_name and r.row == target_node.row), None)
            if fatigue: fsi_val = fatigue.fsi
            row["FSI"] = fmt_val_delta(fsi_val, ref_fsi_val, is_ref)
            
            # Bearing
            bearing_val = fatigue.bearing_load if fatigue else 0.0
            row["Bearing"] = fmt_val_delta(bearing_val, ref_bearing_val, is_ref)
            
            # Bypass
            bypass_val = abs(target_node.net_bypass)
            row["Bypass"] = fmt_val_delta(bypass_val, ref_bypass_val, is_ref)
            
            # Fastener Load
            fast_load = 0.0
            fast = next((f for f in sol.fasteners if f.row == target_node.row), None)
            if fast: fast_load = abs(fast.force)
            row["Fastener Load"] = fmt_val_delta(fast_load, ref_fast_load, is_ref)
            
        else:
            row["Node"] = "N/A"
            row["FSI"] = "N/A"
            row["Bearing"] = "N/A"
            row["Bypass"] = "N/A"
            row["Fastener Load"] = "N/A"
            
        inspector_data.append(row)
        
    # Apply Highlight Tolerance to Inspector Table
    if inspector_data:
        df_insp = pd.DataFrame(inspector_data)
        # Reorder columns to put Node right after Model
        cols = ["Model", "Node", "FSI", "Bearing", "Bypass", "Fastener Load"]
        # Ensure all cols exist
        cols = [c for c in cols if c in df_insp.columns]
        df_insp = df_insp[cols]
        
        st.table(df_insp.style.map(highlight_high_delta))


    # C. Charts
    
    # Helper for adding traces
    def add_bar_trace(fig, x, y, label, is_ref, color_ref, color_comp, text_template="%{y:.1f}"):
        opacity = 1.0 if is_ref else 0.7 
        color = color_ref if is_ref else color_comp
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=label,
            opacity=opacity,
            marker_color=color,
            text=y,
            texttemplate=text_template,
            textposition='auto'
        ))

    # FSI Chart
    if show_fsi:
        st.subheader(f"FSI Distribution - {target_element}")
        fig_fsi = go.Figure()
        
        all_y = []
        for m in selected_models:
            sol = solutions[m.model_id]
            
            if target_element == "All Elements":
                relevant_results = sol.fatigue_results
                relevant_results.sort(key=lambda r: (r.plate_name, r.row))
                x_vals = [f"{r.plate_name} - {r.row}" for r in relevant_results]
                y_vals = [r.fsi for r in relevant_results]
            else:
                relevant_results = [r for r in sol.fatigue_results if r.plate_name == target_element]
                relevant_results.sort(key=lambda r: r.row)
                x_vals = [r.row for r in relevant_results]
                y_vals = [r.fsi for r in relevant_results]
            
            all_y.extend(y_vals)
            add_bar_trace(fig_fsi, x_vals, y_vals, m.label, m.model_id == ref_model.model_id, col_ref, col_comp, text_template="%{y:.0f}")
            
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            y_range = y_max - y_min
            if y_range > 0:
                fig_fsi.update_yaxes(range=[max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])

        fig_fsi.update_layout(
            xaxis_title="Node Location",
            yaxis_title="FSI",
            barmode='group',
            title=f"FSI Distribution ({target_element})",
        )
        st.plotly_chart(fig_fsi, use_container_width=True)

    # Load Chart
    if show_loads:
        st.subheader("Fastener Load Distribution")
        fig_load = go.Figure()
        
        all_y = []
        for m in selected_models:
            sol = solutions[m.model_id]
            fasteners = sorted(sol.fasteners, key=lambda f: f.row)
            
            x_vals = [f.row for f in fasteners]
            y_vals = [abs(f.force) for f in fasteners]
            all_y.extend(y_vals)
            
            add_bar_trace(fig_load, x_vals, y_vals, m.label, m.model_id == ref_model.model_id, col_ref, col_comp)
            
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            y_range = y_max - y_min
            if y_range > 0:
                fig_load.update_yaxes(range=[max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])

        fig_load.update_layout(
            xaxis_title="Fastener Row",
            yaxis_title=f"Load [{current_units}]",
            barmode='group',
            title="Fastener Loads"
        )
        st.plotly_chart(fig_load, use_container_width=True)

    # Bypass Chart
    if show_bypass:
        st.subheader(f"Bypass Load Distribution - {target_element}")
        fig_bypass = go.Figure()
        
        all_y = []
        for m in selected_models:
            sol = solutions[m.model_id]
            
            if target_element == "All Elements":
                nodes = sol.nodes
                nodes.sort(key=lambda n: (n.plate_name, n.row))
                x_vals = [f"{n.plate_name} - {n.row}" for n in nodes]
                y_vals = [abs(n.net_bypass) for n in nodes]
            else:
                nodes = [n for n in sol.nodes if n.plate_name == target_element]
                nodes.sort(key=lambda n: n.row)
                x_vals = [n.row for n in nodes]
                y_vals = [abs(n.net_bypass) for n in nodes]
                
            all_y.extend(y_vals)
            add_bar_trace(fig_bypass, x_vals, y_vals, m.label, m.model_id == ref_model.model_id, col_ref, col_comp)
            
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            y_range = y_max - y_min
            if y_range > 0:
                fig_bypass.update_yaxes(range=[max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])

        fig_bypass.update_layout(
            xaxis_title="Node Location",
            yaxis_title=f"Bypass Load [{current_units}]",
            barmode='group',
            title=f"Bypass Loads ({target_element})"
        )
        st.plotly_chart(fig_bypass, use_container_width=True)

    # Displacement Chart
    if show_disp:
        st.subheader(f"Displacement Distribution - {target_element}")
        fig_disp = go.Figure()
        
        all_y = []
        for m in selected_models:
            sol = solutions[m.model_id]
            
            if target_element == "All Elements":
                nodes = sol.nodes
                nodes.sort(key=lambda n: (n.plate_name, n.row))
                x_vals = [f"{n.plate_name} - {n.row}" for n in nodes]
                y_vals = [n.displacement for n in nodes]
            else:
                nodes = [n for n in sol.nodes if n.plate_name == target_element]
                nodes.sort(key=lambda n: n.row)
                x_vals = [n.row for n in nodes]
                y_vals = [n.displacement for n in nodes]
                
            all_y.extend(y_vals)
            add_bar_trace(fig_disp, x_vals, y_vals, m.label, m.model_id == ref_model.model_id, col_ref, col_comp, text_template="%{y:.4f}")
            
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            y_range = y_max - y_min
            if y_range > 0:
                 fig_disp.update_yaxes(range=[y_min - 0.1 * y_range, y_max + 0.1 * y_range])

        fig_disp.update_layout(
            xaxis_title="Node Location",
            yaxis_title=f"Displacement [{current_units}]",
            barmode='group',
            title=f"Displacement ({target_element})"
        )
        st.plotly_chart(fig_disp, use_container_width=True)

    # --- 7. Diagram Selection ---
    st.subheader("Joint Diagrams")
    
    diagram_types = ["Scheme", "Internal Loads", "Fatigue Analysis"]
    selected_diagrams = st.multiselect(
        "Select Diagrams to Display",
        options=diagram_types,
        default=diagram_types,
        key="comparison_diagram_selector"
    )
    
    if not selected_diagrams:
        st.info("Select at least one diagram type to view.")
        return

    n_models = len(selected_models)
    cols = 2 if n_models <= 4 else 3
    grid_cols = st.columns(cols)
    
    unit_labels = UnitConverter.get_labels(current_units)

    for i, m in enumerate(selected_models):
        col_idx = i % cols
        with grid_cols[col_idx]:
            # Add border container
            with st.container(border=True):
                st.markdown(f"**{m.label}**")
                sol = solutions[m.model_id]
                
                # 1. Scheme
                if "Scheme" in selected_diagrams:
                    st.caption("Scheme Overview")
                    fig_scheme = render_joint_diagram_plotly(
                        pitches=m.pitches, plates=m.plates, fasteners=m.fasteners, supports=m.supports,
                        solution=sol, units=unit_labels, mode="scheme", font_size=10
                    )
                    if fig_scheme: st.plotly_chart(fig_scheme, use_container_width=True, key=f"viz_scheme_{m.model_id}")
                
                # 2. Loads
                if "Internal Loads" in selected_diagrams:
                    st.caption("Internal Loads")
                    fig_load = render_joint_diagram_plotly(
                        pitches=m.pitches, plates=m.plates, fasteners=m.fasteners, supports=m.supports,
                        solution=sol, units=unit_labels, mode="loads", font_size=10
                    )
                    if fig_load: st.plotly_chart(fig_load, use_container_width=True, key=f"viz_load_{m.model_id}")
                
                # 3. Fatigue
                if "Fatigue Analysis" in selected_diagrams:
                    st.caption("Fatigue Analysis")
                    fig_fatigue = render_joint_diagram_plotly(
                        pitches=m.pitches, plates=m.plates, fasteners=m.fasteners, supports=m.supports,
                        solution=sol, units=unit_labels, mode="fatigue", font_size=10
                    )
                    if fig_fatigue: st.plotly_chart(fig_fatigue, use_container_width=True, key=f"viz_fatigue_{m.model_id}")


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


def _render_jolt_benchmark_section(current_model: Optional[Dict[str, Any]], current_units: str):
    """Render the JOLT benchmark comparison section.
    
    This function is only called when ENABLE_JOLT_BENCHMARKS is True.
    
    Args:
        current_model: Dict with current model definition (pitches, plates, fasteners, supports, point_forces)
        current_units: Current unit system string
    """
    with st.expander("🔬 Topology / JOLT Benchmark Comparison", expanded=False):
        st.markdown("""
        Compare different fastener topology variants against Boeing JOLT reference values.
        This feature runs the current model with multiple topology settings and shows how 
        each compares to the JOLT reference data for Case D06 Row A.
        """)
        
        # Check if we have a valid model to work with
        if current_model is None:
            st.info("No model loaded. Please configure a model in the sidebar first.")
            return
        
        pitches = current_model.get("pitches", [])
        plates = current_model.get("plates", [])
        fasteners = current_model.get("fasteners", [])
        supports = current_model.get("supports", [])
        point_forces = current_model.get("point_forces", [])
        
        if not plates or not fasteners:
            st.info("Model is incomplete. Please define plates and fasteners in the sidebar.")
            return
        
        # Case label selection (locked to D06 Row A for now)
        col1, col2 = st.columns([1, 2])
        with col1:
            case_label = st.text_input(
                "Case Label",
                value=D06_ROW_A_LABEL,
                disabled=True,
                help="Currently locked to Case_5_3_elements_row_a (Boeing JOLT D06 Row A)"
            )
        
        with col2:
            # Topology variant selection
            selected_topologies = st.multiselect(
                "Select Topology Variants",
                options=TOPOLOGY_VARIANTS_BOEING,
                default=TOPOLOGY_VARIANTS_BOEING,
                help="Choose which topology variants to compare"
            )
        
        if not selected_topologies:
            st.warning("Select at least one topology variant to compare.")
            return
        
        # Run comparison button
        if st.button("🚀 Run Topology Comparison", key="run_jolt_benchmark"):
            with st.spinner("Running solver for each topology..."):
                try:
                    # Run solve_with_topologies
                    results = solve_with_topologies(
                        pitches=pitches,
                        plates=plates,
                        fasteners=fasteners,
                        supports=supports,
                        point_forces=point_forces,
                        topology_variants=selected_topologies,
                    )
                    
                    # Store results in session state
                    st.session_state["jolt_benchmark_results"] = results
                    st.success(f"Completed analysis for {len(results)} topology variants.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return
        
        # Display results if available
        if "jolt_benchmark_results" in st.session_state:
            results = st.session_state["jolt_benchmark_results"]
            
            st.subheader("Comparison Results")
            
            # Build comparison table data
            table_data = []
            
            for topo_name, solution in results.items():
                # Compare against JOLT references
                comparisons = compare_solution_to_jolt(solution, case_label)
                
                for (row, plate_i, plate_j), metrics in comparisons.items():
                    table_data.append({
                        "Topology": topo_name,
                        "Row": row,
                        "Plate_i": plate_i,
                        "Plate_j": plate_j,
                        "JOLT_ref [lb]": f"{metrics['ref']:.1f}",
                        "Model [lb]": f"{metrics['model']:.1f}",
                        "Δ [lb]": f"{metrics['abs_err']:+.1f}",
                        "Δ [%]": f"{metrics['rel_err_pct']:+.1f}%",
                    })
            
            if table_data:
                df = pd.DataFrame(table_data)
                
                # Styling function for highlighting large deltas
                def highlight_delta(val):
                    if isinstance(val, str) and val.endswith("%"):
                        try:
                            pct = float(val.replace("%", "").replace("+", ""))
                            if abs(pct) > 10:
                                return "color: red; font-weight: bold"
                            elif abs(pct) > 5:
                                return "color: orange"
                        except:
                            pass
                    return ""
                
                st.dataframe(
                    df.style.map(highlight_delta, subset=["Δ [%]"]),
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Summary statistics per topology
                st.subheader("Summary by Topology")
                summary_data = []
                
                for topo_name, solution in results.items():
                    comparisons = compare_solution_to_jolt(solution, case_label)
                    if comparisons:
                        abs_errors = [abs(m["abs_err"]) for m in comparisons.values()]
                        rel_errors = [abs(m["rel_err_pct"]) for m in comparisons.values()]
                        
                        summary_data.append({
                            "Topology": topo_name,
                            "Max |Δ| [lb]": f"{max(abs_errors):.1f}",
                            "Mean |Δ| [lb]": f"{sum(abs_errors)/len(abs_errors):.1f}",
                            "Max |Δ| [%]": f"{max(rel_errors):.1f}%",
                            "RMS Δ [%]": f"{(sum(e**2 for e in rel_errors)/len(rel_errors))**0.5:.1f}%",
                        })
                
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
            else:
                st.warning("No matching interfaces found in JOLT reference data. "
                          "Make sure the model matches Case_5_3_elements_row_a geometry.")
        
        # Show reference data
        with st.expander("📋 JOLT Reference Values", expanded=False):
            st.markdown("**Boeing JOLT D06 Row A Reference Fastener Loads:**")
            ref_data = []
            for (case, row, pi, pj), load in JOLT_FASTENER_REFS.items():
                if case == case_label:
                    ref_data.append({
                        "Row": row,
                        "Plate i": pi,
                        "Plate j": pj,
                        "Reference Load [lb]": load,
                    })
            if ref_data:
                st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

