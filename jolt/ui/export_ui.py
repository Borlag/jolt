import streamlit as st
import pandas as pd
from jolt.export import JoltExcelExporter

def render_export_tab(pitches, plates, fasteners, supports, point_forces, units):
    st.header("Generate Engineering Reports")
    
    saved_models = st.session_state.get("saved_models", [])
    if not saved_models:
        st.warning("No saved models found. Please Define, Solve, and Save at least one case.")
        return

    # --- Configuration Column ---
    c_conf, c_preview = st.columns([1, 2])
    
    with c_conf:
        st.subheader("1. Select Content")
        
        # Select Models
        model_labels = [f"{m.label} (ID: {m.model_id[:6]})" for m in saved_models]
        selected_indices = st.multiselect(
            "Models to Include", 
            range(len(saved_models)), 
            default=range(len(saved_models)), 
            format_func=lambda i: model_labels[i]
        )
        
        st.markdown("---")
        st.subheader("2. Report Options")
        
        # Table Options
        st.markdown("**Tables**")
        opt_elements = st.checkbox("Elements / Plates", value=True, key="exp_opt_elements")
        opt_fasteners = st.checkbox("Fasteners", value=True, key="exp_opt_fasteners")
        opt_loads = st.checkbox("Internal Loads", value=True, key="exp_opt_loads")
        opt_fatigue = st.checkbox("Fatigue Analysis", value=True, key="exp_opt_fatigue")
        opt_nodes = st.checkbox("Nodes (Detailed)", value=False, key="exp_opt_nodes")
        opt_supports = st.checkbox("Supports", value=False, key="exp_opt_supports")
        opt_disp_table = st.checkbox("Displacements (Table)", value=False, key="exp_opt_disp_table")
        
        # Diagram Options
        st.markdown("**Diagrams**")
        include_images = st.checkbox("Include Diagrams (Slow)", value=True, help="Requires 'kaleido' installed.", key="exp_include_images")
        
        img_scheme = False
        img_loads = False
        img_disp = False
        img_fatigue = False
        label_spacing = 1.0
        
        if include_images:
            c_img1, c_img2 = st.columns(2)
            with c_img1:
                img_scheme = st.checkbox("Scheme", value=True, key="exp_img_scheme")
                img_loads = st.checkbox("Loads", value=True, key="exp_img_loads")
            with c_img2:
                img_disp = st.checkbox("Displacements", value=False, key="exp_img_disp")
                img_fatigue = st.checkbox("Fatigue", value=False, key="exp_img_fatigue")
            
            label_spacing = st.slider(
                "Label Scale", 
                0.5, 2.0, 1.0, 0.1, 
                help="Adjust to prevent overlapping text."
            )

        st.markdown("**Comparison**")
        include_comparison = st.checkbox("Include Comparison Sheet", value=True, disabled=(len(selected_indices) < 2), key="exp_include_comparison")
        
        comp_dist_fsi = False
        comp_dist_loads = False
        comp_dist_bypass = False
        comp_dist_disp = False
        
        if include_comparison:
             st.caption("Comparison Distribution Charts")
             c_cd1, c_cd2 = st.columns(2)
             with c_cd1:
                 comp_dist_fsi = st.checkbox("FSI Dist.", value=True, key="exp_comp_fsi")
                 comp_dist_loads = st.checkbox("Load Dist.", value=True, key="exp_comp_loads")
             with c_cd2:
                 comp_dist_bypass = st.checkbox("Bypass Dist.", value=False, key="exp_comp_bypass")
                 comp_dist_disp = st.checkbox("Disp. Dist.", value=False, key="exp_comp_disp")

        st.markdown("---")
        st.subheader("3. Generate")
        
        if st.button("Generate Excel Report", type="primary"):
            if not selected_indices:
                st.error("Select at least one model.")
            else:
                with st.spinner("Compiling Engineering Data..."):
                    try:
                        target_models = [saved_models[i] for i in selected_indices]
                        
                        # Build Options Dict
                        options = {
                            "include_elements": opt_elements,
                            "include_fasteners": opt_fasteners,
                            "include_loads": opt_loads,
                            "include_fatigue": opt_fatigue,
                            "include_nodes": opt_nodes,
                            "include_supports": opt_supports,
                            "include_displacements": opt_disp_table,
                            "include_images": include_images,
                            "img_scheme": img_scheme,
                            "img_loads": img_loads,
                            "img_disp": img_disp,
                            "img_fatigue": img_fatigue,
                            "label_spacing": label_spacing,
                            # Comparison Options
                            "comp_dist_fsi": comp_dist_fsi,
                            "comp_dist_loads": comp_dist_loads,
                            "comp_dist_bypass": comp_dist_bypass,
                            "comp_dist_disp": comp_dist_disp
                        }
                        
                        exporter = JoltExcelExporter(target_models, st.session_state.unit_system)
                        
                        for m in target_models:
                            exporter.export_model(m, options)
                            
                        if include_comparison and len(target_models) > 1:
                            exporter.export_comparison(target_models, options)
                            
                        excel_data = exporter.close()
                        
                        st.success("Report generated successfully!")
                        st.download_button(
                            label="⬇️ Download .xlsx",
                            data=excel_data,
                            file_name=f"JOLT_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())

    with c_preview:
        st.info("ℹ️ The Excel export generates a multi-sheet workbook compatible with certification standards.")
        st.markdown("""
        **Features included:**
        * **Standard Formatting:** Frozen headers, 3-decimal precision, aerospace units.
        * **Traceability:** Includes full input echo (Materials, Geometry, Fasteners).
        * **Analysis:** Full internal load distribution and fatigue factors.
        * **Visualization:** High-res snapshots of the JOLT diagrams.
        """)
