"""Streamlit application for the JOLT 1D joint model."""
from __future__ import annotations

import streamlit as st

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional for tests
    plt = None  # type: ignore

from jolt import Joint1D
from jolt.ui import (
    initialize_session_state,
    render_sidebar,
    render_solution_tables,
    render_save_section,
)


st.set_page_config(page_title="JOLT 1D Joint", layout="wide")

# Initialize session state
initialize_session_state()

# Render sidebar
pitches, plates, fasteners, supports, point_forces = render_sidebar()

# Main content
st.title("JOLT 1D Joint — Bars + Springs")

if st.button("Solve", type="primary"):
    if not plates:
        st.error("No plates defined. Please check your configuration (Nodes/Elements).")
    else:
        try:
            model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
            solution = model.solve(supports=supports, point_forces=point_forces or None)
            st.session_state["solution"] = solution
        except (ValueError, RuntimeError) as exc:
            st.error(f"Analysis failed: {exc}")

if "solution" in st.session_state:
    solution = st.session_state["solution"]
    
    # Try importing Plotly renderer
    try:
        from jolt.visualization_plotly import render_joint_diagram_plotly
        has_plotly = True
    except ImportError:
        has_plotly = False

    if has_plotly:
        col_viz, col_ctrl = st.columns([4, 1])
        with col_ctrl:
            font_size = st.slider("Annotation Font Size", 8, 24, 10, key="viz_font_size")
        
        tabs = st.tabs(["Scheme", "Displacements", "Loads"])
        modes = ["scheme", "displacements", "loads"]
        for tab, mode in zip(tabs, modes):
            with tab:
                fig = render_joint_diagram_plotly(
                    pitches=pitches,
                    plates=plates,
                    fasteners=fasteners,
                    supports=supports,
                    solution=solution,
                    mode=mode,
                    font_size=font_size,
                )
                if fig:
                    # config={'editable': True} enables dragging annotations (text)
                    st.plotly_chart(fig, width="stretch", config={'editable': True, 'scrollZoom': True})
                else:
                    st.error("Failed to render Plotly figure.")
    else:
        st.warning("Plotly is not installed. Please install 'plotly' to view interactive diagrams.")

    render_solution_tables(solution)
    render_save_section(pitches, plates, fasteners, supports, point_forces)


elif not st.session_state.get("solution"):
    st.info("Соберите схему в левой панели и нажмите **Solve**. Для воспроизведения скрина используйте **Load ▶ JOLT Figure 76**.")
