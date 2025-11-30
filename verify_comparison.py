import pandas as pd
import plotly.graph_objects as go
from jolt import JointConfiguration, JointSolution, Plate, FastenerRow
from jolt.ui.comparison import render_comparison_tab

# Mock Streamlit
import streamlit as st
from unittest.mock import MagicMock
st.header = MagicMock()
st.info = MagicMock()
st.warning = MagicMock()
st.error = MagicMock()
st.multiselect = MagicMock(return_value=["Model A (ID: 12345678)", "Model B (ID: 87654321)"])
st.selectbox = MagicMock(side_effect=["Model A", "All Elements"]) # Select Model A as ref, then "All Elements"
st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
st.checkbox = MagicMock(return_value=True) # Enable all charts
st.slider = MagicMock(return_value=5.0)
st.markdown = MagicMock()
st.subheader = MagicMock()
st.dataframe = MagicMock()
st.plotly_chart = MagicMock()
st.container = MagicMock()
st.caption = MagicMock()

def verify():
    # Create mock data
    sol_data = {
        "displacements": [0.0, 0.1],
        "stiffness_matrix": [],
        "force_vector": [],
        "fasteners": [{"row": 1, "force": 100.0, "plate_i": 0, "plate_j": 1, "compliance": 1e-6, "stiffness": 1e6, "dof_i": 0, "dof_j": 1}],
        "bearing_bypass": [],
        "nodes": [
            {"plate_name": "P1", "row": 1, "net_bypass": 50.0, "displacement": 0.01, "legacy_id": 1001, "plate_id": 0, "local_node": 0, "x": 0.0, "thickness": 0.1, "bypass_area": 0.1},
            {"plate_name": "P2", "row": 1, "net_bypass": 40.0, "displacement": 0.02, "legacy_id": 2001, "plate_id": 1, "local_node": 0, "x": 0.0, "thickness": 0.1, "bypass_area": 0.1}
        ],
        "bars": [],
        "reactions": [],
        "dof_map_items": [],
        "applied_forces": [],
        "fatigue_results": [
            {"plate_name": "P1", "row": 1, "fsi": 0.5, "node_id": 1001, "ktg": 1.0, "ktn": 1.0, "ktb": 1.0, "theta": 0.0, "ssf": 1.0, "bearing_load": 100.0, "bypass_load": 50.0, "sigma_ref": 1000.0, "term_bearing": 1.0, "term_bypass": 1.0},
            {"plate_name": "P2", "row": 1, "fsi": 0.6, "node_id": 2001, "ktg": 1.0, "ktn": 1.0, "ktb": 1.0, "theta": 0.0, "ssf": 1.0, "bearing_load": 100.0, "bypass_load": 40.0, "sigma_ref": 1000.0, "term_bearing": 1.0, "term_bypass": 1.0}
        ],
        "critical_points": [],
        "critical_node_id": 1001
    }
    
    m1 = JointConfiguration(
        pitches=[1.0], 
        plates=[Plate("P1", 10e6, 0.1, 1, 2, [0.1]), Plate("P2", 10e6, 0.1, 1, 2, [0.1])], 
        fasteners=[FastenerRow(1, 0.25, 10e6, 0.3)], 
        supports=[], 
        label="Model A", 
        model_id="12345678-1234-1234-1234-1234567890ab",
        results=sol_data
    )
    
    m2 = JointConfiguration(
        pitches=[1.0], 
        plates=[Plate("P1", 10e6, 0.1, 1, 2, [0.1]), Plate("P2", 10e6, 0.1, 1, 2, [0.1])], 
        fasteners=[FastenerRow(1, 0.25, 10e6, 0.3)], 
        supports=[], 
        label="Model B", 
        model_id="87654321-4321-4321-4321-ba0987654321",
        results=sol_data
    )
    
    try:
        render_comparison_tab([m1, m2], "Imperial")
        print("Verification SUCCESS: render_comparison_tab ran without error.")
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
