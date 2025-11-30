import sys
from unittest.mock import MagicMock

# Mock plotly and streamlit before importing module
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["plotly.subplots"] = MagicMock()
sys.modules["streamlit"] = MagicMock()

try:
    import jolt.ui.comparison
    print("Syntax check passed: jolt.ui.comparison imported successfully.")
except Exception as e:
    print(f"Syntax check FAILED: {e}")
    import traceback
    traceback.print_exc()
