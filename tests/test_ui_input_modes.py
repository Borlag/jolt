import importlib.util
from pathlib import Path
import sys
import types

import pytest


# Provide a lightweight Streamlit stub so ui.state can import without optional GUI deps.
sys.modules.setdefault("streamlit", types.SimpleNamespace())


def _load_state_module():
    state_path = Path(__file__).resolve().parent.parent / "jolt" / "ui" / "state.py"
    spec = importlib.util.spec_from_file_location("ui_state_for_tests", state_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


state = _load_state_module()


def test_standard_mode_is_supported():
    assert "Standard" in state.SUPPORTED_INPUT_MODES


@pytest.mark.parametrize(
    "raw_value",
    ["Refined Row", "Node-based", "legacy", None, 123],
)
def test_normalize_input_mode_fallbacks(raw_value):
    assert state.normalize_input_mode(raw_value) == "Standard"


def test_normalize_input_mode_preserves_standard():
    assert state.normalize_input_mode("Standard") == "Standard"
