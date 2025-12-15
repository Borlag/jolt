from pathlib import Path
import sys
import types
import importlib.util

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


def test_supported_methods():
    assert "Graphical" in state.SUPPORTED_INPUT_METHODS
    assert "Standard" in state.SUPPORTED_INPUT_METHODS


@pytest.mark.parametrize(
    "raw_value",
    ["legacy", None, 123, "Refined"],
)
def test_normalize_input_method_fallbacks(raw_value):
    assert state.normalize_input_method(raw_value) == "Standard"


def test_normalize_input_method_preserves_valid():
    for method in state.SUPPORTED_INPUT_METHODS:
        assert state.normalize_input_method(method) == method
