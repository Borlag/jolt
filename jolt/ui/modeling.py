"""Lightweight graphical modeling scaffolding for the Streamlit UI.

This module provides a click-first grid experience that mirrors the existing
model primitives. The initial version focuses on node toggling and grid sizing
so the team can iterate toward the full interaction model described in
``docs/graphical-modeling.md`` without blocking the main app.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Set, Tuple

import streamlit as st

from jolt import Plate, case_5_3_elements_example
from .state import normalize_input_method


def _layout_from_primitives() -> Dict[str, Any]:
    """Seed the graphical layout from the current configuration."""

    pitches = st.session_state.get("pitches", [])
    plates: List[Plate] = list(st.session_state.get("plates", []))

    stations = len(pitches) or 1
    layout_nodes: Set[Tuple[int, int]] = set()

    for plate_idx, plate in enumerate(plates):
        start = max(0, int(plate.first_row) - 1)
        end = max(start, int(plate.last_row) - 1)
        for station in range(start, min(end + 1, stations)):
            layout_nodes.add((plate_idx, station))

    if not layout_nodes:
        layout_nodes.add((0, 0))

    return {
        "plates": max(1, len(plates)),
        "stations": stations,
        "nodes": sorted(layout_nodes),
    }


def _ensure_graphical_layout() -> None:
    """Guarantee layout metadata exists and is internally consistent."""

    if "graphical_layout" not in st.session_state:
        st.session_state.graphical_layout = _layout_from_primitives()

    layout = st.session_state.graphical_layout
    layout.setdefault("plates", max(1, len(st.session_state.get("plates", []))))
    layout.setdefault("stations", max(1, len(st.session_state.get("pitches", []))))
    layout.setdefault("nodes", [(0, 0)])

    st.session_state.graphical_layout = layout


def _sync_pitches(stations: int) -> None:
    """Resize pitch array to match the requested station count."""

    pitches: List[float] = list(st.session_state.get("pitches", []))
    if not pitches:
        # Seed from example if nothing exists
        pitches = list(case_5_3_elements_example()[0])

    if stations > len(pitches):
        pitches.extend([pitches[-1]] * (stations - len(pitches)))
    elif stations < len(pitches):
        pitches = pitches[:stations]

    st.session_state.pitches = pitches
    st.session_state["n_rows"] = len(pitches)


def _sync_plate_count(target: int, stations: int) -> None:
    """Grow/shrink the plate list to match the grid."""

    plates: List[Plate] = list(st.session_state.get("plates", []))
    if not plates:
        _, plates, _, _ = case_5_3_elements_example()

    current = len(plates)
    if target > current:
        template = plates[-1]
        for idx in range(current, target):
            first_row = min(max(int(template.first_row), 1), stations)
            last_row = min(max(int(template.last_row), first_row), stations)
            plates.append(
                replace(
                    template,
                    name=f"Layer {idx + 1}",
                    first_row=first_row,
                    last_row=last_row,
                )
            )
    elif target < current:
        plates = plates[:target]

    st.session_state.plates = plates


def _prune_nodes(nodes: Iterable[Tuple[int, int]], plates: int, stations: int) -> List[Tuple[int, int]]:
    return sorted((p, r) for p, r in nodes if p < plates and r < stations)


def _toggle_node(p: int, r: int) -> None:
    layout = st.session_state.graphical_layout
    nodes = set(tuple(item) for item in layout.get("nodes", []))

    if (p, r) in nodes:
        nodes.remove((p, r))
    else:
        nodes.add((p, r))

    layout["nodes"] = sorted(nodes)
    st.session_state.graphical_layout = layout


def render_modeling_tab(units: Dict[str, str]) -> None:  # pragma: no cover - Streamlit UI
    """Render the graphical modeling tab.

    The tab offers a grid of clickable nodes. It intentionally focuses on
    high-signal primitives (grid sizing + node toggling) to lay the groundwork
    for the richer behaviors outlined in the design document.
    """

    _ensure_graphical_layout()
    layout = st.session_state.graphical_layout
    current_method = normalize_input_method(st.session_state.get("input_method", "Standard"))

    st.caption("Use the grid to place nodes before defining plates/fasteners along the lines.")
    if current_method != "Graphical":
        st.info("Switch the sidebar Method to Graphical to manage the model here.")

    c1, c2 = st.columns(2)
    plates_count = int(
        c1.number_input("Plate lines", 1, 50, int(layout.get("plates", 1)), key="graphical_plate_lines")
    )
    stations_count = int(
        c2.number_input("Stations", 1, 50, int(layout.get("stations", 1)), key="graphical_stations")
    )

    # Keep primitives in sync with the grid size so Solve keeps working.
    if plates_count != layout.get("plates") or stations_count != layout.get("stations"):
        _sync_pitches(stations_count)
        _sync_plate_count(plates_count, stations_count)
        layout["plates"] = plates_count
        layout["stations"] = stations_count
        layout["nodes"] = _prune_nodes(layout.get("nodes", []), plates_count, stations_count)
        st.session_state.graphical_layout = layout

    nodes = set(tuple(n) for n in st.session_state.graphical_layout.get("nodes", []))

    st.write("Click a node to toggle it on/off:")
    grid = st.container()
    for plate_idx in range(plates_count):
        cols = grid.columns(stations_count)
        for station_idx in range(stations_count):
            active = (plate_idx, station_idx) in nodes
            label = "●" if active else "○"
            help_text = f"Plate {plate_idx + 1}, Station {station_idx + 1}"
            if cols[station_idx].button(label, key=f"node_{plate_idx}_{station_idx}", help=help_text):
                _toggle_node(plate_idx, station_idx)
                nodes.symmetric_difference_update({(plate_idx, station_idx)})

    st.write(
        f"Active nodes: {len(st.session_state.graphical_layout.get('nodes', []))}"
    )
    st.json(st.session_state.graphical_layout)
