"""Reference model configurations."""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from .model import FastenerRow, Plate



def case_5_3_elements_example() -> Tuple[List[float], List[Plate], List[FastenerRow], List[Tuple[int, int, float]]]:
    """Return the Case 5.3 Elements configuration (based on Case_5_3_elements_row_a.json)."""

    pitches = [1.125] * 5
    E_sheet = 1.0e7
    E_bolt_1 = 1.03e7
    E_bolt_2 = 1.04e7
    nu_bolt = 0.30
    diameter = 0.188

    # Note: Indicies for connections: 0=Doubler, 1=Skin, 2=Strap
    doubler = Plate(
        name="Doubler",
        E=E_sheet,
        t=0.071,
        first_row=2,
        last_row=5,
        A_strip=[0.067, 0.067, 0.067],
    )
    skin = Plate(
        name="Skin",
        E=E_sheet,
        t=0.063,
        first_row=2,
        last_row=4,
        A_strip=[0.059, 0.059],
    )
    strap = Plate(
        name="Strap",
        E=E_sheet,
        t=0.063,
        first_row=1,
        last_row=3,
        A_strip=[0.059, 0.059],
        Fx_left=-1000.0,
    )

    plates = [doubler, skin, strap]

    fasteners = [
        FastenerRow(
            row=2,
            D=diameter,
            Eb=E_bolt_1,
            nu_b=nu_bolt,
            method="Boeing69",
            connections=[(0, 1), (1, 2)],
        ),
        FastenerRow(
            row=3,
            D=diameter,
            Eb=E_bolt_1,
            nu_b=nu_bolt,
            method="Boeing69",
            connections=[(0, 1), (1, 2)],
        ),
        FastenerRow(
            row=4,
            D=diameter,
            Eb=E_bolt_2,
            nu_b=nu_bolt,
            method="Boeing69",
            connections=[(0, 1)],
        ),
    ]

    supports = [
        (0, 3, 0.0), # Doubler (index 0), local node 3 (end of plate)
    ]

    return pitches, plates, fasteners, supports


__all__ = ["case_5_3_elements_example"]
