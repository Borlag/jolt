"""Reference model configurations."""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

from .model import FastenerRow, Plate


def figure76_example() -> Tuple[List[float], List[Plate], List[FastenerRow], List[Tuple[int, int, float]]]:
    """Return the configuration shown in the original Figure 76 screenshot."""

    pitches = [1.128] * 7
    E_sheet = 1.05e7
    E_bolt = 1.04e7
    nu_bolt = 0.30
    diameter = 0.188
    plates = [
        Plate(name="Tripler", E=E_sheet, t=0.063, first_row=1, last_row=3, A_strip=[0.071, 0.071]),
        Plate(name="Doubler", E=E_sheet, t=0.040, first_row=1, last_row=7, A_strip=[0.045] * 6),
        Plate(name="Skin", E=E_sheet, t=0.040, first_row=1, last_row=7, A_strip=[0.045] * 6, Fx_left=1000.0),
    ]
    fasteners = [
        FastenerRow(row=row, D=diameter, Eb=E_bolt, nu_b=nu_bolt, method="Boeing69")
        for row in range(1, 8)
    ]
    supports = [
        (0, 2, 0.0),
        (1, 6, 0.0),
    ]
    return pitches, plates, fasteners, supports


def figure76_beam_idealized_example() -> Tuple[List[float], List[Plate], List[FastenerRow], List[Tuple[int, int, float]]]:
    """Return the configuration for the beam-idealized walkthrough."""

    pitches, plates, fasteners, supports = figure76_example()
    beam_plates = [
        replace(
            plate,
            Fx_left=0.0,
            Fx_right=1000.0 if plate.name == "Skin" else 0.0,
        )
        for plate in plates
    ]
    beam_fasteners = [replace(fastener) for fastener in fasteners]
    return pitches, beam_plates, beam_fasteners, list(supports)


__all__ = ["figure76_example", "figure76_beam_idealized_example"]
