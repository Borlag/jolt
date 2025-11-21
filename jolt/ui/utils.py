"""Utility functions for the JOLT UI."""
from typing import List, Sequence, Tuple, Set
from jolt import Plate, FastenerRow


def plates_present_at_row(plates: Sequence[Plate], row_index: int) -> List[int]:
    present = [
        idx
        for idx, plate in enumerate(plates)
        if plate.first_row <= row_index <= plate.last_row
    ]
    present.sort()
    return present


def available_fastener_pairs(fastener: FastenerRow, plates: Sequence[Plate]) -> List[Tuple[int, int]]:
    row_index = int(fastener.row)
    present = plates_present_at_row(plates, row_index)
    return list(zip(present[:-1], present[1:]))


def resolve_fastener_connections(
    fastener: FastenerRow, plates: Sequence[Plate]
) -> List[Tuple[int, int]]:
    row_index = int(fastener.row)
    present = plates_present_at_row(plates, row_index)
    if len(present) < 2:
        return []
    if fastener.connections is None:
        return list(zip(present[:-1], present[1:]))

    order = {plate_idx: position for position, plate_idx in enumerate(present)}
    resolved: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    for pair in fastener.connections:
        if len(pair) != 2:
            continue
        raw_top, raw_bottom = int(pair[0]), int(pair[1])
        if raw_top not in order or raw_bottom not in order:
            continue
        top_idx, bottom_idx = (raw_top, raw_bottom)
        if order[top_idx] > order[bottom_idx]:
            top_idx, bottom_idx = bottom_idx, top_idx
        if abs(order[top_idx] - order[bottom_idx]) != 1:
            continue
        key = (top_idx, bottom_idx)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)

    return resolved
