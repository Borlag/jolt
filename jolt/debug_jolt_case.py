"""Debug harness for comparing the assembled system against Boeing JOLT."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from jolt.model import FastenerRow, Joint1D, Plate


# Hard-coded configuration for Case_4_4_elements
CASE_4_4_ELEMENTS = {
    "label": "Case_4_4_elements",
    "unloading": "",
    "pitches": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "plates": [
        {
            "name": "sheartie",
            "E": 10300000.0,
            "t": 0.063,
            "first_row": 2,
            "last_row": 5,
            "A_strip": [0.063, 0.063, 0.063],
            "Fx_left": 0.0,
            "Fx_right": 0.0,
        },
        {
            "name": "Tearstap",
            "E": 10300000.0,
            "t": 0.071,
            "first_row": 1,
            "last_row": 6,
            "A_strip": [0.071, 0.071, 0.071, 0.071, 0.071],
            "Fx_left": 0.0,
            "Fx_right": -710.0,
        },
        {
            "name": "Skin",
            "E": 10300000.0,
            "t": 0.056,
            "first_row": 1,
            "last_row": 6,
            "A_strip": [0.056, 0.0, 0.056, 0.056, 0.056],
            "Fx_left": 0.0,
            "Fx_right": -560.0,
        },
        {
            "name": "Doubler",
            "E": 10300000.0,
            "t": 0.063,
            "first_row": 1,
            "last_row": 5,
            "A_strip": [0.063, 0.063, 0.063, 0.063],
            "Fx_left": 0.0,
            "Fx_right": 0.0,
        },
    ],
    "fasteners": [
        {
            "row": 2,
            "D": 0.188,
            "Eb": 10300000.0,
            "nu_b": 0.3,
            "method": "Boeing69",
            "connections": [[0, 1], [1, 2], [2, 3]],
        },
        {
            "row": 3,
            "D": 0.188,
            "Eb": 10300000.0,
            "nu_b": 0.3,
            "method": "Boeing69",
            "connections": [[0, 1], [1, 2], [2, 3]],
        },
        {
            "row": 4,
            "D": 0.188,
            "Eb": 10300000.0,
            "nu_b": 0.3,
            "method": "Boeing69",
            "connections": [[0, 1], [1, 2], [2, 3]],
        },
        {
            "row": 5,
            "D": 0.188,
            "Eb": 10300000.0,
            "nu_b": 0.3,
            "method": "Boeing69",
            "connections": [[0, 1], [1, 2], [2, 3]],
        },
    ],
    "supports": [
        [1, 0, 0.0],
        [2, 0, 0.0],
        [3, 0, 0.0],
    ],
}


def _build_model(case_data: Dict) -> Tuple[Joint1D, List[Tuple[int, int, float]]]:
    plates = [
        Plate(
            name=p["name"],
            E=p["E"],
            t=p["t"],
            first_row=p["first_row"],
            last_row=p["last_row"],
            A_strip=p["A_strip"],
            Fx_left=p.get("Fx_left", 0.0),
            Fx_right=p.get("Fx_right", 0.0),
        )
        for p in case_data["plates"]
    ]

    fasteners = [
        FastenerRow(
            row=f["row"],
            D=f["D"],
            Eb=f["Eb"],
            nu_b=f.get("nu_b", 0.3),
            method=f.get("method", "Boeing69"),
            connections=f.get("connections"),
        )
        for f in case_data["fasteners"]
    ]

    joint = Joint1D(case_data["pitches"], plates, fasteners)
    supports = [(int(s[0]), int(s[1]), float(s[2])) for s in case_data["supports"]]
    return joint, supports


def _write_matrix(path: Path, matrix: List[List[float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def _write_vector(path: Path, vector: List[float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for value in vector:
            writer.writerow([value])


def main() -> None:
    joint, supports = _build_model(CASE_4_4_ELEMENTS)
    K, F, dof_map = joint.debug_system(supports)
    solution = joint.solve(supports)

    output_dir = Path(__file__).parent / "debug_outputs"
    output_dir.mkdir(exist_ok=True)

    _write_matrix(output_dir / "Case_4_4_elements_K.csv", K)
    _write_vector(output_dir / "Case_4_4_elements_F.csv", F)
    displacements = getattr(solution, "full_displacements", solution.displacements)
    _write_vector(output_dir / "Case_4_4_elements_u.csv", displacements)

    # Results
    with (output_dir / "Case_4_4_elements_fasteners.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solution.fasteners[0].as_dict().keys())
        writer.writeheader()
        for fast in solution.fasteners:
            writer.writerow(fast.as_dict())

    with (output_dir / "Case_4_4_elements_bearing_bypass.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solution.bearing_bypass[0].as_dict().keys())
        writer.writeheader()
        for bb in solution.bearing_bypass:
            writer.writerow(bb.as_dict())

    with (output_dir / "Case_4_4_elements_reactions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solution.reactions[0].as_dict().keys())
        writer.writeheader()
        for rxn in solution.reactions:
            writer.writerow(rxn.as_dict())

    with (output_dir / "Case_4_4_elements_nodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solution.nodes[0].as_dict().keys())
        writer.writeheader()
        for node in solution.nodes:
            writer.writerow(node.as_dict())

    with (output_dir / "Case_4_4_elements_bars.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solution.bars[0].as_dict().keys())
        writer.writeheader()
        for bar in solution.bars:
            writer.writerow(bar.as_dict())

    print("DOF map (dof -> (plate, node)):")
    for key, value in sorted(dof_map.items(), key=lambda kv: kv[1]):
        print(f"  {value}: {key}")

    print(f"\nMatrices and result tables written to {output_dir}")


if __name__ == "__main__":
    main()
