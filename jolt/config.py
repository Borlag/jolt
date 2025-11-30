"""Serialization helpers for joint configurations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, IO, List, Optional, Sequence, Tuple, Union

from .model import FastenerRow, Joint1D, Plate

_DEFAULT_FASTENER_METHOD = "Boeing69"
_FASTENER_METHOD_ALIASES = {
    "boeing69": "Boeing69",
    "boeing_69": "Boeing69",
    "boeing-69": "Boeing69",
    "boeing": "Boeing69",
    "huth_metal": "Huth_metal",
    "huth-metal": "Huth_metal",
    "huthmetal": "Huth_metal",
    "huth_graphite": "Huth_graphite",
    "huth-graphite": "Huth_graphite",
    "huthgraphite": "Huth_graphite",
    "grumman": "Grumman",
    "manual": "Manual",
}


def _normalize_fastener_method(value: Any) -> str:
    """Return a canonical fastener method name.

    Saved configurations created by older versions of the app (or edited by hand)
    may store method identifiers that do not exactly match the keys used by the
    UI.  Falling back to a known method avoids hard failures when loading such
    data.  The Boeing 69 formulation is used as a conservative default when the
    value cannot be resolved.
    """

    if not isinstance(value, str):
        return _DEFAULT_FASTENER_METHOD

    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    return _FASTENER_METHOD_ALIASES.get(normalized, _DEFAULT_FASTENER_METHOD)

Supports = List[Tuple[int, int, float]]
PointForces = List[Tuple[int, int, float]]
_JSONSource = Union[str, Path, IO[str]]


def _to_float_list(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values]


def plate_to_dict(plate: Plate) -> Dict[str, Any]:
    """Serialize a :class:`Plate` instance to a JSON-compatible dictionary."""

    data = {
        "name": plate.name,
        "E": plate.E,
        "t": plate.t,
        "first_row": plate.first_row,
        "last_row": plate.last_row,
        "A_strip": list(plate.A_strip),
        "Fx_left": plate.Fx_left,
        "Fx_right": plate.Fx_right,
        "fatigue_strength": getattr(plate, "fatigue_strength", None),
    }
    return data


def fastener_to_dict(fastener: FastenerRow) -> Dict[str, Any]:
    """Serialize a :class:`FastenerRow` to a JSON-compatible dictionary."""

    data: Dict[str, Any] = {
        "row": fastener.row,
        "D": fastener.D,
        "Eb": fastener.Eb,
        "nu_b": fastener.nu_b,
        "method": fastener.method,
        # New fields
        "name": fastener.name,
        "marker_symbol": fastener.marker_symbol,
    }
    if fastener.k_manual is not None:
        data["k_manual"] = fastener.k_manual
    if fastener.connections is not None:
        data["connections"] = [list(pair) for pair in fastener.connections]
    return data


def plate_from_dict(data: Dict[str, Any]) -> Plate:
    """Create a :class:`Plate` instance from JSON data."""
    
    fs = data.get("fatigue_strength")
    return Plate(
        name=str(data.get("name", "Plate")),
        E=float(data.get("E", 0.0)),
        t=float(data.get("t", 0.0)),
        first_row=int(data.get("first_row", 1)),
        last_row=int(data.get("last_row", 1)),
        A_strip=_to_float_list(data.get("A_strip", [])),
        Fx_left=float(data.get("Fx_left", 0.0)),
        Fx_right=float(data.get("Fx_right", 0.0)),
        fatigue_strength=float(fs) if fs is not None else None
    )


def fastener_from_dict(data: Dict[str, Any]) -> FastenerRow:
    """Create a :class:`FastenerRow` instance from JSON data."""

    connections_raw = data.get("connections")
    connections: Optional[List[Tuple[int, int]]] = None
    if isinstance(connections_raw, Sequence):
        resolved: List[Tuple[int, int]] = []
        for item in connections_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            resolved.append((int(item[0]), int(item[1])))
        if resolved:
            connections = resolved

    k_manual = data.get("k_manual")
    method = _normalize_fastener_method(data.get("method", _DEFAULT_FASTENER_METHOD))

    # Load new fields with safe defaults
    name = str(data.get("name", ""))
    marker_symbol = str(data.get("marker_symbol", "circle"))

    return FastenerRow(
        row=int(data.get("row", 1)),
        D=float(data.get("D", 0.0)),
        Eb=float(data.get("Eb", 0.0)),
        nu_b=float(data.get("nu_b", 0.0)),
        method=method,
        k_manual=float(k_manual) if k_manual is not None else None,
        connections=connections,
        name=name,
        marker_symbol=marker_symbol,
    )


def _supports_from_data(entries: Iterable[Any]) -> Supports:
    supports: Supports = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            continue
        plate_idx, local_node, value = entry
        supports.append((int(plate_idx), int(local_node), float(value)))
    return supports


def _point_forces_from_data(entries: Iterable[Any]) -> PointForces:
    forces: PointForces = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            continue
        plate_idx, local_node, magnitude = entry
        forces.append((int(plate_idx), int(local_node), float(magnitude)))
    return forces


@dataclass
class JointConfiguration:
    """Container for a full joint definition."""

    pitches: List[float]
    plates: List[Plate]
    fasteners: List[FastenerRow]
    supports: Supports
    point_forces: PointForces = field(default_factory=list)
    label: str = ""
    unloading: str = ""
    description: str = ""
    units: str = "Imperial"

    def build_model(self) -> Joint1D:
        """Create a :class:`Joint1D` model for this configuration."""

        return Joint1D(pitches=self.pitches, plates=self.plates, fasteners=self.fasteners)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representing the configuration."""

        payload: Dict[str, Any] = {
            "label": self.label,
            "unloading": self.unloading,
            "description": self.description,
            "units": self.units,
            "pitches": list(self.pitches),
            "plates": [plate_to_dict(plate) for plate in self.plates],
            "fasteners": [fastener_to_dict(fastener) for fastener in self.fasteners],
            "supports": [list(item) for item in self.supports],
        }
        if self.point_forces:
            payload["point_forces"] = [list(item) for item in self.point_forces]
        return payload

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize the configuration to a JSON string."""

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JointConfiguration":
        """Create a configuration from a dictionary."""

        pitches = _to_float_list(data.get("pitches", []))
        plates = [plate_from_dict(item) for item in data.get("plates", [])]
        fasteners = [fastener_from_dict(item) for item in data.get("fasteners", [])]
        supports = _supports_from_data(data.get("supports", []))
        point_forces = _point_forces_from_data(data.get("point_forces", []))
        label = str(data.get("label", ""))
        unloading = str(data.get("unloading", ""))
        description = str(data.get("description", ""))
        units = str(data.get("units", "Imperial"))
        return cls(
            pitches=pitches,
            plates=plates,
            fasteners=fasteners,
            supports=supports,
            point_forces=point_forces,
            label=label,
            unloading=unloading,
            description=description,
            units=units,
        )

    @classmethod
    def from_json(cls, source: _JSONSource) -> "JointConfiguration":
        """Load a configuration from a JSON file path or file-like object."""

        if hasattr(source, "read"):
            data = json.load(source)  # type: ignore[arg-type]
        else:
            path = Path(source)
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Joint configuration JSON must contain an object at the top level")
        return cls.from_dict(data)

    def save(self, target: Union[str, Path, IO[str]], *, indent: Optional[int] = 2) -> None:
        """Write the configuration to disk or a file-like object."""

        payload = self.to_json(indent=indent)
        if hasattr(target, "write"):
            target.write(payload)  # type: ignore[arg-type]
        else:
            path = Path(target)
            path.write_text(payload, encoding="utf-8")


def load_joint_from_json(source: _JSONSource) -> Tuple[Joint1D, Supports, PointForces, JointConfiguration]:
    """Load a :class:`Joint1D` model and associated data from JSON."""

    config = JointConfiguration.from_json(source)
    model = config.build_model()
    return model, config.supports, config.point_forces, config


__all__ = [
    "JointConfiguration",
    "fastener_from_dict",
    "fastener_to_dict",
    "load_joint_from_json",
    "plate_from_dict",
    "plate_to_dict",
]
