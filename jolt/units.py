"""Unit conversion logic for JOLT."""
from enum import Enum
from typing import Dict, Tuple

class UnitSystem(str, Enum):
    IMPERIAL = "Imperial"
    SI = "SI"

class UnitConverter:
    """Handles conversion between Imperial and SI units."""
    
    # Conversion factors (Imperial -> SI)
    # Length: in -> mm
    FACTOR_LENGTH = 25.4
    # Force: lb -> N
    FACTOR_FORCE = 4.4482216152605
    # Stress/Modulus: psi -> MPa
    FACTOR_STRESS = 0.006894757293168
    # Area: in^2 -> mm^2
    FACTOR_AREA = 645.16
    # Stiffness: lb/in -> N/mm
    # (lb -> N) / (in -> mm) = 4.448... / 25.4 = 0.175126835
    FACTOR_STIFFNESS = 0.17512683524647637

    @staticmethod
    def get_labels(system: UnitSystem) -> Dict[str, str]:
        if system == UnitSystem.SI:
            return {
                "length": "mm",
                "force": "N",
                "stress": "MPa",
                "area": "mm²",
                "stiffness": "N/mm",
            }
        else:
            return {
                "length": "in",
                "force": "lb",
                "stress": "psi",
                "area": "in²",
                "stiffness": "lb/in",
            }

    @classmethod
    def convert_length(cls, value: float, to_system: UnitSystem) -> float:
        if to_system == UnitSystem.SI:
            return value * cls.FACTOR_LENGTH
        return value / cls.FACTOR_LENGTH

    @classmethod
    def convert_force(cls, value: float, to_system: UnitSystem) -> float:
        if to_system == UnitSystem.SI:
            return value * cls.FACTOR_FORCE
        return value / cls.FACTOR_FORCE

    @classmethod
    def convert_stress(cls, value: float, to_system: UnitSystem) -> float:
        if to_system == UnitSystem.SI:
            return value * cls.FACTOR_STRESS
        return value / cls.FACTOR_STRESS

    @classmethod
    def convert_area(cls, value: float, to_system: UnitSystem) -> float:
        if to_system == UnitSystem.SI:
            return value * cls.FACTOR_AREA
        return value / cls.FACTOR_AREA

    @classmethod
    def convert_stiffness(cls, value: float, to_system: UnitSystem) -> float:
        if to_system == UnitSystem.SI:
            return value * cls.FACTOR_STIFFNESS
        return value / cls.FACTOR_STIFFNESS
