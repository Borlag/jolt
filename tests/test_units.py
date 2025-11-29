import unittest
from dataclasses import replace
from jolt.units import UnitSystem, UnitConverter
from jolt.config import JointConfiguration
from jolt import Plate, FastenerRow

class TestUnitConverter(unittest.TestCase):
    def test_length_conversion(self):
        # 1 inch = 25.4 mm
        self.assertAlmostEqual(UnitConverter.convert_length(1.0, UnitSystem.SI), 25.4)
        self.assertAlmostEqual(UnitConverter.convert_length(25.4, UnitSystem.IMPERIAL), 1.0)

    def test_force_conversion(self):
        # 1 lb = 4.44822 N
        self.assertAlmostEqual(UnitConverter.convert_force(1.0, UnitSystem.SI), 4.4482216, places=5)
        self.assertAlmostEqual(UnitConverter.convert_force(4.4482216, UnitSystem.IMPERIAL), 1.0, places=5)

    def test_stress_conversion(self):
        # 1 psi = 0.00689476 MPa
        self.assertAlmostEqual(UnitConverter.convert_stress(1000.0, UnitSystem.SI), 6.894757, places=4)
        self.assertAlmostEqual(UnitConverter.convert_stress(6.894757, UnitSystem.IMPERIAL), 1000.0, places=4)

    def test_area_conversion(self):
        # 1 in^2 = 645.16 mm^2
        self.assertAlmostEqual(UnitConverter.convert_area(1.0, UnitSystem.SI), 645.16)
        self.assertAlmostEqual(UnitConverter.convert_area(645.16, UnitSystem.IMPERIAL), 1.0)

    def test_stiffness_conversion(self):
        # 1 lb/in = 0.175127 N/mm
        self.assertAlmostEqual(UnitConverter.convert_stiffness(1000.0, UnitSystem.SI), 175.1268, places=3)
        self.assertAlmostEqual(UnitConverter.convert_stiffness(175.1268, UnitSystem.IMPERIAL), 1000.0, places=3)

class TestConfigSerialization(unittest.TestCase):
    def test_units_persistence(self):
        config = JointConfiguration(
            pitches=[1.0],
            plates=[],
            fasteners=[],
            supports=[],
            units="SI"
        )
        data = config.to_dict()
        self.assertEqual(data["units"], "SI")
        
        new_config = JointConfiguration.from_dict(data)
        self.assertEqual(new_config.units, "SI")
        
        # Default check
        config_def = JointConfiguration(pitches=[], plates=[], fasteners=[], supports=[])
        self.assertEqual(config_def.units, "Imperial")
        data_def = config_def.to_dict()
        self.assertEqual(data_def["units"], "Imperial")

if __name__ == '__main__':
    unittest.main()
