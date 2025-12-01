import pytest

from dataclasses import replace

from jolt import (
    Joint1D,
    Plate,
    FastenerRow,
    JointConfiguration,
    boeing69_compliance,
    figure76_example,
    load_joint_from_json,
    figure76_beam_idealized_example,
)
from jolt.config import fastener_from_dict


def test_boeing69_compliance_matches_reference_value():
    cf = boeing69_compliance(0.04, 1.05e7, 0.04, 1.05e7, 1.04e7, 0.30, 0.188)
    assert cf == pytest.approx(9.919919174397126e-06, rel=1e-9)
    assert 1.0 / cf == pytest.approx(100807.27296457776, rel=1e-9)


def test_figure76_solution_matches_expected_response():
    pitches, plates, fasteners, supports = figure76_example()
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)

    assert len(solution.displacements) == 14
    assert max(abs(u) for u in solution.displacements) == pytest.approx(0.008286685664326896)

    expected_fastener_forces = [
        270.5876215042927,
        253.4028932996332,
        286.86951216761577,
        166.13305439399576,
        209.90628522003996,
        156.15057452385952,
        232.63658110803945,
    ]
    computed_forces = [fastener["F [lb]"] for fastener in solution.fasteners_as_dicts()]
    assert computed_forces == pytest.approx(expected_fastener_forces)

    reactions = {
        (item["Plate"], item["Global node"]): item["Reaction [lb]"]
        for item in solution.reactions_as_dicts()
    }
    assert reactions[("Tripler", 6)] == pytest.approx(-575.6865222175309)
    assert reactions[("Doubler", 6)] == pytest.approx(-424.3134777825026)

    row4 = {
        item["Plate"]: (item["Bearing [lb]"], item["Bypass [lb]"])
        for item in solution.bearing_bypass_as_dicts()
        if item["Row"] == 4
    }
    assert row4["Doubler"] == pytest.approx((43.7732308260442, -304.0542403722854), rel=1e-12)
    assert row4["Skin"] == pytest.approx((-209.90628522003206, -442.5428663280849), rel=1e-12)

    tripler_left = next(
        item for item in solution.nodes_as_dicts() if item["Plate"] == "Tripler" and item["local_node"] == 0
    )
    assert tripler_left["u [in]"] == pytest.approx(0.0018892681552002413)


def test_fastener_from_dict_normalizes_method_names():
    base_data = {"row": 2, "D": 0.25, "Eb": 1.0e7, "nu_b": 0.3}

    manual = fastener_from_dict({**base_data, "method": "manual"})
    assert manual.method == "Manual"

    graphite = fastener_from_dict({**base_data, "method": "Huth-Graphite"})
    assert graphite.method == "Huth_graphite"

    defaulted = fastener_from_dict({**base_data, "method": "unknown"})
    assert defaulted.method == "Boeing69"


def test_figure76_beam_idealized_solution_matches_expected_response():
    pitches, plates, fasteners, supports = figure76_beam_idealized_example()
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)

    assert len(solution.displacements) == 14
    assert max(abs(u) for u in solution.displacements) == pytest.approx(0.004434798215029945)

    expected_fastener_forces = [
        67.94083852804458,
        117.86438668355679,
        159.57395791753444,
        172.69650425547712,
        268.1313060438172,
        307.86938710892196,
        504.3538975106004,
    ]
    computed_forces = [fastener["F [lb]"] for fastener in solution.fasteners_as_dicts()]
    assert computed_forces == pytest.approx(expected_fastener_forces)

    reactions = {
        (item["Plate"], item["Global node"]): item["Reaction [lb]"]
        for item in solution.reactions_as_dicts()
    }
    assert reactions[("Tripler", 6)] == pytest.approx(-598.4302780479945)
    assert reactions[("Doubler", 6)] == pytest.approx(-401.5697219520472)

    row4 = {
        item["Plate"]: (item["Bearing [lb]"], item["Bypass [lb]"])
        for item in solution.bearing_bypass_as_dicts()
        if item["Row"] == 4
    }
    assert row4["Doubler"] == pytest.approx((95.43480178834004, -109.65040976203721), rel=1e-12)
    assert row4["Skin"] == pytest.approx((-268.1313060438097, 227.51479644558472), rel=1e-12)

    tripler_left = next(
        item for item in solution.nodes_as_dicts() if item["Plate"] == "Tripler" and item["local_node"] == 0
    )
    assert tripler_left["u [in]"] == pytest.approx(0.0015234514645156728)


def test_fastener_custom_interfaces():
    pitches = [1.0, 1.0]
    plates = [
        Plate(name="Top", E=1.0e7, t=0.05, first_row=1, last_row=3, A_strip=[0.05, 0.05]),
        Plate(name="Middle", E=1.0e7, t=0.05, first_row=1, last_row=3, A_strip=[0.05, 0.05]),
        Plate(name="Bottom", E=1.0e7, t=0.05, first_row=1, last_row=3, A_strip=[0.05, 0.05], Fx_right=100.0),
    ]
    fastener = FastenerRow(row=2, D=0.25, Eb=1.0e7, nu_b=0.3, method="Manual", k_manual=1.0e5)
    supports = [(0, 0, 0.0), (1, 0, 0.0), (2, 0, 0.0)]

    model_default = Joint1D(pitches=pitches, plates=plates, fasteners=[fastener])
    solution_default = model_default.solve(supports=supports)
    interfaces_default = sorted((item.plate_i, item.plate_j) for item in solution_default.fasteners)
    assert interfaces_default == [(0, 1), (1, 2)]

    fastener_upper_only = FastenerRow(
        row=2,
        D=0.25,
        Eb=1.0e7,
        nu_b=0.3,
        method="Manual",
        k_manual=1.0e5,
        connections=[(0, 1)],
    )
    model_upper = Joint1D(pitches=pitches, plates=plates, fasteners=[fastener_upper_only])
    solution_upper = model_upper.solve(supports=supports)
    assert [(item.plate_i, item.plate_j) for item in solution_upper.fasteners] == [(0, 1)]

    fastener_lower_only = FastenerRow(
        row=2,
        D=0.25,
        Eb=1.0e7,
        nu_b=0.3,
        method="Manual",
        k_manual=1.0e5,
        connections=[(1, 2)],
    )
    model_lower = Joint1D(pitches=pitches, plates=plates, fasteners=[fastener_lower_only])
    solution_lower = model_lower.solve(supports=supports)
    assert [(item.plate_i, item.plate_j) for item in solution_lower.fasteners] == [(1, 2)]

    fastener_invalid = FastenerRow(
        row=1,
        D=0.25,
        Eb=1.0e7,
        nu_b=0.3,
        method="Manual",
        k_manual=1.0e5,
        connections=[(0, 2)],
    )
    model_invalid = Joint1D(pitches=pitches, plates=plates, fasteners=[fastener_invalid])
    with pytest.raises(ValueError):
        model_invalid.solve(supports=supports)


def test_joint_configuration_roundtrip(tmp_path):
    pitches, plates, fasteners, supports = figure76_example()
    config = JointConfiguration(
        pitches=list(pitches),
        plates=[replace(plate) for plate in plates],
        fasteners=[replace(fastener) for fastener in fasteners],
        supports=list(supports),
        label="Figure 76",
        unloading="Reference",
    )
    json_path = tmp_path / "figure76.json"
    config.save(json_path)

    loaded_model, loaded_supports, loaded_point_forces, loaded_config = load_joint_from_json(json_path)
    assert loaded_config.label == "Figure 76"
    assert loaded_supports == list(supports)
    assert loaded_point_forces == []

    base_solution = Joint1D(pitches, plates, fasteners).solve(supports=supports)
    loaded_solution = loaded_model.solve(supports=loaded_supports, point_forces=loaded_point_forces or None)
    assert loaded_solution.displacements == pytest.approx(base_solution.displacements)
    assert [fast.force for fast in loaded_solution.fasteners] == pytest.approx(
        [fast.force for fast in base_solution.fasteners]
    )


def test_stiffer_plate_carries_more_load():
    pitches = [1.0, 1.0]
    plates = [
        Plate(name="soft", E=1.0e6, t=0.1, first_row=1, last_row=3, A_strip=[0.1, 0.1]),
        Plate(name="stiff", E=1.0e7, t=0.1, first_row=1, last_row=3, A_strip=[0.1, 0.1], Fx_right=1000.0),
    ]
    fasteners = [
        FastenerRow(row=1, D=0.25, Eb=1.0e7, nu_b=0.3, method="Manual", k_manual=1.0e5),
        FastenerRow(row=2, D=0.25, Eb=1.0e7, nu_b=0.3, method="Manual", k_manual=1.0e5),
        FastenerRow(row=3, D=0.25, Eb=1.0e7, nu_b=0.3, method="Manual", k_manual=1.0e5),
    ]
    supports = [(0, 0, 0.0), (1, 0, 0.0), (0, 2, 0.0)]

    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)

    soft_forces = [abs(bar.axial_force) for bar in solution.bars if bar.plate_name == "soft"]
    stiff_forces = [abs(bar.axial_force) for bar in solution.bars if bar.plate_name == "stiff"]
    assert stiff_forces and soft_forces
    assert max(stiff_forces) > max(soft_forces)
    assert any(abs(fast.force) > 1e-6 for fast in solution.fasteners)
