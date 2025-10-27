import pytest

from jolt import Joint1D, Plate, FastenerRow, boeing69_compliance, figure76_example


def test_boeing69_compliance_matches_reference_value():
    cf = boeing69_compliance(0.04, 1.05e7, 0.04, 1.05e7, 1.04e7, 0.30, 0.188)
    assert cf == pytest.approx(9.919919174397126e-06, rel=1e-9)
    assert 1.0 / cf == pytest.approx(100807.27296457776, rel=1e-9)


def test_figure76_solution_matches_expected_response():
    pitches, plates, fasteners, supports = figure76_example()
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)

    assert len(solution.displacements) == 14
    assert max(abs(u) for u in solution.displacements) == pytest.approx(0.005474220733198422)

    expected_fastener_forces = [
        -56.401704454461,
        -85.01609588915109,
        -156.76187616308238,
        -358.5903307098406,
        -218.7686399729117,
        -184.24351713687625,
        -238.3975121803719,
    ]
    computed_forces = [fastener["F [lb]"] for fastener in solution.fasteners_as_dicts()]
    assert computed_forces == pytest.approx(expected_fastener_forces)

    reactions = {(item["Plate"], item["Global node"]): item["Reaction [lb]"] for item in solution.reactions_as_dicts()}
    assert reactions[("Tripler", 3)] == pytest.approx(-298.1796765066945)
    assert reactions[("Doubler", 7)] == pytest.approx(-701.8203234933058)

    row4 = {item["Plate"]: (item["Bearing [lb]"], item["Bypass [lb]"]) for item in solution.bearing_bypass_as_dicts() if item["Row"] == 4}
    assert row4["Doubler"] == pytest.approx((-358.5903307098407, -60.41065420314609), rel=1e-12)
    assert row4["Skin"] == pytest.approx((-641.4096692901595, 0.0), rel=1e-12)

    tripler_left = next(item for item in solution.nodes_as_dicts() if item["Plate"] == "Tripler" and item["local_node"] == 0)
    assert tripler_left["u [in]"] == pytest.approx(0.00029931643381921725)


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
    interfaces_default = sorted(item.interface for item in solution_default.fasteners)
    assert interfaces_default == ["Middle-Bottom", "Top-Middle"]

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
    assert [item.interface for item in solution_upper.fasteners] == ["Top-Middle"]

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
    assert [item.interface for item in solution_lower.fasteners] == ["Middle-Bottom"]

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
