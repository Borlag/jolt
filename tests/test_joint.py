import pytest

from jolt import Joint1D, boeing69_compliance, figure76_example


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

    row4 = {item["Plate"]: (item["Bearing [lb]"], item["Bypass [lb]"]) for item in solution.bearing_bypass_as_dicts() if item["Row"] == 4}
    assert row4["Doubler"] == pytest.approx((-358.5903307098407, -60.41065420314609), rel=1e-12)
    assert row4["Skin"] == pytest.approx((-641.4096692901595, 0.0), rel=1e-12)

    tripler_left = next(item for item in solution.nodes_as_dicts() if item["Plate"] == "Tripler" and item["local_node"] == 0)
    assert tripler_left["u [in]"] == pytest.approx(0.00029931643381921725)
