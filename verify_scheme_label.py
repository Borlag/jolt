
from jolt import Joint1D, Plate, FastenerRow, JointSolution
from jolt.visualization_plotly import render_joint_diagram_plotly

def test_scheme_label():
    pitches = [1.0, 1.0, 1.0]
    plates = [
        Plate(name="P1", E=1e7, t=0.1, first_row=1, last_row=3, A_strip=[0.1, 0.1]),
        Plate(name="P2", E=1e7, t=0.1, first_row=1, last_row=3, A_strip=[0.1, 0.1]),
    ]
    # Fastener at row 2
    fasteners = [
        FastenerRow(row=2, D=0.2, Eb=1e7, nu_b=0.3)
    ]
    
    # Dummy solution (needed for the function signature, though scheme mode might not use it much)
    solution = JointSolution(
        displacements=[], stiffness_matrix=[], force_vector=[], 
        fasteners=[], bearing_bypass=[], nodes=[], bars=[], reactions=[], dof_map={}
    )
    
    supports = []
    
    fig = render_joint_diagram_plotly(pitches, plates, fasteners, supports, solution, mode="scheme")
    
    # Check annotations
    found = False
    for layout_name in fig.layout:
        if layout_name == "annotations":
            for annotation in fig.layout.annotations:
                if annotation.text == "F2":
                    found = True
                    print("PASS: Found annotation 'F2'")
                    break
    
    if not found:
        print("FAIL: Did not find annotation 'F2'")
        # Print all annotations to debug
        if hasattr(fig.layout, "annotations"):
            print("Annotations found:", [a.text for a in fig.layout.annotations])

if __name__ == "__main__":
    test_scheme_label()
