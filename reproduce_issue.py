from jolt.model import Plate, FastenerRow, Joint1D

def reproduce():
    # Case 1 Configuration from User
    pitches = [1.128] * 6  # User JSON has 6 pitches
    E_sheet = 1.05e7
    E_bolt = 1.04e7
    nu_bolt = 0.30
    diameter = 0.188

    # Note: User JSON indices are 0-based for plates in connections/supports, 
    # but 1-based for rows in Plate definitions? 
    # Let's look at the JSON:
    # Tripler: first_row=3, last_row=6. 
    # Doubler: first_row=2, last_row=6.
    # Skin: first_row=1, last_row=5.
    # Fasteners at rows 2, 3, 4, 5.
    
    # Plate order in JSON: Tripler (0), Doubler (1), Skin (2)
    
    tripler = Plate(
        name="Tripler",
        E=E_sheet,
        t=0.063,
        first_row=3,
        last_row=6,
        A_strip=[0.071, 0.071, 0.071],
    )
    doubler = Plate(
        name="Doubler",
        E=E_sheet,
        t=0.040,
        first_row=2,
        last_row=6,
        A_strip=[0.045, 0.045, 0.045, 0.045],
    )
    skin = Plate(
        name="Skin",
        E=E_sheet,
        t=0.040,
        first_row=1,
        last_row=5,
        A_strip=[0.045, 0.045, 0.045, 0.045],
        Fx_left=1000.0,
    )

    plates = [tripler, doubler, skin]

    # Fasteners
    # Row 2: Connects Doubler(1) and Skin(2)
    # Row 3, 4, 5: Connects Tripler(0)-Doubler(1) and Doubler(1)-Skin(2)
    
    # Note: The current FastenerRow.connections expects plate indices.
    # In the JSON: 
    # Row 2 connections: [[1, 2]] -> Doubler-Skin
    # Row 3 connections: [[0, 1], [1, 2]] -> Tripler-Doubler, Doubler-Skin
    
    fasteners = []
    
    # Row 2
    fasteners.append(FastenerRow(
        row=2, D=diameter, Eb=E_bolt, nu_b=nu_bolt, method="Boeing69",
        connections=[(1, 2)]
    ))
    
    # Rows 3, 4, 5
    for r in [3, 4, 5]:
        fasteners.append(FastenerRow(
            row=r, D=diameter, Eb=E_bolt, nu_b=nu_bolt, method="Boeing69",
            connections=[(0, 1), (1, 2)]
        ))

    # Supports
    # Tripler (0) at local node 3 (end) -> Fixed
    # Doubler (1) at local node 4 (end) -> Fixed
    # Note: segment_count = last_row - first_row.
    # Tripler: 6-3 = 3 segments. Local nodes 0,1,2,3.
    # Doubler: 6-2 = 4 segments. Local nodes 0,1,2,3,4.
    supports = [
        (0, 3, 0.0), # Tripler end
        (1, 4, 0.0), # Doubler end
    ]

    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)
    
    tripler_reaction = next(r.reaction for r in solution.reactions if r.plate_name == "Tripler")
    doubler_reaction = next(r.reaction for r in solution.reactions if r.plate_name == "Doubler")
    
    print(f"TR: {tripler_reaction:.2f} (Target: ~570)")
    print(f"DR: {doubler_reaction:.2f} (Target: ~430)")
    
    # Check loads at fasteners if needed
    for f in solution.fasteners:
        print(f"Row {f.row} Force: {f.force:.2f}")

if __name__ == "__main__":
    reproduce()
