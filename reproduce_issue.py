from jolt.model import Plate, FastenerRow, Joint1D

def reproduce():
    pitches = [1.128] * 5
    E_sheet = 1.05e7
    E_bolt = 1.04e7
    nu_bolt = 0.30
    diameter = 0.188

    skin = Plate(
        name="Skin",
        E=E_sheet,
        t=0.040,
        first_row=1,
        last_row=5,
        A_strip=[0.045, 0.045, 0.045, 0.045],
        Fx_left=1000.0,
    )
    doubler = Plate(
        name="Doubler",
        E=E_sheet,
        t=0.040,
        first_row=2,
        last_row=6,
        A_strip=[0.045, 0.045, 0.045, 0.045],
    )
    tripler = Plate(
        name="Tripler",
        E=E_sheet,
        t=0.063,
        first_row=3,
        last_row=6,
        A_strip=[0.071, 0.071, 0.071],
    )

    # SWAPPED ORDER: Doubler, Tripler, Skin
    # Original was: Tripler, Doubler, Skin
    plates = [doubler, tripler, skin]

    fasteners = [
        FastenerRow(row=row, D=diameter, Eb=E_bolt, nu_b=nu_bolt, method="Boeing69")
        for row in range(2, 6)
    ]

    supports = [
        (plates.index(tripler), tripler.segment_count(), 0.0),
        (plates.index(doubler), doubler.segment_count(), 0.0),
    ]

    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)
    
    tripler_reaction = next(r.reaction for r in solution.reactions if r.plate_name == "Tripler")
    doubler_reaction = next(r.reaction for r in solution.reactions if r.plate_name == "Doubler")
    f1_load = solution.fasteners[0].force
    f2_load = solution.fasteners[1].force
    f3_load = solution.fasteners[2].force
    
    print(f"TR: {tripler_reaction}")
    print(f"DR: {doubler_reaction}")
    print(f"F1: {f1_load}")
    print(f"F2: {f2_load}")
    print(f"F3: {f3_load}")

if __name__ == "__main__":
    reproduce()
