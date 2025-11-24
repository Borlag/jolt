import json
from jolt import Joint1D, Plate, FastenerRow
from jolt import Joint1D, Plate, FastenerRow

def verify():
    with open("case1.json", "r") as f:
        data = json.load(f)
    
    # Manually reconstruct objects because JointConfiguration might be UI specific
    # Actually let's try to use JointConfiguration.from_dict if possible, but it's easier to just parse manually here to be safe and quick.
    
    pitches = data["pitches"]
    
    plates = []
    for p_data in data["plates"]:
        plates.append(Plate(
            name=p_data["name"],
            E=p_data["E"],
            t=p_data["t"],
            first_row=p_data["first_row"],
            last_row=p_data["last_row"],
            A_strip=p_data["A_strip"],
            Fx_left=p_data.get("Fx_left", 0.0),
            Fx_right=p_data.get("Fx_right", 0.0)
        ))
        
    fasteners = []
    for f_data in data["fasteners"]:
        fasteners.append(FastenerRow(
            row=f_data["row"],
            D=f_data["D"],
            Eb=f_data["Eb"],
            nu_b=f_data["nu_b"],
            method=f_data["method"],
            connections=f_data.get("connections")
        ))
        
    supports = [tuple(s) for s in data["supports"]]
    
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)
    
    # print("Fastener Results:")
    # print(f"{'Row':<5} {'Pair':<20} {'Stiffness':<15} {'Force':<15}")
    # print("-" * 60)
    
    # for res in solution.fasteners:
    #     p_i = plates[res.plate_i].name
    #     p_j = plates[res.plate_j].name
    #     pair_name = f"{p_i}-{p_j}"
    #     print(f"{res.row:<5} {pair_name:<20} {res.stiffness:<15.4e} {res.force:<15.2f}")

if __name__ == "__main__":
    verify()
