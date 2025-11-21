
from jolt import Joint1D, Plate, FastenerRow

def run_case(t_tripler, t_doubler, case_name):
    print(f"\n--- {case_name} ---")
    print(f"Tripler t={t_tripler}, Doubler t={t_doubler}")
    
    # Geometry
    pitches = [1.75] * 5
    
    # Plates
    # Skin: 0.429 (eff 0.344 for fasteners)
    skin = Plate(name="Skin", E=1.03e7, t=0.429, first_row=1, last_row=6, A_strip=[0.695]*5, Fx_right=-728.0)
    
    # Doubler
    doubler = Plate(name="Doubler", E=1.03e7, t=t_doubler, first_row=2, last_row=6, A_strip=[0.26]*4, Fx_right=-272.0)
    
    # Tripler
    tripler = Plate(name="Tripler", E=1.03e7, t=t_tripler, first_row=1, last_row=5, A_strip=[0.279]*4)
    # Note: User JSON says Tripler 1-5. Screenshot says 3-6. 
    # Let's match the User's JSON topology first since that's what produced their result.
    # JSON: Tripler 1-5. Doubler 2-6. Skin 1-6.
    
    plates = [tripler, doubler, skin]
    
    # Fasteners
    # Rows 2, 3, 4, 5
    fasteners = []
    for r in [2, 3, 4, 5]:
        f = FastenerRow(
            row=r, 
            D=0.344, 
            Eb=1.06e7, 
            nu_b=0.33, 
            method="Boeing69",
            connections=[(0, 1), (1, 2)] # Tripler-Doubler, Doubler-Skin
        )
        fasteners.append(f)
        
    # Supports
    # User screenshot shows supports at Node 6 for Tripler and Doubler.
    # But Tripler ends at 5 in JSON? 
    # If Tripler ends at 5, it can't be supported at 6.
    # Let's assume the JSON meant Tripler 3-6 (length 3) or 2-6?
    # Boeing screenshot: Tripler nodes 1003, 1004, 1005, 1006. (Rows 3, 4, 5, 6).
    # Doubler nodes 2002, 2003, 2004, 2005, 2006. (Rows 2, 3, 4, 5, 6).
    # Skin nodes 3001...3006. (Rows 1...6).
    
    # Let's adjust topology to match Boeing Screenshot exactly.
    # Pitches: 1.128 (from screenshot)
    pitches = [1.128] * 5 # 5 bays: 1-2, 2-3, 3-4, 4-5, 5-6
    
    # Skin: Row 1 to 6.
    skin = Plate(name="Skin", E=1.03e7, t=0.429, first_row=1, last_row=6, A_strip=[0.695]*5)
    # Doubler: Row 2 to 6.
    doubler = Plate(name="Doubler", E=1.03e7, t=t_doubler, first_row=2, last_row=6, A_strip=[0.26]*4)
    # Tripler: Row 3 to 6.
    tripler = Plate(name="Tripler", E=1.03e7, t=t_tripler, first_row=3, last_row=6, A_strip=[0.279]*3)
    
    plates = [tripler, doubler, skin]
    
    # Fasteners at 2, 3, 4, 5?
    # Screenshot:
    # Row 2: Skin-Doubler. (No Tripler).
    # Row 3: Skin-Doubler-Tripler.
    # Row 4: Skin-Doubler-Tripler.
    # Row 5: Skin-Doubler-Tripler.
    # Row 6: Support.
    
    fasteners = []
    # Row 2: Doubler(1) - Skin(2)
    fasteners.append(FastenerRow(row=2, D=0.344, Eb=1.06e7, nu_b=0.33, method="Boeing69", connections=[(1, 2)]))
    # Row 3, 4, 5: Tripler(0)-Doubler(1)-Skin(2)
    for r in [3, 4, 5]:
        fasteners.append(FastenerRow(row=r, D=0.344, Eb=1.06e7, nu_b=0.33, method="Boeing69", connections=[(0, 1), (1, 2)]))
        
    # Supports at Node 6 (End of plates)
    # Tripler (0) at local node 3 (start=3, 3+3=6)
    # Doubler (1) at local node 4 (start=2, 2+4=6)
    supports = [
        (0, 3, 0.0), # Tripler right end
        (1, 4, 0.0)  # Doubler right end
    ]
    
    # Load: 1000 lb on Skin at Node 1 (Left end)
    # Skin (2) at local node 0
    point_forces = [
        (2, 0, 1000.0)
    ]
    
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports, point_forces=point_forces)
    
    # Get reactions
    react_tripler = solution.reactions.get((0, 3), 0.0)
    react_doubler = solution.reactions.get((1, 4), 0.0)
    
    print(f"Reaction Tripler: {react_tripler:.2f}")
    print(f"Reaction Doubler: {react_doubler:.2f}")
    print(f"Ratio (Tripler/Doubler): {react_tripler/react_doubler:.2f}")

# Case A: User Inputs (0.14, 0.13)
run_case(0.14, 0.13, "User Inputs (0.14 / 0.13)")

# Case B: Boeing Nodes Table (0.063, 0.040)
run_case(0.063, 0.040, "Boeing Nodes (0.063 / 0.040)")
