from jolt.model import Plate, FastenerRow, Joint1D
import pandas as pd

def reproduce_inbd():
    # Reconstructed from screenshots
    # Pitches: 1.750
    # Rows: 7 (based on x=12.25 max, 1.75 step)
    # x coords: 1.75, 3.5, 5.25, 7.0, 8.75, 10.5, 12.25
    pitches = [1.750] * 7
    
    E_plate = 1.030e7
    E_bolt = 1.600e7
    diameter = 0.344
    
    # Plate 1 (Top / Tripler equivalent?): 200x series
    # Nodes 2002-2006. x=1.75 to 8.75?
    # Screenshot 2: 2002 (1.75), 2003 (3.5), 2004 (5.25), 2005 (7.0), 2006 (8.75)
    # Rows 1 to 5?
    # t=0.140
    # A_strip: Screenshot 2 shows "Area" 0.279 for 2003, 2004, 2005, 2006.
    # 2002 area is 0.279?
    
    plate1 = Plate(
        name="Tripler",
        E=E_plate,
        t=0.140,
        first_row=1,
        last_row=5,
        A_strip=[0.279] * 4 # 4 segments for 5 nodes
    )
    
    # Plate 2 (Mid / Doubler equivalent?): 300x series
    # Nodes 3003-3006. x=3.5 to 8.75?
    # Screenshot 2: 3003 (3.5), 3004 (5.25), 3005 (7.0), 3006 (8.75), 3007 (10.5)?
    # Wait, Screenshot 2 shows 3003, 3004, 3005, 3006, 3007 (10.5)
    # Rows 2 to 6?
    # t=0.130
    # A_strip: 0.260
    
    plate2 = Plate(
        name="Doubler",
        E=E_plate,
        t=0.130,
        first_row=2,
        last_row=6,
        A_strip=[0.260] * 4
    )
    
    # Plate 3 (Bot / Skin equivalent?): 400x series
    # Nodes 4002-4007. x=1.75 to 10.5?
    # Screenshot 2: 4002 (1.75) ... 4007 (10.5)
    # Rows 1 to 6?
    # t=0.429
    # A_strip: 0.695
    
    plate3 = Plate(
        name="Skin",
        E=E_plate,
        t=0.429,
        first_row=1,
        last_row=6,
        A_strip=[0.695] * 5,
        Fx_left=0.0, # Where is load applied?
        Fx_right=1000.0 # Guessing load application based on arrows
    )
    
    # Adjust load application based on Screenshot 1
    # Arrows at right end of 300x (Doubler) and 400x (Skin)?
    # "Net Bypass Load" at 3007 is 0? at 4007 is 0?
    # Screenshot 1 shows arrows pointing RIGHT at node 7 (x=12.25?)
    # But my nodes end at 6 (x=10.5) for Skin?
    # Let's check x locations again.
    # 2002=1.75 (Row 1)
    # 2006=8.75 (Row 5)
    # 3003=3.5 (Row 2)
    # 3007=10.5 (Row 6)
    # 4002=1.75 (Row 1)
    # 4007=10.5 (Row 6)
    
    # Fasteners:
    # Rows 2, 3, 4, 5.
    # Screenshot 1 shows fasteners at 2, 3, 4, 5, 6?
    # Screenshot 5 (My App) shows F1..F6.
    # F1 at Row 2.
    # F2 at Row 2.
    # Wait, my app shows F1, F2 at same x? No, F1 at n2, F2 at n2?
    # Ah, my app shows 3 plates.
    # F1 connects Tripler-Doubler at n2.
    # F2 connects Doubler-Skin at n2.
    # This means separate fasteners?
    # Screenshot 2 (Reference) shows:
    # 2003-3003 (Row 2, Top-Mid)
    # 3003-4003 (Row 2, Mid-Bot)
    # So they are modeled as separate fasteners in the reference?
    
    # My code models a "FastenerRow" which connects all plates at that row.
    # If I use FastenerRow, it creates springs between all adjacent plates.
    # And my fix sets shear_planes = len(pairs).
    # Here len(pairs) = 2 (Top-Mid, Mid-Bot).
    # So shear_planes = 2.
    
    # But if the reference models them as independent 2-plate connections, then shear_planes should be 1 for each!
    
    plates = [plate1, plate2, plate3]
    
    fasteners = []
    # Rows 2, 3, 4, 5 have all 3 plates?
    # Row 2 (x=3.5): 2003, 3003, 4003. Yes.
    # Row 3 (x=5.25): 2004, 3004, 4004. Yes.
    # Row 4 (x=7.0): 2005, 3005, 4005. Yes.
    # Row 5 (x=8.75): 2006, 3006, 4006. Yes.
    
    # Row 6 (x=10.5): 3007, 4007. Only Doubler and Skin.
    # 200x ends at 2006 (Row 5).
    
    for r in range(2, 6):
        fasteners.append(FastenerRow(row=r, D=diameter, Eb=E_bolt, nu_b=0.3, method="Boeing69"))
        
    # Row 6
    fasteners.append(FastenerRow(row=6, D=diameter, Eb=E_bolt, nu_b=0.3, method="Boeing69"))
    
    # Supports
    # Screenshot 1: Green triangle at 2002 (Row 1, Tripler)
    # Green triangle at 4002 (Row 1, Skin)
    supports = [
        (0, 0, 0.0), # Tripler, local node 0 (Row 1)
        (2, 0, 0.0)  # Skin, local node 0 (Row 1)
    ]
    
    # Loads
    # Screenshot 1: Arrows at right.
    # 3007 (Doubler end) -> 272.0 lb?
    # 4007 (Skin end) -> 728.0 lb?
    # Total 1000 lb.
    
    # Doubler is plate 1. Last node is local node 4 (Row 6 - Row 2 = 4).
    # Skin is plate 2. Last node is local node 5 (Row 6 - Row 1 = 5).
    
    plate2.Fx_right = 272.0
    plate3.Fx_right = 728.0
    
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    solution = model.solve(supports=supports)
    
    print("Fastener Stiffnesses and Loads:")
    for f in solution.fasteners:
        print(f"Row {f.row} ({f.plate_i}-{f.plate_j}): k={f.stiffness:.2e}, F={f.force:.2f}")

if __name__ == "__main__":
    reproduce_inbd()
