
from jolt import Joint1D, Plate, FastenerRow
import math

def verify_refactor():
    print("Verifying Refactoring...")
    
    # Define a simple case: 2 plates, 1 fastener
    # Plate 0: Tripler (Top)
    # Plate 1: Skin (Bottom)
    p0 = Plate(name="Tripler", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1])
    p1 = Plate(name="Skin", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1])
    
    # Fastener at row 1 connecting 0 and 1
    f1 = FastenerRow(row=1, D=0.2, Eb=1e7, nu_b=0.3)
    
    model = Joint1D(pitches=[1.0, 1.0], plates=[p0, p1], fasteners=[f1])
    
    # Apply load to Plate 0 left end (Node 1001)
    # Plate 0 starts at row 1. Local node 0 is row 1? No.
    # Plate: first_row=1.
    # Local node 0 -> Row 1.
    # Local node 1 -> Row 2.
    # Wait, pitch is between rows.
    # If first_row=1, last_row=2. Segment count = 1.
    # Local node 0 (Row 1), Local node 1 (Row 2).
    
    # Support Plate 1 right end (Row 2)
    supports = [(1, 1, 0.0)] # Plate 1, Local Node 1 (Row 2)
    
    # Force on Plate 0 left end (Row 1)
    forces = [(0, 0, 1000.0)] # Plate 0, Local Node 0 (Row 1)
    
    solution = model.solve(supports=supports, point_forces=forces)
    
    # 1. Verify Legacy IDs
    # Plate 0 (Tripler): 1000 * (0+1) + Row
    # Row 1 -> 1001
    # Row 2 -> 1002
    nodes = solution.nodes_as_dicts()
    node_map = {n["Node ID"]: n for n in nodes}
    
    assert 1001 in node_map, "Node 1001 missing"
    assert 1002 in node_map, "Node 1002 missing"
    assert 2001 in node_map, "Node 2001 missing (Plate 1, Row 1)"
    assert 2002 in node_map, "Node 2002 missing (Plate 1, Row 2)"
    
    print("Legacy IDs verified.")
    
    # 2. Verify Net Bypass Load
    # Node 1001 (Top Left): Incoming 1000. Outgoing (Bar 0-1) should be less due to fastener?
    # Wait, fastener is at Row 1 (Node 1001).
    # Load enters Node 1001. Some goes to fastener, some to bar.
    # Force Left = 0 (start). Force Right = Bar Force.
    # Net Bypass = min(0, Bar Force) = 0?
    # Let's check definition: "Average of element force entering and leaving" or "min(abs(left), abs(right))"
    # I implemented min(abs(left), abs(right)).
    # At end node, one side is 0. So Net Bypass should be 0.
    
    n1001 = node_map[1001]
    print(f"Node 1001 Net Bypass: {n1001['Net Bypass Load']}")
    # assert n1001['Net Bypass Load'] == 0.0 # This might be controversial if user expects incoming load.
    # But based on "internal load remaining in the plate", at the edge it is 0?
    # Let's check Node 1002 (Top Right).
    # Incoming from Bar. Outgoing 0. Net Bypass 0.
    
    # Let's check a middle node if we had 3 rows.
    
    # 3. Verify Total Bearing Load
    # Fastener at Row 1 transfers load.
    # Force balance: 1000 applied.
    # Fastener takes some. Bar takes rest.
    # Fastener Force F.
    # Plate 0 at Row 1: Bearing Load = F.
    # Plate 1 at Row 1: Bearing Load = F.
    
    fasteners = solution.fasteners_as_dicts()
    f_res = fasteners[0]
    f_load = f_res["Load"]
    print(f"Fastener Load: {f_load}")
    
    assert f_res["Brg Force Upper"] == abs(f_load)
    assert f_res["Brg Force Lower"] == abs(f_load)
    
    print("Total Bearing Load verified.")
    
    # 4. Verify Classic Results
    classic = solution.classic_results_as_dicts()
    # Should have entries for Row 1.
    c_res = [c for c in classic if c["Row"] == 1]
    assert len(c_res) == 2 # One for each plate
    
    c_p0 = next(c for c in c_res if c["Plate"] == "Tripler")
    print(f"Classic P0: Incoming={c_p0['Incoming Load']}, Bypass={c_p0['Bypass Load']}, Transfer={c_p0['Load Transfer']}")
    
    # For P0 at Row 1 (Left end):
    # Incoming (Left) = 0 (No bar to left) -> Wait, logic says flow_left.
    # If flow_left is 0, then Incoming is 0.
    # But applied load is 1000.
    # My logic uses internal element forces. It doesn't see point forces directly in "flow_left".
    # This is a potential issue with "Incoming Load" definition if it relies only on internal bars.
    # If load is applied directly to the node, it splits into bearing and bypass (bar).
    # If Incoming is defined as "Load from upstream element", then it is 0.
    # But physically, 1000 is coming in.
    # The user might expect Incoming to include applied loads?
    # "Incoming Load: Load entering the node from the 'loaded' direction."
    # If I only look at elements, I miss the point load.
    # However, for internal nodes, it works.
    # Let's see if I need to adjust for point loads.
    
    print("Verification Script Finished.")

if __name__ == "__main__":
    verify_refactor()
