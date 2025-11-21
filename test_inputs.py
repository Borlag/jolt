from jolt import Plate, FastenerRow
from jolt.inputs import process_refined_rows, process_node_based

def test_refined_rows():
    print("Testing Refined Rows...")
    # Original: Nodes at 0 (Row 1), 10 (Row 2). Pitch 10.
    pitches = [10.0] 
    plates = [Plate("Skin", 1e7, 0.1, 1, 2, [1.0])] # Spans 0 -> 10
    fasteners = [FastenerRow(row=2, D=0.2, Eb=1e7, nu_b=0.3)] # At Row 2 (10.0)
    
    extra_nodes = [5.0] # Add node at 5.0
    
    new_pitches, new_plates, new_fasteners = process_refined_rows(pitches, plates, fasteners, extra_nodes)
    
    # Expected nodes: 0, 5, 10.
    # Expected pitches: 5, 5.
    print(f"Pitches: {new_pitches}")
    assert new_pitches == [5.0, 5.0]
    
    # Plate was 1->2 (0->10). Now should be 1->3 (0->10).
    print(f"Plate rows: {new_plates[0].first_row} -> {new_plates[0].last_row}")
    assert new_plates[0].first_row == 1
    assert new_plates[0].last_row == 3
    
    # Fastener was at 2 (10.0). Now should be at 3 (10.0).
    print(f"Fastener row: {new_fasteners[0].row}")
    assert new_fasteners[0].row == 3
    
    print("Refined Rows Test Passed!")

def test_node_based():
    print("\nTesting Node-based...")
    nodes = {0: 0.0, 1: 5.0, 2: 10.0}
    elements = [
        {"layer_name": "Skin", "start_node": 0, "end_node": 1, "E": 1e7, "t": 0.1, "width": 1.0},
        {"layer_name": "Skin", "start_node": 1, "end_node": 2, "E": 1e7, "t": 0.1, "width": 1.0},
    ]
    fasteners = [
        {"node_id": 1, "diameter": 0.2, "connected_layers": ["Skin", "Doubler"]}
    ]
    
    pitches, plates, new_fasteners = process_node_based(nodes, elements, fasteners)
    
    # Expected pitches: 5.0, 5.0
    print(f"Pitches: {pitches}")
    assert pitches == [5.0, 5.0]
    
    # Elements should be merged into one Plate spanning 0->2 (Row 1->3)
    print(f"Plates: {len(plates)}")
    assert len(plates) == 1
    print(f"Plate rows: {plates[0].first_row} -> {plates[0].last_row}")
    assert plates[0].first_row == 1
    assert plates[0].last_row == 3
    
    # Fastener at Node 1 (x=5.0) -> Row 2
    # But connected_layers=["Skin", "Doubler"]. We only have "Skin".
    # So fastener should NOT be created (needs >= 2 plates).
    print(f"Fasteners: {len(new_fasteners)}")
    assert len(new_fasteners) == 0
    
    print("Node-based Test Passed!")

if __name__ == "__main__":
    test_refined_rows()
    test_node_based()
