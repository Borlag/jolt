from jolt import Joint1D, Plate, FastenerRow
from jolt.linalg import LinearSystemError

def test_singular_matrix():
    # Create a model with a floating plate (no supports)
    pitches = [1.0]
    plates = [
        Plate(name="Floating", E=1e7, t=0.1, first_row=1, last_row=2, A_strip=[0.1])
    ]
    fasteners = []
    # Support is empty -> Rigid body motion -> Singular matrix
    supports = []
    
    model = Joint1D(pitches=pitches, plates=plates, fasteners=fasteners)
    
    try:
        model.solve(supports=supports)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except LinearSystemError as e:
        print(f"Caught expected LinearSystemError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")
    else:
        print("Failed to catch error! Model solved unexpectedly.")

def test_invalid_fastener_inputs():
    from jolt.fasteners import boeing69_compliance
    try:
        boeing69_compliance(0.1, 1e7, 0.1, 1e7, 1e7, 0.3, 0.0) # Diameter 0
    except ValueError as e:
        print(f"Caught expected ValueError for diameter: {e}")
    else:
        print("Failed to catch error for invalid diameter!")

if __name__ == "__main__":
    test_singular_matrix()
    test_invalid_fastener_inputs()
