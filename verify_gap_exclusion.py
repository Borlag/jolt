
from jolt import Joint1D, Plate, FastenerRow
import math

def verify_gap_exclusion():
    print("Verifying Gap Exclusion from Min/Max...")
    
    # Define a case with a gap
    # Plate 0: Continuous
    # Plate 1: Has a gap in second segment
    p0 = Plate(name="P0", E=1e7, t=0.1, first_row=1, last_row=3, A_strip=[0.1, 0.1])
    p1 = Plate(name="P1", E=1e7, t=0.1, first_row=1, last_row=3, A_strip=[0.1, 0.0]) # Gap in segment 2
    
    # Fasteners
    f1 = FastenerRow(row=1, D=0.2, Eb=1e7, nu_b=0.3)
    f2 = FastenerRow(row=2, D=0.2, Eb=1e7, nu_b=0.3)
    
    model = Joint1D(pitches=[1.0, 1.0], plates=[p0, p1], fasteners=[f1, f2])
    
    supports = [(0, 0, 0.0), (0, 2, 0.0), (1, 2, 0.0)] # Fix P0 ends and P1 right end (after gap)
    forces = [(1, 0, 1000.0)] # Pull P1
    
    solution = model.solve(supports=supports, point_forces=forces)
    
    # Check Bars
    bars = solution.bars_as_dicts()
    
    # P1 Segment 2 (Gap) should have 0 force and 0 stiffness
    gap_bar = next(b for b in bars if b["Plate"] == "P1" and b["seg"] == 1)
    print(f"Gap Bar Stiffness: {gap_bar['Stiffness']}")
    print(f"Gap Bar Force: {gap_bar['Force']}")
    
    assert gap_bar['Stiffness'] == 0.0
    
    # P1 Segment 1 should have force
    loaded_bar = next(b for b in bars if b["Plate"] == "P1" and b["seg"] == 0)
    print(f"Loaded Bar Force: {loaded_bar['Force']}")
    
    # Min/Max Logic Simulation
    active_bars = [b for b in solution.bars if b.stiffness > 1e-9]
    
    max_load = max([abs(b.axial_force) for b in active_bars]) if active_bars else 0.0
    min_load = min([abs(b.axial_force) for b in active_bars]) if active_bars else 0.0
    
    print(f"Calculated Max Load: {max_load}")
    print(f"Calculated Min Load: {min_load}")
    
    # Min load should NOT be 0 (unless a real bar has 0 force, but here P0 carries load too)
    # P0 segments carry reaction.
    # P1 seg 1 carries load.
    # Gap carries 0.
    # If gap was included, min load would be 0.
    # If gap excluded, min load should be > 0 (assuming all other bars are loaded).
    
    assert min_load > 0.0, "Min load is 0.0, meaning gap might be included or a real bar has 0 force."
    
    print("Gap Exclusion Verified.")

if __name__ == "__main__":
    verify_gap_exclusion()
