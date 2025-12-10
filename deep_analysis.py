"""
Deep analysis of the branch decomposition algorithm.
This script traces through the exact same logic as the solver but with detailed output.
"""
import json
import math
from pathlib import Path

def boeing69_compliance(ti, Ei, tj, Ej, Eb, nu_b, diameter, shear_planes=1):
    """Return fastener compliance according to Boeing (1969) (D6-29942)."""
    area_bolt = math.pi * diameter**2 / 4.0
    inertia_bolt = math.pi * diameter**4 / 64.0
    shear_modulus = Eb / (2.0 * (1.0 + nu_b))

    # Shear Term
    term_shear = 4.0 * (ti + tj) / (9.0 * shear_modulus * area_bolt)

    # Bending Term
    term_bending = (
        ti**3 + 5.0*ti**2*tj + 5.0*ti*tj**2 + tj**3
    ) / (40.0 * Eb * inertia_bolt)

    # Bearing Term
    div = float(1 if shear_planes <= 1 else 2 * shear_planes)
    term_bearing = (
        (1.0 / ti) * (1.0/Eb + 1.0/Ei) + 
        (1.0 / tj) * (1.0/Eb + 1.0/Ej)
    ) / div

    return term_shear + term_bending + term_bearing


def calculate_base_compliance(t, E_plate, D, Eb, nu_b):
    """Calculate the 'Base' (Single) compliance of a plate branch."""
    A_b = math.pi * (D / 2.0) ** 2
    I_b = math.pi * (D / 2.0) ** 4 / 4.0
    G_b = Eb / (2.0 * (1.0 + nu_b))
    
    # Shear Term (Boeing Single Model uses t_shear = min(t, (D+t)/2))
    t_shear = min(t, (D + t) / 2.0)
    C_shear = (4.0 * t_shear) / (9.0 * G_b * A_b)
    
    # Bending Term (Boeing Single Model uses t_bending = min(t, D))
    t_bending = min(t, D)
    C_bending = (6.0 * t_bending**3) / (40.0 * Eb * I_b)
    
    # Bearing Term
    C_bearing = (1.0 / t) * (1.0 / Eb + 1.0 / E_plate)
        
    return C_shear + C_bending + C_bearing


def solve_branch_compliances_current(n_branches, pairwise_compliances, base_compliances):
    """
    Current WLS decomposition implementation.
    """
    n_pairs = len(pairwise_compliances)
    
    # Case 1: 2-Layer Stack
    if n_branches == 2:
        c_pair = pairwise_compliances[0]
        return [c_pair / 2.0, c_pair / 2.0]
    
    # Case 2: 3-Layer Stack (Boeing Symmetric Star / JOLT Logic)
    if n_branches == 3:
        C01, C12 = pairwise_compliances[0], pairwise_compliances[1]
        C_min, C_max = min(C01, C12), max(C01, C12)
        
        # Symmetric Star Decomposition
        C_mid = (2.0 * C_min - C_max) / 2.0
        
        # Enforce non-negative
        if C_mid < 0.0: 
            C_mid = 0.0
            
        branch_compliances = [0.0] * 3
        branch_compliances[1] = C_mid
        branch_compliances[0] = C01 - C_mid
        branch_compliances[2] = C12 - C_mid
        
        return [max(c, 1e-12) for c in branch_compliances]

    # Case 3: N > 3 Layers (Constrained Least Squares)
    # Build matrix M
    matrix_M = [[0.0] * n_pairs for _ in range(n_pairs)]
    for i in range(n_pairs):
        matrix_M[i][i] = 2.0
        if i > 0:
            matrix_M[i][i-1] = 1.0
        if i < n_pairs - 1:
            matrix_M[i][i+1] = 1.0
            
    rhs_vec = []
    for i in range(n_pairs):
        c_base_sum = base_compliances[i] + base_compliances[i+1]
        excess = pairwise_compliances[i] - c_base_sum
        rhs_vec.append(excess)
        
    # Solve using simple Gaussian elimination
    lambdas = solve_dense(matrix_M, rhs_vec)
        
    final_compliances = list(base_compliances)
    final_compliances[0] += lambdas[0]
    for i in range(1, n_branches - 1):
        final_compliances[i] += (lambdas[i-1] + lambdas[i])
    final_compliances[-1] += lambdas[-1]

    for i, c in enumerate(final_compliances):
        if c < 0.0:
            final_compliances[i] = 0.0
    
    return final_compliances


def solve_dense(A, b):
    """Simple Gaussian elimination with partial pivoting."""
    n = len(b)
    # Make copies to avoid mutating originals
    A = [row[:] for row in A]
    b = list(b)
    
    # Forward elimination
    for k in range(n):
        # Partial pivoting
        max_idx = k
        for i in range(k+1, n):
            if abs(A[i][k]) > abs(A[max_idx][k]):
                max_idx = i
        A[k], A[max_idx] = A[max_idx], A[k]
        b[k], b[max_idx] = b[max_idx], b[k]
        
        if abs(A[k][k]) < 1e-15:
            continue
            
        for i in range(k+1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        if abs(A[i][i]) > 1e-15:
            x[i] /= A[i][i]
    
    return x


def analyze_row(config, row_num, reference):
    """Analyze a specific row from the configuration."""
    plates = config["plates"]
    fasteners = config["fasteners"]
    
    # Find fastener at this row
    fastener = None
    for f in fasteners:
        if f["row"] == row_num:
            fastener = f
            break
    
    if not fastener:
        print(f"No fastener at row {row_num}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS: Row {row_num}")
    print(f"{'='*80}")
    
    # Get plates at this row
    plates_at_row = []
    for idx, p in enumerate(plates):
        if p["first_row"] <= row_num <= p["last_row"]:
            plates_at_row.append((idx, p))
    
    print(f"\nPlates at row: {[p['name'] for _, p in plates_at_row]}")
    print(f"Connections: {fastener.get('connections', 'auto')}")
    
    # Get connection pairs
    connections = fastener.get("connections")
    if connections:
        connection_pairs = [tuple(c) for c in connections]
    else:
        connection_pairs = [(plates_at_row[i][0], plates_at_row[i+1][0]) for i in range(len(plates_at_row)-1)]
    
    print(f"Connection pairs: {connection_pairs}")
    
    D = fastener["D"]
    Eb = fastener["Eb"]
    nu_b = fastener.get("nu_b", 0.3)
    
    # Calculate pairwise compliances
    print(f"\n--- Pairwise Compliances ---")
    pairwise_compliances = []
    plate_lookup = {idx: p for idx, p in plates_at_row}
    
    for idx_i, idx_j in connection_pairs:
        p_i = plate_lookup.get(idx_i) or plates[idx_i]
        p_j = plate_lookup.get(idx_j) or plates[idx_j]
        t_i = p_i["t"]
        t_j = p_j["t"]
        E_i = p_i["E"]
        E_j = p_j["E"]
        
        comp = boeing69_compliance(t_i, E_i, t_j, E_j, Eb, nu_b, D, shear_planes=1)
        k = 1.0 / comp
        pairwise_compliances.append(comp)
        
        print(f"  {p_i['name']:>12} - {p_j['name']:<12}: C = {comp:.6e}, K = {k:.0f}")
    
    # Get ordered plates
    ordered_indices = list(set([i for pair in connection_pairs for i in pair]))
    ordered_indices.sort()
    
    # Calculate base compliances
    print(f"\n--- Base Compliances ---")
    base_compliances = []
    for idx in ordered_indices:
        p = plate_lookup.get(idx) or plates[idx]
        t = p["t"]
        E_plate = p["E"]
        
        base_c = calculate_base_compliance(t, E_plate, D, Eb, nu_b)
        base_compliances.append(base_c)
        
        print(f"  {p['name']:>12}: C_base = {base_c:.6e}, K_base = {1.0/base_c:.0f}")
    
    # Solve branch compliances
    n_branches = len(ordered_indices)
    print(f"\n--- Branch Decomposition (N={n_branches}) ---")
    
    branch = solve_branch_compliances_current(n_branches, pairwise_compliances, base_compliances)
    
    print(f"\n  Current Method:")
    for idx, c in zip(ordered_indices, branch):
        p = plate_lookup.get(idx) or plates[idx]
        print(f"    {p['name']:>12}: C_branch = {c:.6e}, K_branch = {1.0/c:.0f}")
    
    # Verify sum constraints
    print(f"\n  Constraint Verification:")
    for i, (idx_i, idx_j) in enumerate(connection_pairs):
        p_i = plate_lookup.get(idx_i) or plates[idx_i]
        p_j = plate_lookup.get(idx_j) or plates[idx_j]
        
        # Find branch indices
        bi = ordered_indices.index(idx_i)
        bj = ordered_indices.index(idx_j)
        
        sum_branch = branch[bi] + branch[bj]
        target = pairwise_compliances[i]
        error = 100 * abs(sum_branch - target) / target if target > 0 else 0
        
        print(f"    C_{p_i['name'][:3]} + C_{p_j['name'][:3]} = {sum_branch:.6e} (target: {target:.6e}, error: {error:.2f}%)")
    
    # Compare with reference
    ref_boeing = reference.get("formulas", {}).get("boeing", {})
    ref_fasteners = ref_boeing.get("fasteners", [])
    
    print(f"\n--- Comparison with Boeing JOLT Reference ---")
    
    for rf in ref_fasteners:
        if rf["row"] != row_num:
            continue
        
        ref_force = abs(rf.get("force", 0))
        ref_k = rf.get("stiffness", 0)
        
        print(f"  {rf['plate_i']:>12} - {rf['plate_j']:<12}: Force = {ref_force:.1f}, K = {ref_k:.0f}")


def main():
    # Load D06_4 which has the most complex topology
    test_values_dir = Path(__file__).parent / "test_values"
    
    config_path = test_values_dir / "D06_4_config.json"
    ref_path = test_values_dir / "D06_4_reference.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    with open(ref_path) as f:
        reference = json.load(f)
    
    print(f"Analyzing: {config.get('label', config_path.stem)}")
    print(f"Plates: {[p['name'] for p in config['plates']]}")
    
    # Analyze each row
    for fastener in config["fasteners"]:
        analyze_row(config, fastener["row"], reference)


if __name__ == "__main__":
    main()
