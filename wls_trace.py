"""Full WLS trace for D06_3."""
import json
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jolt.fasteners import boeing69_compliance


def solve_dense(A, b):
    """Simple Gaussian elimination for small dense systems."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        
        for row in range(col + 1, n):
            if M[col][col] != 0:
                factor = M[row][col] / M[col][col]
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]
    
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if M[i][i] != 0:
            x[i] /= M[i][i]
    return x


def wls_trace():
    config_path = Path(__file__).parent / "test_values" / "D06_3_config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Extract fastener params (row 2)
    fastener = config["fasteners"][0]
    D = fastener["D"]
    Eb = fastener["Eb"]
    nu_b = fastener.get("nu_b", 0.3)
    
    # Extract plates at row 2
    plates = []
    for p in config["plates"]:
        if p["first_row"] <= 2 <= p["last_row"]:
            plates.append(p)
    
    print(f"=== WLS TRACE FOR D06_3 ROW 2 ===")
    print(f"Plates: {[p['name'] for p in plates]}")
    
    # Compute pairwise compliances (Boeing 69)
    print("\n--- PAIRWISE COMPLIANCES ---")
    pairwise = []
    for i in range(len(plates) - 1):
        p_i = plates[i]
        p_j = plates[i + 1]
        C_pair = boeing69_compliance(
            ti=p_i["t"], Ei=p_i["E"],
            tj=p_j["t"], Ej=p_j["E"],
            Eb=Eb, nu_b=nu_b,
            diameter=D, shear_planes=1
        )
        pairwise.append(C_pair)
        print(f"  {p_i['name'][:8]:8}-{p_j['name'][:8]:8}: C_pair = {C_pair:.6e}  K_pair = {1.0/C_pair:.0f}")
    
    # Compute base compliances (with double shear fix)
    print("\n--- BASE COMPLIANCES (with fix) ---")
    G_b = Eb / (2.0 * (1.0 + nu_b))
    A_b = math.pi * (D / 2.0) ** 2
    I_b = math.pi * (D / 2.0) ** 4 / 4.0
    
    base = []
    for i, p in enumerate(plates):
        t = p["t"]
        E = p["E"]
        
        t_shear = min(t, (D + t) / 2.0)
        C_s = (4.0 * t_shear) / (9.0 * G_b * A_b)
        
        t_bend = min(t, D)
        C_b = (6.0 * t_bend**3) / (40.0 * Eb * I_b)
        
        C_brg = (1.0 / t) * (1.0 / Eb + 1.0 / E)
        
        is_internal = (0 < i < len(plates) - 1)
        
        if is_internal:
            C_base = C_s + C_b + (C_brg / 2.0)  # Double shear: halved bearing
        else:
            C_base = C_s + C_b + C_brg  # Single shear
        
        base.append(C_base)
        print(f"  Branch {i} ({p['name'][:8]:8}): C_base = {C_base:.6e}  K_base = {1.0/C_base:.0f}  {'(internal)' if is_internal else '(external)'}")
    
    # Build and solve WLS system
    print("\n--- WLS SYSTEM ---")
    n_pairs = len(pairwise)
    n_branches = n_pairs + 1
    
    # Build matrix M
    matrix_M = [[0.0] * n_pairs for _ in range(n_pairs)]
    for i in range(n_pairs):
        matrix_M[i][i] = base[i] + base[i+1]
        if i > 0:
            matrix_M[i][i-1] = base[i]
        if i < n_pairs - 1:
            matrix_M[i][i+1] = base[i+1]
    
    print("Matrix M:")
    for row in matrix_M:
        print("  " + "  ".join(f"{x:.3e}" for x in row))
    
    # Build RHS
    rhs_vec = []
    for i in range(n_pairs):
        c_base_sum = base[i] + base[i+1]
        excess = c_base_sum - pairwise[i]
        rhs_vec.append(excess)
    
    print(f"\nRHS (excess = base_sum - pairwise):")
    for i, x in enumerate(rhs_vec):
        print(f"  [{i}] = {base[i]:.3e} + {base[i+1]:.3e} - {pairwise[i]:.3e} = {x:.3e}")
    
    # Solve
    lambdas = solve_dense(matrix_M, rhs_vec)
    print(f"\nLambdas (Lagrange multipliers):")
    for i, lam in enumerate(lambdas):
        print(f"  lambda[{i}] = {lam:.6f}")
    
    # Apply corrections
    print("\n--- FINAL BRANCH COMPLIANCES ---")
    final = list(base)
    final[0] -= base[0] * lambdas[0]
    for i in range(1, n_branches - 1):
        final[i] -= base[i] * (lambdas[i-1] + lambdas[i])
    final[-1] -= base[-1] * lambdas[-1]
    
    # Clamp negatives
    for i, c in enumerate(final):
        if c < 0:
            final[i] = 0.0
    
    for i, p in enumerate(plates):
        print(f"  Branch {i} ({p['name'][:8]:8}): C_branch = {final[i]:.6e}  K_branch = {1.0/final[i] if final[i] > 0 else float('inf'):.0f}")
    
    # Verify constraints
    print("\n--- CONSTRAINT VERIFICATION ---")
    for i in range(n_pairs):
        c_sum = final[i] + final[i+1]
        c_target = pairwise[i]
        error_pct = 100 * abs(c_sum - c_target) / c_target if c_target > 0 else 0
        print(f"  Pair {i}: C[{i}]+C[{i+1}] = {c_sum:.6e} vs target {c_target:.6e}  (error = {error_pct:.2f}%)")


if __name__ == "__main__":
    wls_trace()
