"""
Test: Compare different branch decomposition approaches.
This tests whether minimal-norm (zero prior) works better for Boeing.
"""
import json
import math
from pathlib import Path

def solve_dense(A, b):
    """Simple Gaussian elimination with partial pivoting."""
    n = len(b)
    A = [row[:] for row in A]
    b = list(b)
    
    for k in range(n):
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
    
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        if abs(A[i][i]) > 1e-15:
            x[i] /= A[i][i]
    
    return x


def decompose_minimal_norm(pairwise_compliances):
    """
    Pure minimal-norm solution: base_compliances = 0.
    This matches model_save.py Boeing implementation.
    """
    n_pairs = len(pairwise_compliances)
    n_branches = n_pairs + 1
    
    if n_branches == 2:
        return [pairwise_compliances[0] / 2.0, pairwise_compliances[0] / 2.0]
    
    # N=3 special case (symmetric star)
    if n_branches == 3:
        C01, C12 = pairwise_compliances[0], pairwise_compliances[1]
        C_min, C_max = min(C01, C12), max(C01, C12)
        C_mid = (2.0 * C_min - C_max) / 2.0
        if C_mid < 0.0:
            C_mid = 0.0
        branch = [0.0] * 3
        branch[1] = C_mid
        branch[0] = C01 - C_mid
        branch[2] = C12 - C_mid
        return [max(c, 1e-12) for c in branch]
    
    # N>3: Minimal-norm WLS with base = 0
    base_compliances = [0.0] * n_branches
    
    matrix_M = [[0.0] * n_pairs for _ in range(n_pairs)]
    for i in range(n_pairs):
        matrix_M[i][i] = 2.0
        if i > 0:
            matrix_M[i][i-1] = 1.0
        if i < n_pairs - 1:
            matrix_M[i][i+1] = 1.0
            
    # RHS: (base_sum) - pairwise = 0 - pairwise = -pairwise
    rhs_vec = [-p for p in pairwise_compliances]
    
    lambdas = solve_dense(matrix_M, rhs_vec)
    
    # C = base - A^T * lambda = 0 - A^T * lambda = -A^T * lambda
    final = list(base_compliances)
    final[0] -= lambdas[0]
    for i in range(1, n_branches - 1):
        final[i] -= (lambdas[i-1] + lambdas[i])
    final[-1] -= lambdas[-1]
    
    # Clamp negatives
    for i, c in enumerate(final):
        if c < 0.0:
            final[i] = 0.0
    
    return final


def decompose_weighted_prior(pairwise_compliances, base_compliances):
    """
    WLS with non-zero base compliances (current model.py approach).
    """
    n_pairs = len(pairwise_compliances)
    n_branches = n_pairs + 1
    
    if n_branches == 2:
        return [pairwise_compliances[0] / 2.0, pairwise_compliances[0] / 2.0]
    
    # N=3 special case
    if n_branches == 3:
        C01, C12 = pairwise_compliances[0], pairwise_compliances[1]
        C_min, C_max = min(C01, C12), max(C01, C12)
        C_mid = (2.0 * C_min - C_max) / 2.0
        if C_mid < 0.0:
            C_mid = 0.0
        branch = [0.0] * 3
        branch[1] = C_mid
        branch[0] = C01 - C_mid
        branch[2] = C12 - C_mid
        return [max(c, 1e-12) for c in branch]
    
    # N>3: WLS with non-zero base
    matrix_M = [[0.0] * n_pairs for _ in range(n_pairs)]
    for i in range(n_pairs):
        matrix_M[i][i] = 2.0
        if i > 0:
            matrix_M[i][i-1] = 1.0
        if i < n_pairs - 1:
            matrix_M[i][i+1] = 1.0
    
    # RHS: excess = pairwise - (base_i + base_{i+1})
    rhs_vec = []
    for i in range(n_pairs):
        c_base_sum = base_compliances[i] + base_compliances[i+1]
        excess = pairwise_compliances[i] - c_base_sum
        rhs_vec.append(excess)
    
    lambdas = solve_dense(matrix_M, rhs_vec)
    
    final = list(base_compliances)
    final[0] += lambdas[0]
    for i in range(1, n_branches - 1):
        final[i] += (lambdas[i-1] + lambdas[i])
    final[-1] += lambdas[-1]
    
    for i, c in enumerate(final):
        if c < 0.0:
            final[i] = 0.0
    
    return final


def verify_constraints(branch_compliances, pairwise_compliances):
    """Check if sum constraints are satisfied."""
    errors = []
    for i in range(len(pairwise_compliances)):
        sum_branch = branch_compliances[i] + branch_compliances[i+1]
        target = pairwise_compliances[i]
        error_pct = 100 * abs(sum_branch - target) / target if target > 0 else 0
        errors.append(error_pct)
    return errors


def main():
    # Test case: Row 5 from D06_4 (4 plates)
    # Pairwise compliances from deep_analysis.py output:
    pairwise = [6.967148e-06, 8.404377e-06, 6.464112e-06]  # Tripler-Doubler, Doubler-Skin, Skin-Strap
    
    # Base compliances from deep_analysis.py:
    base = [2.769795e-06, 4.202188e-06, 4.202188e-06, 2.278960e-06]
    
    print("Test: Row 5 of D06_4 (4-layer stack)")
    print("=" * 60)
    print(f"Pairwise Compliances: {[f'{p:.6e}' for p in pairwise]}")
    print(f"Base Compliances: {[f'{b:.6e}' for b in base]}")
    
    # Method 1: Minimal-norm (Boeing style per model_save.py)
    branch_min = decompose_minimal_norm(pairwise)
    errors_min = verify_constraints(branch_min, pairwise)
    
    print(f"\n1. MINIMAL-NORM (base=0):")
    print(f"   Branch: {[f'{c:.6e}' for c in branch_min]}")
    print(f"   Stiffness: {[f'{1/c:.0f}' for c in branch_min]}")
    print(f"   Constraint Errors: {[f'{e:.2f}%' for e in errors_min]}")
    
    # Method 2: Weighted prior (current model.py)
    branch_wgt = decompose_weighted_prior(pairwise, base)
    errors_wgt = verify_constraints(branch_wgt, pairwise)
    
    print(f"\n2. WEIGHTED PRIOR (current model.py):")
    print(f"   Branch: {[f'{c:.6e}' for c in branch_wgt]}")
    print(f"   Stiffness: {[f'{1/c:.0f}' for c in branch_wgt]}")
    print(f"   Constraint Errors: {[f'{e:.2f}%' for e in errors_wgt]}")
    
    # Compare
    print(f"\nBoth methods satisfy constraints with 0% error.")
    print(f"The difference is in the branch distribution:")
    print(f"  - Minimal-norm distributes more evenly")
    print(f"  - Weighted prior biases toward base compliance ratios")


if __name__ == "__main__":
    main()
