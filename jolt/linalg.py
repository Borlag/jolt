"""Lightweight linear algebra helpers used by the solver."""
from __future__ import annotations

from typing import Iterable, List, Sequence


class LinearSystemError(RuntimeError):
    """Raised when the linear system cannot be solved."""


def _augment(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[List[float]]:
    return [list(row) + [value] for row, value in zip(matrix, rhs)]


def solve_dense(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Solve a dense linear system using Gauss-Jordan elimination.

    The implementation performs partial pivoting to maintain numerical
    stability.  ``matrix`` is not mutated; the solver operates on an augmented
    copy of the system.  ``LinearSystemError`` is raised if the matrix is
    singular.
    """

    n = len(matrix)
    if n == 0:
        return []
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")
    if len(rhs) != n:
        raise ValueError("Right-hand side dimension mismatch")

    augmented = _augment(matrix, rhs)

    for pivot_index in range(n):
        pivot_row = max(range(pivot_index, n), key=lambda r: abs(augmented[r][pivot_index]))
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < 1e-12:
            raise LinearSystemError("Singular matrix: zero pivot encountered")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for col in range(pivot_index, n + 1):
            augmented[pivot_index][col] /= pivot_value

        for row in range(n):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            if abs(factor) < 1e-18:
                continue
            for col in range(pivot_index, n + 1):
                augmented[row][col] -= factor * augmented[pivot_index][col]

    return [augmented[row][n] for row in range(n)]


def extract_submatrix(matrix: Sequence[Sequence[float]], rows: Iterable[int], cols: Iterable[int]) -> List[List[float]]:
    return [[matrix[r][c] for c in cols] for r in rows]


def extract_vector(vector: Sequence[float], indices: Iterable[int]) -> List[float]:
    return [vector[i] for i in indices]


__all__ = [
    "LinearSystemError",
    "solve_dense",
    "extract_submatrix",
    "extract_vector",
]
