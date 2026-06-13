"""Numerical diagnostics for precomputed kernel matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KernelDiagnostics:
    """Compact numerical health checks for a kernel matrix."""

    shape: tuple[int, int]
    symmetry_error: float
    min_diagonal: float
    max_diagonal: float
    approximate_min_eigenvalue: float | None
    condition_warning: str | None


def diagnose_kernel_matrix(
    kernel: np.ndarray,
    eigenvalue_limit: int = 500,
) -> KernelDiagnostics:
    """Compute readable numerical checks for a square kernel matrix."""

    matrix = np.asarray(kernel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Kernel matrix must be square.")
    if matrix.shape[0] == 0:
        raise ValueError("Kernel matrix must not be empty.")

    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    diagonal = np.diag(matrix)
    min_eigenvalue = None
    if matrix.shape[0] <= eigenvalue_limit:
        symmetric_matrix = (matrix + matrix.T) / 2
        min_eigenvalue = float(np.linalg.eigvalsh(symmetric_matrix)[0])

    warnings = []
    if not np.all(np.isfinite(matrix)):
        warnings.append("contains non-finite values")
    if symmetry_error > 1e-8:
        warnings.append("is not numerically symmetric")
    if float(diagonal.min()) <= 0:
        warnings.append("has non-positive diagonal entries")
    if min_eigenvalue is not None and min_eigenvalue < -1e-8:
        warnings.append("has a materially negative eigenvalue")
    max_absolute = float(np.max(np.abs(matrix)))
    nonzero_absolute = np.abs(matrix[np.nonzero(matrix)])
    if nonzero_absolute.size and max_absolute / float(nonzero_absolute.min()) > 1e12:
        warnings.append("has a very large value range")

    return KernelDiagnostics(
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        symmetry_error=symmetry_error,
        min_diagonal=float(diagonal.min()),
        max_diagonal=float(diagonal.max()),
        approximate_min_eigenvalue=min_eigenvalue,
        condition_warning="; ".join(warnings) if warnings else None,
    )
