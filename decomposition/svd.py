"""Singular value decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np


def decompose(A: np.ndarray, *, full_matrices: bool = False) -> dict[str, Any]:
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)

    if full_matrices:
        k = min(m, n)
        S = np.zeros((m, n), dtype=s.dtype)
        S[:k, :k] = np.diag(s)
    else:
        S = np.diag(s)

    reconstructed = U @ S @ Vt
    return {
        "method": "SVD",
        "factors": {"U": U, "Sigma": S, "V.T": Vt},
        "reconstructed": reconstructed,
        "explain_steps": [
            "**A = U Σ Vᵀ** where **U** and **V** have orthonormal columns/rows and **Σ** contains nonnegative singular values.",
            "Reduced mode keeps Σ diagonal with shape min(m,n) for a compact factorization.",
        ],
        "metadata": {"singular_values": s, "full_matrices": full_matrices},
    }
