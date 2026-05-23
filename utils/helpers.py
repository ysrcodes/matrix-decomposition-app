from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np

from utils.validators import DEFAULT_ATOL, DEFAULT_RTOL


def round_matrix(A: np.ndarray, decimals: int = 6) -> np.ndarray:
    if np.iscomplexobj(A):
        return np.round(A.real, decimals) + 1j * np.round(A.imag, decimals)
    return np.round(A, decimals).astype(float)


def reconstruction_metrics(
    A: np.ndarray,
    A_hat: np.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    if np.iscomplexobj(A_hat) and not np.iscomplexobj(A):
        A = A.astype(np.complex128)
    diff = A - A_hat
    max_abs = float(np.max(np.abs(diff)))
    frob = float(np.linalg.norm(diff, ord="fro"))
    close = bool(np.allclose(A, A_hat, rtol=rtol, atol=atol))
    return {
        "allclose": close,
        "max_abs_error": max_abs,
        "frobenius_error": frob,
        "rtol": rtol,
        "atol": atol,
    }


def matrix_to_csv_bytes(A: np.ndarray, *, name: str = "matrix") -> tuple[bytes, str]:
    """Return CSV bytes and suggested filename for download."""
    buf = StringIO()
    if np.iscomplexobj(A):
        stacked = np.vstack([A.real, A.imag])
        np.savetxt(buf, stacked, delimiter=",")
    else:
        np.savetxt(buf, A, delimiter=",")
    data = buf.getvalue().encode("utf-8")
    fname = f"{name}.csv"
    return data, fname
