from __future__ import annotations

from io import BytesIO, StringIO
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
MAX_DIM = 20
MIN_DIM = 1


class ValidationError(ValueError):
    "Raised when user matrix or file input fails validation."


def ensure_finite_numeric(A: np.ndarray) -> np.ndarray:
    if A.dtype.kind not in "iufc":
        raise ValidationError("Matrix entries must be numeric.")
    if not np.isfinite(A).all():
        raise ValidationError("Matrix contains NaN or non-finite values.")
    return A.astype(np.float64, copy=False)


def is_square(A: np.ndarray) -> bool:
    return A.ndim == 2 and A.shape[0] == A.shape[1]


def is_symmetric(A: np.ndarray, tol: float = DEFAULT_ATOL) -> bool:
    if not is_square(A):
        return False
    return bool(np.allclose(A, A.T, rtol=DEFAULT_RTOL, atol=tol))


def is_spd(A: np.ndarray, tol: float = DEFAULT_ATOL) -> bool:
    """True if A is (numerically) symmetric and strictly positive definite."""
    if not is_symmetric(A, tol=tol):
        return False
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(w > tol))


def validate_matrix(
    A: np.ndarray,
    *,
    require_square: bool = False,
    min_rows: int = MIN_DIM,
    max_rows: int = MAX_DIM,
    min_cols: int = MIN_DIM,
    max_cols: int = MAX_DIM,
) -> np.ndarray:
    if A.ndim != 2:
        raise ValidationError("Matrix must be 2-dimensional.")
    m, n = A.shape
    if m < min_rows or m > max_rows or n < min_cols or n > max_cols:
        raise ValidationError(
            f"Dimensions must be between {min_rows}×{min_cols} and "
            f"{max_rows}×{max_cols} (got {m}×{n})."
        )
    A = ensure_finite_numeric(A)
    if require_square and not is_square(A):
        raise ValidationError("This decomposition requires a square matrix.")
    return A


def load_matrix_from_csv(
    file_bytes: bytes,
    *,
    header: str = "infer",
) -> np.ndarray:
    """Load a numeric matrix from CSV bytes. Uses pandas for robust parsing."""
    try:
        bio = BytesIO(file_bytes)
        df = pd.read_csv(bio, header=header if header != "none" else None)
    except Exception as e:
        raise ValidationError(f"Could not parse CSV: {e}") from e

    # If header=None, columns are 0..n-1; coerce all to numeric
    try:
        mat = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    except Exception as e:
        raise ValidationError(f"CSV could not be converted to numeric array: {e}") from e

    if mat.ndim != 2:
        raise ValidationError("CSV must produce a 2D table.")
    if np.isnan(mat).any():
        raise ValidationError("CSV contains non-numeric or empty cells.")
    return mat


def dataframe_to_matrix(df: Any) -> np.ndarray:
    """Convert Streamlit data_editor DataFrame to float ndarray."""
    try:
        mat = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    except Exception as e:
        raise ValidationError(f"Could not read matrix from editor: {e}") from e
    if np.isnan(mat).any():
        raise ValidationError("Manual matrix has empty or non-numeric cells.")
    return ensure_finite_numeric(mat)
