"""PCA via SVD (registered in decomposition.DECOMPOSERS; UI may omit this method)."""

from __future__ import annotations

from typing import Any

import numpy as np


def decompose(A: np.ndarray) -> dict[str, Any]:
    raise NotImplementedError("PCA (SVD) from DECOMPOSERS is not wired in the Streamlit METHOD_ORDER.")
