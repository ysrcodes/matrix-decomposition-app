from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg
import streamlit as st


def decompose(A: np.ndarray) -> dict[str, Any]:
    A = np.asarray(A, dtype=float)
    P, L, U = scipy.linalg.lu(A)
    reconstructed = P @ L @ U
    return {
        "method": "LU",
        "factors": {"P": P, "L": L, "U": U},
        "reconstructed": reconstructed,
        "explain_steps": [
            "**A = P·L·U** with **P** a permutation matrix, **L** unit lower triangular, **U** upper triangular.",
        ],
        "metadata": {},
    }


def run_lu_workflow(
    matrix: np.ndarray,
    *,
    result: dict[str, Any] | None = None,
    embedded: bool = False,
    display_factor: Callable[[str, np.ndarray], None] | None = None,
) -> None:
    res = result
    if res is None:
        if matrix.shape[0] != matrix.shape[1]:
            st.error(
                "❌ Validation Error: LU Decomposition can only be performed on a square matrix ($N \\times N$)."
            )
            return
        try:
            res = decompose(np.asarray(matrix, dtype=float))
        except Exception as e:
            st.error(f"❌ An unexpected numerical error occurred during computation: {str(e)}")
            return

    P, L, U = res["factors"]["P"], res["factors"]["L"], res["factors"]["U"]
    reconstructed = np.asarray(res["reconstructed"])
    matrix_f = np.asarray(matrix, dtype=float)

    try:

        def _show_p_l_u() -> None:
            col1, col2, col3 = st.columns(3)
            with col1:
                if display_factor is not None:
                    display_factor("P", np.asarray(P))
                else:
                    st.write("**Permutation Matrix ($P$)**")
                    st.dataframe(P)
            with col2:
                if display_factor is not None:
                    display_factor("L", np.asarray(L))
                else:
                    st.write("**Lower Triangular ($L$)**")
                    st.dataframe(L)
            with col3:
                if display_factor is not None:
                    display_factor("U", np.asarray(U))
                else:
                    st.write("**Upper Triangular ($U$)**")
                    st.dataframe(U)

        if embedded:
            _show_p_l_u()
            return

        st.markdown("---")
        st.subheader("🤖 LU Decomposition Results")
        st.success("🎉 LU Decomposition completed successfully!")

        _show_p_l_u()

        rec_error = float(np.linalg.norm(matrix_f - reconstructed))
        st.markdown("### 🔍 Reconstruction Verification")
        st.metric(label="Reconstruction Error (Frobenius Norm)", value=f"{rec_error:.2e}")

        if np.allclose(matrix_f, reconstructed, atol=1e-9):
            st.info("✅ **Verification Pass:** $P \\times L \\times U$ perfectly reconstructs the original matrix $A$.")
        else:
            st.warning(
                "⚠️ **Verification Warning:** High reconstruction error detected. Check matrix numerical stability."
            )

    except Exception as e:
        st.error(f"❌ An unexpected numerical error occurred during computation: {str(e)}")
