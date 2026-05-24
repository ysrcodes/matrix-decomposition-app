from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import streamlit as st


def decompose(A: np.ndarray, *, mode: str = "reduced") -> dict[str, Any]:
    A = np.asarray(A, dtype=float)
    nm = "reduced" if mode == "reduced" else "complete"
    Q, R = np.linalg.qr(A, mode=nm)
    reconstructed = Q @ R
    return {
        "method": "QR",
        "factors": {"Q": Q, "R": R},
        "reconstructed": reconstructed,
        "explain_steps": [
            "**A = Q·R** with **Q** having orthonormal columns and **R** upper triangular.",
        ],
        "metadata": {"qr_mode": nm},
    }


def run_qr_workflow(
    matrix: np.ndarray,
    *,
    result: dict[str, Any] | None = None,
    mode: str = "reduced",
    embedded: bool = False,
    display_factor: Callable[[str, np.ndarray], None] | None = None,
) -> None:

    try:
        res = (
            result
            if result is not None
            else decompose(np.asarray(matrix, dtype=float), mode=mode)
        )

        Q, R = res["factors"]["Q"], res["factors"]["R"]
        reconstructed = np.asarray(res["reconstructed"])
        matrix_f = np.asarray(matrix, dtype=float)

        def _show_q_r() -> None:
            col1, col2 = st.columns(2)
            with col1:
                if display_factor is not None:
                    display_factor("Q", np.asarray(Q))
                else:
                    st.write("**Orthogonal Matrix ($Q$)**")
                    st.dataframe(Q)
            with col2:
                if display_factor is not None:
                    display_factor("R", np.asarray(R))
                else:
                    st.write("**Upper Triangular Matrix ($R$)**")
                    st.dataframe(R)

        if embedded:
            _show_q_r()
            return

        st.markdown("---")
        st.subheader("📐 QR Decomposition Results")
        st.success("🎉 QR Decomposition completed successfully!")

        _show_q_r()

        rec_error = float(np.linalg.norm(matrix_f - reconstructed))
        st.markdown("### 🔍 Reconstruction Verification")
        st.metric(label="Reconstruction Error (Frobenius Norm)", value=f"{rec_error:.2e}")

        if np.allclose(matrix_f, reconstructed, atol=1e-9):
            st.info("✅ **Verification Pass:** $Q \\times R$ perfectly reconstructs the original matrix $A$.")
        else:
            st.warning("⚠️ **Verification Warning:** High reconstruction error detected.")

    except Exception as e:
        st.error(f"❌ An unexpected numerical error occurred during computation: {str(e)}")
