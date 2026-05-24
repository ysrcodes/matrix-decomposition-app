"""
Matrix Decomposition Visualizer — Streamlit entrypoint.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from decomposition import DECOMPOSERS
from decomposition.qr import decompose as qr_decompose
from decomposition.svd import decompose as svd_decompose
from utils.helpers import matrix_to_csv_bytes, reconstruction_metrics, round_matrix
from utils.validators import (
    ValidationError,
    dataframe_to_matrix,
    is_square,
    load_matrix_from_csv,
    validate_matrix,
)

SQUARE_METHODS = {"LU", "Eigen", "Cholesky"}
METHOD_ORDER = ["LU", "QR", "Eigen", "SVD", "Cholesky"]


def _display_matrix(label: str, M: np.ndarray, *, decimals: int, show_heatmap: bool) -> None:
    st.subheader(label)
    rounded = round_matrix(M, decimals=decimals)
    if np.iscomplexobj(rounded):
        st.caption("Complex matrix: **real** and **imaginary** parts.")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Real")
            st.dataframe(pd.DataFrame(np.real(rounded)), use_container_width=True)
        with c2:
            st.write("Imag")
            st.dataframe(pd.DataFrame(np.imag(rounded)), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame(rounded), use_container_width=True)

    if show_heatmap and M.ndim == 2 and not np.iscomplexobj(M):
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(M.real.astype(float), cmap="viridis", aspect="auto")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)


def _download_factors(factors: dict[str, np.ndarray], prefix: str, *, widget_nonce: int) -> None:
    cols = st.columns(min(3, len(factors)))
    for i, (name, arr) in enumerate(factors.items()):
        key_safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        data, fname = matrix_to_csv_bytes(np.asarray(arr), name=f"{prefix}_{key_safe}")
        with cols[i % len(cols)]:
            st.download_button(
                label=f"Download {name}",
                data=data,
                file_name=fname,
                mime="text/csv",
                key=f"dl_{widget_nonce}_{prefix}_{key_safe}_{i}",
            )


def _random_matrix(
    rows: int,
    cols: int,
    *,
    rng: np.random.Generator,
    kind: str,
) -> np.ndarray:
    if kind == "uniform [-1, 1]":
        return rng.uniform(-1.0, 1.0, size=(rows, cols))
    if kind == "normal N(0,1)":
        return rng.normal(size=(rows, cols))
    if kind == "SPD (square)":
        if rows != cols:
            raise ValidationError("SPD random matrix requires square dimensions.")
        q, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
        d = np.diag(rng.uniform(0.5, 2.0, size=rows))
        return q @ d @ q.T
    if kind == "rank-deficient (duplicate row)":
        base = rng.normal(size=(rows, cols))
        if rows >= 2:
            base[1] = base[0]
        return base
    if kind == "integer small":
        return rng.integers(-5, 6, size=(rows, cols)).astype(float)
    return rng.uniform(-1.0, 1.0, size=(rows, cols))


def _applicable_methods(A: np.ndarray) -> list[str]:
    sq = is_square(A)
    return [m for m in METHOD_ORDER if m not in SQUARE_METHODS or sq]


def _run_comparison(A: np.ndarray, methods: list[str]) -> list[tuple[str, float, str]]:
    results: list[tuple[str, float, str]] = []
    for name in methods:
        fn: Callable[..., dict[str, Any]] = DECOMPOSERS[name]
        t0 = time.perf_counter()
        err_msg = ""
        try:
            if name == "QR":
                fn(A, mode="reduced")
            elif name == "SVD":
                fn(A, full_matrices=False)
            else:
                fn(A)
        except Exception as e:
            err_msg = str(e)
        dt = time.perf_counter() - t0
        results.append((name, dt, err_msg))
    return results


st.set_page_config(page_title="Matrix Decomposition Visualizer", layout="wide")
st.title("Matrix Decomposition Visualizer")
st.caption("LU · QR · Eigen · SVD · Cholesky")

with st.sidebar:
    st.header("Input")
    rows = st.number_input("Rows", min_value=1, max_value=20, value=3, step=1)
    cols = st.number_input("Columns", min_value=1, max_value=20, value=3, step=1)
    input_mode = st.radio("Matrix source", ["Manual editor", "Upload CSV"], horizontal=False)

    if "matrix_data" not in st.session_state:
        st.session_state.matrix_data = np.zeros((int(rows), int(cols)))
    if "matrix_version" not in st.session_state:
        st.session_state.matrix_version = 0

    if st.button("Sync dimensions to editor"):
        st.session_state.matrix_data = np.zeros((int(rows), int(cols)))
        st.session_state.matrix_version += 1

    rand_col1, rand_col2 = st.columns(2)
    with rand_col1:
        seed = st.number_input("RNG seed", value=42, step=1)
    with rand_col2:
        kind = st.selectbox(
            "Random preset",
            [
                "uniform [-1, 1]",
                "normal N(0,1)",
                "integer small",
                "SPD (square)",
                "rank-deficient (duplicate row)",
            ],
        )
    if st.button("Fill random matrix"):
        rng = np.random.default_rng(int(seed))
        try:
            st.session_state.matrix_data = _random_matrix(int(rows), int(cols), rng=rng, kind=kind)
            st.session_state.matrix_version += 1
            st.success("Random matrix applied.")
        except ValidationError as e:
            st.error(str(e))

manual_matrix = st.session_state.matrix_data
if manual_matrix.shape != (int(rows), int(cols)):
    manual_matrix = np.zeros((int(rows), int(cols)))
    st.session_state.matrix_data = manual_matrix
    st.session_state.matrix_version += 1

A_input: np.ndarray | None = None
edited_manual: pd.DataFrame | None = None

if input_mode == "Manual editor":
    st.subheader("Manual matrix")
    init_df = pd.DataFrame(manual_matrix)
    edited_manual = st.data_editor(
        init_df,
        num_rows="fixed",
        use_container_width=True,
        key=f"matrix_editor_{int(rows)}_{int(cols)}_v{st.session_state.matrix_version}",
    )
    st.caption("Values in the table are used when you click **Compute decomposition**.")
else:
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    header_opt = st.selectbox("CSV header", ["infer", "none"], index=0)
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            A_input = load_matrix_from_csv(raw, header="infer" if header_opt == "infer" else "none")
            if A_input.shape[0] != rows or A_input.shape[1] != cols:
                st.info(
                    f"CSV shape {A_input.shape[0]}×{A_input.shape[1]} — validation uses actual CSV dimensions."
                )
        except ValidationError as e:
            st.error(str(e))
            A_input = None

st.divider()
col_m, col_opt = st.columns((1.2, 1))
with col_m:
    method = st.selectbox("Decomposition method", METHOD_ORDER)
with col_opt:
    show_heatmap = st.checkbox("Heatmap (real matrices)", value=False)
    verify = st.checkbox("Show reconstruction verification", value=True)
    decimals = st.slider("Display decimals", 2, 10, 6)

qr_mode = "reduced"
svd_full = False

with st.expander("Method options"):
    qr_mode = st.select_slider("QR mode", options=["reduced", "complete"], value="reduced")
    svd_full = st.checkbox("SVD full_matrices", value=False)

if st.button("Compute decomposition", type="primary"):
    try:
        if input_mode == "Manual editor":
            if edited_manual is None:
                raise ValidationError("Open the manual matrix editor.")
            A_raw = dataframe_to_matrix(edited_manual)
        else:
            if A_input is None:
                raise ValidationError("Upload a CSV matrix.")
            A_raw = A_input
        req_square = method in SQUARE_METHODS
        A = validate_matrix(A_raw, require_square=req_square)
        if method not in _applicable_methods(A):
            raise ValidationError(
                f"{method} requires a square matrix. Current shape: {A.shape}."
            )

        t0 = time.perf_counter()
        if method == "QR":
            result = qr_decompose(A, mode=qr_mode)
        elif method == "SVD":
            result = svd_decompose(A, full_matrices=svd_full)
        else:
            result = DECOMPOSERS[method](A)
        elapsed = time.perf_counter() - t0

        st.success(f"Computed **{result['method']}** in {elapsed * 1000:.3f} ms")
        _display_matrix("A (input)", A, decimals=decimals, show_heatmap=show_heatmap)

        factors: dict[str, np.ndarray] = result["factors"]
        for name, mat in factors.items():
            _display_matrix(name, np.asarray(mat), decimals=decimals, show_heatmap=show_heatmap)

        if verify and "reconstructed" in result and result["reconstructed"] is not None:
            Ah = np.asarray(result["reconstructed"])
            _display_matrix("Reconstructed A_hat", Ah, decimals=decimals, show_heatmap=show_heatmap)
            metrics = reconstruction_metrics(A, Ah)
            c1, c2, c3 = st.columns(3)
            c1.metric("allclose", str(metrics["allclose"]))
            c2.metric("max |error|", f"{metrics['max_abs_error']:.2e}")
            c3.metric("Frobenius ||A-Â||", f"{metrics['frobenius_error']:.2e}")
            st.caption(
                f"Tolerance rtol={metrics['rtol']}, atol={metrics['atol']} — Eigen may deviate if rank-defective or ill-conditioned."
            )

        with st.expander("Step-by-step explanation"):
            for line in result.get("explain_steps", []):
                st.markdown(f"- {line}")
        if "metadata" in result and result["metadata"]:
            st.json({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result["metadata"].items()})

        st.subheader("Download factors (CSV)")
        if "download_nonce" not in st.session_state:
            st.session_state.download_nonce = 0
        st.session_state.download_nonce += 1
        _download_factors(
            factors,
            prefix=result["method"].replace(" ", "_"),
            widget_nonce=st.session_state.download_nonce,
        )

    except ValidationError as ve:
        st.error(str(ve))
    except np.linalg.LinAlgError as le:
        st.error(f"Linear algebra error: {le}")
    except Exception as ex:
        st.exception(ex)

with st.expander("Benchmark applicable methods on current matrix"):
    st.caption("Uses the matrix from manual editor / last successful CSV parse in session.")
    if st.button("Run timing comparison"):
        try:
            if input_mode == "Manual editor":
                if edited_manual is None:
                    raise ValidationError("No manual matrix in editor.")
                A_raw = dataframe_to_matrix(edited_manual)
            else:
                if A_input is None:
                    raise ValidationError("Upload a CSV first.")
                A_raw = A_input
            A = validate_matrix(A_raw, require_square=False)
            applicable = _applicable_methods(A)
            times = _run_comparison(A, applicable)
            df = pd.DataFrame(
                [{"method": n, "seconds": t, "error": err or None} for n, t, err in times]
            ).sort_values("seconds")
            st.dataframe(df, use_container_width=True)
        except ValidationError as e:
            st.warning(str(e))

st.markdown(
    "---\n**Tips:** Cholesky needs symmetric positive definite **A**. "
    "LU/Eigen/Cholesky expect square **A**. QR and SVD support rectangular matrices."
)
