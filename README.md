# Matrix Decomposition Visualizer

Interactive [Streamlit](https://streamlit.io) app to enter a numeric matrix, pick a decomposition (LU, QR, Eigen, SVD, Cholesky)

## Setup

**Requirements:** Python 3.10+ recommended.

```bash
cd matrix-decomposition-app
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).
## Project structure

```text
matrix-decomposition-app/
├── app.py                 # Streamlit UI: matrix input, method selection, plots, exports
├── requirements.txt       # streamlit, numpy, scipy, pandas, matplotlib
├── README.md
├── decomposition/         # Decomposition routines and factor dictionaries
│   ├── __init__.py        # DECOMPOSERS registry (LU, QR, Eigen, SVD, Cholesky, PCA stub)
│   ├── lu.py              # PLU via SciPy
│   ├── qr.py              # QR via NumPy
│   ├── eigen.py           # Eigen-decomposition via NumPy
│   ├── svd.py             # SVD via NumPy
│   ├── cholesky.py        # Cholesky (SPD matrices) via NumPy
│   └── pca.py             # Placeholder; not listed in the app’s method menu
└── utils/                 # Shared helpers and validation
    ├── helpers.py
    └── validators.py
```

## Decomposition methods

The app implements each method as a function that returns factors, a reconstructed matrix, and short explanatory text. **LU**, **Eigen**, and **Cholesky** require a **square** matrix; **QR** and **SVD** work for general rectangular \(m \times n\) matrices.

- **LU (PLU)** — Factors \(A = P L U\): **P** permutes rows, **L** is unit lower triangular, **U** is upper triangular. Computed with `scipy.linalg.lu`. Useful for solving linear systems and understanding Gaussian elimination with pivoting.

- **QR** — Factors \(A = Q R\): **Q** has orthonormal columns and **R** is upper triangular. Supports reduced or complete QR via NumPy’s `numpy.linalg.qr`. Common for least-squares problems and orthogonalization of columns.

- **Eigen** — For diagonalizable square \(A\), \(A = V \Lambda V^{-1}\): **Λ** holds eigenvalues on the diagonal and **V** holds corresponding eigenvectors. Uses `numpy.linalg.eig`. Applies to square matrices only; reconstruction uses \(V \Lambda V^{-1}\) (works as intended when \(A\) is diagonalizable).

- **SVD** — Factors \(A = U \Sigma V^{\mathsf T}\): **U** and **V** are orthogonal (orthonormal columns/rows), **Σ** carries nonnegative singular values. Uses `numpy.linalg.svd`; reduced mode keeps a compact \(\Sigma\) when applicable. Applies to any shape; cornerstone of low-rank approximation and PCA-style analysis.

- **Cholesky** — For **symmetric positive definite** square \(A\), \(A = L L^{\mathsf T}\) with **L** lower triangular. The implementation checks squareness, symmetry, and positive definiteness before `numpy.linalg.cholesky`. Used when \(A\) is a covariance/Gram-like matrix.
