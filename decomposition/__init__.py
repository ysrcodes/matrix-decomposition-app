"""Matrix decomposition implementations (LU, QR, Eigen, SVD, Cholesky, PCA)."""

from decomposition.cholesky import decompose as cholesky_decompose
from decomposition.eigen import decompose as eigen_decompose
from decomposition.lu import decompose as lu_decompose
from decomposition.pca import decompose as pca_decompose
from decomposition.qr import decompose as qr_decompose
from decomposition.svd import decompose as svd_decompose

DECOMPOSERS = {
    "LU": lu_decompose,
    "QR": qr_decompose,
    "Eigen": eigen_decompose,
    "SVD": svd_decompose,
    "Cholesky": cholesky_decompose,
    "PCA (SVD)": pca_decompose,
}

__all__ = [
    "DECOMPOSERS",
    "cholesky_decompose",
    "eigen_decompose",
    "lu_decompose",
    "pca_decompose",
    "qr_decompose",
    "svd_decompose",
]
