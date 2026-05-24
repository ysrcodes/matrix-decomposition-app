def decompose(A: np.ndarray) -> dict[str, Any]:
    pass
# eigen.py

import numpy as np


def computing_eigen(matrix):

    # convert to numpy array
    A = np.array(matrix, dtype=float)

    rows = A.shape[0]
    cols = A.shape[1]

    # The condition for eigen decomposition is that the amtrix should be square matrix
    if rows != cols:
        raise ValueError("Matrix should be square.")

    # calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # verifying
    D = np.diag(eigenvalues)

    verify = eigenvectors @ D @ np.linalg.inv(eigenvectors)

    return {
        "Eigenvalues": eigenvalues,
        "Eigenvectors": eigenvectors,
        "Verification": verify
    }


# testing

matrix = [
    [5, 7, 10],
    [1, 4, 11],
    [9, 17, 21]
]

try: 
     result = computing_eigen(matrix)

     print("Original Matrix:")
     print(np.array(matrix))

     print("\nEigenvalues:")
     print(result["Eigenvalues"])

     print("\nEigenvectors:")
     print(result["Eigenvectors"])

     print("\nVerifying Matrix:")
     print(result["Verification"])

except Exception as e:

    print("Error:", e)
