def decompose(A: np.ndarray) -> dict[str, Any]:
    pass

# cholesky.py

import numpy as np


def computing_cholesky_decomposition(matrix):

    # convert the matrix into numpy array
    A = np.array(matrix, dtype=float)

    rows = A.shape[0]
    cols = A.shape[1]

    # Cholesky Decomposition requires somme conditions to be fulfilled

    # 1. check if the matrix is a square matrix
    if rows != cols:
        raise ValueError("Matrix should be square.")

    # 2. check if the matrix is symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix should be symmetric.")

    # 3. check if the matrix is positive definite i.e eigenvalues > 0
    eigenvalues = np.linalg.eigvals(A)

    for value in eigenvalues:
        if value <= 0:
            raise ValueError("Matrix is not positive definite.")

    # If all conditions are fulfilled, we can begin cholesky decomposition
    L = np.linalg.cholesky(A)
    
    # confirming matrix to check the decomposition is correct
    L = np.linalg.cholesky(A)
    verify = L @ L.T

    return {
        "L": L,
        "Verify": verify,
    }


# using a matrix to check the output is right

matrix = [
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ]
 

try : 
  
     result = computing_cholesky_decomposition(matrix)

     print("The Original Matrix:")
     print(np.array(matrix))

     print("\nL Matrix:")
     print(result["L"])

     print("\nVerifying Matrix:")
     print(result["Verify"])

except Exception as e:

    print("Error:", e)
