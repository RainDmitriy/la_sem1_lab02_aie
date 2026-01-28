# linalg.py
from CSC import CSCMatrix
from CSR import CSRMatrix
from my_types import Vector
from typing import Tuple, Optional

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    # Decompose A into L and U
    matrix = A.to_dense()
    n = A.shape[0]
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            sum_lu = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = matrix[i][j] - sum_lu
        for j in range(i + 1, n):
            if U[i][i] == 0: return None
            sum_lu = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (matrix[j][i] - sum_lu) / U[i][i]

    return CSCMatrix.from_dense(L), CSCMatrix.from_dense(U)

def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    # Solve Ax = b via LU
    res = lu_decomposition(A)
    if not res: return None
    L_mat, U_mat = res
    L, U = L_mat.to_dense(), U_mat.to_dense()
    n = len(b)

    y = [0.0 for _ in range(n)]
    for i in range(n):
        y[i] = b[i] - sum(L[i][k] * y[k] for k in range(i))

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0: return None
        x[i] = (y[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    # det(A) = det(L) * det(U)
    res = lu_decomposition(A)
    if not res: return 0.0
    _, U_mat = res
    U = U_mat.to_dense()
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return det